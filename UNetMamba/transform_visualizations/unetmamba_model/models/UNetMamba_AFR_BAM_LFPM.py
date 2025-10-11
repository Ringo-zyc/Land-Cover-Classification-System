# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint  # 用于梯度检查点
import os
import math
from functools import partial  # 用于偏函数
from typing import Optional, List, Tuple, Union  # 类型提示

from einops import rearrange  # 张量操作库
from timm.models.layers import DropPath, trunc_normal_  # 来自timm库的常用层

# --- 尝试导入项目特定的模块 ---
try:
    # 假设 VSSM (Vision State Space Model) 相关组件在此路径
    from unetmamba_model.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
except ImportError as e:
    print(f"警告: 无法导入 VSSM 组件: {e}。将使用 Identity 占位符。")
    VSSM = VSSBlock = LayerNorm2d = Permute = nn.Identity

try:
    # 导入 ResT 模型
    from unetmamba_model.models.ResT import ResT
except ImportError as e:
    print(f"警告: 无法导入 ResT: {e}。将使用占位符。")
    # 定义一个基本的 ResT 占位符，以防原始 ResT 无法导入
    class ResT(nn.Module):
        def __init__(self, embed_dims=None, **kwargs):
            super().__init__()
            print("警告: 正在使用 ResT 占位符。")
            self.stages = nn.ModuleList()
            in_c = 3  # 输入通道数
            if embed_dims is None:
                embed_dims = [64, 128, 256, 512]  # 默认嵌入维度
            self.output_channels = embed_dims  # 记录输出通道，方便后续使用
            # 创建模拟的编码器阶段
            for i, out_c in enumerate(embed_dims):
                pool = nn.MaxPool2d(2, 2) if i < len(embed_dims) - 1 else nn.Identity()
                stage = nn.Sequential(nn.Conv2d(in_c, out_c, 3, 1, 1), nn.ReLU(), pool)
                self.stages.append(stage)
                in_c = out_c  # 更新下一阶段的输入通道

        def forward(self, x):
            outputs = []
            current_x = x
            # 依次通过各个阶段
            for stage in self.stages:
                current_x = stage(current_x)
                outputs.append(current_x)
            # 确保返回4个特征图，如果不够则重复最后一个
            while len(outputs) < 4:
                outputs.append(outputs[-1])
            return outputs[:4]  # 返回前4个特征图

# --- 定义辅助模块 ConvBNReLU ---
class ConvBNReLU(nn.Sequential):
    """ 结合了卷积、批归一化和 ReLU 激活的模块 """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, padding=0, groups=1, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)  # inplace=True 节省内存
        )

# --- 定义自适应特征重标定 (AFR) 模块 ---
class AdaptiveFeatureRecalibration(nn.Module):
    """ 自适应特征重标定 (通道注意力模块) """
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.in_channels = in_channels
        reduced_channels = max(1, in_channels // reduction_ratio)  # 确保压缩后的通道数至少为1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化
        # 用于生成通道注意力的共享 MLP
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),  # 1x1卷积降维
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False)  # 1x1卷积升维
        )
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))  # 平均池化路径
        max_out = self.shared_mlp(self.max_pool(x))  # 最大池化路径
        channel_attention = self.sigmoid(avg_out + max_out)  # 结合两条路径并激活
        return x * channel_attention  # 将注意力权重应用到输入特征图

# --- 定义边界感知模块 (BAM) ---
class BoundaryAwareModule(nn.Module):
    """ 边界感知模块 (BAM) """
    def __init__(self, in_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = max(1, in_channels // 2)  # 默认中间通道数

        # 使用固定的 Laplacian 核进行边缘检测 (分组卷积)
        self.edge_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        # 定义 Laplacian 算子
        laplacian_kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # 为每个输入通道重复该核 (分组卷积需要)
        laplacian_kernel_c = laplacian_kernel.repeat(in_channels, 1, 1, 1)
        self.edge_conv.weight.data = laplacian_kernel_c  # 设置卷积核权重
        self.edge_conv.weight.requires_grad = False  # 固定权重，不参与训练

        # 用于处理特征的卷积层
        self.reduce_conv = ConvBNReLU(in_channels, mid_channels, kernel_size=1, padding=0)  # 1x1卷积降低原始特征通道
        self.edge_process_conv = ConvBNReLU(in_channels, mid_channels, kernel_size=1, padding=0)  # 1x1卷积处理边缘特征
        self.fuse_conv = ConvBNReLU(mid_channels * 2, in_channels, kernel_size=3, padding=1)  # 3x3卷积融合拼接后的特征

    def forward(self, x):
        edge_features = self.edge_conv(x)  # 获取边缘特征
        reduced_x = self.reduce_conv(x)  # 处理原始特征
        processed_edge = self.edge_process_conv(edge_features)  # 处理边缘特征
        # 拼接处理后的原始特征和边缘特征
        concat_features = torch.cat([reduced_x, processed_edge], dim=1)
        fused_features = self.fuse_conv(concat_features)  # 融合特征
        out = x + fused_features  # 添加残差连接
        return out

# --- 定义轻量级特征金字塔模块 (LFPM) (优化版) ---
class LightweightFeaturePyramidModule(nn.Module):
    """ 轻量级特征金字塔模块 (LFPM) - 优化版 """
    def __init__(self, in_channels, out_channels=None, compression_ratio=8, dilations=[1, 6, 12]):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels  # 默认输出通道数与输入相同
        mid_channels = max(1, in_channels // compression_ratio)
        num_branches = len(dilations)

        # 创建具有不同膨胀率的并行分支
        self.branches = nn.ModuleList()
        for d in dilations:
            padding = d  # 对于 3x3 卷积，padding=dilation 保持分辨率
            if d == 1:
                self.branches.append(ConvBNReLU(in_channels, mid_channels, kernel_size=1, dilation=d, padding=0))
            else:
                self.branches.append(ConvBNReLU(in_channels, mid_channels, kernel_size=3, dilation=d, padding=padding))

        # 用于融合并行分支输出的 1x1 卷积层
        self.fuse_conv = ConvBNReLU(mid_channels * num_branches, out_channels, kernel_size=1, padding=0)
        self.dropout = nn.Dropout(0.1)  # 添加 Dropout 层
        self.refine_conv = ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1)  # 最终细化卷积层

        print(f"初始化 LFPM: 输入={in_channels}, 中间={mid_channels}, 分支数={num_branches}, 膨胀率={dilations}, 输出={out_channels}")

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        concat_feat = torch.cat(branch_outputs, dim=1)
        fused_feat = self.fuse_conv(concat_feat)
        fused_feat = self.dropout(fused_feat)
        refined_feat = self.refine_conv(fused_feat)
        if refined_feat.shape[1] == x.shape[1]:
            out = x + refined_feat
        else:
            print(f"警告: LFPM 输入通道数 ({x.shape[1]}) != 输出通道数 ({refined_feat.shape[1]})。跳过残差连接。")
            out = refined_feat
        return out

# --- rest_lite 函数 (加载 ResT backbone) ---
def rest_lite(pretrained=True, weight_path='pretrain_weights/rest_lite.pth', embed_dim=64, **kwargs):
    """ 加载 ResT-Lite 模型 """
    # 定义 ResT-Lite 结构参数
    embed_dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
    num_heads = [1, 2, 4, 8]
    mlp_ratios = [4, 4, 4, 4]
    depths = [2, 2, 2, 2]
    sr_ratios = [8, 4, 2, 1]

    try:
        # 实例化 ResT 模型
        model = ResT(
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            qkv_bias=True,
            depths=depths,
            sr_ratios=sr_ratios,
            apply_transform=True,
            **kwargs
        )
    except Exception as e:
        print(f"错误: 无法实例化 ResT: {e}")
        print("警告: 使用 ResT 占位符")
        model = ResT(embed_dims=embed_dims, **kwargs)
        if pretrained:
            print("警告: 无法为 ResT 占位符加载预训练权重。")
        return model

    # 加载预训练权重
    if pretrained and weight_path is not None and os.path.exists(weight_path):
        try:
            checkpoint = torch.load(weight_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            model_dict = model.state_dict()
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k.replace('module.', '') in model_dict}
            state_dict = {k: v for k, v in state_dict.items() if v.shape == model_dict[k].shape}

            load_result = model.load_state_dict(state_dict, strict=False)
            print(f"从 {weight_path} 加载 Backbone 权重。 缺失键: {load_result.missing_keys}。 非预期键: {load_result.unexpected_keys}")
        except Exception as e:
            print(f"错误: 加载 Backbone 权重失败，路径 {weight_path}: {e}")
    elif pretrained:
        print(f"警告: Backbone 权重路径 '{weight_path}' 未找到或未指定。将从头训练。")

    return model

# --- 其他辅助类 ---
class PatchExpand(nn.Module):
    """ 扩展空间分辨率，减少通道维度 """
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim * dim_scale, bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        if x.ndim == 4 and x.shape[-1] == self.dim:
            pass
        elif x.ndim == 4 and x.shape[1] == self.dim:
            x = x.permute(0, 2, 3, 1)
        else:
            raise ValueError(f"PatchExpand 输入形状不匹配: {x.shape}, 期望 C={self.dim}")

        x = self.expand(x)
        B, H, W, C_expanded = x.shape
        p1 = p2 = self.dim_scale
        c_out = C_expanded // (p1 * p2)
        if C_expanded % (p1 * p2) != 0:
            raise ValueError(f"无法使用 scale {self.dim_scale} 重排通道 {C_expanded}")

        x = rearrange(x, 'b h w (p1 p2 c_out) -> b (h p1) (w p2) c_out', p1=p1, p2=p2, c_out=c_out)
        B_new, H_new, W_new, C_new = x.shape
        x = x.view(B_new, H_new * W_new, C_new)
        x = self.norm(x)
        x = x.view(B_new, H_new, W_new, C_new)
        return x

class FinalPatchExpand_X4(nn.Module):
    """ 最终扩展层，空间放大4倍 """
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, (dim_scale**2) * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        if x.ndim != 4 or x.shape[-1] != self.dim:
            raise ValueError(f"FinalPatchExpand_X4 输入形状不匹配: {x.shape}, 期望 C={self.dim}")

        x = self.expand(x)
        B, H, W, C_expanded = x.shape
        p1 = p2 = self.dim_scale
        c_out = C_expanded // (p1 * p2)
        if C_expanded % (p1 * p2) != 0 or c_out != self.output_dim:
            raise ValueError(f"无法使用 scale {self.dim_scale} 将通道 {C_expanded} 重排为输出维度 {self.output_dim}")

        x = rearrange(x, 'b h w (p1 p2 c_out) -> b (h p1) (w p2) c_out', p1=p1, p2=p2, c_out=c_out)
        B_new, H_new, W_new, C_new = x.shape
        x = x.view(B_new, H_new * W_new, C_new)
        x = self.norm(x)
        x = x.view(B_new, H_new, W_new, C_new)
        return x

class VSSLayer(nn.Module):
    """ 视觉状态空间层，包含多个 VSSBlock """
    def __init__(self, dim, depth, attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False, d_state=16, **kwargs):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                **kwargs.get('vssblock_kwargs', {})
            ) for i in range(depth)])
        self.downsample = downsample

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class LocalSupervision(nn.Module):
    """ 用于解码器中间特征的辅助监督头 """
    def __init__(self, in_channels=128, num_classes=6):
        super().__init__()
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU6())
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU6())
        self.drop = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(in_channels, num_classes, 1, bias=False)

    def forward(self, x, h, w):
        local1 = self.conv3(x)
        local2 = self.conv1(x)
        x_out = self.drop(local1 + local2)
        x_out = self.conv_out(x_out)
        x_out = F.interpolate(x_out, size=(h, w), mode='bilinear', align_corners=False)
        return x_out

# --- MambaSegDecoder ---
class MambaSegDecoder(nn.Module):
    """ 解码器，集成了 AFR, 并行的 BAM 和优化后的 LFPM """
    def __init__(
            self, num_classes: int, encoder_channels: Union[Tuple[int, ...], List[int]] = None,
            decode_channels: int = 64, drop_path_rate: float = 0.2, d_state: int = 16,
            afr_reduction_ratio: int = 16, decoder_depths: List[int] = None,
            norm_layer_decoder=nn.LayerNorm, bam_mid_channels: Optional[int] = None,
            lfpm_compression_ratio: int = 8, lfpm_dilations: List[int] = [1, 6, 12],
            **kwargs
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512]
        encoder_output_channels = encoder_channels
        self.num_classes = num_classes
        n_stages_encoder = len(encoder_output_channels)

        if decoder_depths is None:
            decoder_depths = [2] * (n_stages_encoder - 1)
        elif len(decoder_depths) != (n_stages_encoder - 1):
            raise ValueError("解码器深度列表长度必须等于解码器阶段数 (编码器阶段数 - 1)")

        total_decoder_depth = sum(decoder_depths)
        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, total_decoder_depth)]

        stages, expand_layers, lsm_layers, concat_back_dim, afr_modules = [], [], [], [], []
        current_dpr_idx = 0

        for s in range(n_stages_encoder - 1):
            stage_depth = decoder_depths[s]
            stage_dpr = dpr[current_dpr_idx: current_dpr_idx + stage_depth]
            current_dpr_idx += stage_depth

            input_features_skip = encoder_output_channels[-(s + 2)]
            input_features_below = encoder_output_channels[-(s + 1)]

            expand_layers.append(PatchExpand(None, dim=input_features_below, dim_scale=2, norm_layer=nn.LayerNorm))
            upsampled_channels = input_features_below // 2

            stages.append(VSSLayer(
                dim=input_features_skip,
                depth=stage_depth,
                attn_drop=0.,
                drop_path=stage_dpr,
                d_state=math.ceil(input_features_skip / 6) if d_state is None else d_state,
                norm_layer=norm_layer_decoder,
                downsample=None,
                use_checkpoint=kwargs.get('use_checkpoint', False)
            ))

            concat_channels = upsampled_channels + input_features_skip
            concat_back_dim.append(nn.Linear(concat_channels, input_features_skip, bias=False))

            afr_modules.append(AdaptiveFeatureRecalibration(
                in_channels=input_features_skip,
                reduction_ratio=afr_reduction_ratio
            ))

            lsm_layers.append(LocalSupervision(input_features_skip, num_classes))

        expand_layers.append(FinalPatchExpand_X4(None, dim=encoder_output_channels[0], dim_scale=4, norm_layer=nn.LayerNorm))
        stages.append(nn.Identity())

        self.stages = nn.ModuleList(stages)
        self.expand_layers = nn.ModuleList(expand_layers)
        self.concat_back_dim = nn.ModuleList(concat_back_dim)
        self.afr_modules = nn.ModuleList(afr_modules)
        if self.training:
            self.lsm = nn.ModuleList(lsm_layers)

        final_decoder_channels = encoder_output_channels[0]
        self.bam = BoundaryAwareModule(final_decoder_channels, mid_channels=bam_mid_channels)
        self.lfpm = LightweightFeaturePyramidModule(
            in_channels=final_decoder_channels,
            out_channels=final_decoder_channels,
            compression_ratio=lfpm_compression_ratio,
            dilations=lfpm_dilations
        )
        fusion_in_channels = final_decoder_channels * 2
        fusion_out_channels = final_decoder_channels
        self.fusion_conv = ConvBNReLU(fusion_in_channels, fusion_out_channels, kernel_size=1, padding=0)
        self.seg = nn.Conv2d(final_decoder_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.init_weight()

    def forward(self, skips: List[torch.Tensor], h, w):
        lres_input = skips[-1]
        ls_outputs = []

        for s in range(len(self.stages) - 1):
            x_expanded = self.expand_layers[s](lres_input)
            skip_feature = skips[-(s + 2)]
            if s == 0:
                skip_feature_processed = self.afr_modules[s](skip_feature)
            else:
                skip_feature_processed = skip_feature

            skip_feature_permuted = skip_feature_processed.permute(0, 2, 3, 1)
            if x_expanded.shape[1:3] != skip_feature_permuted.shape[1:3]:
                print(f"警告: 第 {s} 阶段空间尺寸不匹配。对 x_expanded 进行插值。")
                x_expanded = F.interpolate(x_expanded.permute(0, 3, 1, 2), size=skip_feature_permuted.shape[1:3], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

            x_concat = torch.cat((x_expanded, skip_feature_permuted), dim=-1)
            B_cat, H_cat, W_cat, C_cat = x_concat.shape
            x_concat_reshaped = x_concat.view(B_cat, H_cat * W_cat, C_cat)
            x_linear = self.concat_back_dim[s](x_concat_reshaped)
            x_linear_reshaped = x_linear.view(B_cat, H_cat, W_cat, skip_feature.shape[1])
            x_stage_out = self.stages[s](x_linear_reshaped)
            x_permuted = x_stage_out.permute(0, 3, 1, 2)

            if self.training and hasattr(self, 'lsm') and s < len(self.lsm):
                ls_outputs.append(self.lsm[s](x_permuted, h, w))

            lres_input = x_stage_out

        x_final_expanded = self.expand_layers[-1](lres_input)
        x = self.stages[-1](x_final_expanded)
        x = x.permute(0, 3, 1, 2)

        bam_out = self.bam(x)
        lfpm_out = self.lfpm(x)
        fused_features = torch.cat([bam_out, lfpm_out], dim=1)
        fused_features = self.fusion_conv(fused_features)
        x_final = x + fused_features
        seg_out = self.seg(x_final)

        if self.training:
            lsm_loss = sum(ls_outputs) if ls_outputs else torch.tensor(0.0, device=seg_out.device)
            return seg_out, lsm_loss
        else:
            return seg_out

    def init_weight(self):
        print("初始化 MambaSegDecoder 权重...")
        for m_name, m in self.named_modules():
            if 'bam.edge_conv.weight' in m_name:
                print(f"跳过固定权重的初始化: {m_name}")
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)

# --- 主模型类: UNetMamba_AFR_BAM_LFPM ---
class UNetMamba_AFR_BAM_LFPM(nn.Module):
    """ UNet Mamba 模型，集成了 ResT, AFR, BAM 和优化版 LFPM """
    def __init__(self,
                 num_classes=6, input_channels=3, embed_dim=64,
                 afr_reduction_ratio=16, bam_mid_channels=None,
                 lfpm_compression_ratio=8, lfpm_dilations=[1, 6, 12],
                 backbone_path='pretrain_weights/rest_lite.pth',
                 decode_channels=64, decoder_depths=[2, 2, 2], drop_path_rate=0.1,
                 d_state=16, use_checkpoint=False, **kwargs):
        super().__init__()

        self.encoder = rest_lite(pretrained=True, weight_path=backbone_path, embed_dim=embed_dim)
        encoder_channels = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        if hasattr(self.encoder, 'output_channels'):
            encoder_channels = self.encoder.output_channels
            print(f"使用来自 ResT 实例的编码器通道: {encoder_channels}")

        self.decoder = MambaSegDecoder(
            num_classes=num_classes,
            encoder_channels=encoder_channels,
            drop_path_rate=drop_path_rate,
            d_state=d_state,
            afr_reduction_ratio=afr_reduction_ratio,
            decoder_depths=decoder_depths,
            norm_layer_decoder=nn.LayerNorm,
            bam_mid_channels=bam_mid_channels,
            lfpm_compression_ratio=lfpm_compression_ratio,
            lfpm_dilations=lfpm_dilations,
            use_checkpoint=use_checkpoint
        )

    def forward(self, x):
        h, w = x.size()[-2:]
        outputs = self.encoder(x)

        if not isinstance(outputs, (list, tuple)) or len(outputs) != 4:
            print(f"警告: 编码器输出格式非预期。类型: {type(outputs)}, 长度: {len(outputs) if isinstance(outputs, (list, tuple)) else 'N/A'}")
            if isinstance(outputs, torch.Tensor) and len(outputs.shape) == 4:
                print("假设编码器输出单个张量，将其复制4次。")
                outputs = [outputs] * 4
            elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                print(f"将编码器输出长度从 {len(outputs)} 调整为 4。")
                while len(outputs) < 4:
                    outputs.append(outputs[-1])
                outputs = outputs[:4]
            else:
                raise TypeError("编码器必须返回包含4个特征图的列表或元组。")

        decoder_output = self.decoder(outputs, h, w)
        return decoder_output

# --- 使用示例 ---
if __name__ == "__main__":
    print("--- 运行模型实例化示例 ---")
    dummy_input = torch.randn(2, 3, 512, 512)

    model = UNetMamba_AFR_BAM_LFPM(
        num_classes=6,
        embed_dim=64,
        backbone_path=None,
        decoder_depths=[2, 2, 2],
        bam_mid_channels=32,
        use_checkpoint=False
    )

    model.train()
    output_train = model(dummy_input)
    model.eval()
    output_eval = model(dummy_input)

    print("\n--- 模型输出形状 ---")
    if isinstance(output_train, tuple):
        print(f"输出 (训练模式): 分割图形状={output_train[0].shape}, 辅助损失类型={type(output_train[1])}")
    else:
        print(f"输出 (训练模式): 分割图形状={output_train.shape}")

    if isinstance(output_eval, tuple):
        print(f"输出 (评估模式): 分割图形状={output_eval[0].shape}")
    else:
        print(f"输出 (评估模式): 分割图形状={output_eval.shape}")

    try:
        from torchinfo import summary
        summary(model, input_size=(2, 3, 512, 512), device="cpu",
                col_names=["input_size", "output_size", "num_params", "mult_adds"], depth=4)
    except ImportError:
        print("\n请安装 torchinfo 以查看模型摘要: pip install torchinfo")
    except Exception as e:
        print(f"torchinfo 摘要生成错误: {e}")

    print("\n--- 示例结束 ---")