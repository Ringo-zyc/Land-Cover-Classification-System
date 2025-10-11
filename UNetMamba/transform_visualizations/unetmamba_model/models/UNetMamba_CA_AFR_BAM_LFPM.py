import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any, Union, Type, List, Tuple, Dict
from collections import OrderedDict

from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# --- 尝试导入项目特定的模块 ---
try:
    from unetmamba_model.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
except ImportError as e:
    print(f"Warning: Could not import VSSM components: {e}.")
    VSSM = VSSBlock = LayerNorm2d = Permute = nn.Identity
try:
    from unetmamba_model.models.ResT import ResT
except ImportError as e:
    print(f"Warning: Could not import ResT: {e}.")
    class ResT(nn.Module): # Placeholder
        def __init__(self, embed_dims=None, **kwargs):
            super().__init__(); print("Warning: Using placeholder ResT.")
            self.stages = nn.ModuleList()
            in_c = 3
            if embed_dims is None: embed_dims = [64, 128, 256, 512]
            for i, out_c in enumerate(embed_dims):
                 stage = nn.Sequential(nn.Conv2d(in_c, out_c, 3, 1, 1), nn.ReLU(),
                                     nn.MaxPool2d(2, 2) if i < len(embed_dims) else nn.Identity())
                 self.stages.append(stage); in_c = out_c
        def forward(self, x):
            outputs = [stage(x if i==0 else outputs[-1]) for i, stage in enumerate(self.stages)]
            if len(outputs) != 4:
                 while len(outputs) < 4: outputs.append(outputs[-1])
                 outputs = outputs[:4]
            return outputs

# --- 定义辅助模块 ConvBNReLU ---
class ConvBNReLU(nn.Sequential):
    """ 组合卷积、批归一化和 ReLU 激活函数 """
    """ Combines Convolution, Batch Normalization, and ReLU activation """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0, groups=1, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

# --- 定义 Adaptive Feature Recalibration (AFR) 模块 ---
class AdaptiveFeatureRecalibration(nn.Module):
    """ 自适应特征重标定 (通道注意力模块) """
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__(); self.in_channels = in_channels
        reduced_channels = max(1, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1); self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential( nn.Conv2d(in_channels, reduced_channels, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False) )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x)); max_out = self.shared_mlp(self.max_pool(x))
        channel_attention = self.sigmoid(avg_out + max_out); return x * channel_attention

# --- 定义 Boundary Aware Module (BAM) 模块 ---
class BoundaryAwareModule(nn.Module):
    """ 边界感知模块 (BAM) """
    def __init__(self, in_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None: mid_channels = max(1, in_channels // 2)
        self.edge_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        laplacian_kernel = torch.tensor([[1, 1, 1], [1,-8, 1], [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        laplacian_kernel_c = laplacian_kernel.repeat(in_channels, 1, 1, 1)
        self.edge_conv.weight.data = laplacian_kernel_c; self.edge_conv.weight.requires_grad = False
        self.reduce_conv = ConvBNReLU(in_channels, mid_channels, 1, padding=0)
        self.edge_process_conv = ConvBNReLU(in_channels, mid_channels, 1, padding=0)
        self.fuse_conv = ConvBNReLU(mid_channels * 2, in_channels, 3, padding=1)
    def forward(self, x):
        edge_features = self.edge_conv(x)
        reduced_x = self.reduce_conv(x); processed_edge = self.edge_process_conv(edge_features)
        concat_features = torch.cat([reduced_x, processed_edge], dim=1)
        fused_features = self.fuse_conv(concat_features); out = x + fused_features
        return out

# --- 定义 Lightweight Feature Pyramid Module (LFPM) 模块 ---
class LightweightFeaturePyramidModule(nn.Module):
    """
    轻量级特征金字塔模块 (LFPM)。
    使用不同膨胀率的卷积捕获多尺度特征。
    Lightweight Feature Pyramid Module (LFPM).
    Uses convolutions with different dilation rates to capture multi-scale features.
    """
    def __init__(self, in_channels, out_channels=None, compression_ratio=4, dilations=[1, 6, 12, 18]): # Common ASPP dilations
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        # 压缩后的中间通道数
        # Intermediate channel count after compression
        mid_channels = max(1, in_channels // compression_ratio)
        num_branches = len(dilations)

        # 创建并行分支
        # Create parallel branches
        self.branches = nn.ModuleList()
        for d in dilations:
            padding = d # 保持分辨率所需的 padding | Padding required to maintain resolution
            if d == 1: # Use 1x1 conv for dilation 1 branch
                 self.branches.append(ConvBNReLU(in_channels, mid_channels, kernel_size=1, dilation=d, padding=0))
            else: # Use 3x3 conv for other dilations
                 self.branches.append(ConvBNReLU(in_channels, mid_channels, kernel_size=3, dilation=d, padding=padding))

        # 融合卷积层
        # Fusion convolution layer
        self.fuse_conv = ConvBNReLU(mid_channels * num_branches, out_channels, kernel_size=1, padding=0)

        # 最终细化卷积层 (可选)
        # Final refinement convolution layer (optional)
        self.refine_conv = ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1)

        print(f"Initialized LFPM: In={in_channels}, Mid={mid_channels} (Ratio={compression_ratio}), Branches={num_branches}, Dilations={dilations}, Out={out_channels}")

    def forward(self, x):
        # 输入 x: (B, C, H, W)
        # Input x: (B, C, H, W)
        b, c, h, w = x.shape

        # 计算各分支特征
        # Calculate features for each branch
        branch_outputs = [branch(x) for branch in self.branches]

        # 拼接特征
        # Concatenate features
        concat_feat = torch.cat(branch_outputs, dim=1) # Shape: (B, mid * num_branches, H, W)

        # 融合特征
        # Fuse features
        fused_feat = self.fuse_conv(concat_feat) # Shape: (B, out_channels, H, W)

        # 细化特征
        # Refine features
        refined_feat = self.refine_conv(fused_feat) # Shape: (B, out_channels, H, W)

        # 添加残差连接 (如果输出通道数与输入相同)
        # Add residual connection (if output channels match input)
        if refined_feat.shape[1] == c:
             out = x + refined_feat
        else:
             # 如果通道数不同，则不使用残差连接或需要一个 1x1 卷积来匹配
             # If channel counts differ, either skip residual or use a 1x1 conv to match
             print(f"Warning: LFPM input channels ({c}) != output channels ({refined_feat.shape[1]}). Skipping residual connection.")
             out = refined_feat

        return out


# --- rest_lite 函数 ---
# (rest_lite function remains the same)
def rest_lite(pretrained=True, weight_path='pretrain_weights/rest_lite.pth', embed_dim=64, **kwargs):
    embed_dims=[embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
    if 'forward' not in dir(ResT) or isinstance(ResT, type(nn.Module)):
         print("Warning: ResT might be a placeholder.");
         if 'Placeholder' in ResT.__name__: return ResT(embed_dims=embed_dims, **kwargs)
    model = ResT(embed_dims=embed_dims, num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], apply_transform=True, **kwargs)
    if pretrained and weight_path is not None and os.path.exists(weight_path):
        try:
            checkpoint = torch.load(weight_path, map_location='cpu')
            if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint: state_dict = checkpoint['model']
            else: state_dict = checkpoint
            model_dict = model.state_dict()
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k.replace('module.', '') in model_dict}
            state_dict = {k: v for k, v in state_dict.items() if v.shape == model_dict[k].shape}
            model_dict.update(state_dict)
            load_result = model.load_state_dict(model_dict, strict=False)
            print(f"Loaded backbone weights from {weight_path}. Missing keys: {load_result.missing_keys}. Unexpected keys: {load_result.unexpected_keys}")
        except Exception as e: print(f"ERROR loading backbone weights from {weight_path}: {e}")
    elif pretrained: print(f"Warning: Backbone weights path '{weight_path}' not found or not specified.")
    return model

# --- 其他辅助类定义 ---
# (PatchExpand, FinalPatchExpand_X4, VSSLayer, LocalSupervision remain the same)
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__(); self.dim = dim; self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim_scale*dim, bias=False) if dim_scale > 1 else nn.Identity()
        self.norm = norm_layer(dim * dim_scale // 4 if dim_scale==2 else dim)
    def forward(self, x):
        if x.ndim == 4: B, C, H, W = x.shape; x = x.permute(0, 2, 3, 1)
        elif x.ndim == 3: raise ValueError("PatchExpand requires 4D input (B, C, H, W)")
        else: raise ValueError(f"PatchExpand input ndim={x.ndim}")
        x = self.expand(x); B, H, W, C_expanded = x.shape
        p1 = p2 = self.dim_scale; c_out = C_expanded // (p1 * p2)
        x = rearrange(x, 'b h w (p1 p2 c_out) -> b (h p1) (w p2) c_out', p1=p1, p2=p2)
        B_new, H_new, W_new, C_new = x.shape
        x = x.view(B_new, H_new * W_new, C_new); x = self.norm(x)
        x = x.view(B_new, H_new, W_new, C_new)
        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__(); self.dim = dim; self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, (dim_scale**2)*dim, bias=False)
        self.output_dim = dim; self.norm = norm_layer(self.output_dim)
    def forward(self, x):
        if x.ndim != 4: raise ValueError(f"FinalPatchExpand_X4 input ndim={x.ndim}")
        x = x.permute(0, 2, 3, 1); x = self.expand(x); B, H, W, C_expanded = x.shape
        p1 = p2 = self.dim_scale; c_out = C_expanded // (p1 * p2)
        x = rearrange(x, 'b h w (p1 p2 c_out) -> b (h p1) (w p2) c_out', p1=p1, p2=p2)
        B_new, H_new, W_new, C_new = x.shape
        x = x.view(B_new, H_new * W_new, C_new); x = self.norm(x)
        x = x.view(B_new, H_new, W_new, C_new)
        return x

class VSSLayer(nn.Module): # Using the version that passes 4D tensor to VSSBlock
    def __init__( self, dim, depth, attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
            downsample=None, use_checkpoint=False, d_state=16, **kwargs ):
        super().__init__(); self.dim = dim; self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([ VSSBlock( hidden_dim=dim, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, attn_drop_rate=attn_drop, d_state=d_state ) for i in range(depth)])
        self.downsample = downsample
    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint: x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else: x = blk(x) # Pass 4D tensor (B, H, W, C)
        if self.downsample is not None: x = self.downsample(x)
        return x

class LocalSupervision(nn.Module):
    def __init__(self, in_channels=128, num_classes=6):
        super().__init__()
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU6())
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU6())
        self.drop = nn.Dropout(0.1); self.conv_out = nn.Conv2d(in_channels, num_classes, 1, bias=False)
    def forward(self, x, h, w):
        local1 = self.conv3(x); local2 = self.conv1(x); x_out = self.drop(local1 + local2)
        x_out = self.conv_out(x_out); x_out = F.interpolate(x_out, size=(h, w), mode='bilinear', align_corners=False)
        return x_out


# --- MambaSegDecoder (加入 AFR, BAM, LFPM) ---
class MambaSegDecoder(nn.Module):
    """ 解码器，集成了 AFR, BAM, 和 LFPM """
    """ Decoder, integrating AFR, BAM, and LFPM """
    def __init__(
            self, num_classes: int, encoder_channels: Union[Tuple[int, ...], List[int]] = None,
            decode_channels: int = 64, drop_path_rate: float = 0.2, d_state: int = 16,
            afr_reduction_ratio: int = 16, decoder_depths: List[int] = None,
            norm_layer_decoder = nn.LayerNorm, bam_mid_channels: Optional[int] = None,
            lfpm_compression_ratio: int = 4, # LFPM 压缩率 | LFPM compression ratio
            lfpm_dilations: List[int] = [1, 6, 12, 18], # LFPM 膨胀率 | LFPM dilations
            **kwargs
    ):
        super().__init__()
        if encoder_channels is None: encoder_channels = [64, 128, 256, 512]
        encoder_output_channels = encoder_channels
        self.num_classes = num_classes
        n_stages_encoder = len(encoder_output_channels)
        if decoder_depths is None: decoder_depths = [2] * (n_stages_encoder - 1)
        elif len(decoder_depths) != (n_stages_encoder - 1): raise ValueError("Decoder depths mismatch")
        total_decoder_depth = sum(decoder_depths)
        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, total_decoder_depth)]

        stages, expand_layers, lsm_layers, concat_back_dim, afr_modules = [], [], [], [], []
        current_dpr_idx = 0

        # --- 构建解码器阶段 (与之前相同) ---
        for s in range(1, n_stages_encoder):
            stage_depth = decoder_depths[s-1]
            stage_dpr = dpr[current_dpr_idx : current_dpr_idx + stage_depth]
            current_dpr_idx += stage_depth
            input_features_skip = encoder_output_channels[-(s + 1)]
            input_features_below = encoder_output_channels[-s] if s == 1 else encoder_output_channels[-s]
            expand_layers.append(PatchExpand(None, dim=input_features_below, dim_scale=2, norm_layer=nn.LayerNorm))
            upsampled_channels = input_features_below * 2 // 4
            stages.append(VSSLayer( dim=input_features_skip, depth=stage_depth, attn_drop=0., drop_path=stage_dpr,
                d_state=math.ceil(input_features_skip / 6) if d_state is None else d_state,
                norm_layer=norm_layer_decoder, downsample=None, use_checkpoint=kwargs.get('use_checkpoint', False) ))
            concat_channels = upsampled_channels + input_features_skip
            concat_back_dim.append(nn.Linear(concat_channels, input_features_skip, bias=False))
            afr_modules.append(AdaptiveFeatureRecalibration( in_channels=input_features_skip, reduction_ratio=afr_reduction_ratio ))
            lsm_layers.append(LocalSupervision(input_features_skip, num_classes))

        # --- 最终扩展层和 Identity Stage (与之前相同) ---
        expand_layers.append(FinalPatchExpand_X4(None, dim=encoder_output_channels[0], dim_scale=4, norm_layer=nn.LayerNorm))
        stages.append(nn.Identity())

        self.stages = nn.ModuleList(stages)
        self.expand_layers = nn.ModuleList(expand_layers)
        self.concat_back_dim = nn.ModuleList(concat_back_dim)
        self.afr_modules = nn.ModuleList(afr_modules)
        self.lsm = nn.ModuleList(lsm_layers)

        # --- 修改: 在分割头之前依次加入 BAM 和 LFPM ---
        # --- Modification: Add BAM and LFPM sequentially before segmentation head ---
        final_decoder_channels = encoder_channels[0] # 通常是 64 | Usually 64
        # 1. 边界感知模块 (BAM)
        # 1. Boundary Aware Module (BAM)
        self.bam = BoundaryAwareModule(final_decoder_channels, mid_channels=bam_mid_channels)
        # 2. 轻量级特征金字塔模块 (LFPM) - 输入通道数与 BAM 输出相同
        # 2. Lightweight Feature Pyramid Module (LFPM) - Input channels same as BAM output
        self.lfpm = LightweightFeaturePyramidModule(
            in_channels=final_decoder_channels,
            out_channels=final_decoder_channels, # LFPM 输出通道数保持不变 | LFPM output channels remain same
            compression_ratio=lfpm_compression_ratio,
            dilations=lfpm_dilations
        )
        # 3. 分割头现在接收 LFPM 的输出
        # 3. Segmentation head now receives output from LFPM
        self.seg = nn.Conv2d(final_decoder_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.init_weight() # 初始化权重 | Initialize weights

    def forward(self, skips: List[torch.Tensor], h, w):
        lres_input = skips[-1]
        ls_outputs = []
        # --- 解码器循环 (与之前相同，直到最后阶段) ---
        for s in range(len(self.stages)):
            x = self.expand_layers[s](lres_input)
            if s < (len(self.stages) - 1):
                skip_feature = skips[-(s + 2)]
                afr_module = self.afr_modules[s]
                skip_feature_afr = afr_module(skip_feature)
                skip_feature_afr_permuted = skip_feature_afr.permute(0, 2, 3, 1)
                if x.shape[1:3] != skip_feature_afr_permuted.shape[1:3]:
                     print(f"Warning: Spatial size mismatch stage {s}. Interpolating x.")
                     x = F.interpolate(x.permute(0, 3, 1, 2), size=skip_feature_afr_permuted.shape[1:3], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                x = torch.cat((x, skip_feature_afr_permuted), dim=-1)
                x = self.concat_back_dim[s](x)
            x = self.stages[s](x) # Output (B, H, W, C)
            x = x.permute(0, 3, 1, 2) # Output (B, C, H, W)

            # --- 修改: 在最后一个阶段依次应用 BAM 和 LFPM ---
            if s == (len(self.stages) - 1):
                x_bam = self.bam(x)     # 应用 BAM | Apply BAM
                x_lfpm = self.lfpm(x_bam) # 应用 LFPM | Apply LFPM
                seg_out = self.seg(x_lfpm) # 应用分割头 | Apply segmentation head
            elif self.training and hasattr(self, 'lsm') and s < len(self.lsm):
                ls_outputs.append(self.lsm[s](x, h, w))
            lres_input = x

        # --- 返回结果 (与之前相同) ---
        if self.training:
            lsm_loss = sum(ls_outputs) if ls_outputs else torch.tensor(0.0, device=seg_out.device)
            return seg_out, lsm_loss
        else:
            return seg_out

    def init_weight(self):
        # --- 权重初始化 (会初始化 LFPM 中的新层) ---
        print("Initializing weights for MambaSegDecoder (including BAM, LFPM)...")
        for m_name, m in self.named_modules():
            if 'bam.edge_conv.weight' in m_name: # 跳过固定权重 | Skip fixed weights
                print(f"Skipping initialization for fixed weights: {m_name}")
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                 trunc_normal_(m.weight, std=.02)
                 if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                 if hasattr(m, 'bias') and m.bias is not None: nn.init.constant_(m.bias, 0)
                 if hasattr(m, 'weight') and m.weight is not None: nn.init.constant_(m.weight, 1.0)


# --- CoordinateAttention 类 (修正版) ---
# (CoordinateAttention class remains the same revised version)
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__(); self.pool_h=nn.AdaptiveAvgPool2d((None, 1)); self.pool_w=nn.AdaptiveAvgPool2d((1, None))
        mip=max(8, in_channels // reduction); self.conv1=nn.Conv2d(in_channels, mip, 1, 1, 0, bias=False); self.bn1=nn.BatchNorm2d(mip); self.act=nn.Hardswish(inplace=True)
        self.conv_h=nn.Conv2d(mip, in_channels, 1, 1, 0); self.conv_w=nn.Conv2d(mip, in_channels, 1, 1, 0); self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        identity=x; B, C, H, W=x.size(); x_h=self.pool_h(x); x_w=self.pool_w(x)
        y=torch.cat((x_h, x_w.permute(0, 1, 3, 2)), dim=2); y=self.act(self.bn1(self.conv1(y)))
        if H > 0 and W > 0: x_h, x_w=torch.split(y, [H, W], dim=2); x_w=x_w.permute(0, 1, 3, 2)
        elif H > 0: x_h=y; x_w=torch.zeros((B, y.shape[1], 1, W), device=x.device, dtype=x.dtype)
        elif W > 0: x_h=torch.zeros((B, y.shape[1], H, 1), device=x.device, dtype=x.dtype); x_w=y.permute(0, 1, 3, 2)
        else: x_h=torch.zeros((B, y.shape[1], H, 1), device=x.device, dtype=x.dtype); x_w=torch.zeros((B, y.shape[1], 1, W), device=x.device, dtype=x.dtype)
        a_h=self.sigmoid(self.conv_h(x_h)); a_w=self.sigmoid(self.conv_w(x_w)); out=identity * a_h * a_w
        return out


# --- 主模型类: UNetMambaCA_AFR_BAM_LFPM ---
# --- Main Model Class: UNetMambaCA_AFR_BAM_LFPM ---
class UNetMambaCA_AFR_BAM_LFPM(nn.Module):
    """
    集成了 ResT, CA, AFR, BAM 和 LFPM 的 UNet Mamba 模型。
    UNet Mamba model integrating ResT, CA, AFR, BAM, and LFPM.
    """
    def __init__(self,
                 # --- Core Args ---
                 num_classes=6, input_channels=3, embed_dim=64,
                 afr_reduction_ratio=16, ca_reduction=32,
                 bam_mid_channels=None,
                 lfpm_compression_ratio=4, # LFPM 参数 | LFPM parameter
                 lfpm_dilations=[1, 6, 12, 18], # LFPM 参数 | LFPM parameter

                 # --- Backbone Args ---
                 backbone_path='pretrain_weights/rest_lite.pth',

                 # --- Decoder/VSSM Args (Pass through if needed) ---
                 decode_channels=64, decoder_depths=[2, 2, 2], drop_path_rate=0.1,
                 d_state=16, patch_size=4, depths=[2, 2, 9, 2], dims=96,
                 ssm_d_state=16, ssm_ratio=2.0, ssm_rank_ratio=2.0, ssm_dt_rank="auto",
                 ssm_act_layer="silu", ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0,
                 ssm_init="v0", forward_type="v4", mlp_ratio=4.0, mlp_act_layer="gelu",
                 mlp_drop_rate=0.0, patch_norm=True, norm_layer="ln",
                 downsample_version="v2", patchembed_version="v2", gmlp=False,
                 use_checkpoint=False, **kwargs
                 ):
        super().__init__()

        # 1. 输入端的坐标注意力 (CA)
        self.CA = CoordinateAttention(input_channels, reduction=ca_reduction)

        # 2. ResT backbone (编码器)
        self.encoder = rest_lite(pretrained=True, weight_path=backbone_path, embed_dim=embed_dim)

        # 3. 确定编码器输出通道数 (基于 ResT)
        encoder_channels = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]

        # 4. 实例化包含 AFR, BAM, LFPM 的解码器
        self.decoder = MambaSegDecoder(
            num_classes=num_classes,
            encoder_channels=encoder_channels,
            drop_path_rate=drop_path_rate,
            d_state=d_state if d_state else ssm_d_state,
            afr_reduction_ratio=afr_reduction_ratio,
            decoder_depths=decoder_depths,
            norm_layer_decoder=nn.LayerNorm if norm_layer=="ln" else nn.BatchNorm2d,
            bam_mid_channels=bam_mid_channels,
            lfpm_compression_ratio=lfpm_compression_ratio, # <<<--- 传递 LFPM 参数 | Pass LFPM parameter
            lfpm_dilations=lfpm_dilations,             # <<<--- 传递 LFPM 参数 | Pass LFPM parameter
            use_checkpoint=use_checkpoint
            # Pass other VSSM params if MambaSegDecoder uses them internally
        )

    def forward(self, x):
        """ 模型前向传播 | Model forward pass """
        x_ca = self.CA(x)
        h, w = x_ca.size()[-2:]
        outputs = self.encoder(x_ca)
        # --- 编码器输出检查 (与之前相同) ---
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 4:
             print(f"Warning: Encoder output format unexpected.")
             if isinstance(outputs, torch.Tensor) and len(outputs.shape) == 4: outputs = [outputs] * 4
             elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                  while len(outputs) < 4: outputs.append(outputs[-1])
                  outputs = outputs[:4]
             else: raise TypeError("Encoder must return a list or tuple of 4 feature maps.")
        # --- 解码器输出 ---
        decoder_output = self.decoder(outputs, h, w)
        return decoder_output

# --- Example Usage (Optional) ---
if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 512, 512)
    # --- 实例化包含 LFPM 的模型 ---
    model = UNetMambaCA_AFR_BAM_LFPM(
        num_classes=6, embed_dim=64, backbone_path=None, decoder_depths=[2, 2, 2],
        bam_mid_channels=32, lfpm_compression_ratio=4, lfpm_dilations=[1, 6, 12, 18]
    )
    model.train()
    output_train = model(dummy_input)
    model.eval()
    output_eval = model(dummy_input)
    print("--- Model Output Shapes (with LFPM) ---")
    if isinstance(output_train, tuple): print(f"Output (Training): Seg Map Shape={output_train[0].shape}, LSM Loss Type={type(output_train[1])}")
    else: print(f"Output (Training): Seg Map Shape={output_train.shape}")
    if isinstance(output_eval, tuple): print(f"Output (Eval): Seg Map Shape={output_eval[0].shape}")
    else: print(f"Output (Eval): Seg Map Shape={output_eval.shape}")
    try:
        from torchinfo import summary
        summary(model, input_size=(2, 3, 512, 512), device="cpu")
    except ImportError: print("\nInstall torchinfo for model summary: pip install torchinfo")
    except Exception as e: print(f"Error during torchinfo summary: {e}")

