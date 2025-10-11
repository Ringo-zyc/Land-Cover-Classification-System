# filename: unetmamba_model/models/UNetMamba_CA_AFR.py

# --- 必要的导入 ---
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
    # 导入 VSSM 相关组件 (如果 MambaSegDecoder 或其子模块需要)
    from unetmamba_model.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
except ImportError as e:
    print(f"Warning: Could not import VSSM components: {e}. Ensure they are available if needed.")
    # 提供占位符以允许代码解析，但如果实际使用会报错
    VSSM = VSSBlock = LayerNorm2d = Permute = nn.Identity

try:
    # 导入 ResT backbone
    from unetmamba_model.models.ResT import ResT
except ImportError as e:
    print(f"Warning: Could not import ResT from 'unetmamba_model.models.ResT': {e}.")
    # 定义一个简单的占位符 ResT，如果导入失败
    class ResT(nn.Module):
        def __init__(self, embed_dims=None, **kwargs):
            super().__init__()
            print("Warning: Using placeholder ResT. Ensure the real ResT model is available.")
            # 创建一个简单的序列，模拟多阶段输出（通道数需要匹配）
            self.stages = nn.ModuleList()
            in_c = 3
            if embed_dims is None: embed_dims = [64, 128, 256, 512] # Default dims
            for i, out_c in enumerate(embed_dims):
                 # 简单的卷积层模拟阶段，加一个下采样
                 stage = nn.Sequential(
                     nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=2, stride=2) if i < len(embed_dims) else nn.Identity() # Downsample
                 )
                 self.stages.append(stage)
                 in_c = out_c # Update input channels for next stage

        def forward(self, x):
            outputs = []
            current_x = x
            for stage in self.stages:
                current_x = stage(current_x)
                outputs.append(current_x)
            # 确保输出是列表/元组
            if len(outputs) != 4:
                 print(f"Warning: Placeholder ResT generated {len(outputs)} outputs, expected 4.")
                 # Pad with last output if needed, though this is incorrect behavior
                 while len(outputs) < 4: outputs.append(outputs[-1])
                 outputs = outputs[:4]
            return outputs

# --- 定义 Adaptive Feature Recalibration (AFR) 模块 ---
# (AFR Module definition remains the same as before)
class AdaptiveFeatureRecalibration(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(AdaptiveFeatureRecalibration, self).__init__()
        self.in_channels = in_channels
        reduced_channels = max(1, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        channel_attention = self.sigmoid(avg_out + max_out)
        return x * channel_attention

# --- rest_lite 函数 (加载 ResT backbone) ---
# (rest_lite function remains the same as before)
def rest_lite(pretrained=True, weight_path='pretrain_weights/rest_lite.pth', embed_dim=64, **kwargs):
    embed_dims=[embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
    if 'forward' not in dir(ResT) or isinstance(ResT, type(nn.Module)):
         print("Warning: ResT might be a placeholder. Attempting to proceed.")
         if 'Placeholder' in ResT.__name__:
              return ResT(embed_dims=embed_dims, **kwargs)
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
    elif pretrained: print(f"Warning: Backbone weights path '{weight_path}' not found or not specified. Training backbone from scratch.")
    return model

# --- 其他辅助类定义 (PatchExpand, FinalPatchExpand_X4, LocalSupervision) ---
# (These helper classes remain the same as before)
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim_scale*dim, bias=False) if dim_scale > 1 else nn.Identity()
        self.norm = norm_layer(dim * dim_scale // 4 if dim_scale==2 else dim)
    def forward(self, x):
        if x.ndim == 4: B, C, H, W = x.shape; x = x.permute(0, 2, 3, 1)
        elif x.ndim == 3: raise ValueError("PatchExpand requires 4D input (B, C, H, W)")
        else: raise ValueError(f"PatchExpand received input with ndim={x.ndim}, expected 3 or 4.")
        x = self.expand(x); B, H, W, C_expanded = x.shape
        p1 = p2 = self.dim_scale; c_out = C_expanded // (p1 * p2)
        x = rearrange(x, 'b h w (p1 p2 c_out) -> b (h p1) (w p2) c_out', p1=p1, p2=p2)
        B_new, H_new, W_new, C_new = x.shape
        x = x.view(B_new, H_new * W_new, C_new); x = self.norm(x)
        x = x.view(B_new, H_new, W_new, C_new)
        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim; self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, (dim_scale**2)*dim, bias=False)
        self.output_dim = dim; self.norm = norm_layer(self.output_dim)
    def forward(self, x):
        if x.ndim != 4: raise ValueError(f"FinalPatchExpand_X4 received input with ndim={x.ndim}, expected 4 (B, C, H, W).")
        x = x.permute(0, 2, 3, 1); x = self.expand(x); B, H, W, C_expanded = x.shape
        p1 = p2 = self.dim_scale; c_out = C_expanded // (p1 * p2)
        x = rearrange(x, 'b h w (p1 p2 c_out) -> b (h p1) (w p2) c_out', p1=p1, p2=p2)
        B_new, H_new, W_new, C_new = x.shape
        x = x.view(B_new, H_new * W_new, C_new); x = self.norm(x)
        x = x.view(B_new, H_new, W_new, C_new)
        return x

class LocalSupervision(nn.Module):
    def __init__(self, in_channels=128, num_classes=6):
        super().__init__()
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=1, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(in_channels), nn.ReLU6())
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, dilation=1, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(in_channels), nn.ReLU6())
        self.drop = nn.Dropout(0.1); self.conv_out = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=False)
    def forward(self, x, h, w):
        local1 = self.conv3(x); local2 = self.conv1(x); x_out = self.drop(local1 + local2)
        x_out = self.conv_out(x_out); x_out = F.interpolate(x_out, size=(h, w), mode='bilinear', align_corners=False)
        return x_out

# --- VSSLayer (修改 forward 方法) ---
# --- VSSLayer (Modified forward method) ---
class VSSLayer(nn.Module):
    """
    包含 VSSBlock (Vision State Space Block) 的层。
    Layer containing VSSBlocks (Vision State Space Blocks).
    """
    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None, # Usually None in decoder
            use_checkpoint=False,
            d_state=16, # State dimension for Mamba-like components
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # 创建 VSSBlock 列表
        # Create a list of VSSBlocks
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                # VSSBlock 现在可能需要处理 4D 输入
                # VSSBlock might now need to handle 4D input
            )
            for i in range(depth)])

        self.downsample = downsample # Typically None in decoder stages

    def forward(self, x):
        """
        Input x: (B, H, W, C) from concat_back_dim
        --- 修改: 直接将 4D 张量传递给 VSSBlock ---
        --- Modification: Pass the 4D tensor directly to VSSBlock ---
        """
        # B, H, W, C = x.shape # Get shape info if needed

        # --- 不再 reshape 成 (B, L, C) ---
        # --- No longer reshape to (B, L, C) ---
        # x_reshaped = x.view(B, H * W, C)

        # 直接处理 4D 张量 x
        # Process the 4D tensor x directly
        for blk in self.blocks:
            if self.use_checkpoint:
                # 确保 checkpoint 函数处理 4D 输入
                # Ensure checkpoint function handles 4D input
                x = checkpoint.checkpoint(blk, x, use_reentrant=False) # Pass 4D tensor
            else:
                x = blk(x) # <<<--- Pass 4D tensor (B, H, W, C)

        # --- 不再需要从 3D reshape 回 4D ---
        # --- No longer need to reshape back from 3D to 4D ---
        # x = x_reshaped.view(B, H, W, C)

        # 下采样（解码器中通常不用）
        # Downsampling (usually not used in decoder)
        if self.downsample is not None:
            x = self.downsample(x) # Assume downsample handles 4D

        # 输出形状仍为 (B, H, W, C)
        # Output shape remains (B, H, W, C)
        return x


# --- MambaSegDecoder (加入 AFR) ---
# (MambaSegDecoder definition remains the same as before)
class MambaSegDecoder(nn.Module):
    def __init__(
            self, num_classes: int, encoder_channels: Union[Tuple[int, ...], List[int]] = None,
            decode_channels: int = 64, drop_path_rate: float = 0.2, d_state: int = 16,
            afr_reduction_ratio: int = 16, decoder_depths: List[int] = None,
            norm_layer_decoder = nn.LayerNorm, **kwargs
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

        for s in range(1, n_stages_encoder):
            stage_depth = decoder_depths[s-1]
            stage_dpr = dpr[current_dpr_idx : current_dpr_idx + stage_depth]
            current_dpr_idx += stage_depth
            input_features_skip = encoder_output_channels[-(s + 1)]
            input_features_below = encoder_output_channels[-s] if s == 1 else encoder_output_channels[-s]

            expand_layers.append(PatchExpand(None, dim=input_features_below, dim_scale=2, norm_layer=nn.LayerNorm))
            # Corrected upsampled_channels calculation based on PatchExpand logic
            upsampled_channels = input_features_below * 2 // 4 # Output C = C_in / 2 if dim_scale=2

            stages.append(VSSLayer( dim=input_features_skip, depth=stage_depth, attn_drop=0., drop_path=stage_dpr,
                d_state=math.ceil(input_features_skip / 6) if d_state is None else d_state,
                norm_layer=norm_layer_decoder, downsample=None, use_checkpoint=kwargs.get('use_checkpoint', False) ))

            concat_channels = upsampled_channels + input_features_skip
            concat_back_dim.append(nn.Linear(concat_channels, input_features_skip, bias=False))

            afr_modules.append(AdaptiveFeatureRecalibration( in_channels=input_features_skip, reduction_ratio=afr_reduction_ratio ))
            lsm_layers.append(LocalSupervision(input_features_skip, num_classes))

        expand_layers.append(FinalPatchExpand_X4(None, dim=encoder_output_channels[0], dim_scale=4, norm_layer=nn.LayerNorm))
        stages.append(nn.Identity())

        self.stages = nn.ModuleList(stages)
        self.expand_layers = nn.ModuleList(expand_layers)
        self.concat_back_dim = nn.ModuleList(concat_back_dim)
        self.afr_modules = nn.ModuleList(afr_modules)
        self.lsm = nn.ModuleList(lsm_layers)
        self.seg = nn.Conv2d(encoder_channels[0], num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.init_weight()

    def forward(self, skips: List[torch.Tensor], h, w):
        lres_input = skips[-1]
        ls_outputs = []
        for s in range(len(self.stages)):
            x = self.expand_layers[s](lres_input) # Output (B, H', W', C_up)
            if s < (len(self.stages) - 1):
                skip_feature = skips[-(s + 2)] # (B, C_skip, H', W')
                afr_module = self.afr_modules[s]
                skip_feature_afr = afr_module(skip_feature) # (B, C_skip, H', W')
                # Permute skip feature for concat
                skip_feature_afr_permuted = skip_feature_afr.permute(0, 2, 3, 1) # (B, H', W', C_skip)
                # Ensure x and skip_feature_afr_permuted have same H', W'
                if x.shape[1:3] != skip_feature_afr_permuted.shape[1:3]:
                     # Add interpolation if spatial sizes mismatch (shouldn't happen with standard U-Net skips)
                     print(f"Warning: Spatial size mismatch before concat in decoder stage {s}. x:{x.shape}, skip:{skip_feature_afr_permuted.shape}. Interpolating x.")
                     x = F.interpolate(x.permute(0, 3, 1, 2), size=skip_feature_afr_permuted.shape[1:3], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

                x = torch.cat((x, skip_feature_afr_permuted), dim=-1) # (B, H', W', C_up + C_skip)
                x = self.concat_back_dim[s](x) # (B, H', W', C_skip)
            # Pass to VSSLayer (now expects B, H, W, C)
            x = self.stages[s](x) # Output (B, H', W', C_skip) or (B, H_orig, W_orig, C=64)
            # Permute for next stage or seg head
            x = x.permute(0, 3, 1, 2) # (B, C_out, H', W')
            if s == (len(self.stages) - 1):
                seg_out = self.seg(x)
            elif self.training and hasattr(self, 'lsm') and s < len(self.lsm):
                ls_outputs.append(self.lsm[s](x, h, w))
            lres_input = x
        if self.training:
            lsm_loss = sum(ls_outputs) if ls_outputs else torch.tensor(0.0, device=seg_out.device)
            return seg_out, lsm_loss
        else:
            return seg_out

    def init_weight(self):
        print("Initializing weights for MambaSegDecoder...")
        for m in self.modules():
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
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x; B, C, H, W = x.size()
        x_h = self.pool_h(x); x_w = self.pool_w(x)
        y = torch.cat((x_h, x_w.permute(0, 1, 3, 2)), dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        if H > 0 and W > 0:
             x_h, x_w = torch.split(y, [H, W], dim=2)
             x_w = x_w.permute(0, 1, 3, 2)
        elif H > 0: x_h = y; x_w = torch.zeros((B, y.shape[1], 1, W), device=x.device, dtype=x.dtype)
        elif W > 0: x_h = torch.zeros((B, y.shape[1], H, 1), device=x.device, dtype=x.dtype); x_w = y.permute(0, 1, 3, 2)
        else: x_h = torch.zeros((B, y.shape[1], H, 1), device=x.device, dtype=x.dtype); x_w = torch.zeros((B, y.shape[1], 1, W), device=x.device, dtype=x.dtype)
        a_h = self.sigmoid(self.conv_h(x_h)); a_w = self.sigmoid(self.conv_w(x_w))
        out = identity * a_h * a_w
        return out


# --- 主模型类: UNetMambaCA_AFR ---
# (Main model class UNetMambaCA_AFR remains the same as before)
class UNetMambaCA_AFR(nn.Module):
    def __init__(self,
                 num_classes=6, input_channels=3, embed_dim=64,
                 afr_reduction_ratio=16, ca_reduction=32,
                 backbone_path='pretrain_weights/rest_lite.pth',
                 decode_channels=64, decoder_depths=[2, 2, 2], drop_path_rate=0.1,
                 d_state=16, patch_size=4, depths=[2, 2, 9, 2], dims=96,
                 ssm_d_state=16, ssm_ratio=2.0, ssm_rank_ratio=2.0, ssm_dt_rank="auto",
                 ssm_act_layer="silu", ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0,
                 ssm_init="v0", forward_type="v4", mlp_ratio=4.0, mlp_act_layer="gelu",
                 mlp_drop_rate=0.0, patch_norm=True, norm_layer="ln",
                 downsample_version="v2", patchembed_version="v2", gmlp=False,
                 use_checkpoint=False, **kwargs ):
        super().__init__()

        self.CA = CoordinateAttention(input_channels, reduction=ca_reduction)
        self.encoder = rest_lite(pretrained=True, weight_path=backbone_path, embed_dim=embed_dim)
        encoder_channels = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]

        self.decoder = MambaSegDecoder(
            num_classes=num_classes, encoder_channels=encoder_channels,
            drop_path_rate=drop_path_rate, d_state=d_state if d_state else ssm_d_state,
            afr_reduction_ratio=afr_reduction_ratio, decoder_depths=decoder_depths,
            norm_layer_decoder=nn.LayerNorm if norm_layer=="ln" else nn.BatchNorm2d,
            use_checkpoint=use_checkpoint
            # Pass VSSM params if needed by MambaSegDecoder
            # Example: ssm_ratio=ssm_ratio, ...
        )

    def forward(self, x):
        x_ca = self.CA(x)
        h, w = x_ca.size()[-2:]
        outputs = self.encoder(x_ca)
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 4:
             print(f"Warning: Encoder output format unexpected. Got type {type(outputs)}, len {len(outputs) if isinstance(outputs, (list, tuple)) else 'N/A'}. Expected list/tuple of 4 features.")
             if isinstance(outputs, torch.Tensor) and len(outputs.shape) == 4: outputs = [outputs] * 4
             elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                  while len(outputs) < 4: outputs.append(outputs[-1])
                  outputs = outputs[:4]
             else: raise TypeError("Encoder must return a list or tuple of 4 feature maps.")
        decoder_output = self.decoder(outputs, h, w)
        return decoder_output

# --- Example Usage (Optional) ---
# (Example usage remains the same as before)
if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 512, 512)
    model = UNetMambaCA_AFR( num_classes=6, embed_dim=64, backbone_path=None, decoder_depths=[2, 2, 2] )
    model.train()
    output_train = model(dummy_input)
    model.eval()
    output_eval = model(dummy_input)
    print("--- Model Output Shapes ---")
    if isinstance(output_train, tuple): print(f"Output (Training): Seg Map Shape={output_train[0].shape}, LSM Loss Type={type(output_train[1])}")
    else: print(f"Output (Training): Seg Map Shape={output_train.shape}")
    if isinstance(output_eval, tuple): print(f"Output (Eval): Seg Map Shape={output_eval[0].shape}")
    else: print(f"Output (Eval): Seg Map Shape={output_eval.shape}")
    try:
        from torchinfo import summary
        summary(model, input_size=(2, 3, 512, 512), device="cpu")
    except ImportError: print("\nInstall torchinfo for model summary: pip install torchinfo")
    except Exception as e: print(f"Error during torchinfo summary: {e}")

