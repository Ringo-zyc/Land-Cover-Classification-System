import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from typing import Optional, Union, Type, List, Tuple, Callable, Dict

from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.utils.checkpoint as checkpoint

# === 从用户提供的原始模型文件导入 VSSBlock 和 ResT ===
try:
    from unetmamba_model.classification.models.vmamba import VSSBlock, LayerNorm2d 
    from unetmamba_model.models.ResT import ResT
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import VSSBlock or ResT. Please ensure these modules are correctly placed.")
    print(f"Import error: {e}")
    print("Expected VSSBlock in: unetmamba_model.classification.models.vmamba")
    print("Expected ResT in: unetmamba_model.models.ResT")
    if 'VSSBlock' not in globals():
        class VSSBlock(nn.Module): # Minimal fallback
            def __init__(self, hidden_dim, **kwargs): super().__init__(); self.fc = nn.Linear(hidden_dim,hidden_dim); print("WARNING: Using MINIMAL FALLBACK VSSBlock.")
            def forward(self, x): return self.fc(x.view(x.size(0),-1,x.size(-1))).view(x.shape) if x.ndim==4 else self.fc(x)
    if 'ResT' not in globals():
        class ResT(nn.Module): # Minimal fallback
            def __init__(self, embed_dims, **kwargs): super().__init__(); self.embed_dims=embed_dims; print("WARNING: Using MINIMAL FALLBACK ResT.")
            def forward(self, x): return [torch.randn(x.size(0), dim, x.size(2)//(2**(i+2)), x.size(3)//(2**(i+2))).to(x.device) for i, dim in enumerate(self.embed_dims)]


def rest_lite(pretrained=True, weight_path='pretrain_weights/rest_lite.pth', embed_dims=None, **kwargs):
    if embed_dims is None: 
        embed_dims = [64, 128, 256, 512]

    rest_constructor_args = {
        'embed_dims': embed_dims,
        'num_heads': kwargs.get('num_heads', [1, 2, 4, 8]),
        'mlp_ratios': kwargs.get('mlp_ratios', [4, 4, 4, 4]),
        'qkv_bias': kwargs.get('qkv_bias', True),
        'depths': kwargs.get('depths', [2, 2, 2, 2]),
        'sr_ratios': kwargs.get('sr_ratios', [8, 4, 2, 1]),
        'apply_transform': kwargs.get('apply_transform', True)
    }
    model = ResT(**rest_constructor_args, **kwargs) 

    if pretrained and weight_path is not None:
        if os.path.exists(weight_path):
            try:
                old_dict = torch.load(weight_path, map_location='cpu')
                model_dict = model.state_dict()
                old_dict = {k: v for k, v in old_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
                if not old_dict:
                    print(f"Warning: No matching keys found in pretrained weights at {weight_path}.")
                model_dict.update(old_dict)
                model.load_state_dict(model_dict)
                print(f"Loaded pretrained weights from {weight_path} for ResT.")
            except Exception as e:
                print(f"Error loading pretrained weights for ResT from {weight_path}: {e}. Training from scratch or with partial weights.")
        else:
            print(f"Pretrained ResT weights not found at {weight_path}. Training ResT from scratch.")
    elif not pretrained:
        print("Training ResT from scratch (pretrained=False).")
    return model


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale) if norm_layer else nn.Identity()

    def forward(self, x):
        if x.ndim == 4 and x.size(1) == self.dim:
             x = x.permute(0, 2, 3, 1)
        elif x.ndim == 3 and x.size(-1) == self.dim :
            pass 
        elif x.ndim == 4 and x.size(3) == self.dim: 
            pass
        else:
            if x.ndim == 4 : 
                 x = x.permute(0, 2, 3, 1) 
                 if x.size(-1) != self.dim: 
                     raise ValueError(f"PatchExpand channel mismatch after permute. Expected {self.dim}, got {x.size(-1)}")
            else:
                raise ValueError(f"PatchExpand received unexpected input shape: {x.shape} for dim {self.dim}")

        x = self.expand(x)
        B, H, W, C_expanded = x.shape
        c_rearranged = self.dim // 2
        x = rearrange(x, 'b h w (p1 p2 c_rearranged)-> b (h p1) (w p2) c_rearranged', p1=2, p2=2, c_rearranged=c_rearranged)
        x_shape_before_norm = x.shape
        x = x.reshape(B, -1, c_rearranged)
        x = self.norm(x)
        x = x.reshape(B, x_shape_before_norm[1], x_shape_before_norm[2], c_rearranged)
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, (dim_scale**2)*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if x.ndim == 4: 
            if x.size(1) == self.dim and x.size(3) != self.dim : 
                x = x.permute(0, 2, 3, 1) 
            elif x.size(3) == self.dim: 
                pass
            else: 
                x = x.permute(0, 2, 3, 1)
                if x.size(-1) != self.dim:
                    raise ValueError(f"FinalPatchExpand_X4 channel mismatch. Expected {self.dim}, got {x.size(-1)}")
        else:
             raise ValueError(f"FinalPatchExpand_X4 received unexpected input ndim: {x.ndim}")

        x = self.expand(x) 
        B, H, W, C_expanded = x.shape
        c_rearranged = self.output_dim
        x = rearrange(x, 'b h w (p1 p2 c_rearranged)-> b (h p1) (w p2) c_rearranged', p1=self.dim_scale, p2=self.dim_scale, c_rearranged=c_rearranged)
        x_shape_before_norm = x.shape
        x = x.reshape(B, -1, self.output_dim)
        x = self.norm(x)
        x = x.reshape(B, x_shape_before_norm[1], x_shape_before_norm[2], self.output_dim)
        return x

class LightweightSpatialAttention(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x): 
        x_permuted = x.permute(0, 3, 1, 2) 
        attn = self.conv(x_permuted)
        attn = self.bn(attn)
        attn = self.sigmoid(attn)
        x_attended = x_permuted * attn
        return x_attended.permute(0, 2, 3, 1) 

class BoundaryAwareModule(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv_boundary = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_main): 
        x_permuted = x_main.permute(0, 3, 1, 2) 
        boundary_features = self.conv_boundary(x_permuted)
        boundary_features = self.bn(boundary_features)
        boundary_features = self.relu(boundary_features) 
        return x_main + boundary_features.permute(0, 2, 3, 1) 


class VSSLayer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm, 
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs, 
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block_drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path
            self.blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=block_drop_path,
                norm_layer=norm_layer, 
                attn_drop_rate=attn_drop, 
                d_state=d_state,
                **kwargs 
            ))
        self.apply(self._init_weights_vsslayer)
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer) 
        else:
            self.downsample = None

    def _init_weights_vsslayer(self, m: nn.Module):
        if hasattr(m, 'out_proj') and isinstance(m.out_proj, nn.Linear):
             nn.init.kaiming_uniform_(m.out_proj.weight, a=math.sqrt(5))

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
    def __init__(self, in_channels=128, num_classes=6):
        super().__init__()
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=1, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU6())
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, dilation=1, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU6())
        self.drop = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(in_channels, num_classes, kernel_size=1, dilation=1, stride=1, padding=0, bias=False)

    def forward(self, x, h, w): 
        local1 = self.conv3(x)
        local2 = self.conv1(x)
        x_out = self.drop(local1 + local2)
        x_out = self.conv_out(x_out)
        x_out = F.interpolate(x_out, size=(h, w), mode='bilinear', align_corners=False)
        return x_out


class MambaSegDecoder(nn.Module):
    def __init__(
            self,
            num_classes: int,
            encoder_channels: Union[Tuple[int, ...], List[int]],
            drop_path_rate: float = 0.2,
            use_lsa: bool = True,
            use_bam: bool = True,
            vss_layer_norm = nn.LayerNorm, 
            vss_block_kwargs: Optional[dict] = None, 
    ):
        super().__init__()
        self.num_classes = num_classes
        n_stages_encoder = len(encoder_channels)
        self.use_lsa = use_lsa
        self.use_bam = use_bam
        self._vss_block_kwargs = vss_block_kwargs if vss_block_kwargs is not None else {}
        num_total_decoder_vss_blocks = (n_stages_encoder - 1) * 2
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_total_decoder_vss_blocks)]
        self.stages_vss = nn.ModuleList()
        self.expand_layers = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        self.lsm_layers = nn.ModuleList() 
        if self.use_lsa:
            self.lsa_layers = nn.ModuleList()
        if self.use_bam:
            self.bam_layers = nn.ModuleList()

        for s in range(n_stages_encoder - 1):
            if s == 0:
                expand_input_dim = encoder_channels[-1] 
            else:
                expand_input_dim = encoder_channels[-(s+1)] 
            self.expand_layers.append(PatchExpand(
                input_resolution=None, 
                dim=expand_input_dim,
                dim_scale=2,
                norm_layer=vss_layer_norm,
            ))
            current_stage_dim = encoder_channels[-(s+2)] 
            self.concat_back_dim.append(nn.Linear( (expand_input_dim//2) + current_stage_dim, current_stage_dim))
            stage_dpr = dpr[2*s : 2*s + 2]
            current_vss_block_kwargs = self._vss_block_kwargs.copy()
            if 'd_state' not in current_vss_block_kwargs: 
                 current_vss_block_kwargs['d_state'] = math.ceil(current_stage_dim / 3) 
            self.stages_vss.append(VSSLayer(
                dim=current_stage_dim,
                depth=2,
                attn_drop=0., 
                drop_path=stage_dpr,
                norm_layer=vss_layer_norm,
                use_checkpoint=False, 
                **current_vss_block_kwargs 
            ))
            if self.use_lsa:
                self.lsa_layers.append(LightweightSpatialAttention(dim=current_stage_dim))
            if self.use_bam:
                self.bam_layers.append(BoundaryAwareModule(dim=current_stage_dim))
            self.lsm_layers.append(LocalSupervision(current_stage_dim, num_classes))

        final_expand_input_dim = encoder_channels[0] 
        self.expand_layers.append(FinalPatchExpand_X4(
            input_resolution=None,
            dim=final_expand_input_dim,
            dim_scale=4,
            norm_layer=vss_layer_norm,
        ))
        self.seg = nn.Conv2d(final_expand_input_dim, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.init_weight()

    def forward(self, skips, h_img, w_img):
        lres_input = skips[-1] 
        ls_outputs = [] 
        num_main_stages = len(self.stages_vss)

        for s in range(num_main_stages):
            x_expanded = self.expand_layers[s](lres_input) 
            skip_feature = skips[-(s+2)].permute(0, 2, 3, 1) 
            if x_expanded.shape[1:3] != skip_feature.shape[1:3]:
                target_h, target_w = skip_feature.shape[1], skip_feature.shape[2]
                x_expanded_permuted = x_expanded.permute(0,3,1,2)
                x_expanded_resized = F.interpolate(x_expanded_permuted, size=(target_h, target_w), mode='bilinear', align_corners=False)
                x_expanded = x_expanded_resized.permute(0,2,3,1)
            x_concat = torch.cat((x_expanded, skip_feature), dim=-1) 
            x = self.concat_back_dim[s](x_concat) 
            x = self.stages_vss[s](x) 
            if self.use_lsa:
                x = self.lsa_layers[s](x) 
            if self.use_bam:
                x = self.bam_layers[s](x) 
            x_for_lsm_and_next_stage = x.permute(0, 3, 1, 2) 
            if self.training and self.lsm_layers and s < len(self.lsm_layers): 
                ls_outputs.append(self.lsm_layers[s](x_for_lsm_and_next_stage, h_img, w_img))
            lres_input = x_for_lsm_and_next_stage

        seg_features = self.expand_layers[-1](lres_input) 
        seg_features_permuted = seg_features.permute(0, 3, 1, 2) 
        seg_out = self.seg(seg_features_permuted)

        if self.training:
            return seg_out, sum(ls_outputs) if ls_outputs else 0.0 
        else:
            return seg_out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)): 
                if m.bias is not None: nn.init.constant_(m.bias, 0)
                if m.weight is not None: nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class UNetMamba(nn.Module):
    def __init__(self,
                 num_classes: int = 6,
                 embed_dim: int = 64, 
                 pretrained_encoder: bool = True,
                 backbone_path: str ='pretrain_weights/rest_lite.pth',
                 rest_embed_dims: Optional[List[int]] = None, 
                 rest_kwargs: Optional[dict] = None, 
                 drop_path_rate_decoder: float = 0.2,
                 use_lsa_in_decoder: bool = True,
                 use_bam_in_decoder: bool = True,
                 vss_layer_norm_in_decoder = nn.LayerNorm, 
                 vss_block_kwargs_in_decoder: Optional[dict] = None, 
                 **kwargs 
                 ):
        super().__init__()
        if rest_embed_dims is None:
            encoder_channels = [embed_dim * (2**i) for i in range(4)] 
        else:
            encoder_channels = rest_embed_dims
        
        _rest_kwargs = rest_kwargs if rest_kwargs is not None else {}
        self.encoder = rest_lite(
            pretrained=pretrained_encoder,
            weight_path=backbone_path,
            embed_dims=encoder_channels, 
            **_rest_kwargs,
            **kwargs 
        )
        self.decoder = MambaSegDecoder(
            num_classes=num_classes,
            encoder_channels=encoder_channels,
            drop_path_rate=drop_path_rate_decoder,
            use_lsa=use_lsa_in_decoder,
            use_bam=use_bam_in_decoder,
            vss_layer_norm=vss_layer_norm_in_decoder,
            vss_block_kwargs=vss_block_kwargs_in_decoder
        )

    def forward(self, x):
        h_img, w_img = x.size()[-2:]
        outputs = self.encoder(x)

        # === 修改点: 接受元组并转换为列表 ===
        if isinstance(outputs, tuple) and len(outputs) == 4:
            outputs = list(outputs) # 将包含4个特征图的元组转换为列表
        elif not (isinstance(outputs, list) and len(outputs) == 4):
            # 如果既不是4元组也不是4列表，则格式不正确
            len_str = str(len(outputs)) if isinstance(outputs, (list, tuple)) else 'N/A'
            raise ValueError(f"Encoder output must be a list or tuple of 4 feature maps. Got: {type(outputs)} with len {len_str}")
        # ====================================

        if self.training:
            seg_map, lsm_loss = self.decoder(outputs, h_img, w_img)
            return seg_map, lsm_loss
        else:
            seg_map = self.decoder(outputs, h_img, w_img)
            return seg_map

if __name__ == '__main__':
    print("Starting UNetMamba_LSA_BAM_Model_V2.py test with corrected imports...")
    decoder_vss_block_config = { "d_state": 16, "ssm_ratio": 2.0, "mlp_ratio": 4.0 }
    example_rest_kwargs = {}
    model = UNetMamba(
        num_classes=6, embed_dim=64, pretrained_encoder=False,
        backbone_path='dummy_rest_lite_weights.pth', use_lsa_in_decoder=True, use_bam_in_decoder=True,
        rest_kwargs=example_rest_kwargs, vss_block_kwargs_in_decoder=decoder_vss_block_config
    )
    model.train()
    print(f"Model Instantiated: UNetMamba with LSA and BAM")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {num_params / 1e6:.2f}M")
    dummy_input = torch.randn(2, 3, 256, 256)
    try:
        print("Testing forward pass in training mode...")
        output_train, lsm_loss_train = model(dummy_input)
        print(f"  Train mode: Output shape: {output_train.shape}, LSM loss type: {type(lsm_loss_train)}")
        model.eval()
        print("Testing forward pass in evaluation mode...")
        output_eval = model(dummy_input)
        print(f"  Eval mode: Output shape: {output_eval.shape}")
        print("Basic model test PASSED.")
    except Exception as e:
        print(f"Error during model test: {e}")
        import traceback
        traceback.print_exc()
    print("\nTesting Baseline Configuration (LSA=False, BAM=False)...")
    model_baseline = UNetMamba(
        num_classes=6, embed_dim=64, pretrained_encoder=False, backbone_path='dummy_rest_lite_weights.pth',
        use_lsa_in_decoder=False, use_bam_in_decoder=False, 
        rest_kwargs=example_rest_kwargs, vss_block_kwargs_in_decoder=decoder_vss_block_config
    )
    model_baseline.eval()
    try:
        output_baseline_eval = model_baseline(dummy_input)
        print(f"  Baseline Eval mode: Output shape: {output_baseline_eval.shape}")
        baseline_params = sum(p.numel() for p in model_baseline.parameters() if p.requires_grad)
        print(f"  Baseline Parameters: {baseline_params / 1e6:.2f}M")
        print("Baseline model test PASSED.")
    except Exception as e:
        print(f"Error during baseline model test: {e}")
        import traceback
        traceback.print_exc()
