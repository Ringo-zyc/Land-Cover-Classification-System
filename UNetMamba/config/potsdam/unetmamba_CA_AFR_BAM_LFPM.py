# filename: config/potsdam/unetmamba_CA_AFR_BAM_LFPM.py

import torch
import os
from torch.utils.data import DataLoader
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# === 1. 导入 Potsdam 数据集类和函数 ===
try:
    from unetmamba_model.datasets.potsdam_dataset import (
        PotsdamPatchesDataset, CLASSES, PALETTE, IGNORE_INDEX, # Assuming PALETTE is defined but maybe not used here
    )
    # Potsdam has 6 classes, ignore index is typically 255
    if len(CLASSES) != 6: print(f"Warning: Expected 6 classes for Potsdam, found {len(CLASSES)}.")
    if IGNORE_INDEX != 255: print(f"Warning: Expected ignore_index 255 for Potsdam, found {IGNORE_INDEX}.")
except ImportError as e: print(f"CRITICAL ERROR: Could not import from potsdam_dataset: {e}"); exit()
except NameError as e: print(f"CRITICAL ERROR: {e}. Check potsdam_dataset definitions."); exit()

# === 2. 导入模型和损失函数 ===
try:
    from unetmamba_model.models.UNetMamba_CA_AFR_BAM_LFPM import UNetMambaCA_AFR_BAM_LFPM
    from unetmamba_model.losses.useful_loss import UnetMambaLoss # <<<--- 使用 UnetMambaLoss
except ImportError as e: print(f"CRITICAL ERROR: Could not import model or loss: {e}"); exit()


# === 配置参数 ===
PATCH_SIZE = 1024 # Using PATCH_SIZE convention from your provided file

# === 3. 数据集路径 ===
# <<<--- !!! 请务必检查这些路径是否指向分块后的数据目录 !!! --->
# <<<--- !!! Please double-check these paths point to the PATCHED data directories !!! --->
POTSDAM_TRAIN_PATCHES_ROOT = f'data/Potsdam/train_{PATCH_SIZE}'
POTSDAM_VAL_PATCHES_ROOT = f'data/Potsdam/val_{PATCH_SIZE}'

# --- 训练超参数 ---
max_epoch = 100
num_classes = len(CLASSES) # Should be 6
ignore_index = IGNORE_INDEX # Should be 255
classes = CLASSES
train_batch_size = 2 # Consistent with your provided file (small for large patches)
val_batch_size = 2   # Consistent with your provided file
lr = 5e-4            # Consistent with your provided file (same as Vaihingen)
weight_decay = 2.5e-4
backbone_lr = 5e-5     # Consistent with your provided file
backbone_weight_decay = 2.5e-4
afr_lr = 5e-4
bam_lr = 5e-4
lfpm_lr = 5e-4
image_size = (PATCH_SIZE, PATCH_SIZE) # For logging/FLOPs consistency

# --- 模型配置 ---
# VSSM parameters
PATCH_SIZE_VSSM = 4; IN_CHANS = 3; DEPTHS = [2, 2, 9, 2]; EMBED_DIM = 96
SSM_D_STATE = 16; SSM_RATIO = 2.0; SSM_RANK_RATIO = 2.0; SSM_DT_RANK = "auto"
SSM_ACT_LAYER = "silu"; SSM_CONV = 3; SSM_CONV_BIAS = True; SSM_DROP_RATE = 0.0
SSM_INIT = "v0"; SSM_FORWARDTYPE = "v4"; MLP_RATIO = 4.0; MLP_ACT_LAYER = "gelu"
MLP_DROP_RATE = 0.0; DROP_PATH_RATE = 0.1; PATCH_NORM = True; NORM_LAYER = "ln"
DOWNSAMPLE = "v2"; PATCHEMBED = "v2"; GMLP = False
USE_CHECKPOINT = True # <<<--- 建议开启 | Recommend enabling

# ResT backbone base dimension
REST_EMBED_DIM = 64

# === 实例化最终模型 ===
net = UNetMambaCA_AFR_BAM_LFPM(
    # --- Core Parameters ---
    num_classes=num_classes, input_channels=IN_CHANS, embed_dim=REST_EMBED_DIM,
    afr_reduction_ratio=16, ca_reduction=32, bam_mid_channels=32,
    lfpm_compression_ratio=4, lfpm_dilations=[1, 6, 12, 18],
    # --- Backbone Path ---
    backbone_path='pretrain_weights/rest_lite.pth', # <<<--- 确保路径正确 | Ensure path is correct
    # --- Other parameters ---
    decoder_depths=[2, 2, 2], drop_path_rate=DROP_PATH_RATE, d_state=SSM_D_STATE,
    patch_size=PATCH_SIZE_VSSM, depths=DEPTHS, dims=EMBED_DIM, ssm_d_state=SSM_D_STATE,
    ssm_ratio=SSM_RATIO, ssm_rank_ratio=SSM_RANK_RATIO, ssm_dt_rank=SSM_DT_RANK,
    ssm_act_layer=SSM_ACT_LAYER, ssm_conv=SSM_CONV, ssm_conv_bias=SSM_CONV_BIAS,
    ssm_drop_rate=SSM_DROP_RATE, ssm_init=SSM_INIT, forward_type=SSM_FORWARDTYPE,
    mlp_ratio=MLP_RATIO, mlp_act_layer=MLP_ACT_LAYER, mlp_drop_rate=MLP_DROP_RATE,
    patch_norm=PATCH_NORM, norm_layer=NORM_LAYER, downsample_version=DOWNSAMPLE,
    patchembed_version=PATCHEMBED, gmlp=GMLP, use_checkpoint=USE_CHECKPOINT
)

# --- Loss Definition ---
loss = UnetMambaLoss(ignore_index=ignore_index)
use_aux_loss = True # UnetMambaLoss uses aux loss internally if model provides it

# --- 权重、日志、监控配置 ---
model_arch_name = "unetmamba_CA_AFR_BAM_LFPM"
dataset_name = "potsdam"
weights_name = f"{model_arch_name}_{dataset_name}_patch{PATCH_SIZE}-e{max_epoch}"
weights_dir = f"model_weights/{dataset_name}_patches_{PATCH_SIZE}/{model_arch_name}"
weights_path = weights_dir
log_name = f'{dataset_name}_patches_{PATCH_SIZE}/{weights_name}'

monitor = 'val_mIoU'; monitor_mode = 'max'; save_top_k = 1; save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None; resume_ckpt_path = None; gpus = 'auto'

# === 4. 数据集和数据加载器实例化 ===
# !! 使用 PotsdamPatchesDataset !!
try:
    print(f"Initializing Potsdam Patches datasets (Patch Size: {PATCH_SIZE})...")
    # Assume PotsdamPatchesDataset handles its own augmentations internally
    train_dataset = PotsdamPatchesDataset(data_root=POTSDAM_TRAIN_PATCHES_ROOT, mode='train')
    val_dataset = PotsdamPatchesDataset(data_root=POTSDAM_VAL_PATCHES_ROOT, mode='val')
    # Using val split for testing as an example, adjust if a separate test split exists
    test_dataset = PotsdamPatchesDataset(data_root=POTSDAM_VAL_PATCHES_ROOT, mode='val')

    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    if len(train_dataset) == 0 or len(val_dataset) == 0: print("CRITICAL ERROR: Datasets are empty."); exit()
except FileNotFoundError as e: print(f"CRITICAL ERROR: Dataset directory not found: {e}. Check POTSDAM_*_PATCHES_ROOT paths."); exit()
except NameError as e: print(f"CRITICAL ERROR: Name not defined: {e}."); exit()
except Exception as e: print(f"Dataset initialization error: {e}"); exit()

# --- 数据加载器定义 ---
pin_memory = True; num_workers = 4 # <<<--- 检查 num_workers 是否合适 | Check if num_workers is suitable
train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True, persistent_workers=True if num_workers > 0 else False)
val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, num_workers=num_workers, shuffle=False, pin_memory=pin_memory, drop_last=False, persistent_workers=True if num_workers > 0 else False)

# --- 优化器与学习率调度器 ---
print("Setting up optimizer with differential learning rates...")
try:
    backbone_params = [p for n, p in net.named_parameters() if n.startswith('encoder.') and p.requires_grad]
    afr_params = [p for n, p in net.named_parameters() if n.startswith('decoder.afr_modules.') and p.requires_grad]
    bam_params = [p for n, p in net.named_parameters() if n.startswith('decoder.bam.') and p.requires_grad and 'edge_conv.weight' not in n]
    lfpm_params = [p for n, p in net.named_parameters() if n.startswith('decoder.lfpm.') and p.requires_grad]
    base_param_ids = set(id(p) for p in net.parameters() if p.requires_grad)
    backbone_param_ids=set(id(p) for p in backbone_params); afr_param_ids=set(id(p) for p in afr_params)
    bam_param_ids=set(id(p) for p in bam_params); lfpm_param_ids=set(id(p) for p in lfpm_params)
    other_params = [p for p in net.parameters() if id(p) in base_param_ids - backbone_param_ids - afr_param_ids - bam_param_ids - lfpm_param_ids]
    print(f"Found {len(backbone_params)} backbone, {len(afr_params)} AFR, {len(bam_params)} BAM, {len(lfpm_params)} LFPM, {len(other_params)} other params.")

    net_params = []
    if backbone_params: net_params.append({'params': backbone_params, 'lr': backbone_lr, 'weight_decay': backbone_weight_decay})
    if afr_params: net_params.append({'params': afr_params, 'lr': afr_lr, 'weight_decay': weight_decay})
    if bam_params: net_params.append({'params': bam_params, 'lr': bam_lr, 'weight_decay': weight_decay})
    if lfpm_params: net_params.append({'params': lfpm_params, 'lr': lfpm_lr, 'weight_decay': weight_decay})
    if other_params: net_params.append({'params': other_params, 'lr': lr, 'weight_decay': weight_decay})

    if not net_params: net_params = net.parameters()
    base_optimizer = torch.optim.AdamW(net_params, lr=lr)
    optimizer = Lookahead(base_optimizer)
    # Using CosineAnnealingLR from your provided file for Potsdam
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
    print("Optimizer and LR Scheduler configured.")
except Exception as e:
    print(f"ERROR setting up optimizer: {e}. Falling back to basic AdamW.")
    optimizer = Lookahead(torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

# --- Final Config Check ---
print(f"\n--- Configuration Summary ---")
print(f"Model: {model_arch_name}")
print(f"Dataset: {dataset_name.capitalize()} Patches ({PATCH_SIZE}x{PATCH_SIZE})")
print(f"Epochs: {max_epoch}")
print(f"Batch Size (Train/Val): {train_batch_size}/{val_batch_size}")
print(f"Learning Rates (Base/Backbone/AFR/BAM/LFPM): {lr}/{backbone_lr}/{afr_lr}/{bam_lr}/{lfpm_lr}")
print(f"Weights Save Directory: {weights_dir}")
print(f"Log Name: {log_name}")
print(f"Backbone Pretrained Weights: pretrain_weights/rest_lite.pth") # Corrected printout
print(f"Resume Checkpoint: {resume_ckpt_path}")
print(f"---------------------------\n")