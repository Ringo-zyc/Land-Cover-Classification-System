import torch
import os
from torch.utils.data import DataLoader
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# === 1. 导入数据集类和函数 ===
try:
    from unetmamba_model.datasets.vaihingen_dataset import (
        VaihingenDataset as YourDatasetClass,
        CLASSES, IGNORE_INDEX, train_aug, val_aug
    )
    dataset_name = "vaihingen"
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import from vaihingen_dataset: {e}")
    exit()
except NameError as e:
    print(f"CRITICAL ERROR: {e}. Check vaihingen_dataset definitions.")
    exit()

# === 2. 导入模型和损失函数 ===
try:
    from unetmamba_model.models.UNetMamba_CA_AFR_BAM_LFPM import UNetMambaCA_AFR_BAM_LFPM
    from unetmamba_model.losses.useful_loss import MSFDL_BAWLLoss
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import model or loss: {e}")
    exit()

# === 配置参数 ===
image_size = 1024
crop_size = 512

# === 3. 数据集路径 ===
VAIHINGEN_TRAIN_DATA_ROOT = f'data/vaihingen/train_{image_size}'
VAIHINGEN_VAL_DATA_ROOT = f'data/vaihingen/val_{image_size}'

# --- 训练超参数 ---
max_epoch = 100
num_classes = len(CLASSES)
ignore_index = IGNORE_INDEX
classes = CLASSES
train_batch_size = 2
val_batch_size = 1
lr = 8e-5
weight_decay = 2.5e-4
backbone_lr = 8e-6
backbone_weight_decay = 2.5e-4
afr_lr = 5e-5
bam_lr = 5e-5
lfpm_lr = 5e-5

# --- 模型配置 ---
PATCH_SIZE_VSSM = 4
IN_CHANS = 3
DEPTHS = [2, 2, 9, 2]
EMBED_DIM = 96
SSM_D_STATE = 16
SSM_RATIO = 2.0
SSM_RANK_RATIO = 2.0
SSM_DT_RANK = "auto"
SSM_ACT_LAYER = "silu"
SSM_CONV = 3
SSM_CONV_BIAS = True
SSM_DROP_RATE = 0.0
SSM_INIT = "v0"
SSM_FORWARDTYPE = "v4"
MLP_RATIO = 4.0
MLP_ACT_LAYER = "gelu"
MLP_DROP_RATE = 0.0
DROP_PATH_RATE = 0.1
PATCH_NORM = True
NORM_LAYER = "ln"
DOWNSAMPLE = "v2"
PATCHEMBED = "v2"
GMLP = False
USE_CHECKPOINT = True
REST_EMBED_DIM = 64

# === 实例化模型 ===
net = UNetMambaCA_AFR_BAM_LFPM(
    num_classes=num_classes,
    input_channels=IN_CHANS,
    embed_dim=REST_EMBED_DIM,
    afr_reduction_ratio=16,
    ca_reduction=32,
    bam_mid_channels=32,
    lfpm_compression_ratio=4,
    lfpm_dilations=[1, 6, 12, 18],
    backbone_path='pretrain_weights/rest_lite.pth',
    decoder_depths=[2, 2, 2],
    drop_path_rate=DROP_PATH_RATE,
    d_state=SSM_D_STATE,
    patch_size=PATCH_SIZE_VSSM,
    depths=DEPTHS,
    dims=EMBED_DIM,
    ssm_d_state=SSM_D_STATE,
    ssm_ratio=SSM_RATIO,
    ssm_rank_ratio=SSM_RANK_RATIO,
    ssm_dt_rank=SSM_DT_RANK,
    ssm_act_layer=SSM_ACT_LAYER,
    ssm_conv=SSM_CONV,
    ssm_conv_bias=SSM_CONV_BIAS,
    ssm_drop_rate=SSM_DROP_RATE,
    ssm_init=SSM_INIT,
    forward_type=SSM_FORWARDTYPE,
    mlp_ratio=MLP_RATIO,
    mlp_act_layer=MLP_ACT_LAYER,
    mlp_drop_rate=MLP_DROP_RATE,
    patch_norm=PATCH_NORM,
    norm_layer=NORM_LAYER,
    downsample_version=DOWNSAMPLE,
    patchembed_version=PATCHEMBED,
    gmlp=GMLP,
    use_checkpoint=USE_CHECKPOINT
)

# === 修改: Loss Definition (使用 MSFDL+BAWL) ===
loss = MSFDL_BAWLLoss(
    ignore_index=ignore_index,
    num_classes=num_classes,
    aux_weight=0.4,
    beta=0.5,
    focal_gamma=2.0,
    focal_alpha=[1.0, 1.0, 1.0, 1.0, 1.0, 2.0],  # 为 Clutter 类设置更高的权重
    dice_smooth=1.0,
    use_bawl=True,
    bawl_factor=2.0,
    bawl_dilation=1,
    aux_loss_type='focal'
)
use_aux_loss = True

# --- 权重、日志、监控配置 ---
dataset_name = "vaihingen"
model_arch_name = "unetmamba_CA_AFR_BAM_LFPM_MSFDL_BAWL"
weights_name = f"{model_arch_name}_{dataset_name}_{image_size}-e{max_epoch}"
weights_dir = f"model_weights/{dataset_name}/{model_arch_name}_{image_size}"
weights_path = weights_dir
log_name = f'{dataset_name}/{weights_name}'

monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
resume_ckpt_path = None
gpus = 'auto'

# === 4. 数据集和数据加载器实例化 ===
IMG_SUFFIX = '.tif'
MASK_SUFFIX = '.png'
try:
    img_size_tuple = (image_size, image_size) if isinstance(image_size, int) else image_size
    train_dataset = YourDatasetClass(
        data_root=VAIHINGEN_TRAIN_DATA_ROOT,
        mode='train',
        transform=train_aug,
        img_size=img_size_tuple
    )
    val_dataset = YourDatasetClass(
        data_root=VAIHINGEN_VAL_DATA_ROOT,
        mode='val',
        transform=val_aug,
        img_size=img_size_tuple
    )
    test_dataset = YourDatasetClass(
        data_root=VAIHINGEN_VAL_DATA_ROOT,
        mode='val',
        transform=val_aug,
        img_size=img_size_tuple
    )
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("CRITICAL ERROR: Datasets are empty.")
        exit()
except Exception as e:
    print(f"Dataset initialization-station error: {e}")
    exit()

# --- 数据加载器定义 ---
pin_memory = True
num_workers = 4
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=True,
    drop_last=True,
    persistent_workers=True if num_workers > 0 else False
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=pin_memory,
    drop_last=False,
    persistent_workers=True if num_workers > 0 else False
)

# --- 优化器与学习率调度器 ---
print("Setting up optimizer with differential learning rates...")
try:
    backbone_params = [p for n, p in net.named_parameters() if n.startswith('encoder.') and p.requires_grad]
    afr_params = [p for n, p in net.named_parameters() if n.startswith('decoder.afr_modules.') and p.requires_grad]
    bam_params = [p for n, p in net.named_parameters() if n.startswith('decoder.bam.') and p.requires_grad and 'edge_conv.weight' not in n]
    lfpm_params = [p for n, p in net.named_parameters() if n.startswith('decoder.lfpm.') and p.requires_grad]
    base_param_ids = set(id(p) for p in net.parameters() if p.requires_grad)
    backbone_param_ids = set(id(p) for p in backbone_params)
    afr_param_ids = set(id(p) for p in afr_params)
    bam_param_ids = set(id(p) for p in bam_params)
    lfpm_param_ids = set(id(p) for p in lfpm_params)
    other_params = [p for p in net.parameters() if id(p) in base_param_ids - backbone_param_ids - afr_param_ids - bam_param_ids - lfpm_param_ids]
    print(f"Found {len(backbone_params)} backbone, {len(afr_params)} AFR, {len(bam_params)} BAM, {len(lfpm_params)} LFPM, {len(other_params)} other params.")

    net_params = []
    if backbone_params:
        net_params.append({'params': backbone_params, 'lr': backbone_lr, 'weight_decay': backbone_weight_decay})
    if afr_params:
        net_params.append({'params': afr_params, 'lr': afr_lr, 'weight_decay': weight_decay})
    if bam_params:
        net_params.append({'params': bam_params, 'lr': bam_lr, 'weight_decay': weight_decay})
    if lfpm_params:
        net_params.append({'params': lfpm_params, 'lr': lfpm_lr, 'weight_decay': weight_decay})
    if other_params:
        net_params.append({'params': other_params, 'lr': lr, 'weight_decay': weight_decay})

    if not net_params:
        net_params = net.parameters()
    base_optimizer = torch.optim.AdamW(net_params, lr=lr)
    optimizer = Lookahead(base_optimizer)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    print("Optimizer and LR Scheduler configured.")
except Exception as e:
    print(f"ERROR setting up optimizer: {e}. Falling back to basic AdamW.")
    optimizer = Lookahead(torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)

# --- Final Config Check ---
print(f"\n--- Configuration Summary ---")
print(f"Model: {model_arch_name}")
print(f"Dataset: {dataset_name}")
print(f"Loss: MSFDL_BAWLLoss (beta={loss.beta if hasattr(loss, 'beta') else 'N/A'}, BAWL={loss.use_bawl if hasattr(loss, 'use_bawl') else 'N/A'})")
print(f"Learning Rates (Base/Backbone/AFR/BAM/LFPM): {lr}/{backbone_lr}/{afr_lr}/{bam_lr}/{lfpm_lr}")
print(f"---------------------------\n")