import torch
import os
from torch.utils.data import DataLoader
from catalyst.contrib.nn import Lookahead
from catalyst import utils
# import albumentations as albu # Not needed if using PotsdamPatchesDataset internal aug

# === 1. 导入 Potsdam 数据集类和函数 ===
try:
    # 假设 potsdam_dataset.py 在 unetmamba_model/datasets/ 目录下
    # Assume potsdam_dataset.py is in unetmamba_model/datasets/
    from unetmamba_model.datasets.potsdam_dataset import (
        PotsdamPatchesDataset, # <<<--- 使用处理 Patches 的类 | Use class for Patches
        CLASSES,
        PALETTE, # Potsdam 通常有调色板 | Potsdam usually has palette
        IGNORE_INDEX, # Potsdam 通常是 255 | Usually 255 for Potsdam
        # train_aug/val_aug 可能在 Dataset 类内部处理 | train_aug/val_aug might be handled inside Dataset class
    )
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import from potsdam_dataset: {e}")
    exit()
except NameError as e:
    print(f"CRITICAL ERROR: {e}. Check potsdam_dataset definitions.")
    exit()

# === 2. 导入模型和损失函数 ===
try:
    from unetmamba_model.models.UNetMamba_CA_AFR_BAM import UNetMambaCA_AFR_BAM
    from unetmamba_model.losses import UnetMambaLoss, CrossEntropyLoss # Ensure necessary losses are imported
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import model 'UNetMambaCA_AFR_BAM' or 'UnetMambaLoss': {e}")
    exit()


# === 配置参数 ===
# !! 与你分块脚本使用的参数保持一致 !!
# !! Keep consistent with your patching script parameters !!
PATCH_SIZE = 1024 # <<<--- 确认这个尺寸 | Confirm this size
# STRIDE = 1024 # If needed by dataset class

# === 3. 数据集路径 ===
# !! 指向包含分块后图像和标签的 *父* 目录 !!
# !! Point to the *parent* directory containing the patched images and masks !!
POTSDAM_TRAIN_PATCHES_ROOT = f'data/Potsdam/train_{PATCH_SIZE}' # <<<--- 检查/修改此路径 | Check/Modify path
POTSDAM_VAL_PATCHES_ROOT = f'data/Potsdam/val_{PATCH_SIZE}'   # <<<--- 检查/修改此路径 | Check/Modify path
# POTSDAM_TEST_PATCHES_ROOT = f'data/Potsdam/test_{PATCH_SIZE}' # <<<--- 如果有测试分块 | If test patches exist

# --- 训练超参数 ---
max_epoch = 100
num_classes = len(CLASSES) # Potsdam 通常是 6 | Usually 6 for Potsdam
ignore_index = IGNORE_INDEX # Potsdam 通常是 255 | Usually 255 for Potsdam
classes = CLASSES
# !! 对于 1024x1024 的 Potsdam 分块，Batch Size 可能需要很小 !!
# !! Batch size might need to be small for 1024x1024 Potsdam patches !!
train_batch_size = 2 # 尝试值，根据显存调整 | Tentative value, adjust based on VRAM
val_batch_size = 2   # 尝试值 | Tentative value
lr = 5e-4 # 参考原始 Potsdam 配置 | Refer to original Potsdam config
weight_decay = 2.5e-4
backbone_lr = 5e-5 # 参考原始 Potsdam 配置 | Refer to original Potsdam config
backbone_weight_decay = 2.5e-4
afr_lr = 5e-4 # 与主 LR 相同或单独调整 | Same as main LR or tune separately
bam_lr = 5e-4 # 与主 LR 相同或单独调整 | Same as main LR or tune separately
# image_size 用于 FLOPs 计算等，设为 patch 尺寸元组
# image_size used for FLOPs etc., set to patch size tuple
image_size = (PATCH_SIZE, PATCH_SIZE)

# --- 模型配置 ---
# VSSM parameters (If needed by the model) - Keep consistent if model uses them
PATCH_SIZE_VSSM = 4; IN_CHANS = 3; DEPTHS = [2, 2, 9, 2]; EMBED_DIM = 96
SSM_D_STATE = 16; SSM_RATIO = 2.0; SSM_RANK_RATIO = 2.0; SSM_DT_RANK = "auto"
SSM_ACT_LAYER = "silu"; SSM_CONV = 3; SSM_CONV_BIAS = True; SSM_DROP_RATE = 0.0
SSM_INIT = "v0"; SSM_FORWARDTYPE = "v4"; MLP_RATIO = 4.0; MLP_ACT_LAYER = "gelu"
MLP_DROP_RATE = 0.0; DROP_PATH_RATE = 0.1; PATCH_NORM = True; NORM_LAYER = "ln"
DOWNSAMPLE = "v2"; PATCHEMBED = "v2"; GMLP = False; USE_CHECKPOINT = False # Potsdam patches might fit without checkpointing

# ResT backbone base dimension
REST_EMBED_DIM = 64

# === 实例化 AFR+BAM 模型 ===
net = UNetMambaCA_AFR_BAM(
    # --- Core Parameters ---
    num_classes=num_classes, # Should be 6 for Potsdam
    input_channels=IN_CHANS,
    embed_dim=REST_EMBED_DIM,
    afr_reduction_ratio=16,
    ca_reduction=32, # Or 1, check your CA implementation
    bam_mid_channels=32,

    # --- Backbone Path ---
    backbone_path='pretrain_weights/rest_lite.pth',

    # --- Other parameters (Pass through if needed) ---
    decoder_depths=[2, 2, 2], # Example decoder depths
    drop_path_rate=DROP_PATH_RATE,
    d_state=SSM_D_STATE,
    # Pass other VSSM params if needed...
    patch_size=PATCH_SIZE_VSSM, depths=DEPTHS, dims=EMBED_DIM, ssm_d_state=SSM_D_STATE,
    ssm_ratio=SSM_RATIO, ssm_rank_ratio=SSM_RANK_RATIO, ssm_dt_rank=SSM_DT_RANK,
    ssm_act_layer=SSM_ACT_LAYER, ssm_conv=SSM_CONV, ssm_conv_bias=SSM_CONV_BIAS,
    ssm_drop_rate=SSM_DROP_RATE, ssm_init=SSM_INIT, forward_type=SSM_FORWARDTYPE,
    mlp_ratio=MLP_RATIO, mlp_act_layer=MLP_ACT_LAYER, mlp_drop_rate=MLP_DROP_RATE,
    patch_norm=PATCH_NORM, norm_layer=NORM_LAYER, downsample_version=DOWNSAMPLE,
    patchembed_version=PATCHEMBED, gmlp=GMLP, use_checkpoint=USE_CHECKPOINT
)

# --- Loss Definition ---
loss = UnetMambaLoss(ignore_index=ignore_index) # ignore_index should be 255 for Potsdam
use_aux_loss = True

# --- 权重、日志、监控配置 ---
model_arch_name = "unetmamba_CA_AFR_BAM"
weights_name = f"{model_arch_name}_potsdam_patch{PATCH_SIZE}-e{max_epoch}" # <<<--- Updated names
weights_dir = f"model_weights/potsdam_patches_{PATCH_SIZE}/{model_arch_name}" # <<<--- Updated names
weights_path = weights_dir
log_name = f'potsdam_patches_{PATCH_SIZE}/{weights_name}' # <<<--- Updated names

monitor = 'val_mIoU'; monitor_mode = 'max'; save_top_k = 1; save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
resume_ckpt_path = None
gpus = 'auto'

# === 4. 数据集和数据加载器实例化 ===
# !! 使用 PotsdamPatchesDataset !!
# !! Use PotsdamPatchesDataset !!
try:
    print(f"Initializing Potsdam Patches datasets...")
    print(f"Train patches root: {os.path.abspath(POTSDAM_TRAIN_PATCHES_ROOT)}")
    print(f"Val patches root: {os.path.abspath(POTSDAM_VAL_PATCHES_ROOT)}")

    # 假设 PotsdamPatchesDataset 内部处理增强 | Assume PotsdamPatchesDataset handles augmentations internally
    train_dataset = PotsdamPatchesDataset(data_root=POTSDAM_TRAIN_PATCHES_ROOT, mode='train')
    val_dataset = PotsdamPatchesDataset(data_root=POTSDAM_VAL_PATCHES_ROOT, mode='val')
    # 实例化测试集 (如果需要) | Instantiate test set (if needed)
    test_dataset = PotsdamPatchesDataset(data_root=POTSDAM_VAL_PATCHES_ROOT, mode='val') # Example using val as test

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("CRITICAL ERROR: Datasets are empty."); exit()
except FileNotFoundError as e: print(f"CRITICAL ERROR: Dataset directory not found: {e}. Check POTSDAM_*_PATCHES_ROOT paths."); exit()
except NameError as e: print(f"CRITICAL ERROR: Name not defined: {e}."); exit()
except Exception as e: print(f"Dataset initialization error: {e}"); exit()

# --- 数据加载器定义 ---
pin_memory = True; num_workers = 4
train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, num_workers=num_workers,
                          pin_memory=pin_memory, shuffle=True, drop_last=True,
                          persistent_workers=True if num_workers > 0 else False)
val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=pin_memory, drop_last=False,
                        persistent_workers=True if num_workers > 0 else False)

# --- 优化器与学习率调度器 ---
print("Setting up optimizer with differential learning rates...")
try:
    backbone_params = [p for n, p in net.named_parameters() if n.startswith('encoder.') and p.requires_grad]
    afr_params = [p for n, p in net.named_parameters() if n.startswith('decoder.afr_modules.') and p.requires_grad]
    bam_params = [p for n, p in net.named_parameters() if n.startswith('decoder.bam.') and p.requires_grad and 'edge_conv.weight' not in n]
    base_param_ids = set(id(p) for p in net.parameters() if p.requires_grad)
    backbone_param_ids = set(id(p) for p in backbone_params)
    afr_param_ids = set(id(p) for p in afr_params)
    bam_param_ids = set(id(p) for p in bam_params)
    other_params = [p for p in net.parameters() if id(p) in base_param_ids - backbone_param_ids - afr_param_ids - bam_param_ids]
    print(f"Found {len(backbone_params)} backbone, {len(afr_params)} AFR, {len(bam_params)} BAM, {len(other_params)} other params.")

    net_params = []
    if backbone_params: net_params.append({'params': backbone_params, 'lr': backbone_lr, 'weight_decay': backbone_weight_decay})
    if afr_params: net_params.append({'params': afr_params, 'lr': afr_lr, 'weight_decay': weight_decay})
    if bam_params: net_params.append({'params': bam_params, 'lr': bam_lr, 'weight_decay': weight_decay})
    if other_params: net_params.append({'params': other_params, 'lr': lr, 'weight_decay': weight_decay})

    if not net_params: net_params = net.parameters()

    base_optimizer = torch.optim.AdamW(net_params, lr=lr)
    optimizer = Lookahead(base_optimizer)
    # 使用原始 Potsdam 配置中的 CosineAnnealingLR 调度器
    # Use CosineAnnealingLR scheduler from original Potsdam config
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
    print("Optimizer and LR Scheduler configured.")

except Exception as e:
    print(f"ERROR setting up optimizer: {e}. Falling back to basic AdamW.")
    optimizer = Lookahead(torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

# --- Final Config Check ---
print(f"\n--- Configuration Summary ---")
print(f"Model: {model_arch_name}")
print(f"Dataset: Potsdam Patches ({PATCH_SIZE}x{PATCH_SIZE})")
print(f"Epochs: {max_epoch}")
print(f"Batch Size (Train/Val): {train_batch_size}/{val_batch_size}")
print(f"Learning Rates (Base/Backbone/AFR/BAM): {lr}/{backbone_lr}/{afr_lr}/{bam_lr}")
print(f"Weights Save Directory: {weights_dir}")
print(f"Log Name: {log_name}")
print(f"Backbone Pretrained Weights: pretrain_weights/rest_lite.pth")
print(f"Resume Checkpoint: {resume_ckpt_path}")
print(f"---------------------------\n")
