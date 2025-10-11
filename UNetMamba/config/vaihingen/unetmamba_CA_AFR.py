# filename: config/vaihingen/unetmamba_CA_AFR.py

import torch
import os
from torch.utils.data import DataLoader
from catalyst.contrib.nn import Lookahead
from catalyst import utils
# fvcore imports removed as FLOPs calculation is removed

# === 1. 导入数据集类和函数 ===
# Ensure these imports match your project structure and file contents
try:
    from unetmamba_model.datasets.vaihingen_dataset import (
        VaihingenDataset,
        CLASSES,
        IGNORE_INDEX,
        train_aug, # Make sure train_aug and val_aug are defined/imported correctly
        val_aug
    )
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import from vaihingen_dataset: {e}")
    print("Please ensure 'unetmamba_model/datasets/vaihingen_dataset.py' exists and defines necessary classes/functions.")
    exit()
except NameError as e:
    print(f"CRITICAL ERROR: {e}. A required name (e.g., CLASSES) is not defined in vaihingen_dataset.")
    exit()

# === 2. 导入模型和损失函数 ===
try:
    # === 使用正确的相对路径导入 AFR 模型 ===
    from unetmamba_model.models.UNetMamba_CA_AFR import UNetMambaCA_AFR # <<<--- CORRECTED IMPORT
    from unetmamba_model.losses import UnetMambaLoss # Keep loss import
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import model 'UNetMambaCA_AFR' from 'unetmamba_model.models' or 'UnetMambaLoss': {e}")
    print("Ensure 'unetmamba_model/models/UNetMamba_CA_AFR.py' exists and defines the UNetMambaCA_AFR class.")
    print("Also ensure losses are correctly defined in 'unetmamba_model.losses'.")
    exit()


# === 配置参数 ===
image_size = 1024 # Or the size you intend to use (e.g., 512)
# PATCH_SIZE_PARAM = image_size # This seems unused later, maybe remove?
crop_size = 512 # Example crop size, adjust if needed based on train_aug/val_aug

# === 3. 数据集路径 (修正) ===
# Use relative paths from your project root or absolute paths
# === 修改: 恢复原始路径格式 ===
VAIHINGEN_TRAIN_DATA_ROOT = f'data/vaihingen/train_{image_size}' # <<<--- CORRECTED PATH
VAIHINGEN_VAL_DATA_ROOT = f'data/vaihingen/val_{image_size}'   # <<<--- CORRECTED PATH

# --- 训练超参数 ---
max_epoch = 100
num_classes = len(CLASSES)
ignore_index = IGNORE_INDEX # Use imported IGNORE_INDEX
classes = CLASSES
train_batch_size = 8
val_batch_size = 1 # Often 1 for validation if images are large
lr = 6e-4 # Main learning rate
weight_decay = 2.5e-4 # Weight decay for non-backbone/non-AFR layers
backbone_lr = 6e-5 # Learning rate for the ResT backbone
backbone_weight_decay = 2.5e-4 # Weight decay for the ResT backbone
afr_lr = 6e-4 # <<--- Learning rate for AFR modules (can tune this)

# --- 模型配置 ---
# VSSM parameters (Check if UNetMambaCA_AFR still uses these)
# If UNetMambaCA_AFR primarily uses ResT + MambaSegDecoder, these might only be relevant
# if MambaSegDecoder internally uses VSSM components. Assuming they are needed.
PATCH_SIZE_VSSM = 4
IN_CHANS = 3
DEPTHS = [2, 2, 9, 2]
EMBED_DIM = 96 # Base dim for VSSM-like parts if used? Or ResT base dim? Clarify needed.
                # Original ResT uses embed_dims=[64, 128, 256, 512]. Let's assume UNetMambaCA_AFR takes embed_dim=64 for ResT.
REST_EMBED_DIM = 64 # <<--- Explicitly define for ResT

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
USE_CHECKPOINT = False # Set based on memory needs

# === 实例化 AFR 模型 ===
net = UNetMambaCA_AFR(
    # --- Core Parameters ---
    num_classes=num_classes,
    input_channels=IN_CHANS,
    embed_dim=REST_EMBED_DIM, # <<--- Use ResT base dim
    afr_reduction_ratio=16, # <<--- Added AFR parameter (tuneable)
    ca_reduction=1, # Keep CA parameter if needed by the model (Original config had 1, maybe should be 32?)

    # --- Backbone Path ---
    # pretrained=None, # Set to None - backbone loading handled internally via backbone_path
    backbone_path='pretrain_weights/rest_lite.pth', # <<--- Path to ResT weights

    # --- Other parameters (VSSM-like, if needed by decoder/model) ---
    # Pass these only if UNetMambaCA_AFR's __init__ expects them
    patch_size=PATCH_SIZE_VSSM,
    # in_chans=IN_CHANS, # Already passed as input_channels
    depths=DEPTHS,
    dims=EMBED_DIM, # If different from ResT embed_dim, clarify which 'dims' this refers to
    ssm_d_state=SSM_D_STATE,
    ssm_ratio=SSM_RATIO,
    ssm_rank_ratio=SSM_RANK_RATIO,
    ssm_dt_rank=("auto" if SSM_DT_RANK == "auto" else int(SSM_DT_RANK)),
    ssm_act_layer=SSM_ACT_LAYER,
    ssm_conv=SSM_CONV,
    ssm_conv_bias=SSM_CONV_BIAS,
    ssm_drop_rate=SSM_DROP_RATE,
    ssm_init=SSM_INIT,
    forward_type=SSM_FORWARDTYPE,
    mlp_ratio=MLP_RATIO,
    mlp_act_layer=MLP_ACT_LAYER,
    mlp_drop_rate=MLP_DROP_RATE,
    drop_path_rate=DROP_PATH_RATE,
    patch_norm=PATCH_NORM,
    norm_layer=NORM_LAYER,
    downsample_version=DOWNSAMPLE,
    patchembed_version=PATCHEMBED,
    gmlp=GMLP,
    use_checkpoint=USE_CHECKPOINT
)

# --- Loss Definition ---
loss = UnetMambaLoss(ignore_index=ignore_index)
use_aux_loss = True # Does UNetMambaCA_AFR return aux loss (LSM)? Assume yes.

# --- 权重、日志、监控配置 ---
model_arch_name = "unetmamba_CA_AFR" # Model name for paths
weights_name = f"{model_arch_name}_vaihingen_{image_size}-e{max_epoch}" # Filename prefix
weights_dir = f"model_weights/vaihingen/{model_arch_name}_{image_size}" # Directory to save weights
weights_path = weights_dir # Lightning Trainer's dirpath often refers to the directory
log_name = f'vaihingen/{weights_name}' # Logger experiment name

monitor = 'val_mIoU' # Metric to monitor
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
# === 移除 base_ckpt_path, 确保 pretrained_ckpt_path 为 None ===
pretrained_ckpt_path = None # No full model checkpoint to load initially
resume_ckpt_path = None # Set this path if resuming training of *this* AFR model
gpus = 'auto'

# === 4. 数据集实例化 ===
# Ensure dataset paths and parameters are correct
IMG_SUFFIX = '.tif'
MASK_SUFFIX = '.png'

try:
    # Assuming train_aug and val_aug handle necessary transformations including ToTensor and Normalize
    img_size_tuple = (image_size, image_size) if image_size else (crop_size, crop_size) # Use image_size if defined
    print(f"Initializing datasets with image size: {img_size_tuple}")
    print(f"Train data root: {os.path.abspath(VAIHINGEN_TRAIN_DATA_ROOT)}")
    print(f"Val data root: {os.path.abspath(VAIHINGEN_VAL_DATA_ROOT)}")

    train_dataset = VaihingenDataset(data_root=VAIHINGEN_TRAIN_DATA_ROOT, mode='train',
                                     img_suffix=IMG_SUFFIX, mask_suffix=MASK_SUFFIX,
                                     transform=train_aug, mosaic_ratio=0.25, img_size=img_size_tuple)
    val_dataset = VaihingenDataset(data_root=VAIHINGEN_VAL_DATA_ROOT, mode='val',
                                   img_suffix=IMG_SUFFIX, mask_suffix=MASK_SUFFIX,
                                   transform=val_aug, img_size=img_size_tuple)
    # test_dataset is defined but not used in standard training loop, keep if needed for inference later
    test_dataset = VaihingenDataset(data_root=VAIHINGEN_VAL_DATA_ROOT, mode='val', # Often test uses val data or separate test set
                                    img_suffix=IMG_SUFFIX, mask_suffix=MASK_SUFFIX,
                                    transform=val_aug, img_size=img_size_tuple)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("CRITICAL ERROR: One or both datasets are empty. Check data paths and dataset implementation.")
        exit()

except FileNotFoundError as e:
    print(f"CRITICAL ERROR: Dataset directory not found: {e}. Check paths '{VAIHINGEN_TRAIN_DATA_ROOT}' and '{VAIHINGEN_VAL_DATA_ROOT}'.")
    exit()
except NameError as e:
    print(f"CRITICAL ERROR: A required name is not defined: {e}.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during dataset initialization: {e}")
    exit()


# --- 数据加载器定义 ---
pin_memory = True
num_workers = 4 # Adjust based on your system
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=num_workers,
                          pin_memory=pin_memory,
                          shuffle=True,
                          drop_last=True,
                          persistent_workers=True if num_workers > 0 else False)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=num_workers,
                        shuffle=False,
                        pin_memory=pin_memory,
                        drop_last=False,
                        persistent_workers=True if num_workers > 0 else False)

# --- 优化器与学习率调度器 ---
# Define parameter groups for differential learning rates
print("Setting up optimizer with differential learning rates...")
try:
    # Group 1: AFR modules
    afr_params = [p for n, p in net.named_parameters() if 'afr_modules' in n and p.requires_grad]
    print(f"Found {len(afr_params)} parameters in AFR modules.")

    # Group 2: Backbone (assuming it's named 'encoder' within UNetMambaCA_AFR)
    backbone_params = [p for n, p in net.named_parameters() if n.startswith('encoder.') and p.requires_grad]
    print(f"Found {len(backbone_params)} parameters in backbone ('encoder.').")
    if not backbone_params:
         print("Warning: No parameters found starting with 'encoder.'. Check model structure if backbone LR is desired.")

    # Group 3: Rest of the network
    base_param_ids = set(id(p) for p in net.parameters() if p.requires_grad)
    afr_param_ids = set(id(p) for p in afr_params)
    backbone_param_ids = set(id(p) for p in backbone_params)
    other_params = [p for p in net.parameters() if id(p) in base_param_ids - afr_param_ids - backbone_param_ids]
    print(f"Found {len(other_params)} parameters in other parts of the network.")

    # Create optimizer parameter groups
    net_params = []
    if backbone_params:
        net_params.append({'params': backbone_params, 'lr': backbone_lr, 'weight_decay': backbone_weight_decay})
    if afr_params:
        net_params.append({'params': afr_params, 'lr': afr_lr, 'weight_decay': weight_decay}) # Use specific AFR LR
    if other_params:
        net_params.append({'params': other_params, 'lr': lr, 'weight_decay': weight_decay}) # Use main LR for the rest

    if not net_params:
         print("Warning: No parameters found for optimizer. Using all parameters with base LR.")
         net_params = net.parameters() # Fallback

    # Define the base optimizer
    base_optimizer = torch.optim.AdamW(net_params, lr=lr) # Base LR might be overridden by group LR
    # Wrap with Lookahead
    optimizer = Lookahead(base_optimizer)
    # Define the learning rate scheduler (Using CosineAnnealingWarmRestarts from original config)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6) # Keep original scheduler settings
    print("Optimizer and LR Scheduler configured.")

except Exception as e:
    print(f"ERROR setting up optimizer: {e}. Check model parameter names ('afr_modules', 'encoder.').")
    print("Falling back to optimizing all parameters with base LR.")
    optimizer = Lookahead(torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)


# --- Final Config Check ---
print(f"\n--- Configuration Summary ---")
print(f"Model: {model_arch_name}")
print(f"Dataset: Vaihingen")
print(f"Image Size: {image_size}")
print(f"Epochs: {max_epoch}")
print(f"Batch Size (Train/Val): {train_batch_size}/{val_batch_size}")
print(f"Learning Rates (Base/Backbone/AFR): {lr}/{backbone_lr}/{afr_lr}")
print(f"Weights Save Directory: {weights_dir}")
print(f"Log Name: {log_name}")
print(f"Backbone Pretrained Weights: pretrain_weights/rest_lite.pth")
print(f"Resume Checkpoint: {resume_ckpt_path}")
print(f"---------------------------\n")
