import torch
import os
from torch.utils.data import DataLoader
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# === 1. 导入数据集类和函数 ===
try:
    from unetmamba_model.datasets.vaihingen_dataset import (
        VaihingenDataset, CLASSES, IGNORE_INDEX, train_aug, val_aug
    )
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import from vaihingen_dataset: {e}")
    exit()
except NameError as e:
    print(f"CRITICAL ERROR: {e}. Check vaihingen_dataset definitions.")
    exit()

# === 2. 导入模型和损失函数 ===
try:
    # === 修改: 导入新的 BAM 模型 ===
    from unetmamba_model.models.UNetMamba_CA_AFR_BAM import UNetMambaCA_AFR_BAM # <<<--- Import BAM model
    from unetmamba_model.losses import UnetMambaLoss
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import model 'UNetMambaCA_AFR_BAM' or 'UnetMambaLoss': {e}")
    print("Ensure 'unetmamba_model/models/UNetMamba_CA_AFR_BAM.py' exists and defines the class.")
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
train_batch_size = 8
val_batch_size = 1
lr = 6e-4 # 主学习率 | Main learning rate
weight_decay = 2.5e-4
backbone_lr = 6e-5 # Backbone 学习率 | Backbone learning rate
backbone_weight_decay = 2.5e-4
afr_lr = 6e-4 # AFR 模块学习率 | AFR module learning rate
bam_lr = 6e-4 # <<<--- 新增: BAM 模块学习率 (可以设为与 lr 相同) | New: BAM module learning rate (can be same as lr)

# --- 模型配置 ---
# VSSM parameters (If needed by the model)
PATCH_SIZE_VSSM = 4; IN_CHANS = 3; DEPTHS = [2, 2, 9, 2]; EMBED_DIM = 96
SSM_D_STATE = 16; SSM_RATIO = 2.0; SSM_RANK_RATIO = 2.0; SSM_DT_RANK = "auto"
SSM_ACT_LAYER = "silu"; SSM_CONV = 3; SSM_CONV_BIAS = True; SSM_DROP_RATE = 0.0
SSM_INIT = "v0"; SSM_FORWARDTYPE = "v4"; MLP_RATIO = 4.0; MLP_ACT_LAYER = "gelu"
MLP_DROP_RATE = 0.0; DROP_PATH_RATE = 0.1; PATCH_NORM = True; NORM_LAYER = "ln"
DOWNSAMPLE = "v2"; PATCHEMBED = "v2"; GMLP = False; USE_CHECKPOINT = False

# ResT backbone base dimension
REST_EMBED_DIM = 64

# === 修改: 实例化 AFR+BAM 模型 ===
net = UNetMambaCA_AFR_BAM(
    # --- Core Parameters ---
    num_classes=num_classes,
    input_channels=IN_CHANS,
    embed_dim=REST_EMBED_DIM,
    afr_reduction_ratio=16,
    ca_reduction=32, # Or 1, check your CA implementation
    bam_mid_channels=32, # <<<--- BAM 中间通道数 (可调) | BAM intermediate channels (tuneable)

    # --- Backbone Path ---
    backbone_path='pretrain_weights/rest_lite.pth', # <<<--- 指定 Backbone 权重路径 | Specify Backbone weight path

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
loss = UnetMambaLoss(ignore_index=ignore_index)
use_aux_loss = True

# --- 权重、日志、监控配置 ---
model_arch_name = "unetmamba_CA_AFR_BAM" # <<<--- Updated model name
weights_name = f"{model_arch_name}_vaihingen_{image_size}-e{max_epoch}"
weights_dir = f"model_weights/vaihingen/{model_arch_name}_{image_size}" # <<<--- Updated directory
weights_path = weights_dir
log_name = f'vaihingen/{weights_name}' # <<<--- Updated log name

monitor = 'val_mIoU'; monitor_mode = 'max'; save_top_k = 1; save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # No full model pretraining
resume_ckpt_path = None
gpus = 'auto'

# === 4. 数据集实例化 ===
IMG_SUFFIX = '.tif'; MASK_SUFFIX = '.png'
try:
    img_size_tuple = (image_size, image_size) if image_size else (crop_size, crop_size)
    print(f"Initializing datasets with image size: {img_size_tuple}")
    print(f"Train data root: {os.path.abspath(VAIHINGEN_TRAIN_DATA_ROOT)}")
    print(f"Val data root: {os.path.abspath(VAIHINGEN_VAL_DATA_ROOT)}")
    train_dataset = VaihingenDataset(data_root=VAIHINGEN_TRAIN_DATA_ROOT, mode='train',
                                     img_suffix=IMG_SUFFIX, mask_suffix=MASK_SUFFIX,
                                     transform=train_aug, mosaic_ratio=0.25, img_size=img_size_tuple)
    val_dataset = VaihingenDataset(data_root=VAIHINGEN_VAL_DATA_ROOT, mode='val',
                                   img_suffix=IMG_SUFFIX, mask_suffix=MASK_SUFFIX,
                                   transform=val_aug, img_size=img_size_tuple)
    test_dataset = VaihingenDataset(data_root=VAIHINGEN_VAL_DATA_ROOT, mode='val',
                                    img_suffix=IMG_SUFFIX, mask_suffix=MASK_SUFFIX,
                                    transform=val_aug, img_size=img_size_tuple)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("CRITICAL ERROR: Datasets are empty."); exit()
except FileNotFoundError as e: print(f"CRITICAL ERROR: Dataset directory not found: {e}."); exit()
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
    # Group 1: Backbone ('encoder.')
    backbone_params = [p for n, p in net.named_parameters() if n.startswith('encoder.') and p.requires_grad]
    print(f"Found {len(backbone_params)} parameters in backbone ('encoder.').")

    # Group 2: AFR modules ('decoder.afr_modules.')
    afr_params = [p for n, p in net.named_parameters() if n.startswith('decoder.afr_modules.') and p.requires_grad]
    print(f"Found {len(afr_params)} parameters in AFR modules.")

    # Group 3: BAM modules ('decoder.bam.') - Exclude fixed edge conv weights
    bam_params = [p for n, p in net.named_parameters() if n.startswith('decoder.bam.') and p.requires_grad and 'edge_conv.weight' not in n]
    print(f"Found {len(bam_params)} trainable parameters in BAM modules.")

    # Group 4: Rest of the network
    base_param_ids = set(id(p) for p in net.parameters() if p.requires_grad)
    backbone_param_ids = set(id(p) for p in backbone_params)
    afr_param_ids = set(id(p) for p in afr_params)
    bam_param_ids = set(id(p) for p in bam_params)
    other_params = [p for p in net.parameters() if id(p) in base_param_ids - backbone_param_ids - afr_param_ids - bam_param_ids]
    print(f"Found {len(other_params)} parameters in other parts.")

    # Create optimizer parameter groups
    net_params = []
    if backbone_params: net_params.append({'params': backbone_params, 'lr': backbone_lr, 'weight_decay': backbone_weight_decay})
    if afr_params: net_params.append({'params': afr_params, 'lr': afr_lr, 'weight_decay': weight_decay})
    if bam_params: net_params.append({'params': bam_params, 'lr': bam_lr, 'weight_decay': weight_decay}) # <<<--- Added BAM group
    if other_params: net_params.append({'params': other_params, 'lr': lr, 'weight_decay': weight_decay})

    if not net_params:
         print("Warning: No parameters found for optimizer. Using all parameters with base LR.")
         net_params = net.parameters()

    base_optimizer = torch.optim.AdamW(net_params, lr=lr)
    optimizer = Lookahead(base_optimizer)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
    print("Optimizer and LR Scheduler configured.")

except Exception as e:
    print(f"ERROR setting up optimizer: {e}. Check model parameter names.")
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
print(f"Learning Rates (Base/Backbone/AFR/BAM): {lr}/{backbone_lr}/{afr_lr}/{bam_lr}") # <<<--- Updated LR printout
print(f"Weights Save Directory: {weights_dir}")
print(f"Log Name: {log_name}")
print(f"Backbone Pretrained Weights: pretrain_weights/rest_lite.pth")
print(f"Resume Checkpoint: {resume_ckpt_path}")
print(f"---------------------------\n")
