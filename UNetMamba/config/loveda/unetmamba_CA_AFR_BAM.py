import torch
import os
from torch.utils.data import DataLoader
from catalyst.contrib.nn import Lookahead
from catalyst import utils
import albumentations as albu # Import albumentations
from tools.data_process import Compose, RandomScale, SmartCropV1 # Import augmentations
import numpy as np # Import numpy

# === 1. 导入 LoveDA 数据集类和函数 ===
try:
    # 假设 loveda_dataset.py 在 unetmamba_model/datasets/ 目录下
    # Assume loveda_dataset.py is in unetmamba_model/datasets/
    from unetmamba_model.datasets.loveda_dataset import (
        LoveDATrainDataset,
        loveda_val_dataset, # Assuming this is a pre-instantiated dataset object or class
        LoveDATestDataset,
        CLASSES, # Should define LoveDA classes (Building, Road, Water, Barren, Forest, Agriculture, Background)
        # LoveDA typically has 7 classes + ignore, check your dataset file
    )
    # LoveDA 通常 ignore_index 是 7 (0-6 是有效类)
    # LoveDA usually has ignore_index = 7 (0-6 are valid classes)
    if len(CLASSES) == 7:
         IGNORE_INDEX = 7
    else:
         # Fallback or raise error if CLASSES definition is unexpected
         print(f"Warning: Expected 7 classes for LoveDA, found {len(CLASSES)}. Setting IGNORE_INDEX based on length.")
         IGNORE_INDEX = len(CLASSES)

except ImportError as e:
    print(f"CRITICAL ERROR: Could not import from loveda_dataset: {e}")
    exit()
except NameError as e:
    print(f"CRITICAL ERROR: {e}. Check loveda_dataset definitions.")
    exit()
except Exception as e:
     print(f"An error occurred during dataset import setup: {e}")
     exit()


# === 2. 导入模型和损失函数 ===
try:
    from unetmamba_model.models.UNetMamba_CA_AFR_BAM import UNetMambaCA_AFR_BAM
    from unetmamba_model.losses import UnetMambaLoss, CrossEntropyLoss # Ensure necessary losses are imported
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import model 'UNetMambaCA_AFR_BAM' or 'UnetMambaLoss': {e}")
    exit()


# === 配置参数 ===
image_size = 1024
crop_size = 512 # 保持与原始 LoveDA 配置一致 | Keep consistent with original LoveDA config

# === 3. 数据集路径 ===
# 检查并修改为你的实际 LoveDA 数据集路径
# Check and modify to your actual LoveDA dataset paths
LOVEDA_TRAIN_DATA_ROOT = 'data/LoveDA/Train' # <<<--- 检查路径 | Check path
LOVEDA_VAL_DATA_ROOT = 'data/LoveDA/Val'     # <<<--- 检查路径 | Check path
# LOVEDA_TEST_DATA_ROOT = 'data/LoveDA/Test' # <<<--- 如果需要测试集 | If test set is needed

# --- 训练超参数 ---
max_epoch = 100
num_classes = len(CLASSES) # LoveDA 通常是 7 | Usually 7 for LoveDA
ignore_index = IGNORE_INDEX # LoveDA 通常是 7 | Usually 7 for LoveDA
classes = CLASSES
train_batch_size = 8 # 与原始 LoveDA 配置一致 | Consistent with original LoveDA config
val_batch_size = 8   # 与原始 LoveDA 配置一致 | Consistent with original LoveDA config
lr = 6e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4
afr_lr = 6e-4
bam_lr = 6e-4

# --- 模型配置 ---
# VSSM parameters (If needed by the model) - Keep consistent if model uses them
PATCH_SIZE_VSSM = 4; IN_CHANS = 3; DEPTHS = [2, 2, 9, 2]; EMBED_DIM = 96
SSM_D_STATE = 16; SSM_RATIO = 2.0; SSM_RANK_RATIO = 2.0; SSM_DT_RANK = "auto"
SSM_ACT_LAYER = "silu"; SSM_CONV = 3; SSM_CONV_BIAS = True; SSM_DROP_RATE = 0.0
SSM_INIT = "v0"; SSM_FORWARDTYPE = "v4"; MLP_RATIO = 4.0; MLP_ACT_LAYER = "gelu"
MLP_DROP_RATE = 0.0; DROP_PATH_RATE = 0.1; PATCH_NORM = True; NORM_LAYER = "ln"
DOWNSAMPLE = "v2"; PATCHEMBED = "v2"; GMLP = False; USE_CHECKPOINT = True # Use checkpointing for larger model

# ResT backbone base dimension
REST_EMBED_DIM = 64

# === 实例化 AFR+BAM 模型 ===
net = UNetMambaCA_AFR_BAM(
    # --- Core Parameters ---
    num_classes=num_classes, # Should be 7 for LoveDA
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
loss = UnetMambaLoss(ignore_index=ignore_index) # ignore_index should be 7 for LoveDA
use_aux_loss = True

# --- 权重、日志、监控配置 ---
model_arch_name = "unetmamba_CA_AFR_BAM"
weights_name = f"{model_arch_name}_loveda_{image_size}-e{max_epoch}" # <<<--- Updated dataset name
weights_dir = f"model_weights/loveda/{model_arch_name}_{image_size}" # <<<--- Updated dataset name
weights_path = weights_dir
log_name = f'loveda/{weights_name}' # <<<--- Updated dataset name

monitor = 'val_mIoU'; monitor_mode = 'max'; save_top_k = 1; save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
resume_ckpt_path = None
gpus = 'auto'

# === 4. 数据集和数据加载器实例化 ===

# --- LoveDA 数据增强函数 (从原始 config 文件复制) ---
# --- LoveDA augmentation functions (copied from original config file) ---
def get_training_transform():
    """ Basic augmentations like flip, normalize """
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        # Use standard ImageNet normalization, adjust if needed
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
    return albu.Compose(train_transform)

def train_aug(img, mask):
    """ Combined augmentation pipeline for LoveDA training """
    # Crop augmentation (scale + crop)
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=crop_size, max_ratio=0.75, ignore_index=ignore_index, nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    # Basic augmentations
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    # Dataset class should handle ToTensor and CHW conversion
    return img, mask

# --- 实例化数据集 ---
# --- Instantiate Datasets ---
try:
    print(f"Initializing LoveDA datasets...")
    print(f"Train data root: {os.path.abspath(LOVEDA_TRAIN_DATA_ROOT)}")
    print(f"Val data root: {os.path.abspath(LOVEDA_VAL_DATA_ROOT)}")

    # 使用导入的类和上面定义的 train_aug
    # Use imported classes and train_aug defined above
    train_dataset = LoveDATrainDataset(transform=train_aug, data_root=LOVEDA_TRAIN_DATA_ROOT)
    # 假设 loveda_val_dataset 是一个需要实例化的类或已准备好的对象
    # Assume loveda_val_dataset is a class to be instantiated or a ready object
    if isinstance(loveda_val_dataset, type): # If it's a class
         val_dataset = loveda_val_dataset(data_root=LOVEDA_VAL_DATA_ROOT) # Basic instantiation, add args if needed
    else: # Assume it's a pre-made object
         val_dataset = loveda_val_dataset
    # 实例化测试集
    # Instantiate test set
    test_dataset = LoveDATestDataset() # Add args if needed (e.g., data_root)

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
    # 使用原始 LoveDA 配置中的 CosineAnnealingLR 调度器
    # Use CosineAnnealingLR scheduler from original LoveDA config
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
    print("Optimizer and LR Scheduler configured.")

except Exception as e:
    print(f"ERROR setting up optimizer: {e}. Falling back to basic AdamW.")
    optimizer = Lookahead(torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

# --- Final Config Check ---
print(f"\n--- Configuration Summary ---")
print(f"Model: {model_arch_name}")
print(f"Dataset: LoveDA")
print(f"Image Size: {image_size}")
print(f"Epochs: {max_epoch}")
print(f"Batch Size (Train/Val): {train_batch_size}/{val_batch_size}")
print(f"Learning Rates (Base/Backbone/AFR/BAM): {lr}/{backbone_lr}/{afr_lr}/{bam_lr}")
print(f"Weights Save Directory: {weights_dir}")
print(f"Log Name: {log_name}")
print(f"Backbone Pretrained Weights: {net.decoder.bam_mid_channels if hasattr(net, 'decoder') and hasattr(net.decoder, 'bam_mid_channels') else 'N/A'}") # Print BAM mid channels instead
print(f"Resume Checkpoint: {resume_ckpt_path}")
print(f"---------------------------\n")

