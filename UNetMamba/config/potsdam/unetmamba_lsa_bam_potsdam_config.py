# config/postdam/unetmamba_lsa_bam_potsdam_config.py
import torch
from torch.utils.data import DataLoader
from catalyst.contrib.nn import Lookahead
from catalyst import utils
# import albumentations as albu # Not explicitly used in unetmamba (6).py for PotsdamPatchesDataset
import os
import copy

# === 1. 导入数据集相关类和函数 ===
try:
    from unetmamba_model.datasets.potsdam_dataset import (
        PotsdamPatchesDataset, # Uses pre-generated patches
        CLASSES,
        PALETTE, # Potsdam might have its own palette
        IGNORE_INDEX,
    )
except ImportError as e:
     print(f"Error importing from potsdam_dataset: {e}")
     exit()

# === 2. 导入模型和损失函数 ===
try:
    from unetmamba_model.models.UNetMamba_LSA_BAM_Model_V2 import UNetMamba # Points to the LSA/BAM model
    from unetmamba_model.losses import UnetMambaLoss
except ImportError as e:
     print(f"Error importing UNetMamba_LSA_BAM_Model_V2 or UnetMambaLoss: {e}")
     exit()

# --- 配置参数 ---
PATCH_SIZE = 1024 # Patch size used during data pre-processing
# STRIDE = 1024   # Stride used during data pre-processing

# --- 数据集路径配置 ---
POTSDAM_TRAIN_PATCHES_ROOT = f'data/Potsdam/train_{PATCH_SIZE}'
POTSDAM_VAL_PATCHES_ROOT = f'data/Potsdam/val_{PATCH_SIZE}'

# --- 训练超参数 ---
max_epoch = 100
num_classes = len(CLASSES)
ignore_index = IGNORE_INDEX # From potsdam_dataset.py (e.g., 255)
classes = CLASSES

train_batch_size = 2 # As per original config
val_batch_size = 2   # As per original config

lr = 5e-4
weight_decay = 2.5e-4
backbone_lr = 5e-5
backbone_weight_decay = 2.5e-4
# image_size = (PATCH_SIZE, PATCH_SIZE) # For reference, dataset uses patches directly

# --- 模型配置 ---
# --- 控制 LSA 和 BAM 模块 ---
USE_LSA_IN_DECODER = True
USE_BAM_IN_DECODER = True
# --------------------------

MODEL_EMBED_DIM = 64 # Base dimension for default encoder_channels calculation
PRETRAINED_ENCODER = True
BACKBONE_PATH = 'pretrain_weights/rest_lite.pth'
REST_EMBED_DIMS = None
REST_KWARGS = {}

DROP_PATH_RATE_DECODER = 0.2
VSS_BLOCK_KWARGS_DECODER = {
    "d_state": 16,
    "ssm_ratio": 2.0,
    "mlp_ratio": 4.0,
}
VSS_LAYER_NORM_IN_DECODER = torch.nn.LayerNorm

net = UNetMamba(
    num_classes=num_classes,
    embed_dim=MODEL_EMBED_DIM, # Corresponds to embed_dim in original UNetMamba call
    pretrained_encoder=PRETRAINED_ENCODER,
    backbone_path=BACKBONE_PATH,
    rest_embed_dims=REST_EMBED_DIMS,
    rest_kwargs=REST_KWARGS,
    drop_path_rate_decoder=DROP_PATH_RATE_DECODER,
    use_lsa_in_decoder=USE_LSA_IN_DECODER,
    use_bam_in_decoder=USE_BAM_IN_DECODER,
    vss_layer_norm_in_decoder=VSS_LAYER_NORM_IN_DECODER,
    vss_block_kwargs_in_decoder=VSS_BLOCK_KWARGS_DECODER
)

loss = UnetMambaLoss(ignore_index=ignore_index)
use_aux_loss = True # Assume LSA/BAM model also supports/uses aux loss

# --- 权重、日志、监控配置 ---
model_variant_name = "unetmamba"
if USE_LSA_IN_DECODER:
    model_variant_name += "_lsa"
if USE_BAM_IN_DECODER:
    model_variant_name += "_bam"

weights_name = f"{model_variant_name}_potsdam_patch{PATCH_SIZE}-e{max_epoch}"
weights_dir = f"model_weights/potsdam_patches_{PATCH_SIZE}_{model_variant_name}"
weights_path = os.path.join(weights_dir, weights_name)
test_weights_name = "last"
log_name = f'potsdam_patches_{PATCH_SIZE}_{model_variant_name}/{weights_name}' # Made log_name more specific

monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
resume_ckpt_path = None
gpus = 'auto'

# --- 数据加载器配置 ---
try:
    # PotsdamPatchesDataset does not take a transform argument in unetmamba (6).py
    # It likely handles its own minimal transformations (e.g., ToTensor, Normalize) internally
    train_dataset = PotsdamPatchesDataset(data_root=POTSDAM_TRAIN_PATCHES_ROOT, mode='train')
    val_dataset = PotsdamPatchesDataset(data_root=POTSDAM_VAL_PATCHES_ROOT, mode='val')
    test_dataset = PotsdamPatchesDataset(data_root=POTSDAM_VAL_PATCHES_ROOT, mode='val') # For test script
except FileNotFoundError as e:
     print(f"Error initializing PotsdamPatchesDataset: {e}. Check POTSDAM_*_PATCHES_ROOT paths.")
     exit()
except Exception as e:
     print(f"An unexpected error occurred during Potsdam dataset initialization: {e}")
     exit()

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True,
                          persistent_workers=True if 4 > 0 else False)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False,
                        persistent_workers=True if 4 > 0 else False)

# --- 优化器与学习率调度器 ---
layerwise_params = {"encoder.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
try:
    net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
except AttributeError:
    print("Warning: Could not find 'net.encoder' for layerwise_params. Using all parameters for optimizer.")
    net_params = net.parameters()

optimizer = Lookahead(torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay))

_lr_scheduler_instance = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, # Pass the Lookahead optimizer
    T_max=max_epoch,
    eta_min=1e-6
)

lr_scheduler_config = {
    'scheduler': _lr_scheduler_instance,
    'interval': 'epoch',
    'frequency': 1,
    'monitor': monitor,
    'strict': True,
}

print(f"Config loaded for Potsdam Patches, Model: {model_variant_name.upper()}")
print(f"  USE_LSA_IN_DECODER: {USE_LSA_IN_DECODER}")
print(f"  USE_BAM_IN_DECODER: {USE_BAM_IN_DECODER}")
print(f"  Train data root: {POTSDAM_TRAIN_PATCHES_ROOT}")
print(f"  Val data root: {POTSDAM_VAL_PATCHES_ROOT}")

