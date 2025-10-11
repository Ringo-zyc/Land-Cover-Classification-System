# config/loveda/unetmamba_lsa_bam_loveda_config.py
import torch
from torch.utils.data import DataLoader
from catalyst.contrib.nn import Lookahead
from catalyst import utils
import albumentations as albu # For transforms if used by dataset
import numpy as np # For transforms if used by dataset
import os # For path joining
import copy

# === 1. 导入数据集相关类和函数 ===
try:
    # 确保这些路径和你的项目结构一致
    from unetmamba_model.datasets.loveda_dataset import (
        LoveDATrainDataset, # Assuming this is your training dataset class
        loveda_val_dataset, # Assuming this is your validation dataset instance or class
        LoveDATestDataset,  # Assuming this is your test dataset class
        CLASSES,            # Should be defined in loveda_dataset.py
        IGNORE_INDEX,       # Should be defined in loveda_dataset.py
        # train_aug,        # If train_aug is complex and defined in dataset file
        # val_aug           # If val_aug is complex and defined in dataset file
    )
    # If train_aug/val_aug are simple or defined here, no need to import them specifically
except ImportError as e:
     print(f"Error importing from loveda_dataset: {e}")
     print("Make sure loveda_dataset.py is in unetmamba_model/datasets/ and relevant classes/variables are defined.")
     exit()

# === 2. 导入模型和损失函数 ===
try:
    from unetmamba_model.models.UNetMamba_LSA_BAM_Model_V2 import UNetMamba # Points to the LSA/BAM model
    from unetmamba_model.losses import UnetMambaLoss # Assuming UnetMambaLoss is general
except ImportError as e:
     print(f"Error importing UNetMamba_LSA_BAM_Model_V2 or UnetMambaLoss: {e}")
     exit()

# --- 数据集相关配置 (LoveDA specific) ---
# LoveDA specific paths (modify as needed)
LOVEDA_TRAIN_DATA_ROOT = 'data/LoveDA/Train' # Example path
LOVEDA_VAL_DATA_ROOT = 'data/LoveDA/Val'     # Example path
LOVEDA_TEST_DATA_ROOT = 'data/LoveDA/Test'   # Example path

# --- 训练超参数 ---
max_epoch = 100
num_classes = len(CLASSES)
ignore_index = IGNORE_INDEX # Make sure this is correct for LoveDA (often 255 or num_classes)
classes = CLASSES # From loveda_dataset.py

train_batch_size = 8
val_batch_size = 8 # Can be same as train or smaller
image_size = 1024 # Used for LoveDA dataset, ensure your dataset class handles this or transforms

lr = 6e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4

# --- 模型配置 ---
# --- 控制 LSA 和 BAM 模块 ---
USE_LSA_IN_DECODER = True
USE_BAM_IN_DECODER = True
# --------------------------

MODEL_EMBED_DIM = 64 # Base dimension for default encoder_channels calculation
PRETRAINED_ENCODER = True
BACKBONE_PATH = 'pretrain_weights/rest_lite.pth' # Path to ResT pretrained weights
REST_EMBED_DIMS = None # Let UNetMamba calculate default based on MODEL_EMBED_DIM
REST_KWARGS = {} # Add ResT specific constructor args if needed

DROP_PATH_RATE_DECODER = 0.2
VSS_BLOCK_KWARGS_DECODER = { # Passed to VSSLayers -> VSSBlocks in decoder
    "d_state": 16,
    "ssm_ratio": 2.0,
    "mlp_ratio": 4.0,
    # Add other params your VSSBlock expects
}
VSS_LAYER_NORM_IN_DECODER = torch.nn.LayerNorm

net = UNetMamba(
    num_classes=num_classes,
    embed_dim=MODEL_EMBED_DIM,
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

loss = UnetMambaLoss(ignore_index=ignore_index) # Ensure this loss is suitable for LoveDA
use_aux_loss = True # If your UNetMamba model returns aux loss

# --- 权重、日志、监控配置 ---
model_variant_name = "unetmamba"
if USE_LSA_IN_DECODER:
    model_variant_name += "_lsa"
if USE_BAM_IN_DECODER:
    model_variant_name += "_bam"

weights_name = f"{model_variant_name}_loveda_{image_size}-e{max_epoch}"
weights_dir = f"model_weights/loveda_{image_size}_{model_variant_name}"
weights_path = os.path.join(weights_dir, weights_name)
test_weights_name = "last"
log_name = f'loveda/{weights_name}'

monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # For resuming UNetMamba training, not ResT
resume_ckpt_path = None
gpus = 'auto'

# --- 数据集和数据增强定义 ---
# Example simple augmentations, replace with your actual `train_aug` if it's complex
# Or ensure `train_aug` is correctly imported from `loveda_dataset.py`
def get_simple_train_transform(img_size):
    return albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.RandomResizedCrop(height=img_size, width=img_size, scale=(0.5, 1.0), ratio=(0.75, 1.33), p=0.5),
        albu.Normalize()
    ])

def get_simple_val_transform(img_size):
    return albu.Compose([
        albu.Resize(height=img_size, width=img_size), # Ensure val images are resized
        albu.Normalize()
    ])

# If your LoveDATrainDataset and loveda_val_dataset already handle transformations internally,
# you might not need to define train_transform and val_transform here.
# Otherwise, define or import them.

# For this example, let's assume LoveDATrainDataset takes a transform function
# and loveda_val_dataset might be an instance already or needs a transform.

# Define your augmentation functions for LoveDA if not imported
# This is a placeholder based on the original unetmamba (5).py if needed by your Dataset class
# If your Dataset classes handle augmentation internally, you can skip this.
# from unetmamba_model.datasets.loveda_dataset import Compose, RandomScale, SmartCropV1 # If these are used
# def train_aug_loveda(img, mask, ignore_idx_param, img_size_param): # Make params explicit
#     # This is just an example structure, adapt from your actual loveda_dataset.py
#     # crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
#     #                     SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=ignore_idx_param, nopad=False)])
#     # img, mask = crop_aug(img, mask)
#     # img, mask = np.array(img), np.array(mask) # Ensure they are numpy arrays for albumentations
#     aug = get_simple_train_transform(img_size_param)(image=img.copy(), mask=mask.copy())
#     img, mask = aug['image'], aug['mask']
#     return img, mask

# def val_aug_loveda(img, mask, img_size_param):
#     aug = get_simple_val_transform(img_size_param)(image=img.copy(), mask=mask.copy())
#     img, mask = aug['image'], aug['mask']
#     return img, mask

# Dataset Instantiation
try:
    # If your Dataset class takes a transform function:
    # train_transform_func = lambda img, mask: train_aug_loveda(img, mask, ignore_index, image_size)
    # val_transform_func = lambda img, mask: val_aug_loveda(img, mask, image_size)

    # How you instantiate depends on your Dataset class implementation
    # Option 1: Dataset class handles transforms or takes file paths and loads
    train_dataset = LoveDATrainDataset(data_root=LOVEDA_TRAIN_DATA_ROOT, image_size=(image_size, image_size)) # Add other necessary args
    # `loveda_val_dataset` might be an instance already from import, or a class
    if isinstance(loveda_val_dataset, torch.utils.data.Dataset):
         val_dataset = loveda_val_dataset # It's already an instance
    else: # It's a class, needs instantiation
         val_dataset = loveda_val_dataset(data_root=LOVEDA_VAL_DATA_ROOT, image_size=(image_size, image_size)) # Add other necessary args

    test_dataset = LoveDATestDataset(data_root=LOVEDA_TEST_DATA_ROOT, image_size=(image_size, image_size)) # Add other necessary args

except Exception as e:
     print(f"Error initializing LoveDA dataset: {e}")
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
# In original unetmamba (5).py, it was "backbone.*". Check your UNetMamba model's encoder attribute name.
# Assuming it's "encoder" as per our LSA/BAM model.
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
    'monitor': monitor, # For ReduceLROnPlateau, not strictly needed for CosineAnnealingLR
    'strict': True,
}

print(f"Config loaded for LoveDA, Model: {model_variant_name.upper()}")
print(f"  USE_LSA_IN_DECODER: {USE_LSA_IN_DECODER}")
print(f"  USE_BAM_IN_DECODER: {USE_BAM_IN_DECODER}")
print(f"  Train data root: {LOVEDA_TRAIN_DATA_ROOT}")
print(f"  Val data root: {LOVEDA_VAL_DATA_ROOT}")

