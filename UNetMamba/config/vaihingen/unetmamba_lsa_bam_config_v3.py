# config/vaihingen/unetmamba_lsa_bam_config_v3.py
import torch
import os
from torch.utils.data import DataLoader
from catalyst.contrib.nn import Lookahead # Assuming catalyst is used
from catalyst import utils # Assuming catalyst is used
# from fvcore.nn import flop_count, parameter_count # Uncomment if you use these
import copy

# === 1. 导入数据集相关类和函数 ===
try:
    from unetmamba_model.datasets.vaihingen_dataset import (
        VaihingenDataset,
        CLASSES,
        IGNORE_INDEX,
        train_aug,
        val_aug
    )
except ImportError as e:
     print(f"Error importing from vaihingen_dataset: {e}")
     exit()
except NameError as e:
     print(f"Error: {e} in vaihingen_dataset.py.")
     exit()


# === 2. 导入模型和损失函数 ===
try:
    # 确保这个路径指向你保存的包含LSA/BAM的最新模型文件
    # 假设模型文件名为 UNetMamba_LSA_BAM_Model_V2.py (与上一轮交互一致)
    from unetmamba_model.models.UNetMamba_LSA_BAM_Model_V2 import UNetMamba
    from unetmamba_model.losses import UnetMambaLoss
except ImportError as e:
     print(f"Error importing UNetMamba_LSA_BAM_Model_V2 or UnetMambaLoss: {e}")
     print("Please ensure UNetMamba_LSA_BAM_Model_V2.py is correctly saved in unetmamba_model/models/.")
     exit()


# === 配置参数 ===
image_size = 1024
PATCH_SIZE_PARAM = image_size

# === 数据集路径 ===
VAIHINGEN_TRAIN_DATA_ROOT = f'data/vaihingen/train_{PATCH_SIZE_PARAM}'
VAIHINGEN_VAL_DATA_ROOT = f'data/vaihingen/val_{PATCH_SIZE_PARAM}'

# --- 训练超参数 ---
max_epoch = 100
num_classes = len(CLASSES)
ignore_index = IGNORE_INDEX
classes = CLASSES

train_batch_size = 8
val_batch_size = 1

lr = 6e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4

# --- 模型配置 ---
USE_LSA_IN_DECODER = True
USE_BAM_IN_DECODER = True

MODEL_EMBED_DIM = 64
PRETRAINED_ENCODER = True
BACKBONE_PATH = 'pretrain_weights/rest_lite.pth'
REST_EMBED_DIMS = None
REST_KWARGS = {
    # "depths": [2, 2, 2, 2],
    # "num_heads": [1, 2, 4, 8],
}
DROP_PATH_RATE_DECODER = 0.2
VSS_BLOCK_KWARGS_DECODER = {
    "d_state": 16,
    "ssm_ratio": 2.0,
    "mlp_ratio": 4.0,
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

loss = UnetMambaLoss(ignore_index=ignore_index)
use_aux_loss = True

# --- 权重、日志、监控配置 ---
model_variant_name = "unetmamba"
if USE_LSA_IN_DECODER:
    model_variant_name += "_lsa"
if USE_BAM_IN_DECODER:
    model_variant_name += "_bam"

weights_name = f"{model_variant_name}_vaihingen_{image_size}-e{max_epoch}"
weights_dir = f"model_weights/vaihingen_{image_size}_{model_variant_name}"
weights_path = os.path.join(weights_dir, weights_name)
test_weights_name = "last"
log_name = f'vaihingen/{weights_name}'

monitor = 'val_mIoU' 
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
resume_ckpt_path = None
gpus = 'auto'

# === 数据集实例化 ===
try:
    img_size_tuple = (image_size, image_size)
    train_dataset = VaihingenDataset(data_root=VAIHINGEN_TRAIN_DATA_ROOT, mode='train', transform=train_aug, mosaic_ratio=0.25, img_size=img_size_tuple)
    val_dataset = VaihingenDataset(data_root=VAIHINGEN_VAL_DATA_ROOT, mode='val', transform=val_aug, img_size=img_size_tuple)
    test_dataset = VaihingenDataset(data_root=VAIHINGEN_VAL_DATA_ROOT, mode='val', transform=val_aug, img_size=img_size_tuple)
except FileNotFoundError as e:
     print(f"CRITICAL ERROR in config (Dataset path): {e}.")
     exit()
except Exception as e:
     print(f"An unexpected error occurred during dataset initialization: {e}")
     exit()

# --- 数据加载器定义 ---
pin_memory = True
num_workers = 4
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
layerwise_params = {"encoder.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
try:
    net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
except AttributeError:
    print("Warning: Could not find 'net.encoder' for layerwise_params. Using all parameters for optimizer.")
    net_params = net.parameters()
except Exception as e:
    print(f"Warning: Error processing layerwise_params: {e}. Using all parameters.")
    net_params = net.parameters()

optimizer = Lookahead(torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay))

# === 修改点: 初始化学习率调度器时使用 Lookahead 优化器实例 ===
_lr_scheduler_instance = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, # 使用 Lookahead 优化器实例本身
    T_0=15,
    T_mult=2
)
# ============================================================

lr_scheduler_config = {
    'scheduler': _lr_scheduler_instance, 
    'interval': 'epoch',                 
    'frequency': 1,                      
    'monitor': monitor,                  
    'strict': True,                      
    'name': None,                        
}

print(f"Config loaded for Vaihingen (Online Augmentation), Model: {model_variant_name.upper()}")
print(f"  USE_LSA_IN_DECODER: {USE_LSA_IN_DECODER}")
print(f"  USE_BAM_IN_DECODER: {USE_BAM_IN_DECODER}")
print(f"  Weights will be saved in: {weights_dir}")

