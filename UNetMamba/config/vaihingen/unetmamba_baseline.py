# config/vaihingen/unetmamba_baseline.py (Using Online Augmentation Dataset)
import torch
import os
from torch.utils.data import DataLoader
from catalyst.contrib.nn import Lookahead
from catalyst import utils
from fvcore.nn import flop_count, parameter_count
import copy

# === 1. 导入类和函数 ===
# 从用户提供的 vaihingen_dataset.py 导入
try:
    from unetmamba_model.datasets.vaihingen_dataset import (
        VaihingenDataset, # 使用用户提供的 Dataset 类
        CLASSES,
        IGNORE_INDEX, # 使用脚本中定义的 IGNORE_INDEX (值为 6)
        train_aug,    # 使用脚本中定义的训练增强函数
        val_aug       # 使用脚本中定义的验证增强函数
    )
except ImportError as e:
     print(f"Error importing from vaihingen_dataset: {e}")
     print("Make sure vaihingen_dataset.py is in unetmamba_model/datasets/ and __init__.py exists.")
     exit()
except NameError as e:
     print(f"Error: {e}. Make sure VaihingenDataset, CLASSES, IGNORE_INDEX, train_aug, val_aug are defined correctly in vaihingen_dataset.py.")
     exit()


# === 2. 导入模型和损失函数 ===
try:
    from unetmamba_model.models.UNetMamba import UNetMamba # Baseline model
    from unetmamba_model.losses import UnetMambaLoss
except ImportError as e:
     print(f"Error importing model or loss: {e}")
     exit()


# === 配置参数 ===
# 使用用户脚本中的 image_size
image_size = 1024
PATCH_SIZE_PARAM = image_size # 用于路径构造

# === 3. 使用正确的相对路径 (无 '../') ===
# (路径相对于 UNetMamba/ 目录)
VAIHINGEN_TRAIN_DATA_ROOT = f'data/vaihingen/train_{PATCH_SIZE_PARAM}' # e.g., data/vaihingen/train_1024
VAIHINGEN_VAL_DATA_ROOT = f'data/vaihingen/val_{PATCH_SIZE_PARAM}'     # e.g., data/vaihingen/val_1024
# VAIHINGEN_TEST_DATA_ROOT = f'data/vaihingen/test_{PATCH_SIZE_PARAM}' # Test data path if needed

# --- 训练超参数 ---
max_epoch = 100
num_classes = len(CLASSES) # 6
ignore_index = IGNORE_INDEX # 6 (来自 vaihingen_dataset.py)
classes = CLASSES

# Batch Size from user's previous config snippet
train_batch_size = 8
val_batch_size = 1   # Keep small for validation with online crop

lr = 6e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4
# crop_size is calculated inside train_aug, no need to define here unless needed elsewhere

# --- 模型配置 ---
# Instantiate Baseline UNetMamba
# Pass VSSM params etc. as defined before
PATCH_SIZE_VSSM = 4
IN_CHANS = 3
DEPTHS = [2, 2, 9, 2]
EMBED_DIM = 96
SSM_D_STATE = 16
# ... (Copy all other VSSM/MLP params from previous correct version) ...
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
USE_CHECKPOINT = False

net = UNetMamba(pretrained=True, # Loads ResT weights
                num_classes=num_classes,
                patch_size=PATCH_SIZE_VSSM, in_chans=IN_CHANS, depths=DEPTHS, dims=EMBED_DIM,
                ssm_d_state=SSM_D_STATE, ssm_ratio=SSM_RATIO, ssm_rank_ratio=SSM_RANK_RATIO,
                ssm_dt_rank=("auto" if SSM_DT_RANK == "auto" else int(SSM_DT_RANK)),
                ssm_act_layer=SSM_ACT_LAYER, ssm_conv=SSM_CONV, ssm_conv_bias=SSM_CONV_BIAS,
                ssm_drop_rate=SSM_DROP_RATE, ssm_init=SSM_INIT, forward_type=SSM_FORWARDTYPE,
                mlp_ratio=MLP_RATIO, mlp_act_layer=MLP_ACT_LAYER, mlp_drop_rate=MLP_DROP_RATE,
                drop_path_rate=DROP_PATH_RATE, patch_norm=PATCH_NORM, norm_layer=NORM_LAYER,
                downsample_version=DOWNSAMPLE, patchembed_version=PATCHEMBED, gmlp=GMLP,
                use_checkpoint=USE_CHECKPOINT,
                backbone_path='pretrain_weights/rest_lite.pth' # Confirm path
               )

loss = UnetMambaLoss(ignore_index=ignore_index) # Pass correct ignore_index (6)
use_aux_loss = True # Assuming UNetMamba uses aux loss

# --- 权重、日志、监控配置 ---
weights_name = f"unetmamba_baseline_vaihingen_{image_size}-e{max_epoch}"
weights_dir = f"model_weights/vaihingen_{image_size}_baseline"
weights_path = os.path.join(weights_dir, weights_name) # Path for ModelCheckpoint dirpath
test_weights_name = "last" # For test script loading from config

log_name = f'vaihingen/{weights_name}'

monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
resume_ckpt_path = None
gpus = 'auto'

# === 4. 在配置文件内部实例化 Dataset ===
try:
    # Pass the specific transform functions imported earlier
    # Also pass img_size needed by mosaic/padding logic inside the class
    img_size_tuple = (image_size, image_size)
    train_dataset = VaihingenDataset(data_root=VAIHINGEN_TRAIN_DATA_ROOT, mode='train', transform=train_aug, mosaic_ratio=0.25, img_size=img_size_tuple)
    val_dataset = VaihingenDataset(data_root=VAIHINGEN_VAL_DATA_ROOT, mode='val', transform=val_aug, img_size=img_size_tuple)
    # Define test_dataset for test script compatibility (using val data and val_aug)
    test_dataset = VaihingenDataset(data_root=VAIHINGEN_VAL_DATA_ROOT, mode='val', transform=val_aug, img_size=img_size_tuple)
except FileNotFoundError as e:
     print(f"CRITICAL ERROR in config: {e}.")
     print("Please double-check VAIHINGEN_*_DATA_ROOT paths (should start with 'data/...') relative to the execution directory.")
     print(f"Expected train data at: {os.path.abspath(VAIHINGEN_TRAIN_DATA_ROOT)}")
     print(f"Expected val data at: {os.path.abspath(VAIHINGEN_VAL_DATA_ROOT)}")
     exit()
except NameError as e:
     print(f"CRITICAL ERROR: {e}. Did you correctly define VaihingenDataset in vaihingen_dataset.py?")
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
# Assuming 'encoder' is the attribute name in UNetMamba
layerwise_params = {"encoder.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
try:
    if hasattr(net, 'encoder'):
         net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
    elif hasattr(net, 'backbone'):
         layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
         net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
    else:
         print("Warning: Cannot find 'backbone' or 'encoder' for layerwise params. Using all params.")
         net_params = net.parameters()
except Exception as e:
    print(f"Warning: Could not process layerwise params, using all params. Error: {e}")
    net_params = net.parameters() # Fallback

base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
# Use CosineAnnealingWarmRestarts as per user's original config snippet
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)


# --- (可选) 打印模型信息 ---
# ...

print(f"Config loaded for Vaihingen (Online Augmentation), Model: UNetMamba (Baseline)")
print(f"Checking Train data root: {os.path.abspath(VAIHINGEN_TRAIN_DATA_ROOT)}")
print(f"Checking Val data root: {os.path.abspath(VAIHINGEN_VAL_DATA_ROOT)}")
