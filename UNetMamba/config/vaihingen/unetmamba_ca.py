# config/vaihingen/unetmamba_ca.py (Corrected Import Case)
import torch
import os
from torch.utils.data import DataLoader
from catalyst.contrib.nn import Lookahead
from catalyst import utils
from fvcore.nn import flop_count, parameter_count
import copy

# === 1. 导入类和函数 ===
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
     print(f"Error: {e}. Make sure classes/functions are defined.")
     exit()

# === 2. 导入模型和损失函数 ===
try:
    # === 修改: 使用正确的大小写导入 CA 模型 ===
    from unetmamba_model.models.UNetMamba_CA import UNetMambaCA # <-- 使用大写 CA 匹配文件名
    from unetmamba_model.losses import UnetMambaLoss
except ImportError as e:
     print(f"Error importing model or loss: {e}")
     print("Ensure UNetMamba_CA.py (case-sensitive) exists in unetmamba_model/models/ and defines UNetMambaCA class.")
     exit()


# === 配置参数 ===
image_size = 1024
PATCH_SIZE_PARAM = image_size
crop_size = int(512*float(image_size/1024)) # 512

# === 3. 使用正确的相对路径 (无 '../') ===
VAIHINGEN_TRAIN_DATA_ROOT = f'data/vaihingen/train_{PATCH_SIZE_PARAM}'
VAIHINGEN_VAL_DATA_ROOT = f'data/vaihingen/val_{PATCH_SIZE_PARAM}'

# --- 训练超参数 ---
max_epoch = 100
num_classes = len(CLASSES)
ignore_index = IGNORE_INDEX
classes = CLASSES
train_batch_size = 8 # Keep same as baseline
val_batch_size = 1   # Keep same as baseline
lr = 6e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4

# --- 模型配置 ---
# Instantiate CA UNetMamba
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
USE_CHECKPOINT = False

net = UNetMambaCA(pretrained=True,
                  num_classes=num_classes,
                  embed_dim=64,
                  patch_size=PATCH_SIZE_VSSM, in_chans=IN_CHANS, depths=DEPTHS, dims=EMBED_DIM,
                  ssm_d_state=SSM_D_STATE, ssm_ratio=SSM_RATIO, ssm_rank_ratio=SSM_RANK_RATIO,
                  ssm_dt_rank=("auto" if SSM_DT_RANK == "auto" else int(SSM_DT_RANK)),
                  ssm_act_layer=SSM_ACT_LAYER, ssm_conv=SSM_CONV, ssm_conv_bias=SSM_CONV_BIAS,
                  ssm_drop_rate=SSM_DROP_RATE, ssm_init=SSM_INIT, forward_type=SSM_FORWARDTYPE,
                  mlp_ratio=MLP_RATIO, mlp_act_layer=MLP_ACT_LAYER, mlp_drop_rate=MLP_DROP_RATE,
                  drop_path_rate=DROP_PATH_RATE, patch_norm=PATCH_NORM, norm_layer=NORM_LAYER,
                  downsample_version=DOWNSAMPLE, patchembed_version=PATCHEMBED, gmlp=GMLP,
                  use_checkpoint=USE_CHECKPOINT,
                  backbone_path='pretrain_weights/rest_lite.pth'
                 )

loss = UnetMambaLoss(ignore_index=ignore_index)
use_aux_loss = True

# --- 权重、日志、监控配置 ---
weights_name = f"unetmamba_CA_vaihingen_{image_size}-e{max_epoch}"
weights_dir = f"model_weights/vaihingen_{image_size}_CA"
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

# === 4. 在配置文件内部实例化 Dataset ===
IMG_SUFFIX = '.tif'
MASK_SUFFIX = '.png'

try:
    img_size_tuple = (image_size, image_size)
    train_dataset = VaihingenDataset(data_root=VAIHINGEN_TRAIN_DATA_ROOT, mode='train',
                                     img_suffix=IMG_SUFFIX, mask_suffix=MASK_SUFFIX,
                                     transform=train_aug, mosaic_ratio=0.25, img_size=img_size_tuple)
    val_dataset = VaihingenDataset(data_root=VAIHINGEN_VAL_DATA_ROOT, mode='val',
                                   img_suffix=IMG_SUFFIX, mask_suffix=MASK_SUFFIX,
                                   transform=val_aug, img_size=img_size_tuple)
    test_dataset = VaihingenDataset(data_root=VAIHINGEN_VAL_DATA_ROOT, mode='val',
                                    img_suffix=IMG_SUFFIX, mask_suffix=MASK_SUFFIX,
                                    transform=val_aug, img_size=img_size_tuple)
except FileNotFoundError as e:
     print(f"CRITICAL ERROR in config: {e}.")
     exit()
except NameError as e:
     print(f"CRITICAL ERROR: {e}.")
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
    if hasattr(net, 'encoder'):
         net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
    elif hasattr(net, 'backbone'):
         layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
         net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
    else:
         print("Warning: Cannot find 'backbone' or 'encoder' for layerwise params.")
         net_params = net.parameters()
except Exception as e:
    print(f"Warning: Could not process layerwise params: {e}")
    net_params = net.parameters()

base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)


# --- (可选) 打印模型信息 ---
# ...

print(f"Config loaded for Vaihingen (Online Augmentation), Model: UNetMambaCA")
print(f"Checking Train data root: {os.path.abspath(VAIHINGEN_TRAIN_DATA_ROOT)}")
print(f"Checking Val data root: {os.path.abspath(VAIHINGEN_VAL_DATA_ROOT)}")
