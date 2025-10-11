# config/postdam/potsdam_config.py (Modified for Patches)
import torch
from torch.utils.data import DataLoader
from catalyst.contrib.nn import Lookahead
from catalyst import utils
import albumentations as albu
import copy

# 导入修改后的 PotsdamPatchesDataset
from unetmamba_model.datasets.potsdam_dataset import (
    PotsdamPatchesDataset, # <--- 使用加载 Patches 的类
    CLASSES,
    PALETTE,
    IGNORE_INDEX,
    # 不再需要导入 train/val aug 函数，Dataset内部根据 mode 选择
)
from unetmamba_model.models.UNetMamba_CA import UNetMambaCA
from unetmamba_model.losses import UnetMambaLoss

# --- 配置参数 ---
PATCH_SIZE = 1024 # 与分块脚本使用的 patch_size 一致
STRIDE = 1024     # 与分块脚本使用的 stride 一致 (无重叠)

# --- 数据集路径配置 ---
# !! ****** 指向存放分块数据集的 *父* 目录 ****** !!
# 例如，如果分块输出到 data/Potsdam/train_1024 和 data/Potsdam/val_1024
POTSDAM_TRAIN_PATCHES_ROOT = f'data/Potsdam/train_{PATCH_SIZE}' # <--- !!! 检查/修改此路径 !!!
POTSDAM_VAL_PATCHES_ROOT = f'data/Potsdam/val_{PATCH_SIZE}'     # <--- !!! 检查/修改此路径 !!!

# --- 训练超参数 ---
max_epoch = 100
num_classes = len(CLASSES) # 6
ignore_index = IGNORE_INDEX # 255
classes = CLASSES

# !! Batch Size for 1024x1024 patches on 24GB GPU will likely be small !!
train_batch_size = 2 # 尝试从 2 开始，如果 OOM 再降为 1
val_batch_size = 2  # 尝试从 2 开始

lr = 5e-4 # 可能需要调整
weight_decay = 2.5e-4
backbone_lr = 5e-5
backbone_weight_decay = 2.5e-4
# image_size 参数现在主要用于FLOPs计算等，设为 patch size
image_size = (PATCH_SIZE, PATCH_SIZE)

# --- 模型与权重配置 ---
net = UNetMambaCA(pretrained=True, num_classes=num_classes, embed_dim=64)
loss = UnetMambaLoss(ignore_index=ignore_index)
use_aux_loss = True # Check if your UNetMambaCA uses aux loss

# --- 权重、日志、监控配置 ---
weights_name = f"unetmamba_CA_potsdam_patch{PATCH_SIZE}-e{max_epoch}"
weights_path = f"model_weights/potsdam_patches_{PATCH_SIZE}/{{}}".format(weights_name) # 使用 f-string
log_name = f'potsdam_patches_{PATCH_SIZE}/{{}}'.format(weights_name) # 使用 f-string
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # Train from scratch on Potsdam patches
resume_ckpt_path = None

# --- GPU 配置 ---
gpus = 'auto'

# --- 数据加载器配置 ---
# !! 使用新的 Dataset 类并指向 Patches 目录 !!
try:
    train_dataset = PotsdamPatchesDataset(data_root=POTSDAM_TRAIN_PATCHES_ROOT, mode='train')
    val_dataset = PotsdamPatchesDataset(data_root=POTSDAM_VAL_PATCHES_ROOT, mode='val')
    # test_dataset = PotsdamPatchesDataset(data_root=POTSDAM_TEST_PATCHES_ROOT, mode='test') # If you create test patches
except FileNotFoundError as e:
     print(f"Error initializing dataset: {e}. Check POTSDAM_*_PATCHES_ROOT paths.")
     exit()


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4, # 可以尝试恢复 > 0
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True,
                          persistent_workers=True if 4 > 0 else False) # Add for efficiency

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4, # 可以尝试恢复 > 0
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False,
                        persistent_workers=True if 4 > 0 else False) # Add for efficiency

test_dataset = PotsdamPatchesDataset(data_root=POTSDAM_VAL_PATCHES_ROOT, mode='val')

# --- 优化器与学习率调度器 ---
layerwise_params = {"encoder.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

# (可选) 打印模型信息、计算参数量等
# (Keep the optional block if desired, ensure image_size is correct)