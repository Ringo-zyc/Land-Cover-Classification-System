# config/postdam/unetmamba.py (For Baseline UNetMamba on Potsdam Patches)
import torch
from torch.utils.data import DataLoader
from catalyst.contrib.nn import Lookahead
from catalyst import utils
import albumentations as albu
import copy

# 导入修改后的 PotsdamPatchesDataset
# !! 确保路径正确 !!
from unetmamba_model.datasets.potsdam_dataset import (
    PotsdamPatchesDataset,
    CLASSES,
    PALETTE,
    IGNORE_INDEX,
)
# === 修改处 1: 导入原始 UNetMamba 模型 ===
# !! 确保路径正确 !!
from unetmamba_model.models.UNetMamba import UNetMamba
from unetmamba_model.losses import UnetMambaLoss

# --- 配置参数 ---
PATCH_SIZE = 1024 # 与分块脚本使用的 patch_size 一致
STRIDE = 1024     # 与分块脚本使用的 stride 一致 (无重叠)

# --- 数据集路径配置 ---
# 指向存放 Patches 的父目录
POTSDAM_TRAIN_PATCHES_ROOT = f'data/Potsdam/train_{PATCH_SIZE}' # 和 CA 配置保持一致
POTSDAM_VAL_PATCHES_ROOT = f'data/Potsdam/val_{PATCH_SIZE}'     # 和 CA 配置保持一致

# --- 训练超参数 ---
max_epoch = 100
num_classes = len(CLASSES) # 6
ignore_index = IGNORE_INDEX # 255
classes = CLASSES

# Batch Size (与 CA 配置保持一致，以便比较，如果 OOM 再调整)
train_batch_size = 2
val_batch_size = 2

lr = 5e-4
weight_decay = 2.5e-4
backbone_lr = 5e-5
backbone_weight_decay = 2.5e-4
image_size = (PATCH_SIZE, PATCH_SIZE) # 主要用于 FLOPs 计算等

# --- 模型与权重配置 ---
# === 修改处 2: 实例化原始 UNetMamba 模型 ===
net = UNetMamba(pretrained=True, # 加载 ResT 预训练权重
                num_classes=num_classes,
                embed_dim=64,
                backbone_path='pretrain_weights/rest_lite.pth' # 确认路径或设为 None 让其使用默认
               )
# 损失函数保持不变
loss = UnetMambaLoss(ignore_index=ignore_index)
use_aux_loss = True # 检查原始 UNetMamba 是否也使用辅助损失

# --- 权重、日志、监控配置 ---
# === 修改处 3: 修改输出名称和路径 ===
weights_name = f"unetmamba_baseline_potsdam_patch{PATCH_SIZE}-e{max_epoch}" # 使用 "baseline" 标识
weights_path = f"model_weights/potsdam_patches_{PATCH_SIZE}_baseline/{{}}".format(weights_name) # 单独目录
log_name = f'potsdam_patches_{PATCH_SIZE}_baseline/{{}}'.format(weights_name) # 单独日志

monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
# === 修改处 4: 确认预训练检查点为 None ===
pretrained_ckpt_path = None # 从 ResT 预训练开始，不加载整个 UNetMamba 检查点
resume_ckpt_path = None

# --- GPU 配置 ---
gpus = 'auto'

# --- 数据加载器配置 ---
# 使用 PotsdamPatchesDataset 并指向 Patches 目录 (与 CA 配置一致)
try:
    train_dataset = PotsdamPatchesDataset(data_root=POTSDAM_TRAIN_PATCHES_ROOT, mode='train')
    val_dataset = PotsdamPatchesDataset(data_root=POTSDAM_VAL_PATCHES_ROOT, mode='val')
except FileNotFoundError as e:
     print(f"Error initializing dataset: {e}. Check POTSDAM_*_PATCHES_ROOT paths.")
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

test_dataset = PotsdamPatchesDataset(data_root=POTSDAM_VAL_PATCHES_ROOT, mode='val')

# --- 优化器与学习率调度器 ---
# 与 CA 配置保持一致
layerwise_params = {"encoder.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)} # 确认原始模型中 backbone 变量名也是 encoder
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

# (可选) 打印模型信息等
# (如果需要，取消注释并确保 image_size 正确)