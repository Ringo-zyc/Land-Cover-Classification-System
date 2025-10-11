# filename: config/loveda/unetmamba_CA_AFR_BAM_LFPM.py

import torch
import os
from torch.utils.data import DataLoader
from catalyst.contrib.nn import Lookahead
from catalyst import utils
# import albumentations as albu # 不再需要在 config 中导入
# import numpy as np # 不再需要在 config 中导入

# === 1. 导入 LoveDA 数据集类和函数 ===
try:
    # 直接导入需要的数据集类/对象和全局变量
    from unetmamba_model.datasets.loveda_dataset import (
        LoveDATrainDataset, loveda_val_dataset, LoveDATestDataset,
        CLASSES # 确保 IGNORE_INDEX 在 loveda_dataset.py 中定义或在这里计算
    )
    # 如果 IGNORE_INDEX 没有从 loveda_dataset 导出，则按之前方式计算
    # if 'IGNORE_INDEX' not in locals():
    #     if len(CLASSES) == 7: IGNORE_INDEX = 7
    #     else: IGNORE_INDEX = len(CLASSES) # Fallback

except ImportError as e: print(f"CRITICAL ERROR: Could not import from loveda_dataset: {e}. Check dataset file and './transform.py'."); exit()
except NameError as e: print(f"CRITICAL ERROR: {e}. Check loveda_dataset definitions."); exit()
except Exception as e: print(f"Dataset import error: {e}"); exit()
if 'CLASSES' in locals() and len(CLASSES) == 7:
     IGNORE_INDEX = 7
     print(f"LoveDA: Detected {len(CLASSES)} classes. Setting IGNORE_INDEX to {IGNORE_INDEX}")
elif 'CLASSES' in locals():
     IGNORE_INDEX = len(CLASSES)
     print(f"Warning: Expected 7 classes for LoveDA, found {len(CLASSES)}. Setting IGNORE_INDEX to {IGNORE_INDEX}.")
else:
     print("CRITICAL ERROR: Cannot determine IGNORE_INDEX because CLASSES was not imported.")
     exit() # 如果 CLASSES 都没导入成功，就退出
# === 2. 导入模型和损失函数 ===
try:
    from unetmamba_model.models.UNetMamba_CA_AFR_BAM_LFPM import UNetMambaCA_AFR_BAM_LFPM
    from unetmamba_model.losses.useful_loss import UnetMambaLoss # 使用 UnetMambaLoss
except ImportError as e: print(f"CRITICAL ERROR: Could not import model or loss: {e}"); exit()


# === 配置参数 ===
image_size = 1024
# crop_size = 512 # crop_size 在 loveda_dataset.py 内部的 train_aug 中使用，这里无需定义

# === 3. 数据集路径 ===
# <<<--- !!! 请务必检查这些路径是否正确 !!! --->
LOVEDA_TRAIN_DATA_ROOT = 'data/LoveDA/Train'
LOVEDA_VAL_DATA_ROOT = 'data/LoveDA/Val' # loveda_val_dataset 初始化时已用

# --- 训练超参数 ---
max_epoch = 100
num_classes = len(CLASSES) # Should be 7
ignore_index = IGNORE_INDEX # Should be 7 (or fallback)
classes = CLASSES
train_batch_size = 4
val_batch_size = 4
lr = 6e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4
afr_lr = 6e-4
bam_lr = 6e-4
lfpm_lr = 6e-4

# --- 模型配置 ---
# VSSM parameters
PATCH_SIZE_VSSM = 4; IN_CHANS = 3; DEPTHS = [2, 2, 9, 2]; EMBED_DIM = 96
SSM_D_STATE = 16; SSM_RATIO = 2.0; SSM_RANK_RATIO = 2.0; SSM_DT_RANK = "auto"
SSM_ACT_LAYER = "silu"; SSM_CONV = 3; SSM_CONV_BIAS = True; SSM_DROP_RATE = 0.0
SSM_INIT = "v0"; SSM_FORWARDTYPE = "v4"; MLP_RATIO = 4.0; MLP_ACT_LAYER = "gelu"
MLP_DROP_RATE = 0.0; DROP_PATH_RATE = 0.1; PATCH_NORM = True; NORM_LAYER = "ln"
DOWNSAMPLE = "v2"; PATCHEMBED = "v2"; GMLP = False
USE_CHECKPOINT = True # 建议开启

# ResT backbone base dimension
REST_EMBED_DIM = 64

# === 实例化最终模型 ===
net = UNetMambaCA_AFR_BAM_LFPM(
    # --- Core Parameters ---
    num_classes=num_classes, input_channels=IN_CHANS, embed_dim=REST_EMBED_DIM,
    afr_reduction_ratio=16, ca_reduction=32, bam_mid_channels=32,
    lfpm_compression_ratio=4, lfpm_dilations=[1, 6, 12, 18],
    # --- Backbone Path ---
    backbone_path='pretrain_weights/rest_lite.pth', # 确保路径正确
    # --- Other parameters ---
    decoder_depths=[2, 2, 2], drop_path_rate=DROP_PATH_RATE, d_state=SSM_D_STATE,
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
model_arch_name = "unetmamba_CA_AFR_BAM_LFPM"
dataset_name = "loveda"
# 使用 image_size 命名，即使它没在 config 中直接用于数据集大小
config_image_size_for_name = 1024
weights_name = f"{model_arch_name}_{dataset_name}_{config_image_size_for_name}-e{max_epoch}"
weights_dir = f"model_weights/{dataset_name}/{model_arch_name}_{config_image_size_for_name}"
weights_path = weights_dir
log_name = f'{dataset_name}/{weights_name}'

monitor = 'val_mIoU'; monitor_mode = 'max'; save_top_k = 1; save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None; resume_ckpt_path = None; gpus = 'auto'

# === 4. 数据集和数据加载器实例化 ===
# --- 数据增强函数定义已移至 loveda_dataset.py ---
# --- Delete the train_aug() and get_training_transform() definitions here ---

# --- 实例化数据集 ---
try:
    print(f"Initializing LoveDA datasets using internal transforms...")
    # 实例化训练集，它将使用 loveda_dataset.py 中默认的 train_aug
    # mosaic_ratio 可以从 loveda_dataset.py 获取默认值或在此处传递
    train_dataset = LoveDATrainDataset(
        data_root=LOVEDA_TRAIN_DATA_ROOT,
        # mosaic_ratio=0.25 # 如果需要覆盖 loveda_dataset.py 中的默认值
    )

    # 直接使用从 loveda_dataset.py 导入的已实例化的验证集对象
    val_dataset = loveda_val_dataset

    # 实例化测试集 (如果需要)
    test_dataset = LoveDATestDataset() # 检查是否需要参数

    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    if len(train_dataset) == 0 or len(val_dataset) == 0: print("CRITICAL ERROR: Datasets are empty."); exit()
except FileNotFoundError as e: print(f"CRITICAL ERROR: Dataset directory not found: {e}. Check LOVEDA_*_DATA_ROOT paths."); exit()
except NameError as e: print(f"CRITICAL ERROR: Name not defined: {e}."); exit()
except Exception as e: print(f"Dataset initialization error: {e}"); exit()

# --- 数据加载器定义 ---
pin_memory = True; num_workers = 4 # 检查 num_workers 是否合适
train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True, persistent_workers=True if num_workers > 0 else False)
val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, num_workers=num_workers, shuffle=False, pin_memory=pin_memory, drop_last=False, persistent_workers=True if num_workers > 0 else False)

# --- 优化器与学习率调度器 ---
print("Setting up optimizer with differential learning rates...")
try:
    backbone_params = [p for n, p in net.named_parameters() if n.startswith('encoder.') and p.requires_grad]
    afr_params = [p for n, p in net.named_parameters() if n.startswith('decoder.afr_modules.') and p.requires_grad]
    bam_params = [p for n, p in net.named_parameters() if n.startswith('decoder.bam.') and p.requires_grad and 'edge_conv.weight' not in n]
    lfpm_params = [p for n, p in net.named_parameters() if n.startswith('decoder.lfpm.') and p.requires_grad]
    base_param_ids = set(id(p) for p in net.parameters() if p.requires_grad)
    backbone_param_ids=set(id(p) for p in backbone_params); afr_param_ids=set(id(p) for p in afr_params)
    bam_param_ids=set(id(p) for p in bam_params); lfpm_param_ids=set(id(p) for p in lfpm_params)
    other_params = [p for p in net.parameters() if id(p) in base_param_ids - backbone_param_ids - afr_param_ids - bam_param_ids - lfpm_param_ids]
    print(f"Found {len(backbone_params)} backbone, {len(afr_params)} AFR, {len(bam_params)} BAM, {len(lfpm_params)} LFPM, {len(other_params)} other params.")

    net_params = []
    if backbone_params: net_params.append({'params': backbone_params, 'lr': backbone_lr, 'weight_decay': backbone_weight_decay})
    if afr_params: net_params.append({'params': afr_params, 'lr': afr_lr, 'weight_decay': weight_decay})
    if bam_params: net_params.append({'params': bam_params, 'lr': bam_lr, 'weight_decay': weight_decay})
    if lfpm_params: net_params.append({'params': lfpm_params, 'lr': lfpm_lr, 'weight_decay': weight_decay})
    if other_params: net_params.append({'params': other_params, 'lr': lr, 'weight_decay': weight_decay})

    if not net_params: net_params = net.parameters()
    base_optimizer = torch.optim.AdamW(net_params, lr=lr)
    optimizer = Lookahead(base_optimizer)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
    print("Optimizer and LR Scheduler configured.")
except Exception as e:
    print(f"ERROR setting up optimizer: {e}. Falling back to basic AdamW.")
    optimizer = Lookahead(torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

# --- Final Config Check ---
print(f"\n--- Configuration Summary ---")
print(f"Model: {model_arch_name}")
print(f"Dataset: {dataset_name.capitalize()}")
# print(f"Image Size (used for naming): {config_image_size_for_name}") # Image size not directly used for dataset now
print(f"Epochs: {max_epoch}")
print(f"Batch Size (Train/Val): {train_batch_size}/{val_batch_size}")
print(f"Learning Rates (Base/Backbone/AFR/BAM/LFPM): {lr}/{backbone_lr}/{afr_lr}/{bam_lr}/{lfpm_lr}")
print(f"Loss: UnetMambaLoss") # Indicate the loss used
print(f"Weights Save Directory: {weights_dir}")
print(f"Log Name: {log_name}")
print(f"Backbone Pretrained Weights: pretrain_weights/rest_lite.pth")
print(f"Resume Checkpoint: {resume_ckpt_path}")
print(f"---------------------------\n")