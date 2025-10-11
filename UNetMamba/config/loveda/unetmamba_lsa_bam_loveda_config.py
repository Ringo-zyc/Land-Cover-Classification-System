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
        LoveDATrainDataset,
        loveda_val_dataset, # This might be an instance or a class
        LoveDATestDataset,
        CLASSES,            # 应该在 loveda_dataset.py 中定义
        # IGNORE_INDEX,     # 已在配置文件中直接定义
        # train_aug,        # 如果需要，从这里导入或在下面定义
        # val_aug
    )
except ImportError as e:
     print(f"Error importing from loveda_dataset: {e}")
     print("Make sure loveda_dataset.py is in unetmamba_model/datasets/ and relevant classes/variables are defined.")
     exit()
except NameError as e: # Handle if CLASSES is not defined in loveda_dataset.py
    print(f"Error: Variable 'CLASSES' not found in unetmamba_model.datasets.loveda_dataset.py. {e}")
    exit()


# === 2. 导入模型和损失函数 ===
try:
    from unetmamba_model.models.UNetMamba_LSA_BAM_Model_V2 import UNetMamba # 指向 LSA/BAM 模型
    from unetmamba_model.losses import UnetMambaLoss
except ImportError as e:
     print(f"Error importing UNetMamba_LSA_BAM_Model_V2 or UnetMambaLoss: {e}")
     exit()

# --- 数据集相关配置 (LoveDA specific) ---
LOVEDA_TRAIN_DATA_ROOT = 'data/LoveDA/Train' # 示例路径，请修改为你的实际路径
LOVEDA_VAL_DATA_ROOT = 'data/LoveDA/Val'     # 示例路径，请修改为你的实际路径
LOVEDA_TEST_DATA_ROOT = 'data/LoveDA/Test'   # 示例路径，请修改为你的实际路径

# --- 训练超参数 ---
max_epoch = 100
num_classes = len(CLASSES)
ignore_index = len(CLASSES)
classes = CLASSES

train_batch_size = 8
val_batch_size = 8
image_size = 1024 # 这个变量仍然有用，例如用于数据增强函数

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
pretrained_ckpt_path = None
resume_ckpt_path = None
gpus = 'auto'

# --- 数据集和数据增强定义 ---
# 假设你的 LoveDATrainDataset, loveda_val_dataset, LoveDATestDataset
# 能够正确处理数据加载和基本转换。
# 如果它们需要显式的 transform 函数，你需要在这里定义或从别处导入。
# 例如，如果需要自定义的 train_aug 和 val_aug：
# from unetmamba_model.datasets.loveda_dataset import train_aug # 如果它们已在数据集中定义
# 或者在这里定义它们：
# def train_aug(img, mask): ... # 确保它使用了全局的 image_size
# def val_aug(img, mask): ...

try:
    # === 修改点: 移除 image_size 参数从数据集类的构造函数 ===
    # 假设 LoveDATrainDataset 的 transform 参数是可选的，或者在内部处理
    # 如果它需要一个 transform，你需要在这里提供，例如：
    # train_dataset = LoveDATrainDataset(data_root=LOVEDA_TRAIN_DATA_ROOT, transform=your_train_transform_function)
    # 这里的 your_train_transform_function 内部可能会使用全局的 image_size 变量
    train_dataset = LoveDATrainDataset(data_root=LOVEDA_TRAIN_DATA_ROOT) # 移除 image_size

    if isinstance(loveda_val_dataset, torch.utils.data.Dataset):
         val_dataset = loveda_val_dataset # 已经是实例，不需要 image_size
    else:
         # 如果 loveda_val_dataset 是一个类，实例化时不传递 image_size
         # val_dataset = loveda_val_dataset(data_root=LOVEDA_VAL_DATA_ROOT, transform=your_val_transform_function)
         val_dataset = loveda_val_dataset(data_root=LOVEDA_VAL_DATA_ROOT) # 移除 image_size

    # test_dataset = LoveDATestDataset(data_root=LOVEDA_TEST_DATA_ROOT, transform=your_test_transform_function)
    test_dataset = LoveDATestDataset(data_root=LOVEDA_TEST_DATA_ROOT) # 移除 image_size
    # ============================================================
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
try:
    net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
except AttributeError:
    print("Warning: Could not find 'net.encoder' for layerwise_params. Using all parameters for optimizer.")
    net_params = net.parameters()
except Exception as e:
    print(f"Warning: Error processing layerwise_params: {e}. Using all parameters.")
    net_params = net.parameters()


optimizer = Lookahead(torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay))

_lr_scheduler_instance = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
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

print(f"Config loaded for LoveDA, Model: {model_variant_name.upper()}")
print(f"  USE_LSA_IN_DECODER: {USE_LSA_IN_DECODER}")
print(f"  USE_BAM_IN_DECODER: {USE_BAM_IN_DECODER}")
print(f"  Train data root: {LOVEDA_TRAIN_DATA_ROOT}")
print(f"  Val data root: {LOVEDA_VAL_DATA_ROOT}")
print(f"  IGNORE_INDEX set to: {ignore_index}")

