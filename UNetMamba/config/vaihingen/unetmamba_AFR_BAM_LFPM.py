import torch
import os
from torch.utils.data import DataLoader
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# === 1. Vaihingen-Datensatzklassen und -funktionen importieren ===
try:
    from unetmamba_model.datasets.vaihingen_dataset import (
        VaihingenDataset, CLASSES, IGNORE_INDEX, train_aug, val_aug
    )
except ImportError as e:
    print(f"KRITISCHER FEHLER: Konnte nicht aus vaihingen_dataset importieren: {e}")
    exit()
except NameError as e:
    print(f"KRITISCHER FEHLER: {e}. Überprüfen Sie die Definitionen in vaihingen_dataset.")
    exit()

# === 2. Modell und Verlustfunktion importieren ===
try:
    from unetmamba_model.models.UNetMamba_AFR_BAM_LFPM import UNetMamba_AFR_BAM_LFPM
    from unetmamba_model.losses.useful_loss import UnetMambaLoss
except ImportError as e:
    print(f"KRITISCHER FEHLER: Konnte Modell oder Verlust nicht importieren: {e}")
    exit()

# === Konfigurationsparameter ===
image_size = 1024

# === 3. Datensatzpfade ===
VAIHINGEN_TRAIN_DATA_ROOT = f'data/vaihingen/train_{image_size}'
VAIHINGEN_VAL_DATA_ROOT = f'data/vaihingen/val_{image_size}'

# --- Trainingshyperparameter ---
max_epoch = 100
num_classes = len(CLASSES)
ignore_index = IGNORE_INDEX
classes = CLASSES
train_batch_size = 4
val_batch_size = 1

# --- Lernraten und Regularisierung ---
lr = 6e-4
weight_decay = 4e-4
backbone_lr = 6e-5
backbone_weight_decay = 4e-4
afr_lr = 4e-4
bam_lr = 3e-4
lfpm_lr = 8e-4

# --- Modellkonfiguration ---
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
USE_CHECKPOINT = True
REST_EMBED_DIM = 64
backbone_path = 'pretrain_weights/rest_lite.pth'

# === Modell instanziieren ===
net = UNetMamba_AFR_BAM_LFPM(
    num_classes=num_classes,
    input_channels=IN_CHANS,
    embed_dim=REST_EMBED_DIM,
    afr_reduction_ratio=16,
    bam_mid_channels=32,
    lfpm_compression_ratio=8,
    lfpm_dilations=[1, 6, 12],
    backbone_path=backbone_path,
    decoder_depths=[2, 2, 2],
    drop_path_rate=DROP_PATH_RATE,
    d_state=SSM_D_STATE,
    patch_size=PATCH_SIZE_VSSM,
    depths=DEPTHS,
    dims=EMBED_DIM,
    ssm_d_state=SSM_D_STATE,
    ssm_ratio=SSM_RATIO,
    ssm_rank_ratio=SSM_RANK_RATIO,
    ssm_dt_rank=SSM_DT_RANK,
    ssm_act_layer=SSM_ACT_LAYER,
    ssm_conv=SSM_CONV,
    ssm_conv_bias=SSM_CONV_BIAS,
    ssm_drop_rate=SSM_DROP_RATE,
    ssm_init=SSM_INIT,
    forward_type=SSM_FORWARDTYPE,
    mlp_ratio=MLP_RATIO,
    mlp_act_layer=MLP_ACT_LAYER,
    mlp_drop_rate=MLP_DROP_RATE,
    patch_norm=PATCH_NORM,
    norm_layer=NORM_LAYER,
    downsample_version=DOWNSAMPLE,
    patchembed_version=PATCHEMBED,
    gmlp=GMLP,
    use_checkpoint=USE_CHECKPOINT
)

# --- Verlustfunktion ---
loss = UnetMambaLoss(ignore_index=ignore_index)
use_aux_loss = True

# --- Gewichte und Protokolle ---
model_arch_name = "unetmamba_modified_v1"
dataset_name = "vaihingen"
weights_name = f"{model_arch_name}_{dataset_name}_{image_size}-e{max_epoch}"
weights_dir = f"model_weights/{dataset_name}/{model_arch_name}_{image_size}"
weights_path = weights_dir
log_name = f'{dataset_name}/{weights_name}'
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
resume_ckpt_path = None
gpus = 'auto'

# === 4. Datensätze instanziieren ===
IMG_SUFFIX = '.tif'
MASK_SUFFIX = '.png'
try:
    img_size_tuple = (image_size, image_size) if isinstance(image_size, int) else image_size
    print(f"Initialisiere Vaihingen-Datensätze mit Bildgröße: {img_size_tuple}")
    train_dataset = VaihingenDataset(data_root=VAIHINGEN_TRAIN_DATA_ROOT, mode='train',
                                     img_suffix=IMG_SUFFIX, mask_suffix=MASK_SUFFIX,
                                     transform=train_aug, mosaic_ratio=0.25, img_size=img_size_tuple)
    val_dataset = VaihingenDataset(data_root=VAIHINGEN_VAL_DATA_ROOT, mode='val',
                                   img_suffix=IMG_SUFFIX, mask_suffix=MASK_SUFFIX,
                                   transform=val_aug, img_size=img_size_tuple)
    test_dataset = VaihingenDataset(data_root=VAIHINGEN_VAL_DATA_ROOT, mode='val',
                                    img_suffix=IMG_SUFFIX, mask_suffix=MASK_SUFFIX,
                                    transform=val_aug, img_size=img_size_tuple)
    print(f"Trainingsdatensatzgröße: {len(train_dataset)}, Validierungsdatensatzgröße: {len(val_dataset)}")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("KRITISCHER FEHLER: Datensätze sind leer.")
        exit()
except FileNotFoundError as e:
    print(f"KRITISCHER FEHLER: Datensatzverzeichnis nicht gefunden: {e}")
    exit()
except NameError as e:
    print(f"KRITISCHER FEHLER: Name nicht definiert: {e}")
    exit()
except Exception as e:
    print(f"Fehler bei der Datensatzinitialisierung: {e}")
    exit()

# --- Datenlader ---
pin_memory = True
num_workers = 4
train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, num_workers=num_workers,
                          pin_memory=pin_memory, shuffle=True, drop_last=True,
                          persistent_workers=True if num_workers > 0 else False)
val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=pin_memory, drop_last=False,
                        persistent_workers=True if num_workers > 0 else False)

# --- Optimierer und Scheduler ---
print("Konfiguriere Optimierer mit differenzierten Lernraten...")
try:
    backbone_params = [p for n, p in net.named_parameters() if n.startswith('encoder.') and p.requires_grad]
    afr_params = [p for n, p in net.named_parameters() if n.startswith('decoder.afr_modules.') and p.requires_grad]
    bam_params = [p for n, p in net.named_parameters() if n.startswith('decoder.bam.') and p.requires_grad and 'edge_conv.weight' not in n]
    lfpm_params = [p for n, p in net.named_parameters() if n.startswith('decoder.lfpm.') and p.requires_grad]
    fusion_conv_params = [p for n, p in net.named_parameters() if n.startswith('decoder.fusion_conv.') and p.requires_grad]
    base_param_ids = set(id(p) for p in net.parameters() if p.requires_grad)
    backbone_param_ids = set(id(p) for p in backbone_params)
    afr_param_ids = set(id(p) for p in afr_params)
    bam_param_ids = set(id(p) for p in bam_params)
    lfpm_param_ids = set(id(p) for p in lfpm_params)
    fusion_conv_param_ids = set(id(p) for p in fusion_conv_params)
    other_params = [p for p in net.parameters() if id(p) in base_param_ids - backbone_param_ids - afr_param_ids - bam_param_ids - lfpm_param_ids - fusion_conv_param_ids]
    print(f"Gefunden: {len(backbone_params)} Backbone, {len(afr_params)} AFR, {len(bam_params)} BAM, {len(lfpm_params)} LFPM, {len(fusion_conv_params)} FusionConv, {len(other_params)} andere Parameter.")
    optimizer_grouped_parameters = []
    learning_rates = []
    if backbone_params:
        optimizer_grouped_parameters.append({'params': backbone_params, 'lr': backbone_lr, 'weight_decay': backbone_weight_decay})
        learning_rates.append(backbone_lr)
    if afr_params:
        optimizer_grouped_parameters.append({'params': afr_params, 'lr': afr_lr, 'weight_decay': weight_decay})
        learning_rates.append(afr_lr)
    if bam_params:
        optimizer_grouped_parameters.append({'params': bam_params, 'lr': bam_lr, 'weight_decay': weight_decay})
        learning_rates.append(bam_lr)
    if lfpm_params:
        optimizer_grouped_parameters.append({'params': lfpm_params, 'lr': lfpm_lr, 'weight_decay': weight_decay})
        learning_rates.append(lfpm_lr)
    other_and_fusion_params = other_params + fusion_conv_params
    if other_and_fusion_params:
        optimizer_grouped_parameters.append({'params': other_and_fusion_params, 'lr': lr, 'weight_decay': weight_decay})
        learning_rates.append(lr)
    if not optimizer_grouped_parameters:
        print("WARNUNG: Keine Parametergruppen gefunden, verwende Standardoptimierer.")
        optimizer_grouped_parameters = net.parameters()
        learning_rates = [lr]
    base_optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    optimizer = Lookahead(base_optimizer)
    warmup_epochs = 10
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * max_epoch
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rates,
        total_steps=total_steps,
        pct_start=warmup_epochs / max_epoch if max_epoch > 0 else 0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )
    lr_scheduler_config = {'scheduler': lr_scheduler, 'interval': 'step'}
    print("Optimierer und LR Scheduler konfiguriert.")
except Exception as e:
    print(f"FEHLER beim Optimierer/Scheduler: {e}. Fallback auf AdamW + Cosine.")
    optimizer = Lookahead(torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
    lr_scheduler_config = {'scheduler': lr_scheduler, 'interval': 'epoch'}

# --- Konfigurationsübersicht ---
print(f"\n--- Konfigurationsübersicht ---")
print(f"Modell: {model_arch_name}")
print(f"Datensatz: {dataset_name.capitalize()}")
print(f"Bildgröße: {image_size}")
print(f"Epochen: {max_epoch}")
print(f"Batch-Größe (Train/Val): {train_batch_size}/{val_batch_size}")
print(f"Max Lernraten (Basis/Backbone/AFR/BAM/LFPM): {lr}/{backbone_lr}/{afr_lr}/{bam_lr}/{lfpm_lr}")
print(f"Scheduler: OneCycleLR (Warmup: {warmup_epochs} Epochen)")
print(f"Gewichtsabnahme (Basis/Backbone): {weight_decay}/{backbone_weight_decay}")
print(f"Speicherort für Gewichte: {weights_dir}")
print(f"Protokollname: {log_name}")
print(f"Backbone vortrainierte Gewichte: {backbone_path}")
print(f"Fortsetzungs-Checkpoint: {resume_ckpt_path}")
print(f"Gradient Checkpoint aktiviert: {USE_CHECKPOINT}")
print(f"---------------------------\n")