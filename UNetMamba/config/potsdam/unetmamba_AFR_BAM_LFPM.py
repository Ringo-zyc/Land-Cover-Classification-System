# filename: config/potsdam/unetmamba_modified_v1.py
# Konfigurationsdatei für das modifizierte UNetMamba-Modell auf dem Potsdam Patch-Dataset
# Version 2: Fügt die Definition von test_dataset hinzu

import torch
import os
from torch.utils.data import DataLoader
from catalyst.contrib.nn import Lookahead # Lookahead-Optimierer-Wrapper
from catalyst import utils
import copy # copy wird hier nicht direkt verwendet, aber Import bleibt bestehen

# === 1. Potsdam Patch-Dataset Klassen und Funktionen importieren ===
try:
    # Annahme: Diese sind in der angegebenen Datei definiert
    from unetmamba_model.datasets.potsdam_dataset import (
        PotsdamPatchesDataset, # <<<--- Geändert auf Potsdam Patch Dataset
        CLASSES,
        PALETTE, # PALETTE wird hier nicht direkt verwendet, aber Import bleibt bestehen
        IGNORE_INDEX
        # Augmentations werden jetzt innerhalb des Datasets gehandhabt
    )
except ImportError as e:
    print(f"KRITISCHER FEHLER: Konnte nicht aus potsdam_dataset importieren: {e}")
    exit()
except NameError as e:
    print(f"KRITISCHER FEHLER: {e}. Überprüfen Sie die Definitionen in potsdam_dataset.")
    exit()

# === 2. Modifiziertes Modell und Verlustfunktion importieren ===
try:
    # === Modifiziertes Modell importieren ===
    # Annahme: Der Code aus unetmamba_modified_v1_zh wird unter diesem Pfad/Namen gespeichert
    from unetmamba_model.models.UNetMamba_AFR_BAM_LFPM import UNetMamba_AFR_BAM_LFPM # <<<--- Importiert das modifizierte Modell
    from unetmamba_model.losses.useful_loss import UnetMambaLoss # Annahme: Verlustfunktion bleibt gleich
except ImportError as e:
    print(f"KRITISCHER FEHLER: Konnte Modell oder Verlust nicht importieren: {e}")
    exit()


# === Konfigurationsparameter ===
PATCH_SIZE = 1024 # Patch-Größe, sollte mit der Erstellung der Patches übereinstimmen
# image_size wird jetzt auf PATCH_SIZE gesetzt, da wir mit Patches arbeiten
image_size = PATCH_SIZE

# === 3. Datensatzpfade (Potsdam Patches) ===
# !! ****** BITTE DIESE PFADE AN DEINE VERZEICHNISSTRUKTUR ANPASSEN ****** !!
POTSDAM_TRAIN_PATCHES_ROOT = f'data/Potsdam/train_{PATCH_SIZE}' # Pfad zu den Trainings-Patches
POTSDAM_VAL_PATCHES_ROOT = f'data/Potsdam/val_{PATCH_SIZE}'   # Pfad zu den Validierungs-Patches
# Optional: Pfad für separate Test-Patches definieren, falls vorhanden
# POTSDAM_TEST_PATCHES_ROOT = f'data/Potsdam/test_{PATCH_SIZE}'
# !! ******************************************************************* !!

# --- Trainingshyperparameter ---
max_epoch = 100 # Maximale Anzahl von Epochen
num_classes = len(CLASSES) # Sollte 6 für Potsdam sein
ignore_index = IGNORE_INDEX # Sollte 255 für Potsdam sein
classes = CLASSES
# --- Angepasste Batch-Größe für Patches ---
train_batch_size = 2 # <<<--- Reduziert für 1024x1024 Patches (ggf. anpassen)
val_batch_size = 2   # <<<--- Reduziert für 1024x1024 Patches (ggf. anpassen)

# --- Lernraten und Regularisierung (aus funktionierender Vaihingen-Konfig übernommen) ---
lr = 6e-4
weight_decay = 4e-4
backbone_lr = 6e-5
backbone_weight_decay = 4e-4
afr_lr = 4e-4
bam_lr = 3e-4
lfpm_lr = 8e-4

# --- Modellkonfiguration (aus funktionierender Vaihingen-Konfig übernommen) ---
PATCH_SIZE_VSSM = 4; IN_CHANS = 3; DEPTHS = [2, 2, 9, 2]; EMBED_DIM = 96
SSM_D_STATE = 16; SSM_RATIO = 2.0; SSM_RANK_RATIO = 2.0; SSM_DT_RANK = "auto"
SSM_ACT_LAYER = "silu"; SSM_CONV = 3; SSM_CONV_BIAS = True; SSM_DROP_RATE = 0.0
SSM_INIT = "v0"; SSM_FORWARDTYPE = "v4"; MLP_RATIO = 4.0; MLP_ACT_LAYER = "gelu"
MLP_DROP_RATE = 0.0; DROP_PATH_RATE = 0.1; PATCH_NORM = True; NORM_LAYER = "ln"
DOWNSAMPLE = "v2"; PATCHEMBED = "v2"; GMLP = False
USE_CHECKPOINT = True # Beibehaltung von Checkpointing
REST_EMBED_DIM = 64
backbone_path = 'pretrain_weights/rest_lite.pth' # Pfad zu den Backbone-Gewichten

# === Modifiziertes Modell instanziieren ===
net = UNetMamba_AFR_BAM_LFPM(
    # --- Kernparameter ---
    num_classes=num_classes, # Korrekt für Potsdam (6)
    input_channels=IN_CHANS,
    embed_dim=REST_EMBED_DIM,
    afr_reduction_ratio=16,
    bam_mid_channels=32,
    lfpm_compression_ratio=8, # Optimierter Wert
    lfpm_dilations=[1, 6, 12], # Optimierter Wert
    backbone_path=backbone_path,
    # --- Andere Parameter (wie in Vaihingen-Konfig) ---
    decoder_depths=[2, 2, 2], drop_path_rate=DROP_PATH_RATE, d_state=SSM_D_STATE,
    patch_size=PATCH_SIZE_VSSM, depths=DEPTHS, dims=EMBED_DIM, ssm_d_state=SSM_D_STATE,
    ssm_ratio=SSM_RATIO, ssm_rank_ratio=SSM_RANK_RATIO, ssm_dt_rank=SSM_DT_RANK,
    ssm_act_layer=SSM_ACT_LAYER, ssm_conv=SSM_CONV, ssm_conv_bias=SSM_CONV_BIAS,
    ssm_drop_rate=SSM_DROP_RATE, ssm_init=SSM_INIT, forward_type=SSM_FORWARDTYPE,
    mlp_ratio=MLP_RATIO, mlp_act_layer=MLP_ACT_LAYER, mlp_drop_rate=MLP_DROP_RATE,
    patch_norm=PATCH_NORM, norm_layer=NORM_LAYER, downsample_version=DOWNSAMPLE,
    patchembed_version=PATCHEMBED, gmlp=GMLP, use_checkpoint=USE_CHECKPOINT
)

# --- Verlustfunktion definieren ---
loss = UnetMambaLoss(ignore_index=ignore_index)
use_aux_loss = True # Annahme, dass das modifizierte Modell Aux-Loss verwendet

# --- Konfiguration für Gewichte, Protokolle, Überwachung ---
model_arch_name = "unetmamba_modified_v1" # Name des modifizierten Modells
dataset_name = "potsdam_patches" # <<<--- Geändert für Potsdam Patches
weights_name = f"{model_arch_name}_{dataset_name}_{PATCH_SIZE}-e{max_epoch}" # <<<--- Patch-Größe im Namen
weights_dir = f"model_weights/{dataset_name}/{model_arch_name}_{PATCH_SIZE}" # <<<--- Verzeichnis angepasst
weights_path = weights_dir # Hauptverzeichnis für Checkpoints
log_name = f'{dataset_name}/{weights_name}' # <<<--- Protokollname angepasst

monitor = 'val_mIoU'; monitor_mode = 'max'; save_top_k = 1; save_last = True
check_val_every_n_epoch = 1 # Validierungshäufigkeit
pretrained_ckpt_path = None # Training von Grund auf für Potsdam Patches
resume_ckpt_path = None # Pfad zum Fortsetzen des Trainings (falls vorhanden)
gpus = 'auto' # GPU-Konfiguration

# === 4. Datensätze instanziieren (Potsdam Patches) ===
try:
    print(f"Initialisiere Potsdam Patch-Datensätze...")
    # Trainingsdatensatz (verwendet interne Augmentationen)
    train_dataset = PotsdamPatchesDataset(data_root=POTSDAM_TRAIN_PATCHES_ROOT, mode='train')
    # Validierungsdatensatz (verwendet interne Augmentationen)
    val_dataset = PotsdamPatchesDataset(data_root=POTSDAM_VAL_PATCHES_ROOT, mode='val')

    # --- Hinzugefügt: Definition für test_dataset ---
    # Verwendet hier die Validierungsdaten für den Testlauf.
    # Wenn du separate Testdaten hast, ändere data_root und ggf. mode='test'.
    print(f"INFO: Definiere test_dataset unter Verwendung von Validierungsdaten aus: {POTSDAM_VAL_PATCHES_ROOT}")
    test_dataset = PotsdamPatchesDataset(data_root=POTSDAM_VAL_PATCHES_ROOT, mode='val')
    # ---------------------------------------------

    print(f"Trainingsdatensatzgröße: {len(train_dataset)}, Validierungsdatensatzgröße: {len(val_dataset)}, Testdatensatzgröße: {len(test_dataset)}")
    # Kritischer Fehler, wenn Datensätze leer sind
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print("KRITISCHER FEHLER: Mindestens ein Datensatz ist leer. Überprüfen Sie die *_PATCHES_ROOT Pfade.")
        exit()
except FileNotFoundError as e:
    print(f"KRITISCHER FEHLER: Datensatzverzeichnis nicht gefunden: {e}. Überprüfen Sie POTSDAM_*_PATCHES_ROOT Pfade.")
    exit()
except NameError as e:
    print(f"KRITISCHER FEHLER: Name nicht definiert: {e}.")
    exit()
except Exception as e:
    print(f"Fehler bei der Datensatzinitialisierung: {e}")
    exit()

# --- Datenlader definieren ---
pin_memory = True # Für schnelleren Datentransfer zur GPU
num_workers = 4 # Anzahl der Worker-Prozesse (an System anpassen)
train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, num_workers=num_workers,
                          pin_memory=pin_memory, shuffle=True, drop_last=True,
                          persistent_workers=True if num_workers > 0 else False) # persistent_workers für Effizienz
val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=pin_memory, drop_last=False,
                        persistent_workers=True if num_workers > 0 else False)
# Optional: Test-Loader definieren, falls benötigt (wird normalerweise vom Test-Skript erstellt)
# test_loader = DataLoader(dataset=test_dataset, batch_size=val_batch_size, num_workers=num_workers, ...)

# --- Optimierer und Lernraten-Scheduler (aus funktionierender Vaihingen-Konfig übernommen) ---
print("Konfiguriere Optimierer mit differenzierten Lernraten...")
try:
    # Parametergruppen definieren (identisch zu Vaihingen-Konfig)
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
    learning_rates = [] # Liste für OneCycleLR max_lr
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

    # Basis-Optimierer (AdamW) und Lookahead-Wrapper
    base_optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr) # Basis-LR hier ist nur Platzhalter
    # --- Optional: Lookahead entfernen, falls es Probleme macht ---
    # optimizer = base_optimizer
    optimizer = Lookahead(base_optimizer)
    # ---------------------------------------------------------

    # Lernraten-Scheduler (OneCycleLR mit Warmup)
    warmup_epochs = 10
    # Berechnet steps_per_epoch basierend auf dem Potsdam train_loader
    steps_per_epoch = len(train_loader) if len(train_loader) > 0 else 1 # Verhindert Division durch Null
    total_steps = steps_per_epoch * max_epoch

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rates, # Übergibt Liste der max. Lernraten pro Gruppe
        total_steps=total_steps,
        pct_start=warmup_epochs / max_epoch if max_epoch > 0 else 0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )
    # Konfigurationsdictionary für Lightning (Scheduler + Update-Intervall)
    lr_scheduler_config = {'scheduler': lr_scheduler, 'interval': 'step'}
    print("Optimierer und LR Scheduler (OneCycleLR mit Warmup) konfiguriert.")

except Exception as e:
    print(f"FEHLER beim Einrichten des Optimierers/Schedulers: {e}. Fallback auf einfachen AdamW + Cosine.")
    optimizer = Lookahead(torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
    lr_scheduler_config = {'scheduler': lr_scheduler, 'interval': 'epoch'} # Fallback: Update pro Epoche

# --- Finale Konfigurationsprüfung (Ausgabe) ---
print(f"\n--- Konfigurationsübersicht ---")
print(f"Modell: {model_arch_name}")
print(f"Datensatz: {dataset_name.capitalize()}")
print(f"Patch Größe: {PATCH_SIZE}")
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

# Wichtig: Stelle sicher, dass train.py das lr_scheduler_config Dictionary korrekt verwendet.
