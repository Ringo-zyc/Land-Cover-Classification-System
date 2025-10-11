# filename: config/loveda/unetmamba_modified_v1.py
# Konfigurationsdatei für das modifizierte UNetMamba-Modell auf dem LoveDA-Dataset
# Version 2: Korrigiert den Import von IGNORE_INDEX

import torch
import os
from torch.utils.data import DataLoader
from catalyst.contrib.nn import Lookahead # Lookahead-Optimierer-Wrapper
from catalyst import utils
import copy # copy wird hier nicht direkt verwendet, aber Import bleibt bestehen

# === 1. LoveDA Dataset Klassen und Funktionen importieren ===
try:
    # Annahme: Diese sind in der angegebenen Datei definiert
    from unetmamba_model.datasets.loveda_dataset import (
        LoveDATrainDataset,
        loveda_val_dataset, # Importiert das bereits instanziierte Val-Dataset
        LoveDATestDataset,
        CLASSES,
        PALETTE # PALETTE wird hier nicht direkt verwendet, aber Import bleibt bestehen
        # --- ENTFERNT: IGNORE_INDEX wird nicht mehr importiert ---
        # IGNORE_INDEX
    )
except ImportError as e:
    print(f"KRITISCHER FEHLER: Konnte nicht aus loveda_dataset importieren: {e}")
    exit()
except NameError as e:
    print(f"KRITISCHER FEHLER: {e}. Überprüfen Sie die Definitionen in loveda_dataset.")
    exit()

# === 2. Modifiziertes Modell und Verlustfunktion importieren ===
try:
    # === Modifiziertes Modell importieren ===
    from unetmamba_model.models.UNetMamba_AFR_BAM_LFPM import UNetMamba_AFR_BAM_LFPM # <<<--- Importiert das modifizierte Modell
    from unetmamba_model.losses.useful_loss import UnetMambaLoss # Annahme: Verlustfunktion bleibt gleich
except ImportError as e:
    print(f"KRITISCHER FEHLER: Konnte Modell oder Verlust nicht importieren: {e}")
    exit()


# === Konfigurationsparameter ===
image_size = 1024 # Bildgröße (relevant für interne Verarbeitung oder Benennung)

# === 3. Datensatzpfade (LoveDA) ===
# !! ****** BITTE DIESE PFADE AN DEINE VERZEICHNISSTRUKTUR ANPASSEN ****** !!
LOVEDA_TRAIN_DATA_ROOT = 'data/LoveDA/Train' # Pfad zum LoveDA Trainingsset
LOVEDA_VAL_DATA_ROOT = 'data/LoveDA/Val'   # Pfad zum LoveDA Validierungsset (wird von loveda_val_dataset verwendet)
# !! ******************************************************************* !!

# --- Trainingshyperparameter ---
max_epoch = 100 # Maximale Anzahl von Epochen
num_classes = len(CLASSES) # Sollte 7 für LoveDA sein
# --- Hinzugefügt: ignore_index direkt definieren ---
ignore_index = len(CLASSES) # Setzt ignore_index auf 7 (konsistent mit alter Konfig)
print(f"INFO: ignore_index wurde auf {ignore_index} gesetzt (basierend auf len(CLASSES)).")
# -------------------------------------------------
classes = CLASSES
# --- Batch-Größe aus alter LoveDA-Konfig ---
train_batch_size = 8 # <<<--- Angepasst für LoveDA
val_batch_size = 8   # <<<--- Angepasst für LoveDA

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
    num_classes=num_classes, # Korrekt für LoveDA (7)
    input_channels=IN_CHANS,
    embed_dim=REST_EMBED_DIM,
    afr_reduction_ratio=16,
    bam_mid_channels=32,
    lfpm_compression_ratio=8, # Optimierter Wert
    lfpm_dilations=[1, 6, 12], # Optimierter Wert
    backbone_path=backbone_path,
    # --- Andere Parameter (wie in Vaihingen/Potsdam-Konfig) ---
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
# Stellt sicher, dass der oben definierte ignore_index verwendet wird
loss = UnetMambaLoss(ignore_index=ignore_index)
use_aux_loss = True # Annahme, dass das modifizierte Modell Aux-Loss verwendet

# --- Konfiguration für Gewichte, Protokolle, Überwachung ---
model_arch_name = "unetmamba_modified_v1" # Name des modifizierten Modells
dataset_name = "loveda" # <<<--- Geändert für LoveDA
weights_name = f"{model_arch_name}_{dataset_name}_{image_size}-e{max_epoch}" # <<<--- Angepasster Name
weights_dir = f"model_weights/{dataset_name}/{model_arch_name}_{image_size}" # <<<--- Verzeichnis angepasst
weights_path = weights_dir # Hauptverzeichnis für Checkpoints
log_name = f'{dataset_name}/{weights_name}' # <<<--- Protokollname angepasst

monitor = 'val_mIoU'; monitor_mode = 'max'; save_top_k = 1; save_last = True
check_val_every_n_epoch = 1 # Validierungshäufigkeit
pretrained_ckpt_path = None # <<<--- Kein vortrainierter Checkpoint für dieses Modell auf LoveDA
resume_ckpt_path = None # Pfad zum Fortsetzen des Trainings (falls vorhanden)
gpus = 'auto' # GPU-Konfiguration

# === 4. Datensätze instanziieren (LoveDA) ===
# Augmentationsfunktionen werden jetzt im Dataset-Modul erwartet

try:
    print(f"Initialisiere LoveDA Datensätze...")
    # Trainingsdatensatz (verwendet interne Augmentationen aus loveda_dataset.py)
    train_dataset = LoveDATrainDataset(
        data_root=LOVEDA_TRAIN_DATA_ROOT,
        # transform=train_aug, # transform wird jetzt intern gehandhabt
        # mosaic_ratio=0.25 # Wird jetzt intern gehandhabt (oder Standardwert verwenden)
    )
    # Validierungsdatensatz (direkt importiert)
    val_dataset = loveda_val_dataset
    # Testdatensatz (falls benötigt)
    # Stellt sicher, dass LoveDATestDataset existiert und korrekt importiert wurde
    if 'LoveDATestDataset' in locals():
         test_dataset = LoveDATestDataset() # Annahme: Keine Argumente benötigt oder Standardwerte ok
         print(f"Testdatensatzgröße: {len(test_dataset)}")
    else:
         print("WARNUNG: LoveDATestDataset nicht gefunden oder importiert. test_dataset wird nicht definiert.")
         test_dataset = None # Definiert test_dataset als None

    print(f"Trainingsdatensatzgröße: {len(train_dataset)}, Validierungsdatensatzgröße: {len(val_dataset)}")
    # Kritischer Fehler, wenn Datensätze leer sind
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("KRITISCHER FEHLER: Trainings- oder Validierungsdatensatz ist leer. Überprüfen Sie die LOVEDA_*_DATA_ROOT Pfade.")
        exit()
except FileNotFoundError as e:
    print(f"KRITISCHER FEHLER: Datensatzverzeichnis nicht gefunden: {e}. Überprüfen Sie LOVEDA_*_DATA_ROOT Pfade.")
    exit()
except NameError as e:
    print(f"KRITISCHER FEHLER: Name nicht definiert (z.B. loveda_val_dataset): {e}.")
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

# --- Optimierer und Lernraten-Scheduler (aus funktionierender Vaihingen/Potsdam-Konfig übernommen) ---
print("Konfiguriere Optimierer mit differenzierten Lernraten...")
try:
    # Parametergruppen definieren (identisch zu Vaihingen/Potsdam-Konfig)
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
    optimizer = Lookahead(base_optimizer) # Behält Lookahead bei (kann bei Bedarf entfernt werden)

    # Lernraten-Scheduler (OneCycleLR mit Warmup)
    warmup_epochs = 10
    # Berechnet steps_per_epoch basierend auf dem LoveDA train_loader
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
print(f"Bildgröße (Referenz): {image_size}")
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
