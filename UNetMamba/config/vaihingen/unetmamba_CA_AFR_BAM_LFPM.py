# filename: config/vaihingen/unetmamba_modified_v1.py
# Konfigurationsdatei für das modifizierte UNetMamba-Modell auf dem Vaihingen-Dataset

import torch
import os
from torch.utils.data import DataLoader
from catalyst.contrib.nn import Lookahead # Lookahead-Optimierer-Wrapper
from catalyst import utils

# === 1. Vaihingen-Datensatzklassen und -funktionen importieren ===
try:
    # Annahme: Diese sind in der angegebenen Datei definiert
    from unetmamba_model.datasets.vaihingen_dataset import (
        VaihingenDataset, CLASSES, IGNORE_INDEX, train_aug, val_aug
    )
except ImportError as e:
    print(f"KRITISCHER FEHLER: Konnte nicht aus vaihingen_dataset importieren: {e}")
    exit()
except NameError as e:
    print(f"KRITISCHER FEHLER: {e}. Überprüfen Sie die Definitionen in vaihingen_dataset.")
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
image_size = 1024 # Bildgröße (wird im Datensatz verwendet)
# crop_size = 512 # Wird jetzt innerhalb von train_aug im Datensatzmodul gehandhabt

# === 3. Datensatzpfade ===
# Stellen Sie sicher, dass diese Pfade korrekt sind!
VAIHINGEN_TRAIN_DATA_ROOT = f'data/vaihingen/train_{image_size}'
VAIHINGEN_VAL_DATA_ROOT = f'data/vaihingen/val_{image_size}'

# --- Trainingshyperparameter ---
max_epoch = 100 # Maximale Anzahl von Epochen
num_classes = len(CLASSES) # Sollte 6 für Vaihingen sein
ignore_index = IGNORE_INDEX # Sollte 255 für Vaihingen sein
classes = CLASSES
train_batch_size = 4 # Beibehalten der kleineren Batch-Größe (ggf. anpassen oder Gradientenakkumulation verwenden)
val_batch_size = 1 # Validierungs-Batch-Größe oft 1
# --- Angepasste Lernraten (basierend auf Referenzmeinung) ---
lr = 6e-4 # Basis-Lernrate für 'andere' Parameter
weight_decay = 4e-4 # <<<--- Erhöhte Gewichtungsabnahme (Regularisierung)
backbone_lr = 6e-5 # Lernrate für den Backbone
backbone_weight_decay = 4e-4 # <<<--- Erhöhte Gewichtungsabnahme
afr_lr = 4e-4 # Lernrate für AFR-Module
bam_lr = 3e-4 # Lernrate für BAM-Module
lfpm_lr = 8e-4 # Lernrate für LFPM-Module
# Beachten Sie: Die Gewichtungsabnahme für AFR/BAM/LFPM verwendet jetzt den erhöhten Basiswert 'weight_decay'

# --- Modellkonfiguration ---
# VSSM-Parameter (falls vom Modell intern benötigt)
PATCH_SIZE_VSSM = 4; IN_CHANS = 3; DEPTHS = [2, 2, 9, 2]; EMBED_DIM = 96
SSM_D_STATE = 16; SSM_RATIO = 2.0; SSM_RANK_RATIO = 2.0; SSM_DT_RANK = "auto"
SSM_ACT_LAYER = "silu"; SSM_CONV = 3; SSM_CONV_BIAS = True; SSM_DROP_RATE = 0.0
SSM_INIT = "v0"; SSM_FORWARDTYPE = "v4"; MLP_RATIO = 4.0; MLP_ACT_LAYER = "gelu"
MLP_DROP_RATE = 0.0; DROP_PATH_RATE = 0.1; PATCH_NORM = True; NORM_LAYER = "ln"
DOWNSAMPLE = "v2"; PATCHEMBED = "v2"; GMLP = False
USE_CHECKPOINT = True # <<<--- Aktiviert für das komplexere Modell

# ResT Backbone Basisdimension
REST_EMBED_DIM = 64

# === Modifiziertes Modell instanziieren ===
net = UNetMamba_AFR_BAM_LFPM(
    # --- Kernparameter ---
    num_classes=num_classes, input_channels=IN_CHANS, embed_dim=REST_EMBED_DIM,
    afr_reduction_ratio=16, # CA-Parameter entfernt
    bam_mid_channels=32, # BAM-Mittelkanäle (kann angepasst werden)
    lfpm_compression_ratio=8, # <<<--- Optimierter LFPM-Standardwert
    lfpm_dilations=[1, 6, 12], # <<<--- Optimierter LFPM-Standardwert

    # --- Backbone-Pfad ---
    backbone_path='pretrain_weights/rest_lite.pth', # Sicherstellen, dass dieser Pfad korrekt ist

    # --- Andere Parameter (falls benötigt durchreichen) ---
    decoder_depths=[2, 2, 2], drop_path_rate=DROP_PATH_RATE, d_state=SSM_D_STATE,
    # VSSM-Parameter werden an __init__ übergeben und sollten intern an VSSLayer weitergegeben werden
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
use_aux_loss = True # Der modifizierte Decoder gibt immer noch einen Aux-Verlust zurück

# --- Konfiguration für Gewichte, Protokolle, Überwachung ---
model_arch_name = "unetmamba_modified_v1" # <<<--- Aktualisierter Modellarchitekturname
dataset_name = "vaihingen"
weights_name = f"{model_arch_name}_{dataset_name}_{image_size}-e{max_epoch}"
weights_dir = f"model_weights/{dataset_name}/{model_arch_name}_{image_size}" # <<<--- Aktualisiertes Verzeichnis
weights_path = weights_dir # Hauptverzeichnis für Checkpoints
log_name = f'{dataset_name}/{weights_name}' # <<<--- Aktualisierter Protokollname

monitor = 'val_mIoU'; monitor_mode = 'max'; save_top_k = 1; save_last = True
check_val_every_n_epoch = 1 # Validierungshäufigkeit
pretrained_ckpt_path = None # Kein vortrainierter Checkpoint für das gesamte modifizierte Modell
resume_ckpt_path = None # Pfad zum Fortsetzen des Trainings (falls vorhanden)
gpus = 'auto' # GPU-Konfiguration (automatisch oder spezifische IDs)

# === 4. Datensätze instanziieren ===
IMG_SUFFIX = '.tif'; MASK_SUFFIX = '.png' # Dateiendungen für Vaihingen
try:
    # Sicherstellen, dass image_size ein Tupel ist
    img_size_tuple = (image_size, image_size) if isinstance(image_size, int) else image_size
    print(f"Initialisiere Vaihingen-Datensätze mit Bildgröße: {img_size_tuple}")
    # Trainingsdatensatz
    train_dataset = VaihingenDataset(data_root=VAIHINGEN_TRAIN_DATA_ROOT, mode='train',
                                     img_suffix=IMG_SUFFIX, mask_suffix=MASK_SUFFIX,
                                     transform=train_aug, # Verwendet Augmentation aus vaihingen_dataset.py
                                     mosaic_ratio=0.25, # Mosaik-Augmentationsverhältnis
                                     img_size=img_size_tuple)
    # Validierungsdatensatz
    val_dataset = VaihingenDataset(data_root=VAIHINGEN_VAL_DATA_ROOT, mode='val',
                                   img_suffix=IMG_SUFFIX, mask_suffix=MASK_SUFFIX,
                                   transform=val_aug, # Verwendet Validierungsaugmentation
                                   img_size=img_size_tuple)
    # Testdatensatz (hier identisch mit Validierung, ggf. anpassen)
    test_dataset = VaihingenDataset(data_root=VAIHINGEN_VAL_DATA_ROOT, mode='val',
                                    img_suffix=IMG_SUFFIX, mask_suffix=MASK_SUFFIX,
                                    transform=val_aug,
                                    img_size=img_size_tuple)
    print(f"Trainingsdatensatzgröße: {len(train_dataset)}, Validierungsdatensatzgröße: {len(val_dataset)}")
    # Kritischer Fehler, wenn Datensätze leer sind
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("KRITISCHER FEHLER: Datensätze sind leer.")
        exit()
except FileNotFoundError as e:
    print(f"KRITISCHER FEHLER: Datensatzverzeichnis nicht gefunden: {e}. Überprüfen Sie VAIHINGEN_*_DATA_ROOT Pfade.")
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
                          pin_memory=pin_memory, shuffle=True, drop_last=True, # drop_last=True für konsistente Batch-Größen
                          persistent_workers=True if num_workers > 0 else False) # persistent_workers für schnellere Starts
val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=pin_memory, drop_last=False,
                        persistent_workers=True if num_workers > 0 else False)

# --- Optimierer und Lernraten-Scheduler ---
print("Konfiguriere Optimierer mit differenzierten Lernraten...")
try:
    # Parametergruppen definieren
    backbone_params = [p for n, p in net.named_parameters() if n.startswith('encoder.') and p.requires_grad]
    afr_params = [p for n, p in net.named_parameters() if n.startswith('decoder.afr_modules.') and p.requires_grad]
    # BAM-Parameter (ohne den fixierten edge_conv.weight)
    bam_params = [p for n, p in net.named_parameters() if n.startswith('decoder.bam.') and p.requires_grad and 'edge_conv.weight' not in n]
    lfpm_params = [p for n, p in net.named_parameters() if n.startswith('decoder.lfpm.') and p.requires_grad]
    # Fusion-Conv-Parameter (gehört jetzt zum Decoder)
    fusion_conv_params = [p for n, p in net.named_parameters() if n.startswith('decoder.fusion_conv.') and p.requires_grad]

    # IDs aller trainierbaren Parameter sammeln
    base_param_ids = set(id(p) for p in net.parameters() if p.requires_grad)
    # IDs der bereits zugeordneten Parameter sammeln
    backbone_param_ids = set(id(p) for p in backbone_params)
    afr_param_ids = set(id(p) for p in afr_params)
    bam_param_ids = set(id(p) for p in bam_params)
    lfpm_param_ids = set(id(p) for p in lfpm_params)
    fusion_conv_param_ids = set(id(p) for p in fusion_conv_params)

    # Restliche Parameter ('andere') identifizieren
    other_params = [p for p in net.parameters() if id(p) in base_param_ids - backbone_param_ids - afr_param_ids - bam_param_ids - lfpm_param_ids - fusion_conv_param_ids]

    print(f"Gefunden: {len(backbone_params)} Backbone, {len(afr_params)} AFR, {len(bam_params)} BAM, {len(lfpm_params)} LFPM, {len(fusion_conv_params)} FusionConv, {len(other_params)} andere trainierbare Parameter.")

    # Parametergruppen für den Optimierer zusammenstellen
    optimizer_grouped_parameters = []
    # Lernraten für jede Gruppe
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
    # Fusion-Conv und andere Parameter verwenden die Basis-Lernrate
    other_and_fusion_params = other_params + fusion_conv_params
    if other_and_fusion_params:
        optimizer_grouped_parameters.append({'params': other_and_fusion_params, 'lr': lr, 'weight_decay': weight_decay})
        learning_rates.append(lr) # Fügt die Basis-LR für diese Gruppe hinzu

    if not optimizer_grouped_parameters: # Fallback, falls keine Parameter gefunden wurden
        print("WARNUNG: Keine spezifischen Parametergruppen gefunden, verwende Standardoptimierer.")
        optimizer_grouped_parameters = net.parameters()
        learning_rates = [lr] # Nur eine Lernrate

    # Basis-Optimierer (AdamW)
    base_optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr) # lr hier ist nur ein Standardwert, die Gruppenspezifischen werden verwendet
    # Lookahead-Wrapper
    optimizer = Lookahead(base_optimizer)

    # --- Lernraten-Scheduler (OneCycleLR mit Warmup) ---
    warmup_epochs = 10 # Anzahl der Aufwärmepochen
    steps_per_epoch = len(train_loader) # Schritte pro Epoche
    total_steps = steps_per_epoch * max_epoch # Gesamte Trainingsschritte

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rates, # Liste der maximalen Lernraten für jede Gruppe
        total_steps=total_steps,
        pct_start=warmup_epochs / max_epoch if max_epoch > 0 else 0.1, # Prozentsatz der Schritte für die Aufwärmphase
        anneal_strategy='cos', # Kosinus-Abklingstrategie
        div_factor=25.0,       # Faktor zur Bestimmung der anfänglichen LR (max_lr / div_factor)
        final_div_factor=1e4   # Faktor zur Bestimmung der minimalen LR (initial_lr / final_div_factor)
    )
    # Wichtig: OneCycleLR sollte pro Schritt aktualisiert werden, nicht pro Epoche.
    # Dies wird normalerweise im LightningModule `lr_scheduler_step` oder durch Setzen von `interval='step'` im `configure_optimizers` Rückgabewert gehandhabt.
    lr_scheduler_config = {'scheduler': lr_scheduler, 'interval': 'step'}


    print("Optimierer und LR Scheduler (OneCycleLR mit Warmup) konfiguriert.")
except Exception as e:
    print(f"FEHLER beim Einrichten des Optimierers/Schedulers: {e}. Fallback auf einfachen AdamW + Cosine.")
    optimizer = Lookahead(torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay))
    # Fallback Scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
    lr_scheduler_config = {'scheduler': lr_scheduler, 'interval': 'epoch'} # Cosine wird pro Epoche aktualisiert

# --- Finale Konfigurationsprüfung (Ausgabe) ---
print(f"\n--- Konfigurationsübersicht ---")
print(f"Modell: {model_arch_name}")
print(f"Datensatz: {dataset_name.capitalize()}")
print(f"Bildgröße: {image_size}")
print(f"Epochen: {max_epoch}")
print(f"Batch-Größe (Train/Val): {train_batch_size}/{val_batch_size}")
# Gibt die *maximalen* Lernraten für OneCycleLR aus
print(f"Max Lernraten (Basis/Backbone/AFR/BAM/LFPM): {lr}/{backbone_lr}/{afr_lr}/{bam_lr}/{lfpm_lr}")
print(f"Scheduler: OneCycleLR (Warmup: {warmup_epochs} Epochen)")
print(f"Gewichtsabnahme (Basis/Backbone): {weight_decay}/{backbone_weight_decay}")
print(f"Speicherort für Gewichte: {weights_dir}")
print(f"Protokollname: {log_name}")
print(f"Backbone vortrainierte Gewichte: {backbone_path}") # Gibt den Pfad aus
print(f"Fortsetzungs-Checkpoint: {resume_ckpt_path}")
print(f"Gradient Checkpoint aktiviert: {USE_CHECKPOINT}")
print(f"---------------------------\n")

# Wichtig: Der Rückgabewert von configure_optimizers im LightningModule muss angepasst werden,
# um das lr_scheduler_config Dictionary zurückzugeben, damit der Scheduler pro Schritt aktualisiert wird.
# Beispiel für configure_optimizers in Supervision_Train:
# def configure_optimizers(self):
#     optimizer = self.config.optimizer
#     scheduler_config = self.config.lr_scheduler_config # Holt das Dictionary
#     return [optimizer], [scheduler_config]
```

**Wichtige Hinweise:**

1.  **`configure_optimizers` in `train.py`:** Wie im Codekommentar am Ende erwähnt, müssen Sie die Methode `configure_optimizers` in Ihrer `Supervision_Train`-Klasse (in `train.py`) anpassen, damit sie das `lr_scheduler_config`-Dictionary zurückgibt. Nur so weiß PyTorch Lightning, dass der `OneCycleLR`-Scheduler pro **Schritt** (`step`) und nicht pro Epoche (`epoch`) aktualisiert werden soll.
    ```python
    # In Ihrer Supervision_Train Klasse in train.py:
    def configure_optimizers(self):
        optimizer = self.config.optimizer
        # Holen Sie das Scheduler-Konfigurationsdictionary aus der Config
        scheduler_config = self.config.lr_scheduler_config
        return [optimizer], [scheduler_config]
    ```
2.  **`lr_scheduler_config`:** Ich habe das Dictionary `lr_scheduler_config` in der Konfigurationsdatei erstellt. Stellen Sie sicher, dass Ihre `Supervision_Train`-Klasse darauf zugreift (z. B. über `self.config.lr_scheduler_config`).
3.  **Parametergruppen:** Ich habe die Definition der Parametergruppen leicht angepasst, um die neue `fusion_conv`-Schicht explizit den 'anderen' Parametern zuzuordnen, die die Basis-Lernrate verwenden.
4.  **Testen:** Führen Sie diese Konfiguration aus und beobachten Sie die Trainingskurven (Verlust, Metriken wie mIoU) und die Lernrate genau, um zu sehen, ob die neuen Einstellungen und das modifizierte Modell zu besseren Ergebnissen führen. Möglicherweise sind weitere Anpassungen der Lernraten oder anderer Hyperparameter erforderli