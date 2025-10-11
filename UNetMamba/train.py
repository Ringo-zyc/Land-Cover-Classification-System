# train.py (Vollständig, mit Metrik-Fix und Verbesserungen)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping # EarlyStopping importieren, falls verwendet
from tools.cfg import py2cfg # Annahme: Werkzeug zum Laden der Konfiguration
import os
import torch
from torch import nn
import cv2 # OpenCV wird hier nicht direkt verwendet, aber Import bleibt bestehen
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator # Importiert die (hoffentlich reparierte) Evaluator-Klasse
from pytorch_lightning.loggers import CSVLogger
import random
# import pdb; pdb.set_trace() # Debugging-Zeile (auskommentiert)

# Optional: Setzt den Hugging Face Endpunkt, falls benötigt
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def seed_everything(seed):
    """Setzt den Seed für Reproduzierbarkeit."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # Wichtig für Multi-GPU
    # Diese Einstellungen können die Reproduzierbarkeit verbessern, aber die Leistung beeinträchtigen
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False # Benchmark auf False setzen für Deterministik

def get_args():
    """Parst Kommandozeilenargumente."""
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Pfad zur Konfigurationsdatei.", required=True)
    return parser.parse_args()

class Supervision_Train(pl.LightningModule):
    """PyTorch Lightning Modul für das Training."""
    def __init__(self, config):
        super().__init__()
        # self.save_hyperparameters(config) # Optional: Hyperparameter im Checkpoint speichern
        self.config = config
        self.net = config.net # Das eigentliche Modell
        self.loss = config.loss # Die Verlustfunktion
        # Metrik-Evaluatoren für Training und Validierung initialisieren
        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

    def forward(self, x):
        """Definiert den Forward-Pass (hauptsächlich für Inferenz)."""
        # Nur das Netzwerk wird für Vorhersagen/Inferenz verwendet
        seg_pre = self.net(x)
        return seg_pre

    def training_step(self, batch, batch_idx):
        """Führt einen einzelnen Trainingsschritt durch."""
        img, mask = batch['img'], batch['gt_semantic_seg'] # Holt Bilder und Masken aus dem Batch

        prediction = self.net(img) # Erhält Modellvorhersagen (kann Tupel sein wg. Aux-Loss)
        loss = self.loss(prediction, mask) # Berechnet den Verlust

        # Loggt den Trainingsverlust pro Schritt und pro Epoche
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # --- Beginn: Trainingsmetriken berechnen und akkumulieren (jetzt aktiviert) ---
        # Ermittelt die Hauptvorhersage (falls Aux-Loss verwendet wird)
        if self.config.use_aux_loss and isinstance(prediction, (list, tuple)):
            main_prediction = prediction[0]
        else:
            main_prediction = prediction

        # Konvertiert die Ausgabe in eine Klassenmaske
        pre_mask = nn.Softmax(dim=1)(main_prediction) # Softmax über Klassen-Dimension
        pre_mask = pre_mask.argmax(dim=1) # Argmax zur Bestimmung der Klasse

        # Akkumuliert Metriken für jeden Batch-Eintrag
        # Wichtig: Konvertiert Tensoren zu CPU NumPy Arrays für den Evaluator
        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())
        # --- Ende: Trainingsmetriken berechnen und akkumulieren ---

        # Gibt das Verlust-Dictionary zurück (erforderlich für Lightning)
        return {"loss": loss}

    def on_train_epoch_end(self):
        """Wird am Ende jeder Trainingsepoche ausgeführt."""
        # Berechnet Metriken basierend auf den akkumulierten Werten in self.metrics_train
        try:
            # Spezifische Behandlung für Vaihingen/Potsdam (ignoriert letzte Klasse 'Clutter')
            is_vai_pots = 'vaihingen' in self.config.log_name or 'potsdam' in self.config.log_name
            iou_all = self.metrics_train.Intersection_over_Union()
            f1_all = self.metrics_train.F1()

            # Berechnet mIoU und F1 (ggf. ohne letzte Klasse)
            mIoU = np.nanmean(iou_all[:-1]) if is_vai_pots and len(iou_all)>1 else np.nanmean(iou_all)
            F1 = np.nanmean(f1_all[:-1]) if is_vai_pots and len(f1_all)>1 else np.nanmean(f1_all)
            OA = self.metrics_train.OA() # OA wird normalerweise über alle Klassen berechnet

            # Stellt sicher, dass NaN durch 0 ersetzt wird, falls np.nanmean NaN zurückgibt
            mIoU = 0.0 if np.isnan(mIoU) else mIoU
            F1 = 0.0 if np.isnan(F1) else F1
            OA = 0.0 if np.isnan(OA) else OA

            eval_value = {'mIoU': mIoU*100.0, 'F1': F1*100.0, 'OA': OA*100.0}
            print('\nTrain Epoch End:', eval_value) # Gibt die durchschnittlichen Metriken aus

            # Gibt IoU für jede Klasse aus
            iou_value = {}
            for i, class_name in enumerate(self.config.classes):
                iou_val = iou_all[i] * 100.0 if i < len(iou_all) else 0.0
                iou_value[f'IoU_{class_name}'] = 0.0 if np.isnan(iou_val) else iou_val # Ersetzt NaN durch 0.0
            print(iou_value)

            # Loggt die durchschnittlichen Metriken für TensorBoard/CSV etc.
            log_dict = {'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}
            self.log_dict(log_dict, prog_bar=True)

        except Exception as e:
            print(f"Fehler bei der Berechnung der Trainingsmetriken am Epochenende: {e}")
        finally:
            # WICHTIG: Setzt den Trainingsmetrik-Evaluator für die nächste Epoche zurück
            self.metrics_train.reset()

    def validation_step(self, batch, batch_idx):
        """Führt einen einzelnen Validierungsschritt durch."""
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.forward(img) # self.forward ruft self.net(img) auf
        loss_val = self.loss(prediction, mask) # Berechnet Validierungsverlust

        # Loggt den Validierungsverlust pro Epoche
        self.log('val_loss', loss_val, on_step=False, on_epoch=True, prog_bar=True)

        # Berechnet Vorhersagemaske für Metriken
        if self.config.use_aux_loss and isinstance(prediction, (list, tuple)):
            main_prediction = prediction[0]
        else:
            main_prediction = prediction

        pre_mask = nn.Softmax(dim=1)(main_prediction)
        pre_mask = pre_mask.argmax(dim=1)

        # Akkumuliert Metriken für die Validierungsepoche
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        # Kein expliziter Return-Wert nötig, da über self.log geloggt wird

    def on_validation_epoch_end(self):
        """Wird am Ende jeder Validierungsepoche ausgeführt."""
        # Berechnet und loggt Validierungsmetriken
        try:
            is_vai_pots = 'vaihingen' in self.config.log_name or 'potsdam' in self.config.log_name
            iou_all = self.metrics_val.Intersection_over_Union()
            f1_all = self.metrics_val.F1()

            mIoU = np.nanmean(iou_all[:-1]) if is_vai_pots and len(iou_all)>1 else np.nanmean(iou_all)
            F1 = np.nanmean(f1_all[:-1]) if is_vai_pots and len(f1_all)>1 else np.nanmean(f1_all)
            OA = self.metrics_val.OA()

            # Stellt sicher, dass NaN durch 0 ersetzt wird
            mIoU = 0.0 if np.isnan(mIoU) else mIoU
            F1 = 0.0 if np.isnan(F1) else F1
            OA = 0.0 if np.isnan(OA) else OA

            eval_value = {'mIoU': mIoU*100.0, 'F1': F1*100.0, 'OA': OA*100.0}
            print('Val Epoch End:', eval_value) # Gibt durchschnittliche Validierungsmetriken aus

            # Gibt IoU für jede Klasse aus
            iou_value = {}
            for i, class_name in enumerate(self.config.classes):
                 iou_val = iou_all[i] * 100.0 if i < len(iou_all) else 0.0
                 iou_value[f'val_IoU_{class_name}'] = 0.0 if np.isnan(iou_val) else iou_val # Präfix 'val_' hinzugefügt
            print(iou_value)

            # Loggt die Hauptmetriken für Checkpointing und Logging
            log_dict = {'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}
            # Wichtig: Sicherstellen, dass der Key für log_dict['val_mIoU'] mit dem `monitor`-Parameter
            # in ModelCheckpoint übereinstimmt (Groß-/Kleinschreibung beachten!)
            self.log_dict(log_dict, prog_bar=True)

        except Exception as e:
            print(f"Fehler bei der Berechnung der Validierungsmetriken am Epochenende: {e}")
        finally:
            # WICHTIG: Setzt den Validierungsmetrik-Evaluator für die nächste Epoche zurück
            self.metrics_val.reset()

    def configure_optimizers(self):
        """Konfiguriert Optimierer und Lernraten-Scheduler."""
        optimizer = self.config.optimizer # Holt den Optimierer aus der Konfiguration

        # Holt die Scheduler-Konfiguration (Dictionary) aus der Konfiguration
        scheduler_config = self.config.lr_scheduler_config

        # Gibt Optimierer und Scheduler-Konfiguration im für Lightning korrekten Format zurück
        return [optimizer], [scheduler_config]

    def train_dataloader(self):
        """Gibt den Trainings-DataLoader zurück."""
        return self.config.train_loader

    def val_dataloader(self):
        """Gibt den Validierungs-DataLoader zurück."""
        return self.config.val_loader

# --- main Funktion ---
def main():
    """Hauptfunktion zum Starten des Trainings."""
    args = get_args() # Parst Kommandozeilenargumente
    config = py2cfg(args.config_path) # Lädt die Konfiguration aus der Datei
    seed_everything(42) # Setzt den globalen Seed

    # --- Hinzugefügt: Tensor Core Precision setzen (optional, aber empfohlen) ---
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium') # 'medium' oder 'high'
        print("INFO: torch.set_float32_matmul_precision('medium') gesetzt.")
    # -------------------------------------------------------------------

    # --- Callbacks konfigurieren ---
    # ModelCheckpoint zum Speichern
    checkpoint_callback = ModelCheckpoint(
        save_top_k=config.save_top_k, # Anzahl der besten Modelle zum Speichern
        monitor=config.monitor, # Zu überwachende Metrik (z.B. 'val_mIoU')
        save_last=config.save_last, # Letztes Modell speichern?
        mode=config.monitor_mode, # 'max' oder 'min' basierend auf der Metrik
        dirpath=config.weights_path, # Speicherverzeichnis
        filename=config.weights_name + "-{epoch}-{" + config.monitor + ":.4f}" # Dateiname für Checkpoints
    )

    # Liste der Callbacks initialisieren
    callbacks_list = [checkpoint_callback]

    # Optional: EarlyStopping hinzufügen (wenn in Konfig definiert)
    # if hasattr(config, 'early_stopping_patience') and config.early_stopping_patience > 0:
    #      early_stopping_callback = EarlyStopping(
    #          monitor=config.monitor, # Dieselbe Metrik überwachen
    #          patience=config.early_stopping_patience, # Anzahl Epochen ohne Verbesserung
    #          verbose=True,
    #          mode=config.monitor_mode
    #      )
    #      callbacks_list.append(early_stopping_callback)
    #      print(f"INFO: EarlyStopping Callback aktiviert mit Geduld={config.early_stopping_patience}.")
    #      # Sie müssten 'early_stopping_patience' in Ihrer Konfigurationsdatei definieren (z.B. early_stopping_patience = 10)

    # Logger konfigurieren (speichert Metriken in CSV-Dateien)
    logger = CSVLogger('lightning_logs', name=config.log_name)

    # Lightning-Modul initialisieren
    model = Supervision_Train(config)

    # Vortrainierten Checkpoint laden, falls Pfad angegeben und vorhanden
    if hasattr(config, 'pretrained_ckpt_path') and config.pretrained_ckpt_path and os.path.exists(config.pretrained_ckpt_path):
        print(f"INFO: Lade vortrainierten Checkpoint von: {config.pretrained_ckpt_path}")
        model = Supervision_Train.load_from_checkpoint(
            config.pretrained_ckpt_path,
            config=config, # Übergibt die aktuelle Konfiguration an __init__
            strict=False   # Erlaubt fehlende/unerwartete Schlüssel im Checkpoint
        )
    elif hasattr(config, 'pretrained_ckpt_path') and config.pretrained_ckpt_path:
        print(f"WARNUNG: Vortrainierter Checkpoint nicht gefunden unter: {config.pretrained_ckpt_path}")

    # PyTorch Lightning Trainer konfigurieren
    trainer = pl.Trainer(
        devices=config.gpus, # Zu verwendende GPUs
        max_epochs=config.max_epoch, # Maximale Trainings-Epochen
        accelerator='auto', # Automatische Wahl: 'gpu', 'cpu', etc.
        check_val_every_n_epoch=config.check_val_every_n_epoch, # Validierungshäufigkeit
        callbacks=callbacks_list, # Liste der Callbacks
        strategy='auto', # Verteilungsstrategie (z.B. 'ddp')
        logger=logger, # Logger für Metriken
        precision='16-mixed' if torch.cuda.is_available() else 32, # Automatische gemischte Präzision (AMP) auf GPU
        # --- Hinzugefügt: Gradient Clipping ---
        gradient_clip_val=1.0 # Wert für Gradient Clipping (kann angepasst werden)
        # ------------------------------------
        # Optional: Gradientenakkumulation
        # accumulate_grad_batches=config.accumulate_grad_batches # Falls in Konfig definiert (z.B. =2 oder 4)
    )

    # Training starten
    print("INFO: Starte Training...")
    trainer.fit(
        model=model,
        ckpt_path=config.resume_ckpt_path # Pfad zum Fortsetzen des Trainings (falls vorhanden)
    )
    print("INFO: Training abgeschlossen.")

    # Optional: Bestes Modell nach dem Training laden und nur Netzwerk-Gewichte speichern
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and os.path.exists(best_model_path):
        print(f"INFO: Lade bestes Modell von: {best_model_path}")
        # Lädt das gesamte LightningModule aus dem besten Checkpoint
        best_lightning_model = Supervision_Train.load_from_checkpoint(best_model_path, config=config)
        # Pfad zum Speichern des reinen Netzwerk-State-Dicts
        final_save_path = os.path.join(config.weights_path, f"{config.weights_name}_final_best_statedict.pth")
        # Speichert nur die Gewichte des eigentlichen Netzwerks (self.net)
        torch.save(best_lightning_model.net.state_dict(), final_save_path)
        print(f"INFO: State dict des besten Netzwerkmodells gespeichert unter: {final_save_path}")
    else:
        # Fallback: Speichere das letzte Modell, wenn kein "bestes" gefunden wurde
        print("WARNUNG: Kein bestes Modell zum Speichern gefunden. Versuche letztes Modell zu speichern.")
        last_model_path = checkpoint_callback.last_model_path
        if last_model_path and os.path.exists(last_model_path):
             last_lightning_model = Supervision_Train.load_from_checkpoint(last_model_path, config=config)
             final_save_path = os.path.join(config.weights_path, f"{config.weights_name}_final_last_statedict.pth")
             torch.save(last_lightning_model.net.state_dict(), final_save_path)
             print(f"INFO: State dict des letzten Netzwerkmodells gespeichert unter: {final_save_path}")
        else:
             print("WARNUNG: Konnte weder bestes noch letztes Modell zum Speichern finden.")

# --- Testteil entfernt, sollte in separatem Skript erfolgen ---

if __name__ == "__main__":
   main() # Führt die main-Funktion aus, wenn das Skript direkt gestartet wird

