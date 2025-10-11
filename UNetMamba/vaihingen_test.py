# vaihingen_test.py (Corrected with --checkpoint and --save-num)
import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import os
import random
import traceback # <-- Import traceback

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Try importing from train.py and cfg.py
try:
    from train import Supervision_Train, Evaluator, get_args as train_get_args # Use alias if needed
    from tools.cfg import py2cfg
except ImportError as e:
    print(f"Error: Could not import from train.py or tools.cfg: {e}. Make sure they are accessible.")
    exit()

# --- Vaihingen Specific Config ---
# Defaults, can be overridden by config file's CLASSES/PALETTE if they exist
# Using palette from user's provided script
CLASSES = ('ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter')
PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 204, 255], [0, 0, 255]] # Check order/colors

SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # Optional determinism settings
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def label2rgb(mask, palette=PALETTE): # Use passed palette
    """Converts indexed mask (0-5) to Vaihingen RGB color mask."""
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    max_index = np.max(mask)
    # Check if mask contains indices outside palette range
    valid_indices = mask < len(palette)
    if not np.all(valid_indices):
         invalid_idx = np.unique(mask[~valid_indices])
         print(f"Warning: Mask contains indices {invalid_idx.tolist()} >= palette length {len(palette)}. Pixels won't be colored.")

    for i, color in enumerate(palette):
        if i < len(palette):
            # Use the RGB values directly from PALETTE
            mask_rgb[mask == i] = color
    return mask_rgb


def img_writer(inp):
    """Saves the prediction mask, optionally as RGB."""
    (mask, mask_output_path, save_rgb, palette_local) = inp # Receive palette
    try:
        if save_rgb:
            mask_to_save = label2rgb(mask, palette=palette_local) # Use correct palette
            # Convert final RGB mask to BGR for cv2.imwrite
            mask_to_save = cv2.cvtColor(mask_to_save, cv2.COLOR_RGB2BGR)
        else:
            mask_to_save = mask.astype(np.uint8)
        # Ensure output directory exists just before writing
        os.makedirs(os.path.dirname(mask_output_path), exist_ok=True)
        success = cv2.imwrite(mask_output_path, mask_to_save)
        if not success: print(f"Warning: cv2.imwrite failed for {mask_output_path}")
    except Exception as e:
        print(f"Error in img_writer for {mask_output_path}: {e}")

def get_args(): # Renamed to avoid conflict
    parser = argparse.ArgumentParser(description="Evaluate trained model on Vaihingen dataset and save limited predictions.")
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to config file (e.g., config/vaihingen/unetmamba_baseline.py)")
    # --- Use --checkpoint ---
    arg("-ckpt", "--checkpoint", type=Path, required=True, help="Path to the trained model checkpoint (.ckpt file)")
    arg("-o", "--output_path", type=Path, required=True, help="Base directory to save resulting masks.")
    arg("-t", "--tta", default=None, choices=[None, "d4", "lr"], help="Test time augmentation (d4 or lr).")
    arg("--rgb", action='store_true', help="Output RGB color masks instead of indexed masks.")
    # --- Add --val flag ---
    arg("--val", action='store_true', help="Evaluate on the validation set (requires gt_semantic_seg in batch).")
    # --- Add --save-num ---
    arg("--save-num", type=int, default=10, help="Number of prediction images to save (default: 10). Set to -1 to save all.")
    return parser.parse_args()

def main():
    seed_everything(SEED)
    args = get_args()
    try:
        config = py2cfg(args.config_path)
        # Use classes/palette from config if available, otherwise use defaults
        global CLASSES, PALETTE
        CLASSES = getattr(config, 'classes', CLASSES)
        PALETTE = getattr(config, 'PALETTE', PALETTE)
        # Validate config consistency
        if len(CLASSES) != getattr(config, 'num_classes', len(CLASSES)):
            print("Warning: Number of classes in config mismatch default CLASSES length.")
        if len(PALETTE) != len(CLASSES):
            print(f"Warning: PALETTE length ({len(PALETTE)}) mismatch number of CLASSES ({len(CLASSES)}). Using default palette.")
            # Reset palette to default if mismatch - Using palette from user script
            PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 204, 255], [0, 0, 255]]


    except Exception as e:
        print(f"Error loading config file {args.config_path}: {e}")
        exit()

    # --- Auto-create output directory ---
    args.output_path.mkdir(exist_ok=True, parents=True)
    print(f"Output will be saved to: {args.output_path.resolve()}")

    # --- Load Model from Checkpoint ---
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        exit()
    try:
        # --- Use load_from_checkpoint with args.checkpoint ---
        # Pass map_location='cpu' first if loading on CPU then moving to GPU, or directly load to GPU
        model_pl = Supervision_Train.load_from_checkpoint(args.checkpoint, config=config, strict=False)
        model_net = model_pl.net # Get the underlying nn.Module
        print(f"Loaded model from: {args.checkpoint}")
    except Exception as e:
        print(f"Error loading checkpoint {args.checkpoint}: {e}")
        exit()

    # Move model to GPU and set to eval mode
    model_net.cuda()
    model_net.eval()

    # --- Configure TTA (applied to model_net) ---
    tta_model = model_net # Start with the base network
    if args.tta == "lr":
        transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
        tta_model = tta.SegmentationTTAWrapper(model_net, transforms, merge_mode='mean')
        print("Using Test Time Augmentation: lr (HFlip, VFlip)")
    elif args.tta == "d4":
         # Use d4 definition from user's provided vaihingen_test.py
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[90]), # User script had only 90?
            # tta.Rotate90(angles=[0, 90, 180, 270]), # Standard d4 rotations
            tta.Scale(scales=[0.5, 0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)
        ])
        tta_model = tta.SegmentationTTAWrapper(model_net, transforms, merge_mode='mean')
        print("Using Test Time Augmentation: d4 (Flips, Rotate90, Scale)") # Updated description
    else:
        print("No Test Time Augmentation used.")

    # --- Load Dataset ---
    # --- Use --val flag to choose dataset ---
    if args.val:
        if not hasattr(config, 'val_dataset'): print(f"Error: '--val' specified, but 'val_dataset' not defined in {args.config_path}"); exit()
        eval_dataset = config.val_dataset
        print(f"Evaluating on Validation Set ({len(eval_dataset)} samples)...")
    else:
        if not hasattr(config, 'test_dataset'): print(f"Error: 'test_dataset' not defined in {args.config_path}"); exit()
        eval_dataset = config.test_dataset
        print(f"Evaluating on Test Set ({len(eval_dataset)} samples)...")


    eval_loader = DataLoader(
        eval_dataset, batch_size=1, num_workers=4, # Adjust num_workers
        pin_memory=True, drop_last=False, persistent_workers=True if 4 > 0 else False
    )

    # --- Initialize Evaluator only if needed ---
    evaluator = Evaluator(num_class=config.num_classes) if args.val else None
    if evaluator: evaluator.reset()
    results_to_save = []
    images_saved_count = 0

    # --- Inference Loop ---
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
            try:
                img = batch['img'].cuda()
                img_id = batch["img_id"][0]

                raw_predictions = tta_model(img) # Use TTA model (might be original net)

                if isinstance(raw_predictions, tuple): raw_predictions = raw_predictions[0]
                pred_probs = torch.softmax(raw_predictions, dim=1)
                pred_mask = torch.argmax(pred_probs, dim=1)
                pred_mask_np = pred_mask.squeeze(0).cpu().numpy()

                # --- Accumulate Metrics only if --val is set ---
                if args.val and evaluator:
                    if 'gt_semantic_seg' not in batch: print(f"Warning: GT not found..."); continue
                    mask_true = batch['gt_semantic_seg']
                    mask_true_np = mask_true.squeeze(0).cpu().numpy()
                    if mask_true_np.shape != pred_mask_np.shape: print(f"Warning: Shape mismatch..."); continue
                    if hasattr(evaluator, 'add_batch'):
                         evaluator.add_batch(pre_image=pred_mask_np, gt_image=mask_true_np)
                    else: print("Error: Evaluator missing 'add_batch'."); break

                # --- Implement Save Limit ---
                if args.save_num == -1 or images_saved_count < args.save_num:
                    # --- Simplify output path ---
                    output_filename = os.path.join(args.output_path, f"{img_id}.png")
                    # Pass palette to img_writer
                    results_to_save.append((pred_mask_np, output_filename, args.rgb, PALETTE))
                    images_saved_count += 1

            except Exception as e:
                 print(f"\nError processing batch {batch_idx} (ID: {batch.get('img_id', ['N/A'])[0]}): {e}")
                 traceback.print_exc() # Use imported traceback

    # --- Calculate Metrics ---
    if args.val and evaluator:
        print("\nCalculating final metrics...")
        try:
            if not hasattr(evaluator, 'Intersection_over_Union') or \
               not hasattr(evaluator, 'F1') or \
               not hasattr(evaluator, 'OA') or \
               not hasattr(evaluator, 'confusion_matrix') or \
               evaluator.confusion_matrix is None or \
               evaluator.confusion_matrix.sum() == 0:
                 print("Warning: Evaluator object is not ready or empty. Skipping metrics.")
            else:
                iou_per_class = evaluator.Intersection_over_Union()
                f1_per_class = evaluator.F1()
                OA = evaluator.OA()

                # ISPRS: Exclude last class (Clutter/background) from mean metrics
                valid_iou = iou_per_class[:-1]
                valid_f1 = f1_per_class[:-1]
                # Use np.nanmean to handle potential NaN if a class is absent
                mean_iou = np.nanmean(valid_iou) * 100.0 if len(valid_iou) > 0 else 0.0
                mean_f1 = np.nanmean(valid_f1) * 100.0 if len(valid_f1) > 0 else 0.0
                overall_acc = OA * 100.0

                print("\n--- Evaluation Results (Validation Set) ---")
                class_names = getattr(config, 'classes', CLASSES)
                for i in range(config.num_classes):
                    c_name = class_names[i] if i < len(class_names) else f"Class_{i}"
                    iou_str = f'{iou_per_class[i]*100.0:.2f}%' if i < len(iou_per_class) else 'N/A'
                    print(f'IOU_{c_name}: {iou_str}')
                print("---------------------------")
                # ... (print F1 per class if needed) ...
                print(f'Mean F1 (excluding last class): {mean_f1:.2f}%')
                print(f'Mean IoU (excluding last class): {mean_iou:.2f}%')
                print(f'Overall Accuracy: {overall_acc:.2f}%')
                print("---------------------------")
        except Exception as e:
             print(f"Error calculating metrics: {e}")
             traceback.print_exc() # Use imported traceback
    elif not args.val: print("\nMetrics skipped because --val flag was not provided.")
    else: print("\nMetrics skipped because evaluator was not properly initialized or populated.")


    # --- Save Images ---
    if results_to_save:
        print(f"\nSaving {len(results_to_save)} prediction masks to {args.output_path}...")
        t0 = time.time()
        try:
            # Pass palette to img_writer via inp tuple modification
            inp_with_palette = [(r[0], r[1], r[2], PALETTE) for r in results_to_save]
            num_processes = max(1, min(mp.cpu_count() // 2, 8))
            pool = mpp.Pool(processes=num_processes)
            pool.map(img_writer, inp_with_palette) # Pass modified input
            pool.close()
            pool.join()
        except Exception as e: print(f"Error during multiprocessing writing: {e}")
        finally:
            if 'pool' in locals() and pool is not None: pool.terminate()
        t1 = time.time()
        img_write_time = t1 - t0
        print(f'Finished saving masks. Time taken: {img_write_time:.2f} s')
    else: print("No prediction masks were saved.")


if __name__ == "__main__":
    main()
