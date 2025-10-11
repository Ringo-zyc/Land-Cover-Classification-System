# potsdam_test.py
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
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 尝试从 train 脚本导入 Supervision_Train, Evaluator, py2cfg
# 假设 train.py 在当前目录或 Python 路径中
try:
    from train import Supervision_Train, Evaluator, py2cfg
except ImportError:
    print("Error: Could not import from train.py. Make sure it's accessible.")
    exit()

# 尝试从数据集配置导入 CLASSES 和 PALETTE
# 这需要 config 文件能被正常执行导入
# from config.postdam.unetmamba import CLASSES, PALETTE # 或者 _ca
# 为了脚本的通用性，我们也可以在这里重新定义 Potsdam 的
CLASSES = ('Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car', 'Clutter/background')
# 使用与 potsdam_patch_split.py 中 rgb_to_index_mask 对应的颜色 (RGB order)
PALETTE = [
    [255, 255, 255], # 0: Imp Surf (White)
    [0, 0, 255],     # 1: Building (Blue)
    [0, 255, 255],   # 2: Low Veg (Cyan)
    [0, 255, 0],     # 3: Tree (Green)
    [255, 255, 0],   # 4: Car (Yellow)
    [255, 0, 0]      # 5: Clutter/BG (Red)
]

SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # 下面两行有时可以提高确定性，但可能影响性能
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def label_to_rgb_potsdam(mask): # Renamed for clarity
    """Converts indexed mask (0-5) to Potsdam RGB color mask."""
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    for i, color in enumerate(PALETTE):
        # Where mask has index i, set corresponding pixels in mask_rgb to color
        mask_rgb[mask == i] = color
    return mask_rgb

def img_writer(inp):
    """Saves the prediction mask, optionally as RGB."""
    (mask, mask_output_path, save_rgb) = inp
    try:
        if save_rgb:
            # Convert index mask back to RGB using Potsdam palette
            mask_to_save = label_to_rgb_potsdam(mask)
            # OpenCV expects BGR, so convert RGB palette result to BGR
            mask_to_save = cv2.cvtColor(mask_to_save, cv2.COLOR_RGB2BGR)
        else:
            # Save as single-channel indexed image (ensure uint8)
            mask_to_save = mask.astype(np.uint8)

        # Ensure output directory exists (though main function already creates base dir)
        os.makedirs(os.path.dirname(mask_output_path), exist_ok=True)
        success = cv2.imwrite(mask_output_path, mask_to_save)
        if not success:
             print(f"Warning: cv2.imwrite failed for {mask_output_path}")
    except Exception as e:
        print(f"Error in img_writer for {mask_output_path}: {e}")

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model on Potsdam dataset and save limited predictions.")
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to config file (e.g., config/postdam/unetmamba.py)")
    arg("-ckpt", "--checkpoint", type=Path, required=True, help="Path to the trained model checkpoint (.ckpt file)")
    arg("-o", "--output_path", type=Path, required=True, help="Base directory to save resulting masks.")
    arg("-t", "--tta", default=None, choices=[None, "d4", "lr"], help="Test time augmentation (d4 or lr).")
    arg("--rgb", action='store_true', help="Output RGB color masks instead of indexed masks.")
    arg("--save-num", type=int, default=5, help="Number of prediction images to save (default: 5). Set to -1 to save all.")
    # 添加一个参数来明确指定使用哪个数据集进行测试 (虽然我们硬编码在config里了)
    # arg("--eval-set", type=str, default="val", choices=["val", "test"], help="Which dataset split to evaluate (defined in config)")
    return parser.parse_args()

def main():
    seed_everything(SEED)
    args = get_args()
    try:
        config = py2cfg(args.config_path)
    except Exception as e:
        print(f"Error loading config file {args.config_path}: {e}")
        exit()

    # --- 自动创建输出目录 ---
    args.output_path.mkdir(exist_ok=True, parents=True)
    print(f"Output will be saved to: {args.output_path.resolve()}")

    # --- 加载模型 ---
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        exit()
    try:
        # 从指定的 checkpoint 加载，将配置也传递过去
        model = Supervision_Train.load_from_checkpoint(args.checkpoint, config=config)
        print(f"Loaded model from: {args.checkpoint}")
    except Exception as e:
        print(f"Error loading checkpoint {args.checkpoint}: {e}")
        exit()

    model.cuda()
    model.eval()

    # --- 配置 TTA ---
    tta_model = model # Start with the original model
    if args.tta == "lr":
        transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
        tta_model = tta.SegmentationTTAWrapper(model, transforms, merge_mode='mean')
        print("Using Test Time Augmentation: lr (HorizontalFlip, VerticalFlip)")
    elif args.tta == "d4":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]), # d4 includes all 90 degree rotations
            # tta.Scale(scales=[0.8, 1.0, 1.2], interpolation='bilinear', align_corners=False) # Example scale TTA
        ])
        tta_model = tta.SegmentationTTAWrapper(model, transforms, merge_mode='mean')
        print("Using Test Time Augmentation: d4 (Flips, Rotations)")
    else:
        print("No Test Time Augmentation used.")


    # --- 加载测试数据集 ---
    # 确保 config 文件中定义了 test_dataset
    if not hasattr(config, 'test_dataset'):
        print(f"Error: 'test_dataset' not defined in the config file: {args.config_path}")
        exit()
    test_dataset = config.test_dataset

    # --- 创建 DataLoader ---
    # 测试时 Batch Size 通常为 1，num_workers 可以适当设置
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4, # 根据需要调整
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if 4 > 0 else False
    )

    # --- 初始化评估器和结果列表 ---
    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()
    results_to_save = []
    images_saved_count = 0

    # --- 推理和评估循环 ---
    print(f"Starting evaluation on {len(test_dataset)} samples...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            try:
                img = batch['img'].cuda()
                mask_true = batch['gt_semantic_seg'] # Keep on CPU for evaluator
                img_id = batch["img_id"][0] # Assuming batch_size=1

                # --- 执行推理 ---
                # raw_predictions: (1, C, H, W), torch.float32, on GPU
                raw_predictions = tta_model(img)

                # --- 后处理 ---
                # Apply softmax only if model doesn't do it internally
                # Check the output shape and type, assuming logits output
                if isinstance(raw_predictions, tuple): # Handle potential auxiliary outputs
                     raw_predictions = raw_predictions[0]

                # Apply softmax and argmax
                pred_probs = torch.softmax(raw_predictions, dim=1) # (1, C, H, W)
                pred_mask = torch.argmax(pred_probs, dim=1)      # (1, H, W), torch.long, on GPU

                # --- 累积评估指标 ---
                # .cpu() before .numpy()
                pred_mask_np = pred_mask.squeeze(0).cpu().numpy() # Remove batch dim -> (H, W)
                mask_true_np = mask_true.squeeze(0).cpu().numpy() # Remove batch dim -> (H, W)
                evaluator.add_batch(pre_image=pred_mask_np, gt_image=mask_true_np)

                # --- 保存指定数量的预测结果 ---
                if args.save_num == -1 or images_saved_count < args.save_num:
                    output_filename = os.path.join(args.output_path, f"{img_id}{config.test_dataset.mask_suffix}") # Use suffix from dataset obj
                    results_to_save.append((pred_mask_np, output_filename, args.rgb))
                    images_saved_count += 1

            except Exception as e:
                 print(f"\nError processing batch {batch_idx} (ID: {batch.get('img_id', ['N/A'])[0]}): {e}")
                 import traceback
                 traceback.print_exc() # Print detailed traceback for the error
                 # Continue to next batch if possible

    # --- 计算并打印最终指标 ---
    print("\nCalculating final metrics...")
    try:
        # ISPRS Potsdam: Usually exclude 'Clutter/background' (last class, index 5) from mean metrics
        iou_per_class = evaluator.Intersection_over_Union()
        f1_per_class = evaluator.F1()
        OA = evaluator.OA()

        valid_iou = iou_per_class[:-1] # Exclude last class (Clutter/background)
        valid_f1 = f1_per_class[:-1]  # Exclude last class

        # Handle potential NaN values if some classes are not present in the test set
        mean_iou = np.nanmean(valid_iou) * 100.0
        mean_f1 = np.nanmean(valid_f1) * 100.0
        overall_acc = OA * 100.0

        print("\n--- Evaluation Results ---")
        for i in range(config.num_classes):
            class_name = config.classes[i] if i < len(config.classes) else f"Class_{i}"
            print(f'IOU_{class_name}: {iou_per_class[i]*100.0:.2f}%')
        print("---------------------------")
        for i in range(config.num_classes):
             class_name = config.classes[i] if i < len(config.classes) else f"Class_{i}"
             print(f'F1_{class_name}: {f1_per_class[i]*100.0:.2f}%')
        print("---------------------------")
        print(f'Mean F1 (excluding last class): {mean_f1:.2f}%')
        print(f'Mean IoU (excluding last class): {mean_iou:.2f}%')
        print(f'Overall Accuracy: {overall_acc:.2f}%')
        print("---------------------------")

    except Exception as e:
         print(f"Error calculating metrics: {e}")

    # --- 保存预测掩码图像 (使用多进程) ---
    if results_to_save:
        print(f"\nSaving {len(results_to_save)} prediction masks...")
        t0 = time.time()
        try:
            num_processes = max(1, min(mp.cpu_count() // 2, 8))
            pool = mpp.Pool(processes=num_processes)
            pool.map(img_writer, results_to_save)
            pool.close()
            pool.join()
        except Exception as e:
            print(f"Error during multiprocessing image writing: {e}")
        finally:
            if 'pool' in locals() and pool is not None: pool.terminate()
        t1 = time.time()
        img_write_time = t1 - t0
        print(f'Finished saving masks. Time taken: {img_write_time:.2f} s')
    else:
         print("No prediction masks were saved (save_num might be 0).")

if __name__ == "__main__":
    main()