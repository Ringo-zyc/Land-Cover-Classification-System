# loveda_test.py (Modified Version)
import traceback # <<<--- 添加: 解决 UnboundLocalError
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

# --- 改进: 使用更明确的导入 ---
try:
    # 假设 Evaluator 和 Supervision_Train 在 train.py 中定义或被其导入
    from train import Supervision_Train, Evaluator
    # 假设 py2cfg 在 tools.cfg 中定义
    from tools.cfg import py2cfg
except ImportError as e:
    print(f"Error: Could not import required components: {e}.")
    print("Please ensure 'train.py' (with Evaluator, Supervision_Train) and 'tools/cfg.py' (with py2cfg) are accessible.")
    exit()

# LoveDA 类和调色板 (保持不变)
CLASSES = ('background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural')
PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
           [159, 129, 183], [0, 255, 0], [255, 195, 128]]

SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # Recommended for multi-GPU
        # Potentially add these for reproducibility, but can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

# --- 使用之前版本中似乎更好的 label_to_rgb ---
def label_to_rgb_loveda(mask):
    """Converts indexed mask (0-6) to LoveDA RGB color mask."""
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    valid_indices = mask < len(PALETTE)
    # Optional: Warn if unexpected indices are found
    # if not np.all(valid_indices):
    #    print(f"Warning: Mask contains indices >= {len(PALETTE)}. These pixels will not be colored.")

    for i, color in enumerate(PALETTE):
        mask_rgb[mask == i] = color # Apply color where index matches
    return mask_rgb

# --- 使用之前版本中似乎更好的 img_writer ---
def img_writer(inp):
    """Saves the prediction mask, optionally as RGB."""
    (mask, mask_output_path, save_rgb) = inp
    try:
        if save_rgb:
            mask_to_save = label_to_rgb_loveda(mask)
            # OpenCV expects BGR format for saving
            mask_to_save = cv2.cvtColor(mask_to_save, cv2.COLOR_RGB2BGR)
        else:
            mask_to_save = mask.astype(np.uint8)

        # Ensure output directory exists (important!)
        os.makedirs(os.path.dirname(mask_output_path), exist_ok=True)
        success = cv2.imwrite(mask_output_path, mask_to_save)
        if not success:
            print(f"Warning: cv2.imwrite failed for {mask_output_path}")
    except Exception as e:
        print(f"Error in img_writer for {mask_output_path}: {e}")
        # traceback.print_exc() # Uncomment for more details if needed

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model on LoveDA dataset.")
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to config file")
    # --- 修改: 添加 -ckpt 参数 ---
    arg("-ckpt", "--checkpoint", type=Path, required=True, help="Path to the trained model checkpoint (.ckpt)")
    arg("-o", "--output_path", type=Path, required=True, help="Base directory to save resulting masks.")
    arg("-t", "--tta", default=None, choices=[None, "d4", "lr"], help="Test time augmentation (d4 or lr).")
    arg("--rgb", action='store_true', help="Output RGB color masks instead of indexed masks.")
    arg("--val", action='store_true', help="Evaluate on the validation set (requires gt_semantic_seg in batch).")
    # --- 修改: 添加 --save-num 参数 ---
    arg("--save-num", type=int, default=-1, help="Number of prediction images to save (default: -1 to save all). Set to 0 to save none.")
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
        # --- 修改: 使用 args.checkpoint 加载模型 ---
        # 假设 checkpoint 是 PyTorch Lightning 保存的
        # strict=False 可能有助于加载稍有不同的旧 checkpoint
        model_pl = Supervision_Train.load_from_checkpoint(args.checkpoint, config=config, strict=False)
        # 获取实际的网络模型 (nn.Module)
        # 注意: 需要确认 Supervision_Train 类中网络模型的属性名是否为 'net'
        if hasattr(model_pl, 'net'):
            model_net = model_pl.net
        else:
            # 如果 Supervision_Train 本身就是 nn.Module 或有其他方式访问网络
            print("Warning: Could not find 'net' attribute in loaded checkpoint. Assuming the loaded object is the model itself.")
            model_net = model_pl

        print(f"Loaded model from: {args.checkpoint}")
    except Exception as e:
        print(f"Error loading checkpoint {args.checkpoint}: {e}")
        # traceback.print_exc() # Uncomment for detailed loading error
        exit()

    # 确保模型在 GPU 上并处于评估模式
    # 检查 model_net 是否是 nn.Module
    if not isinstance(model_net, nn.Module):
         print(f"Error: Loaded object 'model_net' is not an nn.Module. Type: {type(model_net)}")
         exit()

    try:
         model_net.cuda()
         model_net.eval()
    except Exception as e:
         print(f"Error moving model to GPU or setting eval mode: {e}")
         exit()


    # --- 配置 TTA ---
    tta_model = model_net # TTA 包裹实际的网络模型
    if args.tta == "lr":
        transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
        tta_model = tta.SegmentationTTAWrapper(model_net, transforms, merge_mode='mean')
        print("Using Test Time Augmentation: lr (HorizontalFlip, VerticalFlip)")
    elif args.tta == "d4":
        # d4 TTA from your original script
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                # tta.VerticalFlip(), # Original didn't use VFlip for d4?
                # tta.Rotate90(angles=[0, 90, 180, 270]),
                tta.Scale(scales=[0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False),
                # tta.Multiply(factors=[0.8, 1, 1.2])
            ]
        )
        tta_model = tta.SegmentationTTAWrapper(model_net, transforms, merge_mode='mean')
        print("Using Test Time Augmentation: d4 (HFlip, Scale)")
    else:
        print("No Test Time Augmentation used.")

    # --- 加载测试/验证数据集 ---
    # 确保 config 对象中定义了 val_dataset 和 test_dataset
    if args.val:
        if not hasattr(config, 'val_dataset'):
            print(f"Error: '--val' specified, but 'val_dataset' not defined in {args.config_path}")
            exit()
        eval_dataset = config.val_dataset
        print(f"Evaluating on Validation Set ({len(eval_dataset)} samples)...")
    else:
        if not hasattr(config, 'test_dataset'):
             print(f"Error: '--test' specified (default), but 'test_dataset' not defined in {args.config_path}")
             exit()
        eval_dataset = config.test_dataset
        print(f"Evaluating on Test Set ({len(eval_dataset)} samples)...")

    # --- 创建 DataLoader ---
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1, # 推理时通常 batch_size=1
        num_workers=4, # 根据系统调整
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if 4 > 0 else False # Use num_workers here
    )

    # --- 初始化评估器和结果列表 ---
    # --- 修改: 使用 num_classes 初始化 Evaluator ---
    evaluator = Evaluator(num_class=config.num_classes) if args.val else None
    if evaluator: evaluator.reset()
    results_to_save = []
    images_saved_count = 0

    # --- 推理和评估循环 ---
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
            try:
                img = batch['img'].cuda()
                # 确保 'img_id' 存在且是列表或元组
                img_id = batch.get("img_id", [f"unknown_{batch_idx}"])[0]

                # --- 执行推理 (使用 tta_model) ---
                raw_predictions = tta_model(img)

                # --- 后处理 ---
                # 处理模型可能的元组输出 (例如带有辅助输出)
                if isinstance(raw_predictions, (list, tuple)):
                    raw_predictions = raw_predictions[0]

                # 检查输出维度是否合理 (应为 N, C, H, W)
                if raw_predictions.ndim != 4 or raw_predictions.shape[0] != 1:
                     print(f"Warning: Unexpected prediction shape {raw_predictions.shape} for batch {batch_idx}. Skipping.")
                     continue

                pred_probs = torch.softmax(raw_predictions, dim=1)
                pred_mask = torch.argmax(pred_probs, dim=1)
                # 从 GPU 移回 CPU 并转为 NumPy (确保 squeeze(0) 适用于 batch_size=1)
                pred_mask_np = pred_mask.squeeze(0).cpu().numpy()

                # --- 累积评估指标 (如果适用) ---
                if args.val and evaluator:
                    if 'gt_semantic_seg' not in batch:
                        print(f"Warning: Ground truth ('gt_semantic_seg') not found in validation batch {batch_idx}. Skipping metric calculation.")
                    else:
                        mask_true = batch['gt_semantic_seg']
                        # 确保 GT 也是 batch_size=1 并移回 CPU
                        mask_true_np = mask_true.squeeze(0).cpu().numpy()

                        if mask_true_np.shape != pred_mask_np.shape:
                            print(f"Warning: GT mask shape {mask_true_np.shape} != Pred mask shape {pred_mask_np.shape} for {img_id}. Check data loading/padding. Skipping metrics.")
                        else:
                            evaluator.add_batch(pre_image=pred_mask_np, gt_image=mask_true_np)

                # --- 保存指定数量的预测结果 ---
                # --- 修改: 使用 save-num 控制保存 ---
                if args.save_num != 0 and (args.save_num == -1 or images_saved_count < args.save_num):
                    # 直接在输出目录下按 img_id 保存
                    output_filename = os.path.join(args.output_path, f"{img_id}.png")
                    results_to_save.append((pred_mask_np, output_filename, args.rgb))
                    images_saved_count += 1

            except Exception as e:
                print(f"\nError processing batch {batch_idx} (ID: {img_id if 'img_id' in locals() else 'N/A'}): {e}")
                traceback.print_exc() # Now this will work

    # --- 计算并打印最终指标 (如果适用) ---
    if args.val and evaluator:
        print("\nCalculating final metrics...")
        try:
            # 确保 Evaluator 内部有计算好的混淆矩阵等
            # 添加检查，防止在没有有效样本的情况下调用指标计算
            if not hasattr(evaluator, 'confusion_matrix') or evaluator.confusion_matrix is None or evaluator.confusion_matrix.sum() == 0:
                 print("Warning: Evaluator confusion matrix is empty or invalid. Cannot calculate metrics.")
            else:
                iou_per_class = evaluator.Intersection_over_Union()
                f1_per_class = evaluator.F1()
                OA = evaluator.OA()

                # 使用 np.nanmean 安全计算平均值 (忽略 NaN)
                mean_iou_all = np.nanmean(iou_per_class) * 100.0
                mean_f1_all = np.nanmean(f1_per_class) * 100.0
                # 排除背景 (索引 0) 计算前景指标
                mean_iou_fg = np.nanmean(iou_per_class[1:]) * 100.0 if len(iou_per_class) > 1 else 0.0
                mean_f1_fg = np.nanmean(f1_per_class[1:]) * 100.0 if len(f1_per_class) > 1 else 0.0
                overall_acc = OA * 100.0

                print("\n--- Evaluation Results (Validation Set) ---")
                # 从 config 获取类名，如果 config 没有 classes 属性则使用默认名称
                class_names = getattr(config, 'classes', [f"Class_{i}" for i in range(config.num_classes)])
                if len(class_names) != config.num_classes:
                     print(f"Warning: Number of class names ({len(class_names)}) does not match num_classes ({config.num_classes}). Using default names.")
                     class_names = [f"Class_{i}" for i in range(config.num_classes)]

                for i in range(config.num_classes):
                    c_name = class_names[i]
                    # 检查 iou_per_class 和 f1_per_class 是否足够长
                    iou_val = iou_per_class[i] * 100.0 if i < len(iou_per_class) else np.nan
                    f1_val = f1_per_class[i] * 100.0 if i < len(f1_per_class) else np.nan
                    print(f'IOU_{c_name}: {iou_val:.2f}%')
                print("---------------------------")
                # 可以选择性打印 F1
                # for i in range(config.num_classes):
                #     c_name = class_names[i]
                #     f1_val = f1_per_class[i] * 100.0 if i < len(f1_per_class) else np.nan
                #     print(f'F1_{c_name}: {f1_val:.2f}%')
                # print("---------------------------")
                print(f'Mean F1 (All Classes): {mean_f1_all:.2f}%')
                print(f'Mean IoU (All Classes): {mean_iou_all:.2f}%')
                print(f'Mean F1 (Foreground): {mean_f1_fg:.2f}%')
                print(f'Mean IoU (Foreground): {mean_iou_fg:.2f}%')
                print(f'Overall Accuracy: {overall_acc:.2f}%')
                print("---------------------------")

        except Exception as e:
            print(f"Error calculating or printing metrics: {e}")
            traceback.print_exc() # Now this will work correctly
    elif not args.val:
        print("\nEvaluation metrics calculation skipped (Use --val and ensure dataset provides GT).")

    # --- 保存预测掩码图像 (使用多进程) ---
    if results_to_save:
        print(f"\nSaving {len(results_to_save)} prediction masks to {args.output_path}...")
        t0 = time.time()
        # --- 改进: 使用 try/finally 确保 pool 关闭 ---
        pool = None # Initialize pool to None
        try:
            # 限制进程数防止资源耗尽
            num_processes = max(1, min(mp.cpu_count() // 2, 8))
            pool = mpp.Pool(processes=num_processes)
            # 使用 imap_unordered 可能更快，因为它不保证顺序
            # list(tqdm(pool.imap_unordered(img_writer, results_to_save), total=len(results_to_save), desc="Saving Images"))
            pool.map(img_writer, results_to_save) # map 也可以
            pool.close()
            pool.join()
            t1 = time.time()
            img_write_time = t1 - t0
            print(f'Finished saving masks. Time taken: {img_write_time:.2f} s')
        except Exception as e:
            print(f"Error during multiprocessing image writing: {e}")
            traceback.print_exc() # Now this will work
        finally:
            if pool is not None:
                pool.terminate() # Ensure pool is terminated even on error
    else:
        print("No prediction masks were scheduled for saving.")

if __name__ == "__main__":
    main()