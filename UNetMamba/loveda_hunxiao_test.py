# loveda_test.py (最终版本，包含混淆矩阵提取与保存)
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
    from train import Supervision_Train # Supervision_Train 主要用于加载模型结构
    # 假设 Evaluator 在 tools.metric.py 中定义 (根据之前的确认)
    from tools.metric import Evaluator
    # 假设 py2cfg 在 tools.cfg 中定义
    from tools.cfg import py2cfg
except ImportError as e:
    print(f"错误: 无法导入所需组件: {e}")
    print("请确保 'train.py' (含 Supervision_Train), 'tools/metric.py' (含 Evaluator) 和 'tools/cfg.py' (含 py2cfg) 可访问。")
    exit()

# LoveDA 类和调色板 (保持不变)
CLASSES = ('background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural')
PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
           [159, 129, 183], [0, 255, 0], [255, 195, 128]]

SEED = 42

def seed_everything(seed):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 推荐用于多GPU
        # 为了完全可复现性，可能需要牺牲性能
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

# --- 使用之前版本中似乎更好的 label_to_rgb ---
def label_to_rgb_loveda(mask):
    """将索引掩码 (0-6) 转换为 LoveDA RGB 颜色掩码"""
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    valid_indices = mask < len(PALETTE)
    # 可选: 如果发现无效索引则发出警告
    # if not np.all(valid_indices):
    #     print(f"警告: 掩码包含索引 >= {len(PALETTE)}。这些像素将不会被着色。")

    for i, color in enumerate(PALETTE):
        mask_rgb[mask == i] = color # 应用颜色
    return mask_rgb

# --- 使用之前版本中似乎更好的 img_writer ---
def img_writer(inp):
    """保存预测掩码，可选保存为RGB格式"""
    (mask, mask_output_path, save_rgb) = inp
    try:
        if save_rgb:
            mask_to_save = label_to_rgb_loveda(mask)
            # OpenCV 保存图像需要 BGR 格式
            mask_to_save = cv2.cvtColor(mask_to_save, cv2.COLOR_RGB2BGR)
        else:
            mask_to_save = mask.astype(np.uint8) # 索引掩码直接保存为uint8

        # 确保输出目录存在 (非常重要!)
        os.makedirs(os.path.dirname(mask_output_path), exist_ok=True)
        success = cv2.imwrite(str(mask_output_path), mask_to_save) # 使用 str() 转换 Path 对象
        if not success:
            print(f"警告: cv2.imwrite 保存 {mask_output_path} 失败")
    except Exception as e:
        print(f"img_writer 处理 {mask_output_path} 时出错: {e}")
        # traceback.print_exc() # 需要时取消注释以获取详细信息

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="在 LoveDA 数据集上评估训练好的模型")
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="配置文件路径")
    # --- 修改: 添加 -ckpt 参数 ---
    arg("-ckpt", "--checkpoint", type=Path, required=True, help="训练好的模型检查点 (.ckpt) 路径")
    arg("-o", "--output_path", type=Path, required=True, help="保存结果掩码的基础目录")
    arg("-t", "--tta", default=None, choices=[None, "d4", "lr"], help="测试时增强 (Test time augmentation: d4 或 lr)")
    arg("--rgb", action='store_true', help="输出 RGB 彩色掩码而非索引掩码")
    arg("--val", action='store_true', help="在验证集上评估 (需要在 batch 中包含 gt_semantic_seg)")
    # --- 修改: 添加 --save-num 参数 ---
    arg("--save-num", type=int, default=-1, help="要保存的预测图像数量 (默认: -1 保存全部)。设置为 0 则不保存。")
    return parser.parse_args()

def main():
    seed_everything(SEED)
    args = get_args()
    try:
        config = py2cfg(args.config_path)
    except Exception as e:
        print(f"加载配置文件 {args.config_path} 出错: {e}")
        exit()

    # --- 自动创建输出目录 ---
    args.output_path.mkdir(exist_ok=True, parents=True)
    print(f"输出将保存至: {args.output_path.resolve()}")

    # --- 加载模型 ---
    if not args.checkpoint.exists():
        print(f"错误: 检查点文件未在 {args.checkpoint} 找到")
        exit()
    try:
        # --- 修改: 使用 args.checkpoint 加载模型 ---
        # 假设 checkpoint 是 PyTorch Lightning 保存的
        # strict=False 可能有助于加载稍有不同的旧 checkpoint
        # 注意：这里加载的是 Supervision_Train LightningModule
        model_pl = Supervision_Train.load_from_checkpoint(args.checkpoint, config=config, strict=False)
        # 获取实际的网络模型 (nn.Module)
        # 需要确认 Supervision_Train 类中网络模型的属性名是否为 'net'
        if hasattr(model_pl, 'net') and isinstance(model_pl.net, nn.Module):
             model_net = model_pl.net
        # 备用方案：如果 LightningModule 本身就是网络或网络在不同属性下
        elif isinstance(model_pl, nn.Module):
             print("警告: 加载的对象本身是 nn.Module, 假设它就是模型。")
             model_net = model_pl
        else:
             print("错误: 无法从加载的检查点中明确找到 'net' 属性下的 nn.Module。请检查 Supervision_Train 类。")
             exit()

        print(f"模型已从 {args.checkpoint} 加载")
    except Exception as e:
        print(f"加载检查点 {args.checkpoint} 出错: {e}")
        # traceback.print_exc() # 取消注释以获取详细加载错误
        exit()

    # 确保模型在 GPU 上并处于评估模式
    if not isinstance(model_net, nn.Module):
         print(f"错误: 加载的对象 'model_net' 不是 nn.Module 类型: {type(model_net)}")
         exit()

    try:
         if torch.cuda.is_available():
             model_net.cuda()
             print("模型已移至 GPU")
         else:
             print("警告: 未检测到 CUDA，模型将在 CPU 上运行。")
         model_net.eval()
    except Exception as e:
         print(f"移动模型到 GPU 或设置评估模式时出错: {e}")
         exit()


    # --- 配置 TTA ---
    tta_model = model_net # TTA 包裹实际的网络模型
    if args.tta == "lr":
        transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
        tta_model = tta.SegmentationTTAWrapper(model_net, transforms, merge_mode='mean')
        print("使用测试时增强: lr (水平翻转, 垂直翻转)")
    elif args.tta == "d4":
        # d4 TTA (可能需要根据您的原始脚本调整)
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                # tta.VerticalFlip(), # 原脚本d4是否使用了垂直翻转？
                # tta.Rotate90(angles=[0, 90, 180, 270]), # 原脚本d4是否使用了旋转？
                tta.Scale(scales=[0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False),
                # tta.Multiply(factors=[0.8, 1, 1.2]) # 原脚本d4是否使用了亮度调整？
            ]
        )
        tta_model = tta.SegmentationTTAWrapper(model_net, transforms, merge_mode='mean')
        print("使用测试时增强: d4 (水平翻转, 缩放)")
    else:
        print("未使用测试时增强。")

    # --- 加载测试/验证数据集 ---
    # 确保 config 对象中定义了 val_dataset 和 test_dataset
    if args.val:
        if not hasattr(config, 'val_loader') or not hasattr(config.val_loader, 'dataset'):
            print(f"错误: 指定了 '--val'，但在 {args.config_path} 中未定义 'val_loader' 或其 'dataset' 属性")
            exit()
        eval_dataset = config.val_loader.dataset # 从 DataLoader 获取 Dataset
        eval_loader = config.val_loader # 直接使用配置中的 val_loader
        print(f"在验证集上评估 ({len(eval_dataset)} 个样本)...")
    else:
        if not hasattr(config, 'test_loader') or not hasattr(config.test_loader, 'dataset'):
            print(f"错误: 指定了 '--test' (默认)，但在 {args.config_path} 中未定义 'test_loader' 或其 'dataset' 属性")
            exit()
        eval_dataset = config.test_loader.dataset # 从 DataLoader 获取 Dataset
        eval_loader = config.test_loader # 直接使用配置中的 test_loader
        print(f"在测试集上评估 ({len(eval_dataset)} 个样本)...")

    # 如果是从 config 加载的 loader，则不需要再创建 DataLoader
    # eval_loader = DataLoader(...) # 这部分代码现在被上面的逻辑替代

    # --- 初始化评估器和结果列表 ---
    # --- 修改: 使用 num_classes 初始化 Evaluator ---
    evaluator = Evaluator(num_class=config.num_classes) if args.val else None
    if evaluator: evaluator.reset()
    results_to_save = []
    images_saved_count = 0

    # --- 推理和评估循环 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="评估中")):
            try:
                # 将数据移动到正确的设备
                img = batch['img'].to(device)
                # 确保 'img_id' 存在且是列表或元组, 取第一个元素
                img_id = batch.get("img_id", [f"unknown_{batch_idx}"])[0]

                # --- 执行推理 (使用 tta_model) ---
                raw_predictions = tta_model(img)

                # --- 后处理 ---
                # 处理模型可能的元组输出 (例如带有辅助输出)
                if isinstance(raw_predictions, (list, tuple)):
                    raw_predictions = raw_predictions[0]

                # 检查输出维度是否合理 (应为 N, C, H, W)
                if raw_predictions.ndim != 4 or raw_predictions.shape[0] != img.shape[0]: # 检查 batch size
                    print(f"警告: 批次 {batch_idx} 的预测形状 {raw_predictions.shape} 不符合预期。跳过。")
                    continue

                pred_probs = torch.softmax(raw_predictions, dim=1)
                pred_mask = torch.argmax(pred_probs, dim=1)
                # 从 GPU 移回 CPU 并转为 NumPy (因为batch_size可能>1, 需要循环处理)
                pred_mask_np_batch = pred_mask.cpu().numpy()

                # --- 累积评估指标 (如果适用) ---
                if args.val and evaluator:
                    if 'gt_semantic_seg' not in batch:
                        print(f"警告: 验证批次 {batch_idx} 中未找到真实标签 ('gt_semantic_seg')。跳过指标计算。")
                    else:
                        mask_true_batch = batch['gt_semantic_seg']
                        # 确保 GT 也在 CPU 上并转为 NumPy
                        mask_true_np_batch = mask_true_batch.cpu().numpy()

                        # 检查形状并逐个样本添加
                        if mask_true_np_batch.shape[1:] != pred_mask_np_batch.shape[1:]: # 忽略 batch 维度比较 H,W
                            print(f"警告: 图像 {img_id} 的 GT 掩码形状 {mask_true_np_batch.shape} != 预测掩码形状 {pred_mask_np_batch.shape}。检查数据加载/填充。跳过指标。")
                        else:
                            # 假设 batch size > 1 的情况，需要逐个添加
                            for i in range(pred_mask_np_batch.shape[0]):
                                current_img_id = batch.get("img_id", [f"unknown_{batch_idx}_{i}"]*pred_mask_np_batch.shape[0])[i] # 获取当前样本ID
                                evaluator.add_batch(pre_image=pred_mask_np_batch[i], gt_image=mask_true_np_batch[i])

                # --- 保存指定数量的预测结果 ---
                # --- 修改: 使用 save-num 控制保存 ---
                # 处理 batch 中每个样本
                for i in range(pred_mask_np_batch.shape[0]):
                    if args.save_num != 0 and (args.save_num == -1 or images_saved_count < args.save_num):
                        current_img_id = batch.get("img_id", [f"unknown_{batch_idx}_{i}"]*pred_mask_np_batch.shape[0])[i]
                        # 直接在输出目录下按 img_id 保存
                        output_filename = args.output_path / f"{current_img_id}.png"
                        # 保存单个样本的 NumPy 数组
                        results_to_save.append((pred_mask_np_batch[i], output_filename, args.rgb))
                        images_saved_count += 1

            except Exception as e:
                current_img_id_str = "N/A"
                if 'batch' in locals() and batch and 'img_id' in batch:
                     # 处理 batch size > 1 的情况，只显示第一个ID作为参考
                     current_img_id_str = batch['img_id'][0] if isinstance(batch['img_id'], (list, tuple)) and len(batch['img_id']) > 0 else str(batch['img_id'])

                print(f"\n处理批次 {batch_idx} (首个ID: {current_img_id_str}) 时出错: {e}")
                traceback.print_exc() # 现在可以正常工作

    # --- 计算并打印最终指标 (如果适用) ---
    if args.val and evaluator:
        print("\n计算最终指标...")
        try:
            can_calculate_metrics = False # 先假设不能计算
            # !!! --- 添加部分 开始: 获取并保存混淆矩阵 --- !!!
            # 检查混淆矩阵属性是否存在且有效 (属性名已确认为 confusion_matrix)
            if hasattr(evaluator, 'confusion_matrix') and evaluator.confusion_matrix is not None and evaluator.confusion_matrix.sum() > 0:
                 can_calculate_metrics = True # 可以计算指标
                 # 1. 获取混淆矩阵
                 final_confusion_matrix = evaluator.confusion_matrix
                 print("\n--- 最终混淆矩阵 ---")
                 print(final_confusion_matrix)
                 print("------------------------")

                 # 2. (可选但推荐) 保存混淆矩阵到文件
                 cm_save_path_npy = args.output_path / "confusion_matrix.npy"
                 cm_save_path_csv = args.output_path / "confusion_matrix.csv"
                 try:
                     np.save(cm_save_path_npy, final_confusion_matrix)
                     print(f"混淆矩阵已保存至: {cm_save_path_npy}")
                     # 保存为 CSV, 使用整数格式
                     np.savetxt(cm_save_path_csv, final_confusion_matrix, delimiter=",", fmt='%d')
                     print(f"混淆矩阵也已保存为 CSV: {cm_save_path_csv}")
                 except Exception as e:
                     print(f"保存混淆矩阵时出错: {e}")
            else:
                 print("警告: Evaluator 的混淆矩阵属性 ('confusion_matrix') 未找到、为 None 或全为零。无法访问/保存矩阵或准确计算指标。")
            # !!! --- 添加部分 结束 --- !!!

            # 只有在能计算指标时才继续
            if can_calculate_metrics:
                iou_per_class = evaluator.Intersection_over_Union()
                f1_per_class = evaluator.F1()
                OA = evaluator.OA()

                # 使用 np.nanmean 安全计算平均值 (忽略 NaN)
                mean_iou_all = np.nanmean(iou_per_class) * 100.0
                mean_f1_all = np.nanmean(f1_per_class) * 100.0
                # 排除背景 (索引 0) 计算前景指标 (假设背景是第0类)
                mean_iou_fg = np.nanmean(iou_per_class[1:]) * 100.0 if len(iou_per_class) > 1 else np.nan
                mean_f1_fg = np.nanmean(f1_per_class[1:]) * 100.0 if len(f1_per_class) > 1 else np.nan
                overall_acc = OA * 100.0

                print("\n--- 评估结果 (验证集) ---")
                # 从 config 获取类名，如果 config 没有 classes 属性则使用默认名称
                # 注意: 需要确保 config.classes 列表长度与 config.num_classes 匹配
                class_names = getattr(config, 'classes', [f"Class_{i}" for i in range(config.num_classes)])
                if len(class_names) != config.num_classes:
                       print(f"警告: 类别名称数量 ({len(class_names)}) 与 num_classes ({config.num_classes}) 不匹配。使用默认名称。")
                       class_names = [f"Class_{i}" for i in range(config.num_classes)]

                for i in range(config.num_classes):
                       c_name = class_names[i]
                       # 检查 iou_per_class 和 f1_per_class 是否足够长
                       iou_val = iou_per_class[i] * 100.0 if i < len(iou_per_class) else np.nan
                       f1_val = f1_per_class[i] * 100.0 if i < len(f1_per_class) else np.nan
                       print(f'IoU_{c_name}: {iou_val:.2f}%')
                       # 可以取消注释以打印 F1 分数
                       # print(f'F1_{c_name}: {f1_val:.2f}%')
                print("---------------------------")
                print(f'平均 F1 (所有类别): {mean_f1_all:.2f}%')
                print(f'平均 IoU (所有类别): {mean_iou_all:.2f}%')
                print(f'平均 F1 (前景类别): {mean_f1_fg:.2f}%')
                print(f'平均 IoU (前景类别): {mean_iou_fg:.2f}%')
                print(f'总体准确率 (OA): {overall_acc:.2f}%')
                print("---------------------------")

            else:
                 print("由于混淆矩阵问题，跳过指标计算。")

        except Exception as e:
            print(f"计算或打印指标时出错: {e}")
            traceback.print_exc()
    elif not args.val:
        print("\n跳过评估指标计算 (请使用 --val 参数并确保数据集提供 GT)。")

    # --- 保存预测掩码图像 (使用多进程) ---
    if results_to_save:
        print(f"\n正在将 {len(results_to_save)} 个预测掩码保存至 {args.output_path}...")
        t0 = time.time()
        # --- 改进: 使用 try/finally 确保 pool 关闭 ---
        pool = None # 初始化 pool 为 None
        try:
            # 限制进程数防止资源耗尽，例如最多使用 CPU 核心数的一半或 8 个
            num_processes = max(1, min(mp.cpu_count() // 2, 8))
            pool = mpp.Pool(processes=num_processes)
            # 使用 imap_unordered 可能更快，因为它不保证顺序
            # list(tqdm(pool.imap_unordered(img_writer, results_to_save), total=len(results_to_save), desc="保存图像中"))
            # 或者使用 map (保持原始实现)
            list(tqdm(pool.imap(img_writer, results_to_save), total=len(results_to_save), desc="保存图像中"))
            pool.close()
            pool.join()
            t1 = time.time()
            img_write_time = t1 - t0
            print(f'掩码保存完毕。耗时: {img_write_time:.2f} 秒')
        except Exception as e:
            print(f"多进程保存图像时出错: {e}")
            traceback.print_exc()
        finally:
            if pool is not None:
                pool.terminate() # 即使出错也要确保 pool 被终止
    else:
        print("没有需要保存的预测掩码。")

if __name__ == "__main__":
    main()