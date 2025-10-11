import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import random
import glob

# --- 配置区域 (需要您修改) ---

# 1. 定义类别名称
CLASS_NAMES_LOVEDA = ['Background', 'Building', 'Road', 'Water', 'Barren', 'Forest', 'Agricultural', 'Ignore']
CLASS_NAMES_ISPRS = ['Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car', 'Clutter/background']

# 2. 定义颜色图谱 (类别 7 已设为黑色)
COLOR_MAP_LOVEDA = {
    0: [128, 128, 128], 1: [255, 0, 0], 2: [255, 255, 255], 3: [0, 0, 255],
    4: [165, 42, 42], 5: [0, 128, 0], 6: [255, 255, 0], 7: [0, 0, 0]
}
COLOR_MAP_ISPRS = {
    0: [255, 255, 255], 1: [0, 0, 255], 2: [0, 255, 255],
    3: [0, 255, 0], 4: [255, 255, 0], 5: [255, 0, 0]
}

# 3. LoveDA 随机采样配置
DATASET_CONFIG_RANDOM = {
    "LoveDA_Rural": {
        "input_dir": "data/LoveDA/Val/Rural/images_png",
        "gt_dir": "data/LoveDA/Val/Rural/masks_png",
        "mamba_pred_base_dir": "/root/autodl-tmp/UNetMamba-main/fig_results/loveda/unetmamba_rgb",
        "mamba_ca_pred_base_dir": "fig_results/loveda/unetmamba_CA_rgb",
        "sub_dir": "Rural",
        "color_map": COLOR_MAP_LOVEDA,
        "class_names": CLASS_NAMES_LOVEDA,
        "input_ext": ".png", "gt_ext": ".png", "gt_is_rgb": False
    },
     "LoveDA_Urban": {
        "input_dir": "data/LoveDA/Val/Urban/images_png",
        "gt_dir": "data/LoveDA/Val/Urban/masks_png",
        "mamba_pred_base_dir": "/root/autodl-tmp/UNetMamba-main/fig_results/loveda/unetmamba_rgb",
        "mamba_ca_pred_base_dir": "fig_results/loveda/unetmamba_CA_rgb",
        "sub_dir": "Urban",
        "color_map": COLOR_MAP_LOVEDA,
        "class_names": CLASS_NAMES_LOVEDA,
        "input_ext": ".png", "gt_ext": ".png", "gt_is_rgb": False
    },
}

# 4. Vaihingen 和 Potsdam 手动指定样本路径列表
#    **!! 建议为 Vaihingen 和 Potsdam 各添加多个有效条目，以便脚本随机挑选 !!**
MANUAL_SAMPLES = [
    # --- Vaihingen Samples ---
    {
        "dataset_name": "Vaihingen", # 脚本会基于此名称分组和随机选择
        "input_path": "data/vaihingen/val_1024/images/top_mosaic_09cm_area37_0_0.tif",
        "gt_path": "data/vaihingen/val_1024/masks_rgb/top_mosaic_09cm_area37_0_0.png",
        "mamba_pred_path": "fig_results/vaihingen/unetmamba_baseline_rgb_d4/top_mosaic_09cm_area37_0_0.png",
        "mamba_ca_pred_path": "fig_results/vaihingen/unetmamba_CA_rgb_d4/top_mosaic_09cm_area37_0_0.png",
        "color_map": COLOR_MAP_ISPRS, "class_names": CLASS_NAMES_ISPRS, "gt_is_rgb": True
    },
    # { # 再添加一个 Vaihingen 样本...
    #     "dataset_name": "Vaihingen",
    #     "input_path": "data/vaihingen/val_1024/images/ANOTHER_VAIHINGEN_INPUT.tif",
    #     "gt_path": "data/vaihingen/val_1024/masks_rgb/ANOTHER_VAIHINGEN_GT.png",
    #     "mamba_pred_path": "fig_results/vaihingen/unetmamba_baseline_rgb_d4/ANOTHER_VAIHINGEN_PRED1.png",
    #     "mamba_ca_pred_path": "fig_results/vaihingen/unetmamba_CA_rgb_d4/ANOTHER_VAIHINGEN_PRED2.png",
    #     "color_map": COLOR_MAP_ISPRS, "class_names": CLASS_NAMES_ISPRS, "gt_is_rgb": True
    # },

    # --- Potsdam Samples ---
    {
        "dataset_name": "Potsdam", # 脚本会基于此名称分组和随机选择
        "input_path": "data/Potsdam/val_1024/images/top_potsdam_2_12_r0_c0.png",
        "gt_path": "data/Potsdam/val_1024/masks/top_potsdam_2_12_r0_c0.png",
        "mamba_pred_path": "fig_results/potsdam/unetmamba_baseline_rgb_d4/top_potsdam_2_12_r0_c0.png",
        "mamba_ca_pred_path": "fig_results/potsdam/unetmamba_CA_rgb_d4/top_potsdam_2_12_r0_c0.png",
        "color_map": COLOR_MAP_ISPRS, "class_names": CLASS_NAMES_ISPRS, "gt_is_rgb": False
    },
    # { # 再添加一个 Potsdam 样本...
    #     "dataset_name": "Potsdam",
    #     "input_path": "data/Potsdam/val_1024/images/ANOTHER_POTSDAM_INPUT.png",
    #     "gt_path": "data/Potsdam/val_1024/masks/ANOTHER_POTSDAM_GT.png",
    #     "mamba_pred_path": "fig_results/potsdam/unetmamba_baseline_rgb_d4/ANOTHER_POTSDAM_PRED1.png",
    #     "mamba_ca_pred_path": "fig_results/potsdam/unetmamba_CA_rgb_d4/ANOTHER_POTSDAM_PRED2.png",
    #     "color_map": COLOR_MAP_ISPRS, "class_names": CLASS_NAMES_ISPRS, "gt_is_rgb": False
    # },
]

# 5. 要生成的候选图数量
NUM_FIGURES_TO_GENERATE = 3 # <-- 控制生成多少张完整的对比图

# 6. 输出图像设置
OUTPUT_DIR = "comparison_figure_options" # 指定输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
DPI = 300

# --- 辅助函数 (与之前版本相同) ---
def load_image(path):
    try:
        img = Image.open(path).convert('RGB')
        return np.array(img)
    except FileNotFoundError:
        print(f"错误：文件未找到 {path}")
        return None
    except Exception as e:
        print(f"加载图像时出错 {path}: {e}")
        return None

def load_mask(path):
    try:
        mask = Image.open(path)
        if mask.mode == 'L' or mask.mode == 'P':
             return np.array(mask)
        else:
             print(f"警告：掩码 {path} 模式为 {mask.mode}，尝试转换为 'L'。请检查结果是否正确。")
             mask = mask.convert('L')
             return np.array(mask)
    except FileNotFoundError:
        print(f"错误：文件未找到 {path}")
        return None
    except Exception as e:
        print(f"加载掩码时出错 {path}: {e}")
        return None

def colorize_mask(mask, color_map):
    if mask is None: return None
    bg_color = color_map.get(0, [0, 0, 0])
    rgb_mask = np.full((*mask.shape, 3), bg_color, dtype=np.uint8)
    unique_indices = np.unique(mask)
    for class_index, color in color_map.items():
        if class_index == 0: continue
        if class_index in unique_indices:
             rgb_mask[mask == class_index] = color
    defined_indices = list(color_map.keys())
    undefined_indices = unique_indices[~np.isin(unique_indices, defined_indices)]
    if len(undefined_indices) > 0:
        print(f"警告：掩码中包含未在 color_map 中定义的类别索引: {undefined_indices}。这些区域将显示为背景色（或黑色）。")
    return rgb_mask

# --- 主绘图逻辑 ---

# 预先获取 LoveDA 可用文件名列表
loveda_available_files = {}
for config_key, config in DATASET_CONFIG_RANDOM.items():
    input_dir = config["input_dir"]
    input_ext = config["input_ext"]
    if not os.path.isdir(input_dir):
        print(f"错误：LoveDA 输入目录不存在: {input_dir}")
        loveda_available_files[config_key] = []
        continue
    try:
        input_files = glob.glob(os.path.join(input_dir, f"*{input_ext}"))
        loveda_available_files[config_key] = [os.path.splitext(os.path.basename(f))[0] for f in input_files]
        if not loveda_available_files[config_key]:
             print(f"警告：在 {input_dir} 中未找到 {input_ext} 文件。")
    except Exception as e:
        print(f"错误：列出 LoveDA 文件时出错 {input_dir}: {e}")
        loveda_available_files[config_key] = []

# 分离手动样本按数据集名称
manual_samples_by_dataset = {}
valid_manual_samples = [s for s in MANUAL_SAMPLES if s and s.get("input_path") and 'path/to/' not in s.get("input_path", "")]
for sample in valid_manual_samples:
    d_name = sample.get("dataset_name", "Unknown").split('_')[0] # 获取基础数据集名
    if d_name not in manual_samples_by_dataset:
        manual_samples_by_dataset[d_name] = []
    manual_samples_by_dataset[d_name].append(sample)

if not manual_samples_by_dataset.get("Vaihingen"):
     print("警告：MANUAL_SAMPLES 中没有有效的 Vaihingen 条目。")
if not manual_samples_by_dataset.get("Potsdam"):
     print("警告：MANUAL_SAMPLES 中没有有效的 Potsdam 条目。")


print(f"\n准备生成 {NUM_FIGURES_TO_GENERATE} 张候选对比图...")

for fig_idx in range(NUM_FIGURES_TO_GENERATE):
    print(f"\n--- 生成候选图 {fig_idx + 1}/{NUM_FIGURES_TO_GENERATE} ---")
    current_figure_samples = []
    sample_selection_successful = True

    # 1. 选择 LoveDA Rural 样本
    if loveda_available_files.get("LoveDA_Rural"):
        filename_lr = random.choice(loveda_available_files["LoveDA_Rural"])
        config_lr = DATASET_CONFIG_RANDOM["LoveDA_Rural"]
        input_path = os.path.join(config_lr["input_dir"], f"{filename_lr}{config_lr['input_ext']}")
        gt_path = os.path.join(config_lr["gt_dir"], f"{filename_lr}{config_lr['gt_ext']}")
        mamba_pred_path = os.path.join(config_lr["mamba_pred_base_dir"], config_lr["sub_dir"], f"{filename_lr}.png")
        mamba_ca_pred_path = os.path.join(config_lr["mamba_ca_pred_base_dir"], config_lr["sub_dir"], f"{filename_lr}.png")
        current_figure_samples.append({
            "dataset_name": "LoveDA_Rural", "input_path": input_path, "gt_path": gt_path,
            "mamba_pred_path": mamba_pred_path, "mamba_ca_pred_path": mamba_ca_pred_path,
            "color_map": config_lr["color_map"], "class_names": config_lr["class_names"],
            "gt_is_rgb": config_lr.get("gt_is_rgb", False)
        })
        print(f"  选择 LoveDA Rural: {filename_lr}")
    else:
        print("  错误：无法为 LoveDA Rural 选择样本。")
        sample_selection_successful = False

    # 2. 选择 LoveDA Urban 样本
    if loveda_available_files.get("LoveDA_Urban"):
        filename_lu = random.choice(loveda_available_files["LoveDA_Urban"])
        config_lu = DATASET_CONFIG_RANDOM["LoveDA_Urban"]
        input_path = os.path.join(config_lu["input_dir"], f"{filename_lu}{config_lu['input_ext']}")
        gt_path = os.path.join(config_lu["gt_dir"], f"{filename_lu}{config_lu['gt_ext']}")
        mamba_pred_path = os.path.join(config_lu["mamba_pred_base_dir"], config_lu["sub_dir"], f"{filename_lu}.png")
        mamba_ca_pred_path = os.path.join(config_lu["mamba_ca_pred_base_dir"], config_lu["sub_dir"], f"{filename_lu}.png")
        current_figure_samples.append({
            "dataset_name": "LoveDA_Urban", "input_path": input_path, "gt_path": gt_path,
            "mamba_pred_path": mamba_pred_path, "mamba_ca_pred_path": mamba_ca_pred_path,
            "color_map": config_lu["color_map"], "class_names": config_lu["class_names"],
            "gt_is_rgb": config_lu.get("gt_is_rgb", False)
        })
        print(f"  选择 LoveDA Urban: {filename_lu}")
    else:
        print("  错误：无法为 LoveDA Urban 选择样本。")
        sample_selection_successful = False

    # 3. 选择 Vaihingen 样本
    if manual_samples_by_dataset.get("Vaihingen"):
        vai_sample = random.choice(manual_samples_by_dataset["Vaihingen"])
        current_figure_samples.append(vai_sample)
        print(f"  选择 Vaihingen: {os.path.basename(vai_sample['input_path'])}")
    else:
        print("  提示：未提供有效的 Vaihingen 手动样本，将跳过此行。")
        # 添加一个None占位符，或者在绘图时检查长度
        current_figure_samples.append(None)


    # 4. 选择 Potsdam 样本
    if manual_samples_by_dataset.get("Potsdam"):
        pot_sample = random.choice(manual_samples_by_dataset["Potsdam"])
        current_figure_samples.append(pot_sample)
        print(f"  选择 Potsdam: {os.path.basename(pot_sample['input_path'])}")
    else:
        print("  提示：未提供有效的 Potsdam 手动样本，将跳过此行。")
        current_figure_samples.append(None)

    # 过滤掉 None 占位符
    current_figure_samples = [s for s in current_figure_samples if s is not None]
    num_rows_in_figure = len(current_figure_samples)

    if num_rows_in_figure == 0:
        print(f"错误：无法为候选图 {fig_idx + 1} 选择任何有效样本，跳过生成。")
        continue

    # --- 创建并绘制当前候选图 ---
    fig, axes = plt.subplots(num_rows_in_figure, 4, figsize=(12, 3 * num_rows_in_figure + 1))
    if num_rows_in_figure == 1:
        axes = np.array([axes]) # 强制二维

    print(f"  开始绘制候选图 {fig_idx + 1} ({num_rows_in_figure} 行)...")
    plot_successful_this_figure = True
    for i, sample_info in enumerate(current_figure_samples):
        # ... (加载和检查逻辑与上一版本类似) ...
        input_img = load_image(sample_info["input_path"])
        mamba_pred_img = load_image(sample_info["mamba_pred_path"])
        mamba_ca_pred_img = load_image(sample_info["mamba_ca_pred_path"])

        gt_colored = None
        gt_load_success = False
        if sample_info.get("gt_is_rgb", False):
            gt_colored = load_image(sample_info["gt_path"])
            if gt_colored is not None: gt_load_success = True
            else: print(f"    错误：行 {i+1} ({sample_info['dataset_name']}) 彩色 GT 图像加载失败。")
        else:
            gt_mask = load_mask(sample_info["gt_path"])
            if gt_mask is not None:
                gt_colored = colorize_mask(gt_mask, sample_info["color_map"])
                if gt_colored is not None: gt_load_success = True
                else: print(f"    错误：行 {i+1} ({sample_info['dataset_name']}) GT 掩码着色失败。")
            else: print(f"    错误：行 {i+1} ({sample_info['dataset_name']}) GT 索引掩码加载失败。")

        load_success = True
        display_name = sample_info['dataset_name']
        if input_img is None: load_success = False
        if not gt_load_success: load_success = False
        if mamba_pred_img is None: print(f"    警告：行 {i+1} ({display_name}) 的 Mamba 预测图加载失败。")
        if mamba_ca_pred_img is None: print(f"    警告：行 {i+1} ({display_name}) 的 Mamba-CA 预测图加载失败。")

        if not load_success:
             print(f"    错误：行 {i+1} (输入或GT加载失败)，此图可能不完整。")
             # 不跳过整张图，但标记此行绘制失败
             plot_successful_this_figure = False
             for ax_idx in range(4): axes[i, ax_idx].set_visible(False)
             continue # 跳到下一行

        # 绘制
        axes[i, 0].imshow(input_img)
        axes[i, 0].set_title(f"{display_name}\nInput", fontsize=10)
        if gt_colored is not None: axes[i, 1].imshow(gt_colored)
        else: axes[i, 1].text(0.5, 0.5, 'Ground Truth\nLoad/Color Failed', ha='center', va='center', fontsize=9, color='red')
        axes[i, 1].set_title("Ground Truth", fontsize=10)
        if mamba_pred_img is not None: axes[i, 2].imshow(mamba_pred_img)
        else: axes[i, 2].text(0.5, 0.5, 'Mamba Pred\nLoad Failed', ha='center', va='center', fontsize=9, color='red')
        axes[i, 2].set_title("UNetMamba", fontsize=10)
        if mamba_ca_pred_img is not None: axes[i, 3].imshow(mamba_ca_pred_img)
        else: axes[i, 3].text(0.5, 0.5, 'Mamba-CA Pred\nLoad Failed', ha='center', va='center', fontsize=9, color='red')
        axes[i, 3].set_title("UNetMamba-CA", fontsize=10)

        for ax in axes[i]:
            ax.axis('off')
    # --- 单张候选图绘制结束 ---

    if plot_successful_this_figure: # 只有在所有行都基本成功时才保存
        output_filename_figure = os.path.join(OUTPUT_DIR, f"figure_4_3_option_{fig_idx + 1}.png")
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.02, hspace=0.4, wspace=0.05) # 调整间距
        try:
            plt.savefig(output_filename_figure, dpi=DPI, bbox_inches='tight')
            print(f"  候选图 {fig_idx + 1} 已保存至: {output_filename_figure}")
        except Exception as e:
            print(f"  保存候选图 {fig_idx + 1} 时出错: {e}")
    else:
         print(f"  候选图 {fig_idx + 1} 因加载错误未保存。")

    plt.close(fig) # 关闭当前图形

print(f"\n脚本执行完毕，共尝试生成 {NUM_FIGURES_TO_GENERATE} 张候选图。")

