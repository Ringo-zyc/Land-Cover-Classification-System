# check_mask_values.py
import os
import glob
import numpy as np
import time

try:
    import tifffile
except ImportError:
    print("错误：需要安装 tifffile 库才能读取 .tif 文件。")
    print("请运行: pip install tifffile")
    exit()

# --- 配置 ---
# !! 修改为你存放原始 Potsdam 标签文件的目录 !!
MASK_DIR = "data/Potsdam/5_Labels_for_participants"
# !! 确认你的标签文件名后缀 !!
MASK_SUFFIX = "_label.tif"

# --- 脚本逻辑 ---
def find_unique_mask_values(mask_dir, mask_suffix):
    """查找目录下所有掩码文件的唯一像素值"""
    if not os.path.isdir(mask_dir):
        print(f"错误：找不到目录 '{mask_dir}'")
        return None

    mask_files = glob.glob(os.path.join(mask_dir, f"*{mask_suffix}"))

    if not mask_files:
        print(f"错误：在 '{mask_dir}' 中找不到任何后缀为 '{mask_suffix}' 的文件。")
        return None

    print(f"开始检查目录 '{mask_dir}' 中的 {len(mask_files)} 个掩码文件...")

    all_unique_values = set()
    files_processed = 0
    files_with_errors = 0
    start_time = time.time()

    for mask_path in mask_files:
        try:
            mask_array = tifffile.imread(mask_path)
            if mask_array is None:
                 print(f"警告：无法读取文件（可能为空或损坏）：{os.path.basename(mask_path)}")
                 files_with_errors += 1
                 continue

            unique_values_in_file = np.unique(mask_array)
            all_unique_values.update(unique_values_in_file)
            files_processed += 1
            # 打印进度，避免长时间无响应的感觉
            if files_processed % 10 == 0 or files_processed == len(mask_files):
                 print(f"  已处理 {files_processed}/{len(mask_files)} 个文件...", end='\r')

        except Exception as e:
            print(f"\n错误：处理文件 {os.path.basename(mask_path)} 时出错: {e}")
            files_with_errors += 1

    end_time = time.time()
    print(f"\n检查完成。用时 {end_time - start_time:.2f} 秒。")
    if files_with_errors > 0:
        print(f"处理过程中遇到 {files_with_errors} 个错误。")

    return all_unique_values

if __name__ == "__main__":
    unique_values = find_unique_mask_values(MASK_DIR, MASK_SUFFIX)

    if unique_values is not None:
        sorted_values = sorted(list(unique_values))
        print("\n在所有检查的掩码文件中找到的唯一像素值:")
        print(sorted_values)
        print("\n请根据 ISPRS Potsdam 数据集的说明文档，确认这些值代表的含义。")
        print(f"模型期望的类别索引范围是 [0, {6-1}] (共 {6} 个类), 忽略索引是 {255}。")
        # 基础检查
        expected_range_max = 6 - 1 # 类别索引最大值 (0-5)
        ignore_index_val = 255
        has_unexpected = False
        for val in sorted_values:
            if not (0 <= val <= expected_range_max or val == ignore_index_val):
                print(f"*** 警告: 发现值 '{val}' 不在期望的类别范围 [0, {expected_range_max}] 内，也不是忽略值 {ignore_index_val}！ ***")
                has_unexpected = True
        if not has_unexpected:
            print("检查的值都在预期范围内（类别索引 0-5 或忽略值 255）。")
        else:
             print("需要根据实际含义在数据加载或预处理脚本中添加标签映射逻辑。")