# visualize_transforms.py
import os
import random
import numbers
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance
import albumentations as albu

# --- 从 transform.txt 复制过来的类定义 ---
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class RandomCrop(object):
    """
    Take a random crop from the image.
    First the image or crop size may need to be adjusted if the incoming image
    is too small...
    If the image is smaller than the crop, then:
         the image is padded up to the size of the crop
         unless 'nopad', in which case the crop size is shrunk to fit the image
    A random crop is taken such that the crop fits within the image.
    If a centroid is passed in, the crop must intersect the centroid.
    """
    def __init__(self, size=512, ignore_index=12, nopad=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ignore_index = ignore_index
        self.nopad = nopad
        self.pad_color = (0, 0, 0) # Use black for padding color

    def __call__(self, img, mask, centroid=None):
        assert img.size == mask.size
        w, h = img.size
        # ASSUME H, W -> PIL uses W, H
        th, tw = self.size # Target height, target width

        if w == tw and h == th:
            return img, mask

        if self.nopad:
            if th > h or tw > w:
                # Instead of padding, adjust crop size to the shorter edge of image.
                shorter_side = min(w, h)
                th, tw = shorter_side, shorter_side
        else:
            # Check if we need to pad img to fit for crop_size.
            if th > h:
                pad_h = (th - h) // 2
                pad_top = pad_h
                pad_bottom = th - h - pad_top
            else:
                pad_h = 0
                pad_top, pad_bottom = 0, 0

            if tw > w:
                pad_w = (tw - w) // 2
                pad_left = pad_w
                pad_right = tw - w - pad_left
            else:
                pad_w = 0
                pad_left, pad_right = 0, 0

            border = (pad_left, pad_top, pad_right, pad_bottom) # PIL expects left, top, right, bottom
            if pad_h or pad_w:
                # print(f"Padding image from {w}x{h} to {tw}x{th} with border {border}")
                img = ImageOps.expand(img, border=border, fill=self.pad_color)
                mask = ImageOps.expand(mask, border=border, fill=self.ignore_index)
                w, h = img.size # Update dimensions after padding

        if centroid is not None:
            # Need to ensure that centroid is covered by crop and that crop
            # sits fully within the image
            c_x, c_y = centroid
            max_x = w - tw
            max_y = h - th
            # Ensure crop coordinates are valid
            x1_min = max(0, c_x - tw + 1) # +1 because crop includes x1
            x1_max = min(max_x, c_x)
            y1_min = max(0, c_y - th + 1)
            y1_max = min(max_y, c_y)

            if x1_min > x1_max or y1_min > y1_max:
                 # Fallback to random crop if centroid constraints are impossible
                 x1 = random.randint(0, max_x) if max_x > 0 else 0
                 y1 = random.randint(0, max_y) if max_y > 0 else 0
            else:
                 x1 = random.randint(x1_min, x1_max)
                 y1 = random.randint(y1_min, y1_max)
        else:
            if w == tw:
                x1 = 0
            else:
                x1 = random.randint(0, w - tw)
            if h == th:
                y1 = 0
            else:
                y1 = random.randint(0, h - th)
        # print(f"Cropping at ({x1}, {y1}) with size ({tw}, {th}) from image size ({w}, {h})")
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

class RandomScale(object):
    def __init__(self, scale_list=[0.75, 1.0, 1.25], mode='value'):
        self.scale_list = scale_list
        self.mode = mode

    def __call__(self, img, mask):
        # Use img.size which is (width, height) for PIL
        ow, oh = img.size
        scale_amt = 1.0
        if self.mode == 'value':
            scale_amt = np.random.choice(self.scale_list, 1)[0] # Get the scalar value
        elif self.mode == 'range':
            scale_amt = random.uniform(self.scale_list[0], self.scale_list[-1])

        # Calculate new width and height
        w = int(scale_amt * ow)
        h = int(scale_amt * oh)

        # Ensure dimensions are at least 1
        w = max(1, w)
        h = max(1, h)

        # print(f"Scaling from ({ow}, {oh}) to ({w}, {h}) with scale {scale_amt}")
        return img.resize((w, h), Image.BICUBIC), mask.resize((w, h), Image.NEAREST)

class SmartCropV1(object):
    def __init__(self, crop_size=512,
                 max_ratio=0.75,
                 ignore_index=255, nopad=True): # Use ignore_index=255 based on dataset file
        self.crop_size = crop_size
        self.max_ratio = max_ratio
        self.ignore_index = ignore_index
        self.crop = RandomCrop(crop_size, ignore_index=ignore_index, nopad=nopad)

    def __call__(self, img, mask):
        assert img.size == mask.size
        count = 0
        while True:
            img_crop, mask_crop = self.crop(img.copy(), mask.copy())
            count += 1
            mask_np = np.array(mask_crop)
            labels, cnt = np.unique(mask_np, return_counts=True)
            # Filter out ignore index before calculating ratio
            valid_indices = (labels != self.ignore_index)
            cnt = cnt[valid_indices]
            labels = labels[valid_indices] # Keep corresponding labels

            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.max_ratio:
                # print(f"SmartCropV1 found valid crop after {count} tries.")
                break
            if count > 10: # Safety break
                # print(f"SmartCropV1 using fallback crop after {count} tries.")
                # If failed after 10 tries, just return the last crop
                # Or maybe take a center crop as fallback? For now, return last random.
                break

        return img_crop, mask_crop
# --- 定义参数 ---
# 注意：路径相对于你在UNetMamba目录下运行脚本
SAMPLE_IMG_PATH = "data/LoveDA/Train/Rural/images_png/0.png"
SAMPLE_MASK_PATH = "data/LoveDA/Train/Rural/masks_png/0.png" # 使用原始mask
OUTPUT_DIR = "transform_visualizations"

# 从 loveda_dataset (3).txt 和你的确认中得到的参数
IMG_RESIZE_DIM = None # 没有 Resize 步骤
NORM_MEAN = (0.485, 0.456, 0.406) # ImageNet 默认
NORM_STD = (0.229, 0.224, 0.225) # ImageNet 默认
CROP_SIZE = 512
# LoveDA 的 ignore_index, 根据 loveda_dataset.py 中 SmartCropV1 调用确定
# 注意：原始mask可能需要先经过 convert_label 处理才能得到 ignore_index=255 的效果
# 但为了可视化原始增强效果，我们暂时假设 SmartCropV1 用的 ignore index 是 255
IGNORE_INDEX_FOR_CROP = 255
RANDOM_SCALE_LIST = [0.75, 1.0, 1.25, 1.5]
SMARTCROP_MAX_RATIO = 0.75

# --- 创建输出目录 ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"可视化图像将保存在: {os.path.abspath(OUTPUT_DIR)}")

# --- 定义可视化辅助函数 ---
def visualize_step(img_before, img_after, title_before, title_after, suptitle, save_path):
    """显示并保存变换前后的图像对比"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_before)
    axes[0].set_title(title_before)
    axes[0].axis('off')

    axes[1].imshow(img_after)
    axes[1].set_title(title_after)
    axes[1].axis('off')

    fig.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局防止标题重叠
    plt.savefig(save_path)
    plt.close(fig) # 关闭图形，防止累积显示
    print(f"已保存: {os.path.basename(save_path)}")

def visualize_step_mask(img_before, mask_before, img_after, mask_after,
                        title_img_before, title_mask_before,
                        title_img_after, title_mask_after,
                        suptitle, save_path):
    """显示并保存变换前后的图像和掩码对比"""
    # Ensure masks are displayable (e.g., single channel or RGB)
    if len(mask_before.shape) == 3 and mask_before.shape[2] == 1:
        mask_before = mask_before.squeeze(axis=2)
    if len(mask_after.shape) == 3 and mask_after.shape[2] == 1:
        mask_after = mask_after.squeeze(axis=2)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(img_before)
    axes[0, 0].set_title(title_img_before)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(mask_before, cmap='gray') # Display mask in grayscale
    axes[0, 1].set_title(title_mask_before)
    axes[0, 1].axis('off')

    axes[1, 0].imshow(img_after)
    axes[1, 0].set_title(title_img_after)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mask_after, cmap='gray') # Display mask in grayscale
    axes[1, 1].set_title(title_mask_after)
    axes[1, 1].axis('off')

    fig.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"已保存: {os.path.basename(save_path)}")

# --- 加载样本数据 (使用 PIL, 因为自定义变换是基于 PIL 的) ---
try:
    img_pil = Image.open(SAMPLE_IMG_PATH).convert('RGB')
    # 使用原始 masks_png 中的 mask, 因为 convert_label 是后续步骤
    # 注意：原始 masks_png 的标签值可能与最终训练用的不同
    mask_pil = Image.open(SAMPLE_MASK_PATH).convert('L') # 确保是单通道灰度图
except FileNotFoundError as e:
    print(f"错误：无法加载文件 {e.filename}。请确保文件路径正确并且脚本在 UNetMamba 目录下运行。")
    exit()

print(f"加载原始图像尺寸: {img_pil.size}") # PIL size is (width, height)
print(f"加载原始掩码尺寸: {mask_pil.size}")

# 保存原始图像/掩码
plt.figure(figsize=(5, 5))
plt.imshow(img_pil)
plt.title("Original Image")
plt.axis('off')
plt.savefig(os.path.join(OUTPUT_DIR, "0_original_image.png"))
plt.close()

plt.figure(figsize=(5, 5))
plt.imshow(mask_pil, cmap='gray')
plt.title("Original Mask (Raw values)")
plt.axis('off')
plt.savefig(os.path.join(OUTPUT_DIR, "0_original_mask_raw.png"))
plt.close()


# --- 可视化【数据增强】步骤 (独立应用) ---
# 根据 loveda_dataset.py 中的 train_aug 顺序，增强通常在 Normalize 之前
# 我们将对原始 PIL 图像应用这些基于 PIL 的变换
# 对于 Albumentations 变换，我们将 PIL 转换为 NumPy

print("\n--- 开始可视化数据增强步骤 ---")
aug_counter = 1
num_examples = 3 # 为随机增强生成多个例子

# --- 增强 A: RandomScale (来自 transform.py) ---
print(f"\nAugmentation {aug_counter}: RandomScale")
scale_transform = RandomScale(scale_list=RANDOM_SCALE_LIST, mode='value')
for i in range(num_examples):
    # 每次都从原始图像开始
    img_scaled, mask_scaled = scale_transform(img_pil.copy(), mask_pil.copy())
    visualize_step_mask(np.array(img_pil), np.array(mask_pil), # 转为Numpy显示
                        np.array(img_scaled), np.array(mask_scaled),
                        "Original Image", "Original Mask",
                        f"After RandomScale (Example {i+1}) WxH: {img_scaled.size}",
                        f"After RandomScale Mask WxH: {mask_scaled.size}",
                        f"Augmentation {aug_counter}: RandomScale Example {i+1}",
                        os.path.join(OUTPUT_DIR, f"aug_{aug_counter}_random_scale_ex{i+1}.png"))
aug_counter += 1

# --- 增强 B: SmartCropV1 (来自 transform.py) ---
# SmartCropV1 在内部使用了 RandomCrop
print(f"\nAugmentation {aug_counter}: SmartCropV1 (Output size {CROP_SIZE}x{CROP_SIZE})")
# 注意 nopad=True 在 RandomCrop 中使用
crop_transform = SmartCropV1(crop_size=CROP_SIZE, max_ratio=SMARTCROP_MAX_RATIO,
                             ignore_index=IGNORE_INDEX_FOR_CROP, nopad=True)
for i in range(num_examples):
     # 每次都从原始图像开始
    img_cropped, mask_cropped = crop_transform(img_pil.copy(), mask_pil.copy())
    visualize_step_mask(np.array(img_pil), np.array(mask_pil), # 转为Numpy显示
                        np.array(img_cropped), np.array(mask_cropped),
                        "Original Image", "Original Mask",
                        f"After SmartCropV1 (Example {i+1})",
                        f"After SmartCropV1 Mask ({CROP_SIZE}x{CROP_SIZE})",
                        f"Augmentation {aug_counter}: SmartCropV1 Example {i+1}",
                        os.path.join(OUTPUT_DIR, f"aug_{aug_counter}_smart_crop_ex{i+1}.png"))
aug_counter += 1


# --- 接下来是 Albumentations 变换，需要将 PIL 转为 NumPy ---
img_np = np.array(img_pil)
mask_np = np.array(mask_pil)


# --- 增强 C: HorizontalFlip (Albumentations) ---
print(f"\nAugmentation {aug_counter}: HorizontalFlip")
# p=1.0 强制翻转
hflip_transform = albu.HorizontalFlip(p=1.0)
transformed = hflip_transform(image=img_np.copy(), mask=mask_np.copy())
img_hflipped = transformed['image']
mask_hflipped = transformed['mask']
visualize_step_mask(img_np, mask_np,
                    img_hflipped, mask_hflipped,
                    "Original Image (NumPy)", "Original Mask (NumPy)",
                    "After HorizontalFlip", "After HorizontalFlip Mask",
                    f"Augmentation {aug_counter}: HorizontalFlip",
                    os.path.join(OUTPUT_DIR, f"aug_{aug_counter}_horizontal_flip.png"))
aug_counter += 1


# --- 增强 D: VerticalFlip (Albumentations) ---
print(f"\nAugmentation {aug_counter}: VerticalFlip")
# p=1.0 强制翻转
vflip_transform = albu.VerticalFlip(p=1.0)
transformed = vflip_transform(image=img_np.copy(), mask=mask_np.copy())
img_vflipped = transformed['image']
mask_vflipped = transformed['mask']
visualize_step_mask(img_np, mask_np,
                    img_vflipped, mask_vflipped,
                    "Original Image (NumPy)", "Original Mask (NumPy)",
                    "After VerticalFlip", "After VerticalFlip Mask",
                    f"Augmentation {aug_counter}: VerticalFlip",
                    os.path.join(OUTPUT_DIR, f"aug_{aug_counter}_vertical_flip.png"))
aug_counter += 1


# --- 增强 E: RandomBrightnessContrast (Albumentations) ---
# 这个只影响图像，不影响掩码
print(f"\nAugmentation {aug_counter}: RandomBrightnessContrast")
# p=1.0 强制应用
brightcont_transform = albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0)
for i in range(num_examples):
    transformed = brightcont_transform(image=img_np.copy())
    img_bc = transformed['image']
    visualize_step(img_np, img_bc,
                   "Original Image (NumPy)", f"After RandomBrightnessContrast (Example {i+1})",
                   f"Augmentation {aug_counter}: RandomBrightnessContrast Example {i+1}",
                   os.path.join(OUTPUT_DIR, f"aug_{aug_counter}_rand_brightcont_ex{i+1}.png"))
aug_counter += 1


# --- 预处理步骤: Normalize (Albumentations) ---
# 注意：在实际的 train_aug 中，Normalize 是在上述所有增强之后应用的
# 但为了独立展示其效果，我们对原始 NumPy 图像应用它
print(f"\nPreprocessing Step (applied last in train_aug): Normalize")
normalize_transform = albu.Normalize(mean=NORM_MEAN, std=NORM_STD, max_pixel_value=255.0)
normalized_data = normalize_transform(image=img_np.copy())
normalized_img = normalized_data['image']

# 可视化归一化后的图像 (重新缩放到 0-255)
vis_normalized_img = cv2.normalize(normalized_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

visualize_step(img_np, vis_normalized_img,
               "Original Image (NumPy)", "After Normalize (Visualization)",
               f"Preprocessing Step: Normalize",
               os.path.join(OUTPUT_DIR, f"preprocess_normalize.png"))


print("\n--- 可视化完成 ---")