import torch
import numpy as np
import albumentations as albu
from PIL import Image

# 定义颜色表（根据你的数据集调整）
PALETTE = [
    [255, 255, 255],  # 背景
    [255, 0, 0],      # 建筑
    [255, 255, 0],    # 道路
    [0, 0, 255],      # 水
    [159, 129, 183],  # 贫瘠
    [0, 255, 0],      # 森林
    [255, 195, 128]   # 农业
]

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """将输入图片预处理为模型所需的张量格式"""
    image = image.resize((1024, 1024))  # 调整到模型输入大小
    image = np.array(image)
    transform = albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    image = transform(image=image)["image"]
    image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)  # [1, C, H, W]
    if torch.cuda.is_available():
        image = image.cuda()
    return image

def postprocess_output(output: torch.Tensor) -> np.ndarray:
    """将模型输出转换为分割图像"""
    output = torch.softmax(output, dim=1)  # 转换为概率分布
    segmented = output.argmax(dim=1).squeeze().cpu().numpy()
    h, w = segmented.shape
    segmented_image = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(PALETTE):
        segmented_image[segmented == i] = color
    return segmented_image