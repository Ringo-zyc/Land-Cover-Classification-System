# unetmamba_model/datasets/potsdam_dataset.py (Modified for Patches)
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as albu
from PIL import Image
import glob
import random

# Keep CLASSES, PALETTE, IGNORE_INDEX definition
CLASSES = ('Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car', 'Clutter/background')
PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
IGNORE_INDEX = 255

# --- Online Augmentations for Patches ---
def get_potsdam_patch_train_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.25),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
    return albu.Compose(train_transform)

def get_potsdam_patch_val_transform():
    val_transform = [
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
    return albu.Compose(val_transform)

# --- Dataset Class Loading Patches ---
class PotsdamPatchesDataset(Dataset):
    """Loads pre-split Potsdam patches."""

    def __init__(self, data_root, mode='train',
                 img_suffix='.png', mask_suffix='.png', # Match output extension from splitter
                 transform=None):
        """
        Args:
            data_root (string): Directory containing 'images' and 'masks' subfolders with patches.
                                e.g., 'data/Potsdam/train_1024' or 'data/Potsdam/val_1024'
            mode (string): 'train' or 'val'.
            img_suffix (string): Extension of image patch files.
            mask_suffix (string): Extension of mask patch files.
            transform (callable, optional): Albumentations transform to apply.
        """
        self.data_root = data_root
        self.img_dir = osp.join(data_root, 'images')
        self.mask_dir = osp.join(data_root, 'masks')
        self.mode = mode
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix

        if not osp.isdir(self.img_dir): raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not osp.isdir(self.mask_dir): raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        if transform is None:
            if mode == 'train': self.transform = get_potsdam_patch_train_transform()
            else: self.transform = get_potsdam_patch_val_transform()
        else: self.transform = transform

        self.file_list = self._get_file_list()
        if not self.file_list: raise FileNotFoundError(f"No image/mask pairs found in {self.img_dir} / {self.mask_dir}")

        print(f"Initialized PotsdamPatchesDataset from '{data_root}' in '{mode}' mode with {len(self.file_list)} patches.")

    def _get_file_list(self):
        file_list = []
        image_files = sorted(glob.glob(osp.join(self.img_dir, f"*{self.img_suffix}")))
        for img_path in image_files:
            # Use os.path.splitext to handle potential double extensions if needed
            base_name, _ = osp.splitext(osp.basename(img_path))
            # Construct mask path assuming the base name (without ext) matches
            mask_path = osp.join(self.mask_dir, f"{base_name}{self.mask_suffix}")
            if osp.exists(mask_path):
                file_list.append({"img": img_path, "mask": mask_path, "id": base_name})
            # else: # Reduce noise, verification happens in splitter
            #     print(f"Warning: Mask not found for {img_path}")
        return file_list


    def __getitem__(self, index):
        data = self.file_list[index]
        img_path = data['img']
        mask_path = data['mask']
        img_id = data['id']

        try:
            # Load image patch using OpenCV (compatible with Albumentations)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None: raise IOError(f"cv2.imread failed for image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB

            # Load mask patch (ensure grayscale)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: raise IOError(f"cv2.imread failed for mask: {mask_path}")

            mask = mask.astype(np.int64) # Ensure correct type for PyTorch loss

        except Exception as e:
            print(f"Error loading patch {img_id} (Index {index}): {e}")
            patch_size = 1024 # Default or get from config if possible
            return {'img': torch.zeros(3, patch_size, patch_size),
                    'gt_semantic_seg': torch.full((patch_size, patch_size), IGNORE_INDEX, dtype=torch.long),
                    'img_id': img_id + "_load_error"}

        # Apply online transformations (flips, brightness/contrast, normalize)
        if self.transform:
            try:
                augmented = self.transform(image=img, mask=mask)
                img_transformed = augmented['image']
                mask_transformed = augmented['mask']
            except Exception as e:
                 print(f"Error applying transform to {img_id}: {e}. Returning normalized original.")
                 norm_only = get_potsdam_patch_val_transform() # Fallback
                 augmented = norm_only(image=img, mask=mask)
                 img_transformed = augmented['image']
                 mask_transformed = augmented['mask']
        else:
             img_transformed, mask_transformed = img, mask

        # Convert final NumPy arrays to PyTorch tensors
        img_tensor = torch.from_numpy(img_transformed).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_transformed).long()

        results = {'img': img_tensor, 'gt_semantic_seg': mask_tensor, 'img_id': img_id}
        return results

    def __len__(self):
        return len(self.file_list)