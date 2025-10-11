# unetmamba_model/datasets/vaihingen_dataset.py (Corrected - Removed module-level instantiation)
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt # Keep imports if visualization functions are used elsewhere
import albumentations as albu
import matplotlib.patches as mpatches # Keep imports if visualization functions are used elsewhere
from PIL import Image, ImageOps # Keep PIL if load_img_and_mask uses it
import random
try:
    # Try importing custom transforms relative to this file's location
    from .transform import Compose, RandomScale, SmartCropV1
except ImportError:
    try:
        # Fallback if transform.py is in the parent directory or sys.path
        from transform import Compose, RandomScale, SmartCropV1
    except ImportError:
        print("CRITICAL ERROR: Could not import transform classes (Compose, RandomScale, SmartCropV1).")
        print("Ensure transform.py is accessible from vaihingen_dataset.py")
        # Define dummy classes to allow script load, but transforms will fail
        class Compose: pass
        class RandomScale: pass
        class SmartCropV1: pass
        # Or raise the error:
        # raise

# ISPRS Vaihingen Dataset Info (6 Classes)
# Using names from user's provided script
CLASSES = ('ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter')
# Using palette from user's provided script
PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 204, 0], [255, 0, 0]] # Car Yellow adjusted? Check consistency
# IGNORE_INDEX should match the value used in SmartCropV1 and Loss
# User config has ignore_index = len(CLASSES) which is 6
# Let's define it here for consistency if needed by the class, or rely on config
IGNORE_INDEX = len(CLASSES) # 6, based on user's config snippet

# --- Augmentation Functions (from user's provided script) ---
# These will be passed from the config file now

# Note: These transforms expect PIL Image input based on train_aug implementation
def get_training_transform():
    train_transform = [
        albu.RandomRotate90(p=0.5),
        albu.Normalize() # Defaults to ImageNet mean/std
    ]
    return albu.Compose(train_transform)

def train_aug(img, mask):
    # This function expects PIL Images as input
    # It uses custom Compose, RandomScale, SmartCropV1
    image_size = 1024 # Assuming base size for crop calculation
    crop_size = int(512*float(image_size/1024)) # Should be 512 if image_size=1024

    # Check if custom transform classes were loaded
    if 'Compose' not in globals() or 'RandomScale' not in globals() or 'SmartCropV1' not in globals():
        print("Warning: Custom transforms (Compose, RandomScale, SmartCropV1) not loaded. Skipping scale/crop.")
        img_cropped, mask_cropped = img, mask # Pass through if transforms missing
    else:
        # Apply custom scale and crop
        crop_aug = Compose([RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode='value'),
                            SmartCropV1(crop_size=crop_size, max_ratio=0.75,
                                        ignore_index=IGNORE_INDEX, nopad=False)]) # Use IGNORE_INDEX
        try:
            img_cropped, mask_cropped = crop_aug(img, mask)
        except Exception as e:
            print(f"Error during custom Compose/Scale/Crop: {e}. Using original image/mask.")
            img_cropped, mask_cropped = img, mask

    # Convert to NumPy for Albumentations
    img_np = np.array(img_cropped)
    mask_np = np.array(mask_cropped)

    # Apply Albumentations transforms (Rotate, Normalize)
    aug = get_training_transform()(image=img_np.copy(), mask=mask_np.copy())
    img_processed = aug['image']
    mask_processed = aug['mask']
    return img_processed, mask_processed


def get_val_transform():
    val_transform = [
        albu.Normalize() # Defaults to ImageNet mean/std
    ]
    return albu.Compose(val_transform)

def val_aug(img, mask):
    # This function expects PIL Images as input
    img_np = np.array(img)
    mask_np = np.array(mask)
    aug = get_val_transform()(image=img_np.copy(), mask=mask_np.copy())
    img_processed = aug['image']
    mask_processed = aug['mask']
    return img_processed, mask_processed


# --- Dataset Class Definition ---
# Using the class name from user's provided script
class VaihingenDataset(Dataset):
    # Keep the __init__, __getitem__, __len__, get_img_ids, load_img_and_mask, load_mosaic_img_and_mask
    # methods exactly as provided by the user in the previous turn.
    # Ensure load_img_and_mask returns PIL Images as expected by train_aug/val_aug.
    # Ensure get_img_ids correctly finds files based on data_root structure.

    def __init__(self, data_root='data/vaihingen/test_1024', mode='val', img_dir='images', mask_dir='masks',
                 img_suffix='.tif', mask_suffix='.png', transform=val_aug, mosaic_ratio=0.0,
                 img_size=(1024, 1024)): # Using ORIGIN_IMG_SIZE default from user script
        self.data_root = data_root
        # Construct absolute paths or ensure relative paths are handled correctly by caller
        self.img_dir_abs = osp.abspath(osp.join(data_root, img_dir)) # Use absolute path for checking
        self.mask_dir_abs = osp.abspath(osp.join(data_root, mask_dir)) # Use absolute path for checking
        self.img_dir_rel = img_dir # Store relative for joining later
        self.mask_dir_rel = mask_dir # Store relative for joining later

        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform # Will be passed from config
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio if mode == 'train' else 0.0 # Apply mosaic only in train mode
        self.img_size = img_size # Store original/target size if needed by mosaic

        # Check directories exist using absolute paths
        if not osp.isdir(self.img_dir_abs): raise FileNotFoundError(f"Image directory not found: {self.img_dir_abs}")
        if not osp.isdir(self.mask_dir_abs): raise FileNotFoundError(f"Mask directory not found: {self.mask_dir_abs}")

        self.img_ids = self.get_img_ids(self.data_root, self.img_dir_rel, self.mask_dir_rel) # Use relative for scanning
        if not self.img_ids: raise FileNotFoundError(f"No image/mask pairs found in {self.img_dir_abs} / {self.mask_dir_abs}")

        print(f"Initialized VaihingenDataset from '{data_root}' in '{mode}' mode with {len(self.img_ids)} samples.")


    def __getitem__(self, index):
        p_ratio = random.random()
        # Load image/mask (mosaic or single) - returns PIL Images
        if p_ratio < self.mosaic_ratio and self.mode == 'train':
            img, mask = self.load_mosaic_img_and_mask(index)
        else:
            img, mask = self.load_img_and_mask(index)

        # Apply transformations (train_aug or val_aug) - returns NumPy arrays
        if self.transform:
            try:
                img_transformed, mask_transformed = self.transform(img, mask)
            except Exception as e:
                 img_id = self.img_ids[index]
                 print(f"Error applying transform to {img_id}: {e}. Returning untransformed.")
                 # Convert PIL to NumPy and correct type as fallback
                 img_transformed = np.array(img)
                 mask_transformed = np.array(mask).astype(np.int64)
                 # Basic channel check/permute if needed before tensor conversion
                 if img_transformed.ndim == 3 and img_transformed.shape[-1] == 3:
                      pass # Already HWC
                 else: # Handle unexpected shape or return error tensor
                      patch_size_h, patch_size_w = self.img_size
                      return {'img': torch.zeros(3, patch_size_h, patch_size_w),
                              'gt_semantic_seg': torch.full((patch_size_h, patch_size_w), IGNORE_INDEX, dtype=torch.long),
                              'img_id': img_id + "_transform_error"}

        else: # Should not happen if config passes transform
             img_transformed = np.array(img)
             mask_transformed = np.array(mask).astype(np.int64)


        # Convert NumPy arrays to PyTorch tensors
        # Ensure img_transformed is HWC before permute
        if img_transformed.ndim != 3 or img_transformed.shape[-1] != 3: # Check dimensions strictly
             img_id = self.img_ids[index] # Get img_id here
             print(f"Warning: Transformed image shape is not HWC: {img_transformed.shape} for {img_id}")
             # Attempt to fix or return error tensor
             patch_size_h, patch_size_w = self.img_size
             return {'img': torch.zeros(3, patch_size_h, patch_size_w),
                     'gt_semantic_seg': torch.full((patch_size_h, patch_size_w), IGNORE_INDEX, dtype=torch.long),
                     'img_id': img_id + "_shape_error"}

        img_tensor = torch.from_numpy(img_transformed).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_transformed).long()

        img_id = self.img_ids[index]
        results = dict(img_id=img_id, img=img_tensor, gt_semantic_seg=mask_tensor)
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        # This function needs to correctly find files based on data_root structure
        # Assuming data_root points to train_1024 or val_1024 which contain img_dir and mask_dir
        img_folder_path = osp.join(data_root, img_dir)
        mask_folder_path = osp.join(data_root, mask_dir)
        if not osp.isdir(img_folder_path) or not osp.isdir(mask_folder_path):
             print(f"Warning: Cannot find img_dir '{img_dir}' or mask_dir '{mask_dir}' inside data_root '{data_root}'")
             return []

        # Use mask filenames to get IDs, assuming they match image IDs
        mask_filename_list = os.listdir(mask_folder_path)
        # Filter by mask_suffix and remove suffix to get ID
        img_ids = [osp.splitext(id)[0] for id in mask_filename_list if id.endswith(self.mask_suffix)]

        # Optional: Verify corresponding image file exists
        verified_ids = []
        for img_id in img_ids:
            img_path = osp.join(img_folder_path, img_id + self.img_suffix)
            if osp.exists(img_path):
                verified_ids.append(img_id)
            # else: print(f"Warning: Image file not found for mask ID {img_id}")
        return verified_ids


    def load_img_and_mask(self, index):
        # Loads single image/mask based on ID - returns PIL Images
        img_id = self.img_ids[index]
        # Use data_root which is passed during __init__
        img_name = osp.join(self.data_root, self.img_dir_rel, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir_rel, img_id + self.mask_suffix)
        try:
            img = Image.open(img_name).convert('RGB')
            mask = Image.open(mask_name).convert('L') # Load mask as grayscale index
        except FileNotFoundError:
             print(f"Error in load_img_and_mask: Cannot find {img_name} or {mask_name}")
             # Return dummy PIL images or raise error
             img = Image.new('RGB', self.img_size, (0,0,0))
             mask = Image.new('L', self.img_size, IGNORE_INDEX) # Use defined IGNORE_INDEX
        except Exception as e:
             print(f"Error opening image/mask for {img_id}: {e}")
             img = Image.new('RGB', self.img_size, (0,0,0))
             mask = Image.new('L', self.img_size, IGNORE_INDEX)
        return img, mask

    def load_mosaic_img_and_mask(self, index):
        # Mosaic augmentation code as provided by user previously
        # Ensure it returns PIL Images
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])

        img_a, mask_a = np.array(img_a), np.array(mask_a)
        img_b, mask_b = np.array(img_b), np.array(mask_b)
        img_c, mask_c = np.array(img_c), np.array(mask_c)
        img_d, mask_d = np.array(img_d), np.array(mask_d)

        h, w = self.img_size # Use target size for mosaic canvas

        # Mosaic center calculation
        start_x = w // 4
        start_y = h // 4 # Corrected variable name
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(start_y, (h - start_y)) # Corrected variable name

        # Define crop sizes for each quadrant relative to the mosaic center
        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        # Create RandomCrop instances for each quadrant size
        random_crop_a = albu.RandomCrop(height=crop_size_a[1], width=crop_size_a[0])
        random_crop_b = albu.RandomCrop(height=crop_size_b[1], width=crop_size_b[0])
        random_crop_c = albu.RandomCrop(height=crop_size_c[1], width=crop_size_c[0])
        random_crop_d = albu.RandomCrop(height=crop_size_d[1], width=crop_size_d[0])

        # Apply random crops
        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        # Create the mosaic image and mask
        img_mosaic = np.zeros((h, w, 3), dtype=np.uint8)
        mask_mosaic = np.full((h, w), IGNORE_INDEX, dtype=np.uint8) # Use ignore index

        # Place cropped images/masks into the correct quadrants
        img_mosaic[0:offset_y, 0:offset_x, :] = img_crop_a
        mask_mosaic[0:offset_y, 0:offset_x] = mask_crop_a
        img_mosaic[0:offset_y, offset_x:w, :] = img_crop_b
        mask_mosaic[0:offset_y, offset_x:w] = mask_crop_b
        img_mosaic[offset_y:h, 0:offset_x, :] = img_crop_c
        mask_mosaic[offset_y:h, 0:offset_x] = mask_crop_c
        img_mosaic[offset_y:h, offset_x:w, :] = img_crop_d
        mask_mosaic[offset_y:h, offset_x:w] = mask_crop_d

        # Convert final NumPy arrays back to PIL Images
        img = Image.fromarray(img_mosaic)
        mask = Image.fromarray(mask_mosaic)

        return img, mask


# --- Visualization functions (show_img_mask_seg, show_seg, show_mask) ---
# Keep these functions as they were in the user's provided script, if needed elsewhere.
# (Make sure plt and mpatches are imported if these are uncommented)
# def show_img_mask_seg(...): ...
# def show_seg(...): ...
# def show_mask(...): ...

# --- REMOVED MODULE-LEVEL INSTANTIATION ---
# vaihingen_train_dataset = VaihingenDataset(...) # REMOVED
# vaihingen_val_dataset = VaihingenDataset(...)   # REMOVED
# vaihingen_test_dataset = VaihingenDataset(...)  # REMOVED
# --- REMOVED MODULE-LEVEL INSTANTIATION ---
