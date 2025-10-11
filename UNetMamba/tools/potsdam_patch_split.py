# ~/autodl-tmp/UNetMamba-main/UNetMamba/tools/potsdam_patch_split.py (Updated with RGB to Index Mapping)
import glob
import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image # PIL might not be needed anymore if using cv2/tifffile directly
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import albumentations as albu
import random
try:
    import tifffile
except ImportError:
    print("Warning: tifffile library not found. Install using 'pip install tifffile' for reading .tif files.")
    tifffile = None

SEED = 42
IGNORE_INDEX = 255 # Target ignore index for the model loss function

# ISPRS Potsdam Standard Color Palette -> Index Mapping
# Index 0: Imp Surf (White)   -> [255, 255, 255]
# Index 1: Building (Blue)    -> [0, 0, 255]
# Index 2: Low Veg (Cyan)    -> [0, 255, 255]
# Index 3: Tree (Green)     -> [0, 255, 0]
# Index 4: Car (Yellow)      -> [255, 255, 0]
# Index 5: Clutter/BG (Red) -> [255, 0, 0]
# Make sure the BGR order is correct if using OpenCV later
COLOR_TO_INDEX_MAP = {
    (255, 255, 255): 0, # Impervious surfaces (White)
    (0, 0, 255): 1,     # Building (Blue) - OpenCV reads as BGR, so check order
    (0, 255, 255): 2, # Low vegetation (Cyan)
    (0, 255, 0): 3,     # Tree (Green)
    (255, 255, 0): 4, # Car (Yellow)
    (255, 0, 0): 5,     # Clutter/Background (Red)
    # Add other colors present? e.g., Black for boundaries mapped to ignore?
    (0, 0, 0): IGNORE_INDEX # Assuming Black boundaries/ignore
}
# OpenCV reads in BGR order by default, tifffile reads RGB. Let's assume input is RGB.
# Palette for mapping function (RGB order)
RGB_PALETTE = [
    ([255, 255, 255], 0), # Imp Surf
    ([0, 0, 255], 1),     # Building
    ([0, 255, 255], 2), # Low Veg
    ([0, 255, 0], 3),     # Tree
    ([255, 255, 0], 4), # Car
    ([255, 0, 0], 5)      # Clutter/BG
    # Add mapping for black or other ignore colors if they exist
    # ([0, 0, 0], IGNORE_INDEX) # Optional: map black to ignore
]

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Split large Potsdam RGB masks into indexed patches.")
    parser.add_argument("--img-dir", required=True, help="Directory containing original large IRRG images (.tif)")
    parser.add_argument("--mask-dir", required=True, help="Directory containing original large RGB label masks (.tif)")
    parser.add_argument("--output-img-dir", required=True, help="Directory to save image patches")
    parser.add_argument("--output-mask-dir", required=True, help="Directory to save indexed mask patches (.png)") # Output is indexed
    parser.add_argument("--patch-size", type=int, default=1024, help="Size of the square patches (e.g., 1024)")
    parser.add_argument("--stride", type=int, default=1024, help="Stride (equal to patch-size for non-overlapping)")
    parser.add_argument("--img-suffix", default="_IRRG.tif", help="Suffix of input image files")
    parser.add_argument("--mask-suffix", default="_label.tif", help="Suffix of input RGB mask files") # Input is RGB mask
    parser.add_argument("--out-img-ext", default=".png", help="Extension for output image patches (.png or .tif)")
    parser.add_argument("--out-mask-ext", default=".png", help="Extension for output indexed mask patches (.png)")

    return parser.parse_args()

def rgb_to_index_mask(rgb_mask, palette, ignore_index=IGNORE_INDEX):
    """Converts an RGB mask to a single channel index mask based on a palette."""
    # Ensure mask is HWC
    if rgb_mask.ndim != 3 or rgb_mask.shape[2] != 3:
        raise ValueError(f"Input mask must be HWC RGB. Got shape {rgb_mask.shape}")

    h, w, _ = rgb_mask.shape
    index_mask = np.full((h, w), ignore_index, dtype=np.uint8) # Initialize with ignore value

    # Iterate through the palette and create masks for each color
    for color_rgb, index in palette:
        # Create a boolean mask where pixels match the current color
        matches = np.all(rgb_mask == np.array(color_rgb, dtype=rgb_mask.dtype), axis=-1)
        index_mask[matches] = index # Assign the corresponding index

    return index_mask

def get_padded_numpy(image_path, mask_path, patch_size, ignore_value=IGNORE_INDEX):
    """Loads image (IRRG) and mask (RGB), pads them."""
    if tifffile is None: raise ImportError("tifffile library is required.")
    try:
        img_np = tifffile.imread(image_path)    # HWC IRRG
        mask_np_rgb = tifffile.imread(mask_path) # HWC RGB Label
    except Exception as e:
        print(f"Error reading {image_path} or {mask_path}: {e}")
        return None, None

    if img_np is None or mask_np_rgb is None: return None, None
    if img_np.shape[:2] != mask_np_rgb.shape[:2]:
        print(f"Error: Mismatched shapes {img_np.shape} vs {mask_np_rgb.shape} for {os.path.basename(image_path)}")
        return None, None

    # --- Handle Image ---
    if img_np.ndim == 2: img_np = np.stack([img_np]*3, axis=-1)
    if img_np.shape[2] == 4: img_np = img_np[:, :, :3]
    if img_np.shape[2] != 3: print(f"Error: Unexpected image channels..."); return None, None
    if img_np.dtype == np.uint16: img_np = (img_np / 256).astype(np.uint8)
    elif img_np.dtype != np.uint8: img_np = img_np.astype(np.uint8)

    # --- Handle Mask (ensure it's RGB uint8) ---
    if mask_np_rgb.ndim != 3 or mask_np_rgb.shape[2] != 3:
         print(f"Error: Mask {os.path.basename(mask_path)} is not HWC RGB. Shape: {mask_np_rgb.shape}")
         return None, None
    if mask_np_rgb.dtype != np.uint8: mask_np_rgb = mask_np_rgb.astype(np.uint8) # Common format

    # --- Padding ---
    oh, ow = img_np.shape[0], img_np.shape[1]
    rh_patch, rw_patch = oh % patch_size, ow % patch_size
    width_pad = 0 if rw_patch == 0 else patch_size - rw_patch
    height_pad = 0 if rh_patch == 0 else patch_size - rh_patch

    # Pad image with 0, pad mask with a color that maps to ignore_index (e.g., Black if mapped)
    # It's safer to pad the index mask later, let's pad RGB mask with black [0,0,0] for now
    # Or pad with white [255,255,255]? Let's use black.
    pad_mask_value_rgb = [0, 0, 0]

    if height_pad != 0 or width_pad != 0:
        # Pad image
        padder_img = albu.PadIfNeeded(min_height=oh + height_pad, min_width=ow + width_pad,
                                  position='bottom_right', border_mode=cv2.BORDER_CONSTANT,
                                  value=0) # Pad image with 0
        img_pad = padder_img(image=img_np)['image']
        # Pad mask
        padder_mask = albu.PadIfNeeded(min_height=oh + height_pad, min_width=ow + width_pad,
                                  position='bottom_right', border_mode=cv2.BORDER_CONSTANT,
                                  value=pad_mask_value_rgb) # Pad RGB mask with black
        mask_pad_rgb = padder_mask(image=mask_np_rgb)['image'] # Use 'image' key for mask too
    else:
        img_pad, mask_pad_rgb = img_np, mask_np_rgb

    return img_pad, mask_pad_rgb


def split_and_save(inp):
    """Processes one large image/RGB mask pair, converts mask to index, saves patches."""
    (img_path, mask_path, output_img_dir, output_mask_dir,
     patch_size, stride, img_suffix, mask_suffix,
     out_img_ext, out_mask_ext, ignore_value) = inp

    base_filename = os.path.basename(img_path).replace(img_suffix, '')

    img_padded, mask_padded_rgb = get_padded_numpy(img_path, mask_path, patch_size, ignore_value)

    if img_padded is None: return

    img_h, img_w = img_padded.shape[0], img_padded.shape[1]
    patch_count = 0

    for y in range(0, img_h, patch_size): # Non-overlapping stride = patch_size
        for x in range(0, img_w, patch_size):
            if y + patch_size > img_h or x + patch_size > img_w: continue

            img_tile = img_padded[y:y + patch_size, x:x + patch_size]
            mask_tile_rgb = mask_padded_rgb[y:y + patch_size, x:x + patch_size] # RGB mask patch

            if img_tile.shape[0] != patch_size or img_tile.shape[1] != patch_size: continue

            # --- 【新增】将 RGB 掩码块转换为索引掩码块 ---
            try:
                mask_tile_indexed = rgb_to_index_mask(mask_tile_rgb, RGB_PALETTE, ignore_value)
            except ValueError as e:
                print(f"Error converting RGB mask to index for patch at ({y},{x}) for {base_filename}: {e}")
                continue
            # --- 转换结束 ---

            coord_str = f"r{y//patch_size}_c{x//patch_size}"
            out_img_filename = f"{base_filename}_{coord_str}{out_img_ext}"
            out_mask_filename = f"{base_filename}_{coord_str}{out_mask_ext}" # Output is indexed mask

            out_img_path = os.path.join(output_img_dir, out_img_filename)
            out_mask_path = os.path.join(output_mask_dir, out_mask_filename)

            try:
                # Save image patch
                if out_img_ext == '.png':
                    # Ensure uint8 BGR for OpenCV
                    if img_tile.dtype != np.uint8: img_tile = img_tile.astype(np.uint8)
                    cv2.imwrite(out_img_path, cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR))
                elif out_img_ext == '.tif':
                    if tifffile: tifffile.imwrite(out_img_path, img_tile)
                    else: print("Error: tifffile not installed")
                else: print(f"Unsupported image extension: {out_img_ext}"); continue

                # Save the INDEXED mask patch (it's already uint8)
                cv2.imwrite(out_mask_path, mask_tile_indexed) # <-- 保存转换后的索引掩码

                patch_count += 1
            except Exception as e:
                print(f"Error saving patch at ({y},{x}) for {base_filename}: {e}")

    # Suppress print during parallel execution, or add locking
    # print(f"Finished {base_filename}, created {patch_count} patches.")


if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()

    if not osp.isdir(args.img_dir): print(f"Error: Input image directory not found: {args.img_dir}"); exit()
    if not osp.isdir(args.mask_dir): print(f"Error: Input mask directory not found: {args.mask_dir}"); exit()

    os.makedirs(args.output_img_dir, exist_ok=True)
    os.makedirs(args.output_mask_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(args.img_dir, f"*{args.img_suffix}")))

    input_pairs = []
    print("Matching image files with mask files...")
    for img_path in img_paths:
        img_base = os.path.basename(img_path).replace(args.img_suffix, '')
        potential_mask_name = f"{img_base}{args.mask_suffix}" # Assumes mask name matches img base + suffix
        mask_path = osp.join(args.mask_dir, potential_mask_name)
        if osp.exists(mask_path):
            input_pairs.append((img_path, mask_path))
        else:
             print(f"Warning: No corresponding mask found for {os.path.basename(img_path)} at {mask_path}")

    print(f"Found {len(input_pairs)} image/mask pairs to process.")
    if not input_pairs: exit()

    # Pass ignore_index to worker function (label_mapping mode removed, handled internally)
    inp = [(img_p, mask_p, args.output_img_dir, args.output_mask_dir,
            args.patch_size, args.stride, args.img_suffix, args.mask_suffix,
            args.out_img_ext, args.out_mask_ext, IGNORE_INDEX)
           for img_p, mask_p in input_pairs]

    t0 = time.time()
    try:
        num_processes = max(1, min(mp.cpu_count() // 2, 8))
        print(f"Starting patch splitting using {num_processes} processes (with RGB to Index mask conversion)...")
        pool = mpp.Pool(processes=num_processes)
        pool.map(split_and_save, inp)
        pool.close()
        pool.join()
    except Exception as e:
        print(f"Error during multiprocessing: {e}")
    finally:
        if 'pool' in locals() and pool is not None: pool.terminate()

    t1 = time.time()
    split_time = t1 - t0
    print(f'\nFinished splitting. Time taken: {split_time:.2f} s')