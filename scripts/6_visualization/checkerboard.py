"""Checkerboard overlays for visual registration QC (paper Fig. 2a-style figures).

For every ordered pair of sections (i, j) the checkerboard alternates tiles of section i (RGB)
and section j (grayscale), so misalignment shows up as broken structures across tile borders.

Modes
-----
global        sections after global registration:
              <image_root>/<case>/registration_3D_ANTs/*.png -> .../chess_all_pairs_filled/
              (black padding is painted light gray first)
patch-before  each patch stack before patch-level registration:
              <image_root>/<case>/3D_Patch_crop/<x>_<y>/*.png -> .../<x>_<y>/chess_results/
patch-after   each patch stack after patch-level registration:
              <image_root>/<case>/3D_Patch_crop/<x>_<y>/registration_3D_ANTs/*.png -> .../<x>_<y>/chess_results_after/

Usage
-----
    python scripts/6_visualization/checkerboard.py --config configs/example.yaml --mode global

Original scripts: Figure_results_code_Resigration_results_chessfigure_global.py,
Figure_results_code_Resigration_results_chessfigure_patch.py,
Figure_results_code_Resigration_results_chessfigure_patch_after.py
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from thirst.config import list_cases, list_patch_stacks, parse_args  # noqa: E402

tile_size = 256  # Checker tile size

# === PREDEFINED BACKGROUND COLOR (global mode) ===
fill_color = (211, 211, 211)  # Light gray in BGR


# === FUNCTIONS ===
def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def convert_to_gray_rgb(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def fill_black_pixels(img, fill_color):
    """Replace all [0,0,0] pixels with the fill_color."""
    mask = np.all(img == [0, 0, 0], axis=-1)
    img[mask] = fill_color
    return img


def create_checkerboard(rgb_img, gray_img, tile_size):
    h, w = rgb_img.shape[:2]
    chess = np.zeros_like(rgb_img)
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            if (x // tile_size + y // tile_size) % 2 == 0:
                chess[y:y+tile_size, x:x+tile_size] = rgb_img[y:y+tile_size, x:x+tile_size]
            else:
                chess[y:y+tile_size, x:x+tile_size] = gray_img[y:y+tile_size, x:x+tile_size]
    return chess


def all_pairs(image_folder, output_folder, fill_background):
    """Write one checkerboard per ordered pair of images found in ``image_folder``."""
    output_folder.mkdir(exist_ok=True)
    image_files = sorted(image_folder.glob('*.png'))  # or '*.jpg', '*.tif'
    images = [load_rgb(f) for f in image_files]
    filenames = [f.stem for f in image_files]

    for i in range(len(images)):
        for j in range(len(images)):
            if i == j:
                continue

            if fill_background:
                img_rgb = fill_black_pixels(images[i].copy(), fill_color)
                img_gray = fill_black_pixels(convert_to_gray_rgb(images[j].copy()), fill_color)
            else:
                img_rgb = images[i].copy()
                img_gray = convert_to_gray_rgb(images[j].copy())

            if img_rgb.shape != img_gray.shape:
                print(f"Skipping {filenames[i]} vs {filenames[j]} (size mismatch)")
                continue

            chess_img = create_checkerboard(img_rgb, img_gray, tile_size)
            out_path = output_folder / f'chess_{filenames[i]}_RGB_vs_{filenames[j]}_GRAY.png'
            cv2.imwrite(str(out_path), chess_img)


if __name__ == "__main__":
    args, cfg = parse_args(__doc__, lambda p: p.add_argument(
        "--mode", choices=("global", "patch-before", "patch-after"), required=True,
        help="which registered images to compare (see module docstring)"))

    for case in list_cases(cfg):
        if args.mode == "global":
            image_folder = Path(case) / 'registration_3D_ANTs'
            print(f"Using predefined fill color: {fill_color}")
            all_pairs(image_folder, image_folder / 'chess_all_pairs_filled', fill_background=True)
            print("Done generating all RGB vs grayscale chess images with light gray background.")
        else:
            for folder in map(Path, list_patch_stacks(case)):
                print(f"\nProcessing folder: {folder.name}")
                if args.mode == "patch-before":
                    all_pairs(folder, folder / 'chess_results', fill_background=False)
                else:
                    all_pairs(folder / 'registration_3D_ANTs', folder / 'chess_results_after', fill_background=False)
                print(f"Done with folder: {folder.name}")
