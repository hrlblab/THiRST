# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""List the Visium HD bins inside every 3D object, per section (input of the 2D vs 3D DE analysis).

THiRST stage 5 - object-level 3D ST analysis (paper Sec. 2.2, Fig. 3b).

For every track and every HD section (odd positions), the barcodes of HD bins lying entirely inside the
object contour are collected. Aggregating expression over these barcodes gives the object-level profile.

Outputs: ``<contour_root>/<case>/Object_tracking_IoU_affine_patch_gene_barcode<tag>/glomeruli_barcode_per_slice.json``
(``<tag>`` = ``_connectglom`` for the Hungarian tracks, empty for the greedy tracks, see ``--tracks-subdir``)

Usage
-----
    python scripts/5_object_analysis/export_track_barcodes.py --config configs/example.yaml

Original script(s): Step10.5_Match_Object_gene_patchregistration_using_patch_gene_findbarcode.py
"""
import sys
import os
import glob
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from thirst.config import add_tracks_argument, list_cases, parse_args, require_alternating_platforms, section_object_geojsons, single_object_class, tracks_json, tracks_output_dir  # noqa: E402
from thirst.objects import read_object_contours  # noqa: E402


def atoi(text):
    return int(text) if text.isdigit() else text


# Step 3: Compute IoU between each circle and each bin
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


# The original intensity_to_bgr function is replaced with a custom version.
def load_annotations(image_name, mask_json_path, mask_labels):
    """
    Loads a GeoJSON file, extracts contours for given object labels,
    applies a translation transformation based on the image filename,
    and returns a dictionary mapping object type to a list of numpy arrays (contours).
    """
    if not os.path.exists(mask_json_path):
        print(f"Skipping {mask_json_path}, file not found")
        return None

    # {class: [contour, ...]}; reads standard GeoJSON and the earlier per-class layout
    return read_object_contours(mask_json_path, mask_labels)


def contours_to_polygons(annotations_by_object):
    """
    Converts each contour (numpy array) into a Shapely Polygon.
    Returns a list of dictionaries for the slice,
    each with keys: 'object' (type), 'index' (contour index) and 'polygon' (Shapely Polygon).
    """
    slice_contours = []
    for obj, contour_list in annotations_by_object.items():
        for idx, pts in enumerate(contour_list):
            # Here, pts remains as a numpy array.
            poly = pts  # You may convert to a proper Polygon if needed: Polygon(pts)
            slice_contours.append({'object': obj, 'index': idx, 'polygon': poly})
    return slice_contours


# New function to save images with a color bar.
if __name__ == '__main__':
    args, cfg = parse_args(__doc__, lambda p: add_tracks_argument(p, default='Object_IoU_patch_connectglom'))
    # Directories and folder setup.
    case_folder = cfg['paths']['image_root']
    point_folder = cfg['paths']['point_root']
    contour_folder = cfg['paths']['contour_root']

    cases = list_cases(cfg)
    for ci in range(len(cases)):
        case = cases[ci]
        now_case = os.path.basename(case)
        lbl_input_dir = case
        image_input_dir = case
        image_output_dir = os.path.join(case, 'registration_ANTs')
        reconstruction_dir = os.path.join(case.replace(case_folder, point_folder), 'registration_3D_patch_ANTs')
        reconstruction_contour_dir = os.path.join(case.replace(case_folder, contour_folder),
                                                  'registration_3D_patch_ANTs')
        IoU_dir = os.path.join(case.replace(case_folder, contour_folder), args.tracks_subdir)
        global_reconstruction_dir = os.path.join(case.replace(case_folder, point_folder), 'registration_3D_ANTs')
        global_reconstruction_contour_dir = os.path.join(case.replace(case_folder, contour_folder),
                                                         'registration_3D_ANTs')
        after_annotation_dir = os.path.join(reconstruction_contour_dir, 'annotation_QA_generegions')
        tracking_output = tracks_output_dir(case.replace(case_folder, contour_folder),
                                            'Object_tracking_IoU_affine_patch_gene_barcode',
                                            args.tracks_subdir, args.tracks_suffix)

        if not os.path.exists(tracking_output):
            os.makedirs(tracking_output)

        big_images = glob.glob(
                os.path.join(os.path.join(case, 'registration_3D_ANTs'),
                             'image_*.png'))
        big_images.sort(key=natural_keys)

        images = glob.glob(os.path.join(image_input_dir, 'image_*.png'))
        images.sort(key=natural_keys)

        jsons = section_object_geojsons(images, after_annotation_dir, now_case)  # one per section, in stack order
        require_alternating_platforms(cfg, images)  # V2 at even, HD at odd stack positions

        csv_after = glob.glob(os.path.join(reconstruction_contour_dir, '*.csv'))
        csv_after.sort(key=natural_keys)

        middle_idx = int(len(images) / 2)
        middle_ref_x = int(os.path.basename(images[middle_idx]).split('_')[2])
        middle_ref_y = int(os.path.basename(images[middle_idx]).split('_')[3].replace('.png', ''))

        patch_level_folder = os.path.join(case, '3D_Patch_crop')
        patch_list = glob.glob(os.path.join(patch_level_folder, '*'))
        patch_array = np.zeros((len(patch_list), 1, 2)).astype(np.float32)
        for pi in range(len(patch_list)):
            if '3D' in os.path.basename(patch_list[pi]):
                continue

            patch_name = os.path.basename(patch_list[pi])
            patch_abs_x, patch_abs_y = patch_name.split('_')
            patch_array[pi, :, 0] = int(patch_abs_x)
            patch_array[pi, :, 1] = int(patch_abs_y)

        mask_labels = single_object_class(cfg)  # the Omni-Seg class reconstructed in 3D (one class)
        # Build a dictionary that holds all contours (as polygons) for each slice.
        all_slices = {}  # key: slice index, value: list of dicts with keys 'object' and 'polygon'
        for idx, (img_name, json_path) in enumerate(zip(images, jsons)):
            ann = load_annotations(img_name, json_path, mask_labels)
            if ann is None:
                continue
            all_slices[idx] = contours_to_polygons(ann)

        images_slice = {}
        for xi, img in enumerate(images):
            real_x = int(os.path.basename(images[xi]).split('_')[2])
            real_y = int(os.path.basename(images[xi]).split('_')[3].replace('.png', ''))
            now_img = plt.imread(images[xi])[:, :, :3]
            images_slice[xi] = {'img': now_img, 'real_x': real_x, 'real_y': real_y}

        spot_list_after = {}

        for si, csv in enumerate(csv_after):
            now_df = pd.read_csv(csv)
            spot_list_after[si] = {'spot': now_df}

        trakcing_file = tracks_json(IoU_dir, args.tracks_suffix)
        with open(trakcing_file, 'r') as f:
            tracking_results = json.load(f)

        # Final output: list of glomerulus tracking info
        track_barcode_map = []

        for ti in range(len(tracking_results)):
            now_tracking = tracking_results[ti]
            track_id = now_tracking['track_id']
            barcodes_per_slice = []

            # Collect per-slice barcodes
            for si in range(len(now_tracking['contours'])):
                now_obj = now_tracking['contours'][si]
                now_idx = now_obj['index']
                now_slice = now_obj['slice_index']

                if now_slice % 2 == 0:
                    continue

                now_contour = all_slices[now_slice][now_idx]['polygon']

                try:
                    polygon_now = Polygon(now_contour)
                except Exception:
                    continue

                now_df_after = spot_list_after[now_slice]['spot']

                # Prepare batch_array_i for shapely vectorized query
                num_points = 4
                batch_array_i = np.zeros((len(now_df_after), num_points, 1, 2), dtype=np.float32)
                for ni in range(num_points):
                    batch_array_i[:, ni, :, 0] = np.expand_dims(now_df_after[f'now_y_{ni}'], 1)
                    batch_array_i[:, ni, :, 1] = np.expand_dims(now_df_after[f'now_x_{ni}'], 1)

                import shapely  # contains_xy replaces the deprecated shapely.vectorized.contains

                coords = batch_array_i[:, :, 0, :]
                x_coords = coords[:, :, 0]
                y_coords = coords[:, :, 1]

                contained = shapely.contains_xy(polygon_now, x_coords.ravel(), y_coords.ravel()).reshape(
                        len(now_df_after), num_points)
                inside_mask = np.all(contained, axis=1)
                inside_indices = np.nonzero(inside_mask)[0]

                barcodes = now_df_after.iloc[inside_indices]['barcode'].tolist()

                # Only save if we found barcodes
                if barcodes:
                    barcodes_per_slice.append({
                            "slice_index": now_slice,
                            "barcodes": sorted(barcodes)  # Optional: sort for consistency
                    })

            # Sort by slice index
            barcodes_per_slice.sort(key=lambda x: x["slice_index"])

            track_barcode_map.append({
                    "track_id": track_id,
                    "barcodes_per_slice": barcodes_per_slice
            })

        # Save JSON
        output_json_path = os.path.join(tracking_output, 'glomeruli_barcode_per_slice.json')
        with open(output_json_path, 'w') as f:
            json.dump(track_barcode_map, f, indent=4)

        print(f"Saved barcode map by slice to {output_json_path}")