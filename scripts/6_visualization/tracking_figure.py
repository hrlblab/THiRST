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
"""Export the registered patch stacks that contain tracked objects (3D tracking figure).

THiRST stage 6 - figures.

Outputs: ``<contour_root>/<case>/Object_tracking_IoU_affine_patch<tag>_global_vis/<patch>/slice_<k>.png``
(``<tag>`` = ``_connectglom`` for the Hungarian tracks, empty for the greedy tracks, see ``--tracks-subdir``)

Usage
-----
    python scripts/6_visualization/tracking_figure.py --config configs/example.yaml [--tracks-subdir Object_IoU_patch_connectglom]

Original script(s): Figure_results_code_Trackingfigure.py
"""
import sys
import os
import glob
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
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
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


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
            # try:
            poly = pts # Polygon(pts)
            #	if not poly.is_valid:
            #		# Optionally, fix invalid polygons (e.g., self-intersections)
            #		poly = poly.buffer(0)
            slice_contours.append({'object': obj, 'index': idx, 'polygon': poly})
            # except Exception as e:
            #	print("Error converting contour to polygon:", e)
    return slice_contours


if __name__ == '__main__':
    args, cfg = parse_args(__doc__, lambda p: add_tracks_argument(p, default='Object_IoU_patch_connectglom'))
    case_folder =  cfg['paths']['image_root']
    point_folder = cfg['paths']['point_root']
    contour_folder = cfg['paths']['contour_root']

    cases = list_cases(cfg)
    for ci in range(len(cases)):
        case = cases[ci]
        now_case = os.path.basename(case)
        lbl_input_dir = case
        image_input_dir = case
        image_output_dir = os.path.join(case, 'registration_ANTs')
        # image_badcase_dir = os.path.join(case, 'registration_badcase_2048')
        reconstruction_dir = os.path.join(case.replace(case_folder,point_folder), 'registration_3D_patch_ANTs')
        reconstruction_contour_dir = os.path.join(case.replace(case_folder,contour_folder), 'registration_3D_patch_ANTs')
        IoU_dir = os.path.join(case.replace(case_folder, contour_folder), args.tracks_subdir)
        # global_reconstruction_dir = os.path.join(case.replace(case_folder,point_folder), 'registration_3D_ANTs')
        # global_reconstruction_contour_dir = os.path.join(case.replace(case_folder,contour_folder),'registration_3D_ANTs')
        after_annotation_dir = os.path.join(reconstruction_contour_dir, 'annotation_QA_generegions')

        tracking_output = tracks_output_dir(case.replace(case_folder, contour_folder), 'Object_tracking_IoU_affine_patch',
                                            args.tracks_subdir, args.tracks_suffix, tail='_global_vis')

        if not os.path.exists(tracking_output):
            os.makedirs(tracking_output)

        big_images = glob.glob(os.path.join(os.path.join(case, 'registration_3D_ANTs'), 'image_*.png'))
        big_images.sort(key=natural_keys)

        images = glob.glob(os.path.join(image_input_dir, 'image_*.png'))
        images.sort(key=natural_keys)

        jsons = section_object_geojsons(images, after_annotation_dir, now_case)  # one per section, in stack order
        require_alternating_platforms(cfg, images)  # V2 at even, HD at odd stack positions

        csv_after = glob.glob(os.path.join(reconstruction_contour_dir,'*.csv'))
        csv_after.sort(key=natural_keys)

        # csv_before = glob.glob(os.path.join(case.replace(case_folder,contour_folder),'*.csv'))
        # csv_before.sort(key=natural_keys)

        middle_idx = int(len(images) / 2)

        # print(middle_idx)
        # middle_image = plt.imread(images[middle_idx])[:,:,:3]

        middle_ref_x = int(os.path.basename(images[middle_idx]).split('_')[2])
        middle_ref_y = int(os.path.basename(images[middle_idx]).split('_')[3].replace('.png', ''))

        patch_level_folder = os.path.join(case,'3D_Patch_crop')

        patch_list = glob.glob(os.path.join(patch_level_folder,'*'))

        patch_array = np.zeros((len(patch_list), 1, 2)).astype(np.float32)
        for pi in range(len(patch_list)):
            patch_name = os.path.basename(patch_list[pi])
            patch_abs_x, patch_abs_y = patch_name.split('_')
            patch_array[pi,:,0] = int(patch_abs_x)
            patch_array[pi,:,1] = int(patch_abs_y)

        mask_labels = single_object_class(cfg)  # the Omni-Seg class reconstructed in 3D (one class)
        # Build a dictionary that holds all contours (as polygons) for each slice
        all_slices = {}  # key: slice index, value: list of dicts with keys 'object' and 'polygon'
        for idx, (img_name, json_path) in enumerate(zip(images, jsons)):
            ann = load_annotations(img_name, json_path, mask_labels)
            if ann is None:
                continue  # Skip slices without a valid file
            all_slices[idx] = contours_to_polygons(ann)

        images_slice = {}

        for xi, img in enumerate(images):
            real_x = int(os.path.basename(images[xi]).split('_')[2])
            real_y = int(os.path.basename(images[xi]).split('_')[3].replace('.png', ''))
            now_img = plt.imread(images[xi])[:,:,:3]
            images_slice[xi] = {'img': now_img, 'real_x': real_x, 'real_y': real_y}

        # spot_list_ori = {}
        #
        # for si, csv in enumerate(csv_before):
        #     if si % 2 == 0:
        #         now_df = pd.read_csv(csv)
        #     else:
        #         now_df = pd.read_csv(csv_before[6])
        #     spot_list_ori[si] = {'spot': now_df}

        spot_list_after = {}

        for si, csv in enumerate(csv_after):
            if si % 2 == 0:
                now_df = pd.read_csv(csv)
            else:
                # HD sections (odd positions) reuse the V2 spot grid of the reference (middle) section
                now_df = pd.read_csv(csv_after[middle_idx])
            spot_list_after[si] = {'spot': now_df}

        trakcing_file = tracks_json(IoU_dir, args.tracks_suffix)
        with open(trakcing_file, 'r') as f:
            tracking_results = json.load(f)

            for patch_idx, patch_path in enumerate(patch_list):
                patch_image_folder = os.path.join(patch_path, 'registration_3D_ANTs')
                region_list = glob.glob(os.path.join(patch_image_folder, 'image_*.png'))
                region_list.sort(key=natural_keys)

                if len(region_list) == 0:
                    continue

                patch_images = [plt.imread(img_path)[:, :, :3] for img_path in region_list]
                patch_images = [(img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8) for img in
                                patch_images]

                patch_size = 1024

                # Get the absolute position of this patch center
                abs_patch_x = patch_array[patch_idx, 0, 1]
                abs_patch_y = patch_array[patch_idx, 0, 0]

                now_patch_y = int((abs_patch_x - middle_ref_x) / 4)
                now_patch_x = int((abs_patch_y - middle_ref_y) / 4)

                for track in tracking_results:
                    track_id = track['track_id']
                    for obj in track['contours']:
                        now_slice = obj['slice_index']
                        now_idx = obj['index']

                        polygon = all_slices[now_slice][now_idx]['polygon']
                        if isinstance(polygon, dict):
                            contour_np = np.array(polygon['coordinates'][0])
                        else:
                            contour_np = np.array(polygon)

                        # Translate to patch-local coordinates (2048x2048)
                        local_contour = contour_np.copy()
                        local_contour[:, 0] -= (now_patch_y - patch_size)
                        local_contour[:, 1] -= (now_patch_x - patch_size)

                        # if local_contour.min() < 0 or local_contour.max() > 2048:
                        #     continue  # skip if completely outside

                        if (local_contour[:, 0] < 0).all() or (local_contour[:, 0] > 2048).all() or \
                            (local_contour[:, 1] < 0).all() or (local_contour[:, 1] > 2048).all():
                            continue  # skip if *completely* outside

                        contour_int = np.ascontiguousarray(local_contour.reshape((-1, 1, 2)).astype(np.int32))
                        try:
                            cv2.drawContours(patch_images[now_slice], [contour_int], -1, (255, 255, 0), 10)
                        except:
                            continue

                # Save the overlay result
                patch_name = os.path.basename(patch_path)
                save_patch_folder = os.path.join(tracking_output, patch_name)
                os.makedirs(save_patch_folder, exist_ok=True)

                for si, img in enumerate(patch_images):
                    plt.imsave(os.path.join(save_patch_folder, f'slice_{si}.png'), img)


