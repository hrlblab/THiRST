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
"""Per-object tissue crops on a white background for 3D rendering of reconstructed objects.

THiRST stage 6 - figures (3D glomerulus renderings, paper Fig. 3a).

For every track and section the tissue inside the object contour is cut from the locally registered patch
(background set to white) and saved together with the annotated patch.

Outputs: ``<contour_root>/<case>/Object_tracking_IoU_affine_patch_tissueviual_white<tag>/<track_id>/crops/slice_<k>.png``
(``<tag>`` = empty for the greedy tracks, ``_connectglom`` for the Hungarian tracks, see ``--tracks-subdir``)
         ``.../<track_id>/Glo_slice_<k>.png`` and ``IoU_withspot.csv``

Usage
-----
    python scripts/6_visualization/render_3d_objects.py --config configs/example.yaml [--tracks-subdir Object_IoU_patch]

Original script(s): Step12_3D_sample_visualization_whitebackground.py
"""
import sys
import os
import glob
import json
import re

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import cv2 as cv2
import pandas as pd
from scipy.spatial import distance

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from thirst.config import add_tracks_argument, list_cases, parse_args, require_alternating_platforms, section_object_geojsons, single_object_class, tracks_json, tracks_output_dir  # noqa: E402
from thirst.objects import read_object_contours  # noqa: E402


# ---------------------------
# Helper Functions (Existing)
# ---------------------------
def atoi(text):
    return int(text) if text.isdigit() else text


def compute_iou(poly1, poly2):
    # Quick bounding box check:
    minx1, miny1, maxx1, maxy1 = poly1.bounds
    minx2, miny2, maxx2, maxy2 = poly2.bounds
    if (maxx1 < minx2) or (maxx2 < minx1) or (maxy1 < miny2) or (maxy2 < miny1):
        return 0.0  # No overlap in bounding boxes
    # Now compute the detailed intersection/union
    inter = poly1.intersection(poly2)
    if inter.is_empty:
        return 0.0
    inter_area = inter.area
    union_area = poly1.area + poly2.area - inter_area
    return inter_area / union_area if union_area else 0.0


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


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


def get_nearest_patch_index(contour: np.ndarray, patch_array: np.ndarray) -> int:
    """
    Given a contour (an array of shape (N,2)) and a patch_array of shape (num_patches, 1, 2),
    compute the geometric center of the contour, then use distance.cdist to compute the Euclidean
    distances between this center and each patch center. Return the index of the nearest patch.
    """
    contour_center = np.mean(contour, axis=0, keepdims=True)  # shape: (1,2)
    patch_centers = patch_array[:, 0, :]  # shape: (num_patches, 2)
    dist_matrix = distance.cdist(contour_center, patch_centers)
    return int(np.argmin(dist_matrix))


def contours_to_polygons(annotations_by_object):
    """
    Converts each contour (numpy array) into a Shapely Polygon.
    Returns a list of dictionaries for the slice,
    each with keys: 'object' (type), 'index' (contour index) and 'polygon' (Shapely Polygon).
    """
    slice_contours = []
    for obj, contour_list in annotations_by_object.items():
        for idx, pts in enumerate(contour_list):
            # Here we keep pts as the contour; you may choose to convert to a Shapely Polygon
            poly = pts  # or Polygon(pts) if you want to use polygon methods
            slice_contours.append({'object': obj, 'index': idx, 'polygon': poly})
    return slice_contours


# ---------------------------------------
# New Function: Tissue Extraction and Crop
# ---------------------------------------
def crop_and_extract_tissue(image, contour, crop_size=256):
    """
    For a given image and contour:
      - Compute the contour centroid.
      - Extract a crop_size×crop_size patch centered on that centroid
        (clamped at image edges so the patch is always the right size).
      - Convert the crop to uint8 if needed.
      - Create a mask from the contour and apply it so that pixels outside
        the contour become white (255).

    Returns:
        crop_img       : uint8 crop (crop_size×crop_size[×3]).
        tissue_region  : same as crop_img, but with background=white.
        (cx, cy)       : centroid coordinates in the original image.
    """
    # 1) Compute centroid
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = np.mean(contour.reshape(-1, 2), axis=0).astype(int)

    # 2) Define crop window, clamped so we always get exactly `crop_size`
    half = crop_size // 2
    H, W = image.shape[:2]
    x1 = max(0, min(cx - half, W - crop_size))
    y1 = max(0, min(cy - half, H - crop_size))
    x2, y2 = x1 + crop_size, y1 + crop_size

    # 3) Extract and copy the crop
    crop_img = image[y1:y2, x1:x2].copy()

    # 4) Ensure uint8 in [0…255]
    if crop_img.dtype != np.uint8:
        if np.issubdtype(crop_img.dtype, np.floating):
            # assume float in [0,1]
            crop_img_uint8 = (np.clip(crop_img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            crop_img_uint8 = np.clip(crop_img, 0, 255).astype(np.uint8)
    else:
        crop_img_uint8 = crop_img

    # 5) Build a mask over the full image, then crop it
    full_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(full_mask, [contour], 255)
    crop_mask = full_mask[y1:y2, x1:x2]

    # 6) Apply white background outside the contour
    tissue_region = crop_img_uint8.copy()
    if tissue_region.ndim == 3:
        tissue_region[crop_mask == 0] = [255, 255, 255]
    else:
        tissue_region[crop_mask == 0] = 255

    return crop_img_uint8, tissue_region, (cx, cy)

# ---------------------------
# Main Processing Workflow
# ---------------------------
if __name__ == '__main__':
    args, cfg = parse_args(__doc__, lambda p: add_tracks_argument(p, default='Object_IoU_patch'))
    # Define folder paths.
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
                                            'Object_tracking_IoU_affine_patch_tissueviual_white',
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
            patch_name = os.path.basename(patch_list[pi])
            patch_abs_x, patch_abs_y = patch_name.split('_')
            patch_array[pi, :, 0] = int(patch_abs_x)
            patch_array[pi, :, 1] = int(patch_abs_y)

        mask_labels = single_object_class(cfg)  # the Omni-Seg class reconstructed in 3D (one class)
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
            if si % 2 == 0:
                now_df = pd.read_csv(csv)
            else:
                # HD sections (odd positions) reuse the V2 spot grid of the reference (middle) section
                now_df = pd.read_csv(csv_after[middle_idx])
            spot_list_after[si] = {'spot': now_df}

        trakcing_file = tracks_json(IoU_dir, args.tracks_suffix)
        with open(trakcing_file, 'r') as f:
            tracking_results = json.load(f)

        spot_list = pd.DataFrame(columns=['track_id', 'slice_num', 'barcode', 'IoU', 'annotation_index'])

        # Dictionary to accumulate 2D tissue crops for each track.
        tissue_voxels_dict = {}  # key: track_id, value: list of 256x256 tissue regions

        patch_size = 1024  # as used previously
        for ti in range(len(tracking_results)):
            now_tracking = tracking_results[ti]
            track_id = now_tracking['track_id']
            if track_id not in tissue_voxels_dict:
                tissue_voxels_dict[track_id] = []
            for si in range(len(now_tracking['contours'])):
                now_obj = now_tracking['contours'][si]
                now_idx = now_obj['index']
                now_slice = now_obj['slice_index']
                now_contour = all_slices[now_slice][now_idx]['polygon']
                now_image = images_slice[now_slice]['img']

                new_patch_array = np.zeros((patch_array.shape))
                new_patch_array[:, :, 0] = (patch_array[:, :, 0] - middle_ref_y) / 4
                new_patch_array[:, :, 1] = (patch_array[:, :, 1] - middle_ref_x) / 4

                best_index = get_nearest_patch_index(now_contour[:, ::-1], new_patch_array)
                print(best_index)
                if best_index is None:
                    print("No suitable patch found for this contour!")
                    continue

                now_affine_img_folder = os.path.join(patch_list[best_index], 'registration_3D_ANTs')
                region_list = glob.glob(os.path.join(now_affine_img_folder, 'image_*.png'))
                region_list.sort(key=natural_keys)
                now_region = plt.imread(region_list[now_slice])[:, :, :3]

                abs_patch_x = patch_array[best_index, 0, 1]
                abs_patch_y = patch_array[best_index, 0, 0]
                now_patch_y = int((abs_patch_x - middle_ref_x) / 4)
                now_patch_x = int((abs_patch_y - middle_ref_y) / 4)

                center_y = int(now_contour[:, 0].mean() - (now_patch_x - now_region.shape[0]))
                center_x = int(now_contour[:, 1].mean() - (now_patch_y - now_region.shape[1]))

                now_patch = now_region  # In this example, we work on the entire patch.

                now_df_after = spot_list_after[now_slice]['spot']

                num_points = 100
                batch_array_i = np.zeros((len(now_df_after), num_points, 1, 2)).astype(np.float32)
                for ni in range(num_points):
                    batch_array_i[:, ni, :, 0] = np.expand_dims(np.array(now_df_after['now_y_%d' % (ni)]), 1)
                    batch_array_i[:, ni, :, 1] = np.expand_dims(np.array(now_df_after['now_x_%d' % (ni)]), 1)

                # Convert now_contour to a proper shape for OpenCV.
                if now_contour.ndim == 2:
                    now_contour = now_contour.reshape((-1, 1, 2))

                # Adjust contour coordinates relative to the patch.
                now_contour[:, :, 0] = now_contour[:, :, 0] - (now_patch_y - patch_size)
                now_contour[:, :, 1] = now_contour[:, :, 1] - (now_patch_x - patch_size)

                # Convert to int32 for drawing and cropping.
                contour_int = np.ascontiguousarray(now_contour.astype(np.int32))

                # ---- New: Crop and extract tissue region for this contour ----
                crop_img, tissue_voxel, centroid = crop_and_extract_tissue(now_patch, contour_int, crop_size=256)
                # Optionally, save the cropped image to file:
                crop_save_folder = os.path.join(tracking_output, str(track_id), "crops")
                if not os.path.exists(crop_save_folder):
                    os.makedirs(crop_save_folder)
                crop_filename = os.path.join(crop_save_folder, f"slice_{now_slice}.png")
                plt.imsave(crop_filename, tissue_voxel)

                # Store the 256x256 tissue region for later stacking into a 3D voxel.
                tissue_voxels_dict[track_id].append(tissue_voxel)
                # ---------------------------------------------------------------

                # Draw the main contour on now_patch.
                # Ensure now_patch is in the correct format (uint8 and contiguous) for cv2.drawContours.
                if now_patch.dtype != np.uint8:
                    now_patch = (now_patch * 255).astype(np.uint8)
                else:
                    now_patch = np.ascontiguousarray(now_patch)


                cv2.drawContours(now_patch, [contour_int], -1, (255, 255, 0), 20)
                print("draw the contour")
                print(contour_int[:, :, 0].mean(), contour_int[:, :, 1].mean())

                for i in range(batch_array_i.shape[0]):
                    candidate_contour = batch_array_i[i].reshape(-1, 2)
                    candidate_poly = Polygon(candidate_contour)
                    iou = compute_iou(Polygon(now_contour.reshape(-1, 2)), candidate_poly)
                    now_spot = batch_array_i[i]
                    if now_spot.ndim == 2:
                        now_spot = now_spot.reshape((-1, 1, 2))
                    now_spot[:, :, 0] = now_spot[:, :, 0] - (now_patch_y - patch_size)
                    now_spot[:, :, 1] = now_spot[:, :, 1] - (now_patch_x - patch_size)
                    contour_int_spot = np.ascontiguousarray(now_spot.astype(np.int32))
                    if not (now_spot.min() > 0 and now_spot.max() < patch_size * 2):
                        continue
                    if iou > 0:
                        row = len(spot_list)
                        spot_list.loc[row] = [track_id, now_slice, now_df_after['barcode'][i], iou, now_idx]
                        cv2.drawContours(now_patch, [contour_int_spot], -1, (0, 255, 0), 10)
                    else:
                        cv2.drawContours(now_patch, [contour_int_spot], -1, (0, 0, 255), 10)

                save_folder = os.path.join(tracking_output, str(track_id))
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                plt.imsave(os.path.join(save_folder, f'Glo_slice_{now_slice}.png'), now_patch)

        csv_root = os.path.join(tracking_output, 'IoU_withspot.csv')
        spot_list.to_csv(csv_root)

        # Optional: Save the 3D voxel for each track by stacking the 2D tissue regions.
        for track_id, tissue_list in tissue_voxels_dict.items():
            if len(tissue_list) > 0:
                voxel_3d = np.stack(tissue_list, axis=0)
                np.save(os.path.join(tracking_output, f"track_{track_id}_voxel.npy"), voxel_3d)
