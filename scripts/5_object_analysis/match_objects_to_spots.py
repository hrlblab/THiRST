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
"""Collect the ST units inside every 3D object and export per-object image stacks.

THiRST stage 5 - object-level 3D ST analysis (paper Sec. 2.2).

For each track and section, the ST units whose polygon lies inside the object contour are listed and the
locally registered patch around the object is saved with contour and units drawn on it. HD sections reuse
the V2 spot grid of the reference section ("HD to V2").

Outputs: ``<contour_root>/<case>/Object_tracking_IoU_affine_patch<tag>/<track_id>/Glo_slice_<k>.png``
(``<tag>`` = ``_connectglom`` for the Hungarian tracks, empty for the greedy tracks, see ``--tracks-subdir``)
         ``.../IoU_withspot.csv`` (track_id, slice_num, barcode, IoU, annotation_index)

Usage
-----
    python scripts/5_object_analysis/match_objects_to_spots.py --config configs/example.yaml [--tracks-subdir Object_IoU_patch_connectglom]

Original script(s): Step10_Match_Object_Spot_patchregistration_using_patch.py
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


def atoi(text):
    return int(text) if text.isdigit() else text

# Step 3: Compute IoU between each circle and each bin
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


def get_nearest_patch_index(contour: np.ndarray, patch_array: np.ndarray) -> int:
    """
    Given a contour (an array of shape (N,2)) and a patch_array of shape (num_patches, 1, 2),
    compute the geometric center of the contour, then use distance.cdist to compute the Euclidean
    distances between this center and each patch center. Return the index of the nearest patch.

    Parameters:
            contour (np.ndarray): Array of shape (N, 2) representing contour points.
            patch_array (np.ndarray): Array of shape (num_patches, 1, 2) with patch coordinates.

    Returns:
            int: The index of the patch in patch_array that is closest to the contour center.
    """
    # Compute the contour's geometric center; keep dims to get shape (1,2)
    contour_center = np.mean(contour, axis=0, keepdims=True)  # shape: (1,2)

    # Extract patch centers from patch_array (from shape (num_patches,1,2) to (num_patches,2))
    patch_centers = patch_array[:, 0, :]  # shape: (num_patches, 2)

    # Compute the distance matrix between the contour center and all patch centers.
    # This results in a (1, num_patches) matrix.
    dist_matrix = distance.cdist(contour_center, patch_centers)

    # if dist_matrix.min() > 1024:
    #     print('yes')

    # Return the index (as an int) of the patch with the smallest distance.
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
                                            args.tracks_subdir, args.tracks_suffix)

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

        spot_list = pd.DataFrame(columns=['track_id','slice_num','barcode','IoU','annotation_index'])

        for ti in range(len(tracking_results)):
            now_tracking = tracking_results[ti]
            track_id = now_tracking['track_id']
            # if len(now_tracking['contours']) < 10:
            # 	continue
            for si in range(len(now_tracking['contours'])):
                now_obj = now_tracking['contours'][si]
                now_idx = now_obj['index']
                now_slice = now_obj['slice_index']
                now_contour = all_slices[now_slice][now_idx]['polygon']

                now_image = images_slice[now_slice]['img']

                # now_contour[:,0] = (now_contour[:,0] - images_slice[now_slice]['real_x']) / 4
                # now_contour[:,1] = (now_contour[:,1] - images_slice[now_slice]['real_y']) / 4


                new_patch_array = np.zeros((patch_array.shape))
                new_patch_array[:,:,0] = (patch_array[:,:,0] - middle_ref_y) / 4
                new_patch_array[:,:,1] = (patch_array[:,:,1] - middle_ref_x) / 4

                #try:
                best_index = get_nearest_patch_index(now_contour[:,::-1], new_patch_array)
                # best_index_2 = get_nearest_patch_index(now_contour, new_patch_array)
                #except:
                #	continue
                print(best_index)
                if best_index is None:
                    print("No suitable patch found for this contour!")
                    continue  # Skip if no patch is found

                now_affine_img_folder = os.path.join(patch_list[best_index], 'registration_3D_ANTs')
                # now_affine_img_folder = os.path.join(patch_list[best_index])
                region_list = glob.glob(os.path.join(now_affine_img_folder, 'image_*.png'))
                region_list.sort(key=natural_keys)

                now_region = plt.imread(region_list[now_slice])[:,:,:3]

                abs_patch_x = patch_array[best_index,0,1]
                abs_patch_y = patch_array[best_index,0,0]

                now_patch_y = int((abs_patch_x - middle_ref_x) / 4)
                now_patch_x = int((abs_patch_y - middle_ref_y) / 4)

                center_y = int(now_contour[:,0].mean() - (now_patch_x - now_region.shape[0]))
                center_x = int(now_contour[:,1].mean() - (now_patch_y - now_region.shape[1]))

                patch_size = 1024
                now_patch = now_region # [center_x - patch_size: center_x + patch_size, center_y - patch_size: center_y + patch_size,:]
                # now_patch = now_region[center_y - patch_size: center_y + patch_size, center_x - patch_size: center_x + patch_size,:]

                # now_big_images = plt.imread(big_images[now_slice])[:,:,:3]
                # region_check = now_big_images[now_patch_x - patch_size: now_patch_x + patch_size, now_patch_y - patch_size:now_patch_y + patch_size]

                now_df_after = spot_list_after[now_slice]['spot']
                # now_df_ori = spot_list_ori[now_slice]['spot']

                # assert (now_df_after['barcode'] == now_df_ori['barcode']).all()

                num_points = 100
                batch_array_i = np.zeros((len(now_df_after), 100, 1, 2)).astype(np.float32)
                for ni in range(num_points):
                    batch_array_i[:, ni, :, 0] = np.expand_dims(np.array(now_df_after['now_y_%d' % (ni)]), 1)
                    batch_array_i[:, ni, :, 1] = np.expand_dims(np.array(now_df_after['now_x_%d' % (ni)]), 1)
                #
                # num_points = 100
                # batch_array_i_ori = np.zeros((len(now_df_ori), 100, 1, 2)).astype(np.float32)
                # for ni in range(num_points):
                #     batch_array_i_ori[:, ni, :, 0] = np.expand_dims(np.array(now_df_ori['now_y_%d' % (ni)]), 1)
                #     batch_array_i_ori[:, ni, :, 1] = np.expand_dims(np.array(now_df_ori['now_x_%d' % (ni)]), 1)

                #try:
                polygon_now = Polygon(now_contour)
                #except:
                 #   continue

                # Ensure now_spot has shape (n_points, 1, 2)
                if now_contour.ndim == 2:
                    now_contour = now_contour.reshape((-1, 1, 2))

                # Adjust the coordinates relative to the patch
                'big image'
                # contour_int = np.ascontiguousarray(now_contour.astype(np.int32))
                #
                # now_big_images = (now_big_images * 255).astype(np.uint8)
                # # try:
                # cv2.drawContours(now_big_images, [contour_int], -1, (255, 255, 0), 20)

                now_contour[:, :, 0] = now_contour[:, :, 0] - (now_patch_y - patch_size)
                now_contour[:, :, 1] = now_contour[:, :, 1] - (now_patch_x - patch_size)

                # print(now_contour[:, :, :].max(), now_contour[:, :, :].min())
                # assert now_contour.min() > 0 and now_contour.max() < 2048

                # # Print shape and nonzero count for debugging
                # print("now_spot shape after adjustment:", now_spot.shape)
                # print("Non-zero points in now_spot:", np.count_nonzero(now_spot))

                # Convert contour to int32 and ensure it is contiguous
                # now_contour_swapped = now_contour[:, :, ::-1]

                contour_int = np.ascontiguousarray(now_contour.astype(np.int32))

                now_patch = (now_patch * 255).astype(np.uint8)
                # try:
                cv2.drawContours(now_patch, [contour_int], -1, (255, 255, 0), 20)
                print("draw the contour")
                print(contour_int[:,:,0].mean(), contour_int[:,:,1].mean())

                # region_check = (region_check * 255).astype(np.uint8)
                # cv2.drawContours(region_check, [contour_int], -1, (255, 255, 0), 20)

                # except:
                #    continue

                for i in range(batch_array_i.shape[0]):
                    # Remove the extra dimension: reshape from (100, 1, 2) to (100, 2)
                    candidate_contour = batch_array_i[i].reshape(-1, 2)
                    # candidate_contour = candidate_contour[:, ::-1]

                    # Convert candidate_contour to a Shapely Polygon.
                    candidate_poly = Polygon(candidate_contour)

                    # Compute IoU using your defined function.
                    iou = compute_iou(polygon_now, candidate_poly)

                    now_spot = batch_array_i[i]
                    # now_spot = now_spot[:, ::-1]

                    if now_spot.ndim == 2:
                        now_spot = now_spot.reshape((-1, 1, 2))

                    # Adjust the coordinates relative to the patch
                    now_spot[:, :, 0] = now_spot[:, :, 0] - (now_patch_y - patch_size)
                    now_spot[:, :, 1] = now_spot[:, :, 1] - (now_patch_x - patch_size)

                    # # Print shape and nonzero count for debugging
                    # print("now_spot shape after adjustment:", now_spot.shape)
                    # print("Non-zero points in now_spot:", np.count_nonzero(now_spot))

                    # Convert contour to int32 and ensure it is contiguous
                    contour_int = np.ascontiguousarray(now_spot.astype(np.int32))

                    if not (now_spot.min() > 0 and now_spot.max() < patch_size * 2):
                        continue

                    if iou > 0:
                        row = len(spot_list)
                        spot_list.loc[row] = [track_id,now_slice,now_df_after['barcode'][i],iou,now_idx]
                        cv2.drawContours(now_patch, [contour_int], -1, (0, 255, 0), 10)
                    else:
                        cv2.drawContours(now_patch, [contour_int], -1, (0, 0, 255), 10)

                save_folder = os.path.join(tracking_output, str(track_id))

                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                plt.imsave(os.path.join(save_folder, 'Glo_slice_%d.png' % (now_slice)), now_patch)
                # plt.imsave(os.path.join(save_folder, 'Glo_slice_%d_region_check.png' % (now_slice)), region_check)
                # plt.imsave(os.path.join(save_folder, 'Glo_slice_%d_big.png' % (now_slice)), now_big_images)

        csv_root = os.path.join(tracking_output,'IoU_withspot.csv')
        spot_list.to_csv(csv_root)


