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
"""Spot-to-bin IoU between every Visium V2 section and every Visium HD section.

THiRST stage 3 - cross-resolution association (paper Sec. 4.1.7, Algorithm 4; ISBI Eq. 4).

For each V2 (even position) x HD (odd position) section pair, batches of polygons are skipped when all
centroid distances exceed 30 px; otherwise IoUs are computed. Results are written per batch of 1000
V2 spots so long runs can be resumed; merge them with ``merge_v2_hd_iou.py``.

Outputs: ``<contour_root>/<case>/IoU_ST_to_HD_distance_<stage>reg/row_<i>_col_<j>_<batch>.json``

Usage
-----
    python scripts/3_st_association/v2_hd_iou.py --config configs/example.yaml --registration patch

Original script(s): Step9.7_contourMatching_parellel_json_ST_to_HD_sep_usedistance_patchreg.py
"""
import sys
import os
import glob
import json
import re
import gc

import numpy as np
import shapely
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from thirst.config import add_registration_argument, list_cases, parse_args, registered_dir, require_alternating_platforms  # noqa: E402
from thirst.st import is_circle_table, n_batches, registered_rows, require_same_barcodes, selected_barcodes  # noqa: E402


def atoi(text):
    return int(text) if text.isdigit() else text

def flatten_contours(contours_array):
    # Reshape to [num_contours, num_points, 2]
    return contours_array.reshape(contours_array.shape[0], -1, 2)


# Step 3: Compute IoU between each circle and each bin
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def process_batch_pair(bi, bj):
    # Compute batch indices
    i_start = bi * batch_size_i
    i_end = min((bi + 1) * batch_size_i, batch_array_i.shape[0])
    j_start = bj * batch_size_j
    j_end = min((bj + 1) * batch_size_j, batch_array_j.shape[0])


    # calculate center distance
    batch_center_bi = batch_array_center_i[i_start:i_end, 0, :]
    batch_center_bj = batch_array_center_j[j_start:j_end, 0, :]

    # Compute pairwise distances between centers
    distances = cdist(batch_center_bi, batch_center_bj, metric='euclidean')

    # Check if all distances are greater than 30
    if np.all(distances > 30):
        # Create a zero IoU matrix
        iou_matrix = np.zeros((len(batch_center_bi), len(batch_center_bj)), dtype=np.float32)
        # Get column names
        column_names = barcodes_j[j_start:j_end]
        # Return results early with zero IoU
        return i_start, i_end, column_names, iou_matrix

    else:
        # Extract batches
        batch_array_bi = batch_array_i[i_start:i_end]
        batch_array_bj = batch_array_j[j_start:j_end]

        # Flatten contours
        flattened_i = flatten_contours(batch_array_bi)
        flattened_j = flatten_contours(batch_array_bj)

        # Create polygons
        polygons_i = shapely.polygons(flattened_i)
        polygons_j = shapely.polygons(flattened_j)

        # Compute IoUs
        intersections = shapely.intersection(polygons_i[:, None], polygons_j[None, :])
        unions = shapely.union(polygons_i[:, None], polygons_j[None, :])
        intersection_areas = shapely.area(intersections)
        union_areas = shapely.area(unions)
        iou_matrix = np.divide(
            intersection_areas,
            union_areas,
            out=np.zeros_like(intersection_areas),
            where=union_areas != 0
        )

        # Get column names
        column_names = barcodes_j[j_start:j_end]

        # Return results
        return i_start, i_end, column_names, iou_matrix


if __name__ == '__main__':
    args, cfg = parse_args(__doc__, add_registration_argument)
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
        # ST unit centres / polygons after the selected registration stage (--registration)
        reconstruction_dir = registered_dir(case.replace(case_folder,point_folder), args.registration)
        reconstruction_contour_dir = registered_dir(case.replace(case_folder,contour_folder), args.registration)
        IoU_dir = os.path.join(case.replace(case_folder,contour_folder), 'IoU_ST_to_HD_distance_%sreg' % args.registration.replace('none', 'no'))


        if not os.path.exists(IoU_dir):
            os.makedirs(IoU_dir)

        images = glob.glob(os.path.join(image_input_dir, 'image_*.png'))
        images.sort(key=natural_keys)
        require_alternating_platforms(cfg, images)  # V2 at even, HD at odd stack positions

        middle_idx = int(len(images) / 2)

        # print(middle_idx)
        # middle_image = plt.imread(images[middle_idx])[:,:,:3]

        middle_ref_x = int(os.path.basename(images[middle_idx]).split('_')[2])
        middle_ref_y = int(os.path.basename(images[middle_idx]).split('_')[3].replace('.png', ''))

        patch_level_folder = os.path.join(case,'3D_Patch_crop')

        patch_list = glob.glob(os.path.join(patch_level_folder,'*'))

        patch_array = np.zeros((len(patch_list), 1, 2)).astype(np.float32)
        for pi in range(len(patch_list)):
            if '3D_voxel_visualizations' in patch_list[pi]:
                continue

            patch_name = os.path.basename(patch_list[pi])
            patch_abs_x, patch_abs_y = patch_name.split('_')
            patch_array[pi,:,0] = int(patch_abs_y)
            patch_array[pi,:,1] = int(patch_abs_x)

        achieve_pair = []


        for ii in range(0,len(images),2):
            #now_idx = ii
            real_ind_ii = int(os.path.basename(images[ii]).split('_')[1])

            data_i = pd.read_csv(images[ii].replace(case, reconstruction_dir).replace('.png', '_center.csv'))
            data_contour_i = pd.read_csv(images[ii].replace(case, reconstruction_contour_dir).replace('.png', '_spot.csv'))
            require_same_barcodes(data_i, data_contour_i, os.path.basename(images[ii]))

            for jj in range(1,len(images),2):   # HD sections (odd positions)
                # now_idx = jj
                real_ind_jj = int(os.path.basename(images[jj]).split('_')[1])

                data_j = pd.read_csv(images[jj].replace(case, reconstruction_dir).replace('.png', '_center.csv'))
                data_contour_j = pd.read_csv(images[jj].replace(case, reconstruction_contour_dir).replace('.png', '_spot.csv'))
                require_same_barcodes(data_j, data_contour_j, os.path.basename(images[jj]))

                if ii == jj:
                    continue

                elif ((ii,jj) in achieve_pair) or ((jj,ii) in achieve_pair):
                    continue

                else:
                    achieve_pair.append((ii,jj))
                    achieve_pair.append((jj,ii))
                    #
                    # if os.path.exists(os.path.join(IoU_dir, 'row_%d_col_%d.csv' % (real_ind_ii, real_ind_jj))):
                    #     continue

                    # units with registered coordinates (row positions) and their barcodes, in the same
                    # order as the polygon arrays below: entry k of barcodes_i belongs to polygon k
                    inds_i = registered_rows(data_i)
                    inds_j = registered_rows(data_j)
                    barcodes_i = selected_barcodes(data_contour_i, inds_i)
                    barcodes_j = selected_barcodes(data_contour_j, inds_j)

                    print('now_i: ', os.path.basename(images[ii]), 'now_j: ', os.path.basename(images[jj]))


                    batch_array_center_i = np.zeros((len(inds_i), 1, 2)).astype(np.float32)
                    batch_array_center_i[:, :, 0] = np.expand_dims(np.array(data_i['now_y'][inds_i]), 1)
                    batch_array_center_i[:, :, 1] = np.expand_dims(np.array(data_i['now_x'][inds_i]), 1)

                    batch_array_center_j = np.zeros((len(inds_j), 1, 2)).astype(np.float32)
                    batch_array_center_j[:, :, 0] = np.expand_dims(np.array(data_j['now_y'][inds_j]), 1)
                    batch_array_center_j[:, :, 1] = np.expand_dims(np.array(data_j['now_x'][inds_j]), 1)


                    if is_circle_table(data_contour_i, images[ii]):
                        num_points = 100
                        batch_array_i = np.zeros((len(inds_i), 100, 1, 2)).astype(np.float32)
                        for ni in range(num_points):
                            batch_array_i[:, ni, :, 0] = np.expand_dims(np.array(data_contour_i['now_y_%d' % (ni)][inds_i]), 1)
                            batch_array_i[:, ni, :, 1] = np.expand_dims(np.array(data_contour_i['now_x_%d' % (ni)][inds_i]), 1)

                    else:
                        num_points = 4
                        batch_array_i = np.zeros((len(inds_i), 4, 1, 2)).astype(np.float32)
                        for ni in range(num_points):
                            batch_array_i[:, ni, :, 0] = np.expand_dims(np.array(data_contour_i['now_y_%d' % (ni)][inds_i]), 1)
                            batch_array_i[:, ni, :, 1] = np.expand_dims(np.array(data_contour_i['now_x_%d' % (ni)][inds_i]), 1)

                    if is_circle_table(data_contour_j, images[jj]):
                        num_points = 100
                        batch_array_j = np.zeros((len(inds_j), 100, 1, 2)).astype(np.float32)
                        for ni in range(num_points):
                            batch_array_j[:, ni, :, 0] = np.expand_dims(np.array(data_contour_j['now_y_%d' % (ni)][inds_j]), 1)
                            batch_array_j[:, ni, :, 1] = np.expand_dims(np.array(data_contour_j['now_x_%d' % (ni)][inds_j]), 1)

                    else:
                        num_points = 4
                        batch_array_j = np.zeros((len(inds_j), 4, 1, 2)).astype(np.float32)
                        for ni in range(num_points):
                            batch_array_j[:, ni, :, 0] = np.expand_dims(np.array(data_contour_j['now_y_%d' % (ni)][inds_j]), 1)
                            batch_array_j[:, ni, :, 1] = np.expand_dims(np.array(data_contour_j['now_x_%d' % (ni)][inds_j]), 1)

                    # Convert contours to Shapely polygons
                    batch_size_i = 1000
                    batch_size_j = 1000
                    batch_i = n_batches(batch_array_i.shape[0], batch_size_i)
                    batch_j = n_batches(batch_array_j.shape[0], batch_size_j)

                    # output = pd.DataFrame(columns=['barcode'] + (data_contour_j['barcode'][inds_j].unique().tolist()))
                    # output['barcode'] = data_contour_i['barcode'][inds_i]

                    # Create a dictionary to store IoU values
                    # output_dict = {}

                    # Parallel processing
                    batch_pairs = [(bi, bj) for bi in range(batch_i) for bj in range(batch_j)]

                    for bi in range(batch_i):
                        # Process only the current bi with all bjs

                        if os.path.exists(os.path.join(IoU_dir, 'row_%d_col_%d_%d.json' % (real_ind_ii, real_ind_jj, bi))):
                            print('pass', os.path.join(IoU_dir, 'row_%d_col_%d_%d.json' % (real_ind_ii, real_ind_jj, bi)))
                            continue

                        print('generate', os.path.join(IoU_dir, 'row_%d_col_%d_%d.json' % (real_ind_ii, real_ind_jj, bi)))
                        results = Parallel(n_jobs=-1)(
                            delayed(process_batch_pair)(bi, bj) for bj in range(batch_j)
                        )

                        # Save results for the current bi
                        output_dict = {}
                        for i_start, i_end, column_names, iou_matrix in results:
                            # Convert iou_matrix to a NumPy array for efficient processing
                            iou_matrix = np.array(iou_matrix)

                            # Find the indices where IoU is greater than 0
                            rows, cols = np.where(iou_matrix > 0)

                            # Process all positive IoUs efficiently
                            for row_idx, col_idx in zip(rows, cols):
                                barcode_i = barcodes_i[i_start + row_idx]
                                barcode_j = column_names[col_idx]
                                iou_value = iou_matrix[row_idx, col_idx]

                                if barcode_i not in output_dict:
                                    output_dict[barcode_i] = []

                                output_dict[barcode_i].append({
                                    "barcode_j": barcode_j,
                                    "iou": iou_value
                                })
                            # Clean up
                            del iou_matrix

                        # Save the output dictionary for the current `bi`
                        if output_dict:
                            with open(os.path.join(IoU_dir, 'row_%d_col_%d_%d.json' % (real_ind_ii, real_ind_jj, bi)),'w') as json_file:
                                json.dump(output_dict, json_file, indent=4)

                        # Delete variables to free up memory
                        del output_dict, results
                        gc.collect()  # Force garbage collection to free memory


