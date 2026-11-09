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
"""Refine the ST unit polygons with the nearest patch-level registration.

THiRST stage 2 - local refinement of ST coordinates (paper Sec. 4.1.6, Algorithm 2).

Outputs: ``<contour_root>/<case>/registration_3D_patch_ANTs/image_*_spot.csv``

Usage
-----
    python scripts/2_coordinate_transfer/patch_spot_contours.py --config configs/example.yaml

Original script(s): Step8.5_registertoMiddle_ANTs_points_contour_patchlevel_fast.py
"""
import sys
import os
import glob
import re

import numpy as np
import cv2 as cv2
import pandas as pd
import SimpleITK as sitk
from scipy.spatial import distance

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from thirst.config import list_cases, parse_args, require_platform  # noqa: E402
from thirst.transforms import require_patch_chains, require_transform  # noqa: E402
from thirst.st import is_circle_table, require_same_barcodes  # noqa: E402


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


if __name__ == '__main__':
    args, cfg = parse_args(__doc__)
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
        global_reconstruction_dir = os.path.join(case.replace(case_folder,point_folder), 'registration_3D_ANTs')
        global_reconstruction_contour_dir = os.path.join(case.replace(case_folder,contour_folder), 'registration_3D_ANTs')


        if not os.path.exists(reconstruction_contour_dir):
            os.makedirs(reconstruction_contour_dir)

        images = glob.glob(os.path.join(image_input_dir, 'image_*.png'))
        images.sort(key=natural_keys)
        require_patch_chains(case, images)  # every pair of every patch stack, before anything is written

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
            patch_array[pi,:,0] = int(patch_abs_y)
            patch_array[pi,:,1] = int(patch_abs_x)


        for ii in range(len(images)):
            now_idx = ii

            if now_idx == middle_idx:

                data = pd.read_csv(images[now_idx].replace(case, global_reconstruction_dir).replace('.png', '_center.csv'))
                data_contour = pd.read_csv(images[now_idx].replace(case, global_reconstruction_contour_dir).replace('.png', '_spot.csv'))

                new_data = data_contour

                # Add two new columns with NaN values
                new_data['affine_patch_abs_x'] = pd.Series(np.nan, index=new_data.index, dtype=object)  # holds patch names
                new_data['affine_patch_abs_y'] = pd.Series(np.nan, index=new_data.index, dtype=object)  # holds patch names

                ori_array = np.zeros((len(data), 1, 2)).astype(np.float32)
                ori_array[:, :, 0] = np.expand_dims(np.array(data['abs_y']), 1)
                ori_array[:, :, 1] = np.expand_dims(np.array(data['abs_x']), 1)

                dist_matrix = distance.cdist(ori_array[:, 0, :], patch_array[:, 0, :])
                min_indices = np.argmin(dist_matrix, axis=1)

                # closest_pairs = [(tuple(ori_array[i]), tuple(patch_array[min_indices[i]]), dist_matrix[i][min_indices[i]]) for i in range(len(ori_array))]

                closest_pairs_array = np.zeros((len(ori_array), 5))
                for oi in range(len(ori_array)):
                    closest_pairs_array[oi, 0] = ori_array[oi][0, 0]
                    closest_pairs_array[oi, 1] = ori_array[oi][0, 1]
                    closest_pairs_array[oi, 2] = patch_array[min_indices[oi]][0, 0]
                    closest_pairs_array[oi, 3] = patch_array[min_indices[oi]][0, 1]
                    closest_pairs_array[oi, 4] = dist_matrix[oi][min_indices[oi]]

                for pi in range(len(patch_list)):
                    patch_name = os.path.basename(patch_list[pi])
                    patch_abs_x, patch_abs_y = patch_name.split('_')
                    patch_now_x = int(np.round((int(patch_abs_x) - middle_ref_y) / 4))
                    patch_now_y = int(np.round((int(patch_abs_y) - middle_ref_x) / 4))

                    batch_ind = np.where((closest_pairs_array[:, 2].astype(int) == int(patch_abs_y)) & (
                                    closest_pairs_array[:, 3].astype(int) == int(patch_abs_x)) & (closest_pairs_array[:, 4].astype(int) < 1024 * 2))

                    new_data.loc[batch_ind[0], 'affine_patch_abs_x'] = patch_abs_x
                    new_data.loc[batch_ind[0], 'affine_patch_abs_y'] = patch_abs_y

                new_data.to_csv(os.path.join(reconstruction_contour_dir, os.path.basename(images[middle_idx]).replace('.png', '_spot.csv')), index = False)
                continue

            data = pd.read_csv(images[now_idx].replace(case, global_reconstruction_dir).replace('.png', '_center.csv'))
            data_contour = pd.read_csv(images[now_idx].replace(case, global_reconstruction_contour_dir).replace('.png', '_spot.csv'))

            now_ref_x = int(os.path.basename(images[now_idx]).split('_')[2])
            now_ref_y = int(os.path.basename(images[now_idx]).split('_')[3].replace('.png', ''))

            # V2 spots are 100-vertex polygons, HD bins 4-corner squares; decided from the polygon table
            # itself (the number of units says nothing about the platform)
            circle = is_circle_table(data_contour, images[now_idx])
            require_platform(cfg, images[now_idx], circle)   # must agree with st_sections
            require_same_barcodes(data, data_contour, os.path.basename(images[now_idx]))

            ori_array = np.zeros((len(data), 1, 2)).astype(np.float32)
            ori_array[:, :, 0] = np.expand_dims(np.array(data['abs_y']), 1)
            ori_array[:, :, 1] = np.expand_dims(np.array(data['abs_x']), 1)

            dist_matrix = distance.cdist(ori_array[:,0,:], patch_array[:,0,:])
            min_indices = np.argmin(dist_matrix, axis=1)

            # closest_pairs = [(tuple(ori_array[i]), tuple(patch_array[min_indices[i]]), dist_matrix[i][min_indices[i]]) for i in range(len(ori_array))]

            closest_pairs_array = np.zeros((len(ori_array), 5))
            for oi in range(len(ori_array)):
                closest_pairs_array[oi,0] = ori_array[oi][0,0]
                closest_pairs_array[oi,1] = ori_array[oi][0,1]
                closest_pairs_array[oi,2] = patch_array[min_indices[oi]][0,0]
                closest_pairs_array[oi,3] = patch_array[min_indices[oi]][0,1]
                closest_pairs_array[oi,4] = dist_matrix[oi][min_indices[oi]]

            if circle:
                num_points = 100 # int((len(data_contour.columns) - 2) / 2)
                columns = ['barcode','affine_patch_abs_x','affine_patch_abs_y'] + \
                          ['now_x_%d' % ki for ki in range(num_points)] + \
                          ['now_y_%d' % ki for ki in range(num_points)]

                new_data = pd.DataFrame(columns=columns)
                new_data['barcode'] = data['barcode']

                # make everything zero to mask out
                all_array = np.zeros((len(new_data), 100, 1, 2)).astype(np.float32)
                for ni in range(num_points):
                    new_data['now_x_%d' % (ni)] = np.round(all_array[:, ni, 0, 1]).astype(int).tolist()
                    new_data['now_y_%d' % (ni)] = np.round(all_array[:, ni, 0, 0]).astype(int).tolist()

            else:
                new_data = pd.DataFrame(
                        columns=['barcode', 'now_x_0', 'now_y_0', 'now_x_1', 'now_y_1', 'now_x_2', 'now_y_2', 'now_x_3',
                                 'now_y_3','affine_patch_abs_x','affine_patch_abs_y'])
                new_data['barcode'] = data['barcode']

                # make everything zero to mask out
                all_array = np.zeros((len(new_data), 4, 1, 2)).astype(np.float32)
                new_data['now_x_0'] = all_array[:, 0, 0, 1].astype(int).tolist()
                new_data['now_x_1'] = all_array[:, 1, 0, 1].astype(int).tolist()
                new_data['now_x_2'] = all_array[:, 2, 0, 1].astype(int).tolist()
                new_data['now_x_3'] = all_array[:, 3, 0, 1].astype(int).tolist()

                new_data['now_y_0'] = all_array[:, 0, 0, 0].astype(int).tolist()
                new_data['now_y_1'] = all_array[:, 1, 0, 0].astype(int).tolist()
                new_data['now_y_2'] = all_array[:, 2, 0, 0].astype(int).tolist()
                new_data['now_y_3'] = all_array[:, 3, 0, 0].astype(int).tolist()

            for pi in range(len(patch_list)):
                patch_name = os.path.basename(patch_list[pi])
                patch_abs_x, patch_abs_y = patch_name.split('_')
                patch_now_x = int(np.round((int(patch_abs_x) - middle_ref_y) / 4))
                patch_now_y = int(np.round((int(patch_abs_y) - middle_ref_x) / 4))

                # batch_ind = np.where((closest_pairs_array[:,2].astype(int)==int(patch_abs_y)) & (closest_pairs_array[:,3].astype(int)==int(patch_abs_x)) & (closest_pairs_array[:,4].astype(int) < 1024 * 5))
                batch_ind = np.where((closest_pairs_array[:,2].astype(int)==int(patch_abs_y)) & (closest_pairs_array[:,3].astype(int)==int(patch_abs_x)))

                if circle:
                    if len(batch_ind[0]) == 0:
                        continue

                    print(closest_pairs_array[batch_ind[0], 4].max())

                    batch_array = np.zeros((len(batch_ind[0]), 100, 1, 2)).astype(np.float32)
                    for ni in range(num_points):
                        batch_array[:, ni, :, 0] = np.expand_dims(np.array(data_contour['now_y_%d' % (ni)][batch_ind[0]]), 1)
                        batch_array[:, ni, :, 1] = np.expand_dims(np.array(data_contour['now_x_%d' % (ni)][batch_ind[0]]), 1)

                    # batch_array[:,:,0] = np.expand_dims(np.array(data['now_y'][batch_ind[0]]),1)
                    # batch_array[:,:,1] = np.expand_dims(np.array(data['now_x'][batch_ind[0]]),1)

                    'get reference location on the image'
                    batch_array[:, :, :, 0] = batch_array[:, :, :, 0] - (patch_now_y - 1024)
                    batch_array[:, :, :, 1] = batch_array[:, :, :, 1] - (patch_now_x - 1024)

                    has_negatives = np.any(batch_array < 0)
                    has_large_values = np.any(batch_array > 2048)

                    # assert has_negatives == 0 and has_large_values == 0

                else:
                    if len(batch_ind[0]) == 0:
                        continue

                    print(closest_pairs_array[batch_ind[0], 4].max())

                    batch_array = np.zeros((len(batch_ind[0]), 4, 1, 2)).astype(np.float32)

                    batch_array[:, 0, 0, 1] = np.array(data_contour['now_x_0'])[batch_ind[0]]
                    batch_array[:, 1, 0, 1] = np.array(data_contour['now_x_1'])[batch_ind[0]]
                    batch_array[:, 2, 0, 1] = np.array(data_contour['now_x_2'])[batch_ind[0]]
                    batch_array[:, 3, 0, 1] = np.array(data_contour['now_x_3'])[batch_ind[0]]

                    batch_array[:, 0, 0, 0] = np.array(data_contour['now_y_0'])[batch_ind[0]]
                    batch_array[:, 1, 0, 0] = np.array(data_contour['now_y_1'])[batch_ind[0]]
                    batch_array[:, 2, 0, 0] = np.array(data_contour['now_y_2'])[batch_ind[0]]
                    batch_array[:, 3, 0, 0] = np.array(data_contour['now_y_3'])[batch_ind[0]]

                    'get reference location on the image'
                    batch_array[:, :, :, 0] = batch_array[:, :, :, 0] - (patch_now_y - 1024)
                    batch_array[:, :, :, 1] = batch_array[:, :, :, 1] - (patch_now_x - 1024)

                    has_negatives = np.any(batch_array < 0)
                    has_large_values = np.any(batch_array > 2048)

                    # assert has_negatives == 0 and has_large_values == 0

                if now_idx < middle_idx:
                    M_new = np.zeros((3, 3))
                    M_new[2, 2] = 1.

                    for ri in range(now_idx, middle_idx):

                        moving_index = int(os.path.basename(images[ri+1]).split('.')[0].split('_')[1])
                        fixed_index = int(os.path.basename(images[ri]).split('.')[0].split('_')[1])

                        affine_root = os.path.join(patch_list[pi], 'registration_ANTs' , '%d_to_%d' % (moving_index, fixed_index), 'step2_run_ants_reg', 'output0GenericAffine.mat')
                        m = sitk.ReadTransform(require_transform(affine_root))
                        FixP = m.GetFixedParameters()
                        M_vec = m.GetParameters()
                        M_inv = np.zeros((2, 3))

                        M_inv[0, 0] = M_vec[0]
                        M_inv[0, 1] = M_vec[1]
                        M_inv[1, 0] = M_vec[2]
                        M_inv[1, 1] = M_vec[3]
                        M_inv[0, 2] = M_vec[4] - (FixP[0] * M_inv[0, 0] + FixP[1] * M_inv[0, 1] - FixP[0])
                        M_inv[1, 2] = M_vec[5] - (FixP[0] * M_inv[1, 0] + FixP[1] * M_inv[1, 1] - FixP[1])

                        M1 = np.zeros((3, 3))
                        M1[2, 2] = 1.
                        M1[:2, :] = cv2.invertAffineTransform(M_inv)

                        if ri == now_idx:
                            M_new = M1.copy()
                        else:
                            M_new = M_new.dot(M1)

                    affine_matrix_inv = cv2.invertAffineTransform(M_new[:2, :])
                    M_new[:2, :] = affine_matrix_inv

                    transformed_batch_array = cv2.perspectiveTransform(batch_array.reshape(batch_array.shape[0] * batch_array.shape[1], batch_array.shape[2], batch_array.shape[3]), M_new).reshape(batch_array.shape[0], batch_array.shape[1], batch_array.shape[2], batch_array.shape[3])

                    transformed_batch_array[:, :, :, 0] = transformed_batch_array[:, :, :, 0] + (patch_now_y - 1024)
                    transformed_batch_array[:, :, :, 1] = transformed_batch_array[:, :, :, 1] + (patch_now_x - 1024)

                    if circle:
                        for ni in range(num_points):
                            new_data.loc[batch_ind[0], 'now_x_%d' % (ni)] = np.round(transformed_batch_array[:, ni, 0, 1]).astype(int).tolist()
                            new_data.loc[batch_ind[0], 'now_y_%d' % (ni)] = np.round(transformed_batch_array[:, ni, 0, 0]).astype(int).tolist()

                    else:
                        new_data.loc[batch_ind[0], 'now_x_0'] = np.round(transformed_batch_array[:, 0, 0, 1]).astype(int).tolist()
                        new_data.loc[batch_ind[0], 'now_x_1'] = np.round(transformed_batch_array[:, 1, 0, 1]).astype(int).tolist()
                        new_data.loc[batch_ind[0], 'now_x_2'] = np.round(transformed_batch_array[:, 2, 0, 1]).astype(int).tolist()
                        new_data.loc[batch_ind[0], 'now_x_3'] = np.round(transformed_batch_array[:, 3, 0, 1]).astype(int).tolist()

                        new_data.loc[batch_ind[0], 'now_y_0'] = np.round(transformed_batch_array[:, 0, 0, 0]).astype(int).tolist()
                        new_data.loc[batch_ind[0], 'now_y_1'] = np.round(transformed_batch_array[:, 1, 0, 0]).astype(int).tolist()
                        new_data.loc[batch_ind[0], 'now_y_2'] = np.round(transformed_batch_array[:, 2, 0, 0]).astype(int).tolist()
                        new_data.loc[batch_ind[0], 'now_y_3'] = np.round(transformed_batch_array[:, 3, 0, 0]).astype(int).tolist()

                elif now_idx > middle_idx:
                    M_new = np.zeros((3, 3))
                    M_new[2, 2] = 1.

                    for ri in range(middle_idx, now_idx):

                        moving_index = int(os.path.basename(images[ri+1]).split('.')[0].split('_')[1])
                        fixed_index = int(os.path.basename(images[ri]).split('.')[0].split('_')[1])

                        affine_root = os.path.join(patch_list[pi], 'registration_ANTs' , '%d_to_%d' % (moving_index, fixed_index),'step2_run_ants_reg', 'output0GenericAffine.mat')

                        m = sitk.ReadTransform(require_transform(affine_root))
                        FixP = m.GetFixedParameters()
                        M_vec = m.GetParameters()
                        M_inv = np.zeros((2, 3))

                        M_inv[0, 0] = M_vec[0]
                        M_inv[0, 1] = M_vec[1]
                        M_inv[1, 0] = M_vec[2]
                        M_inv[1, 1] = M_vec[3]
                        M_inv[0, 2] = M_vec[4] - (FixP[0] * M_inv[0, 0] + FixP[1] * M_inv[0, 1] - FixP[0])
                        M_inv[1, 2] = M_vec[5] - (FixP[0] * M_inv[1, 0] + FixP[1] * M_inv[1, 1] - FixP[1])

                        M1 = np.zeros((3, 3))
                        M1[2, 2] = 1.
                        M1[:2, :] = cv2.invertAffineTransform(M_inv)

                        if ri == middle_idx:
                            M_new = M1.copy()
                        else:
                            M_new = M_new.dot(M1)


                    transformed_batch_array = cv2.perspectiveTransform(batch_array.reshape(batch_array.shape[0] * batch_array.shape[1], batch_array.shape[2], batch_array.shape[3]), M_new).reshape(batch_array.shape[0], batch_array.shape[1], batch_array.shape[2], batch_array.shape[3])

                    transformed_batch_array[:, :, :, 0] = transformed_batch_array[:, :, :, 0] + (patch_now_y - 1024)
                    transformed_batch_array[:, :, :, 1] = transformed_batch_array[:, :, :, 1] + (patch_now_x - 1024)

                    if circle:
                        for ni in range(num_points):
                            new_data.loc[batch_ind[0], 'now_x_%d' % (ni)] = np.round(transformed_batch_array[:, ni, 0, 1]).astype(int).tolist()
                            new_data.loc[batch_ind[0], 'now_y_%d' % (ni)] = np.round(transformed_batch_array[:, ni, 0, 0]).astype(int).tolist()

                    else:
                        new_data.loc[batch_ind[0], 'now_x_0'] = np.round(transformed_batch_array[:, 0, 0, 1]).astype(int).tolist()
                        new_data.loc[batch_ind[0], 'now_x_1'] = np.round(transformed_batch_array[:, 1, 0, 1]).astype(int).tolist()
                        new_data.loc[batch_ind[0], 'now_x_2'] = np.round(transformed_batch_array[:, 2, 0, 1]).astype(int).tolist()
                        new_data.loc[batch_ind[0], 'now_x_3'] = np.round(transformed_batch_array[:, 3, 0, 1]).astype(int).tolist()

                        new_data.loc[batch_ind[0], 'now_y_0'] = np.round(transformed_batch_array[:, 0, 0, 0]).astype(int).tolist()
                        new_data.loc[batch_ind[0], 'now_y_1'] = np.round(transformed_batch_array[:, 1, 0, 0]).astype(int).tolist()
                        new_data.loc[batch_ind[0], 'now_y_2'] = np.round(transformed_batch_array[:, 2, 0, 0]).astype(int).tolist()
                        new_data.loc[batch_ind[0], 'now_y_3'] = np.round(transformed_batch_array[:, 3, 0, 0]).astype(int).tolist()

                new_data.loc[batch_ind[0], 'affine_patch_abs_x'] = patch_abs_x
                new_data.loc[batch_ind[0], 'affine_patch_abs_y'] = patch_abs_y

            new_data.to_csv(
                    os.path.join(reconstruction_contour_dir, os.path.basename(images[now_idx]).replace('.png', '_spot.csv')), index =  False)


