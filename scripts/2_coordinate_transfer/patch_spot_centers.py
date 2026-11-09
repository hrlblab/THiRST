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
"""Refine the ST unit centres with the nearest patch-level registration.

THiRST stage 2 - local refinement of ST coordinates (paper Sec. 4.1.6, Algorithm 2).

Every unit is assigned to the nearest patch centre, re-expressed in that patch, warped by the patch's
chained ANTs affines and shifted back to the reference frame. Units without a patch are set to 0.

Inputs : global outputs of ``global_spot_centers.py`` + ``<image_root>/<case>/3D_Patch_crop/*/registration_ANTs``
Outputs: ``<point_root>/<case>/registration_3D_patch_ANTs/image_*_center.csv``
         (+ affine_patch_abs_x / affine_patch_abs_y = patch used)

Usage
-----
    python scripts/2_coordinate_transfer/patch_spot_centers.py --config configs/example.yaml

Original script(s): Step8.5_registertoMiddle_ANTs_points_center_patchlevel_fast.py
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
from thirst.config import list_cases, parse_args  # noqa: E402
from thirst.transforms import require_patch_chains, require_transform  # noqa: E402


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

    cases = list_cases(cfg)
    for ci in range(len(cases)):
        case = cases[ci]
        now_case = os.path.basename(case)
        lbl_input_dir = case
        image_input_dir = case
        image_output_dir = os.path.join(case, 'registration_ANTs')
        # image_badcase_dir = os.path.join(case, 'registration_badcase_2048')
        reconstruction_dir = os.path.join(case.replace(case_folder,point_folder), 'registration_3D_patch_ANTs')
        global_reconstruction_dir = os.path.join(case.replace(case_folder,point_folder), 'registration_3D_ANTs')


        if not os.path.exists(reconstruction_dir):
            os.makedirs(reconstruction_dir)

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
                new_data = pd.read_csv(images[middle_idx].replace(case_folder, point_folder).replace('.png', '_center.csv'))
                new_data.to_csv(os.path.join(reconstruction_dir, os.path.basename(images[middle_idx]).replace('.png', '_center.csv')), index = False)
                continue

            data = pd.read_csv(images[now_idx].replace(case, global_reconstruction_dir).replace('.png', '_center.csv'))

            now_ref_x = int(os.path.basename(images[now_idx]).split('_')[2])
            now_ref_y = int(os.path.basename(images[now_idx]).split('_')[3].replace('.png', ''))

            new_data = pd.DataFrame(columns=['barcode', 'abs_x', 'abs_y', 'now_x', 'now_y', 'affine_patch_abs_x', 'affine_patch_abs_y'])
            new_data['barcode'] = data['barcode']

            ori_array = np.zeros((len(new_data), 1, 2)).astype(np.float32)
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

            # make everything zero to mask out
            all_array = np.zeros((len(new_data), 1, 2)).astype(np.float32)
            new_data['abs_x'] = (all_array[:, :, 0]).astype(int)[:, 0].tolist()
            new_data['abs_y'] = (all_array[:, :, 0]).astype(int)[:, 0].tolist()
            new_data['now_x'] = (all_array[:, :, 0]).astype(int)[:, 0].tolist()
            new_data['now_y'] = (all_array[:, :, 0]).astype(int)[:, 0].tolist()

            for pi in range(len(patch_list)):
                patch_name = os.path.basename(patch_list[pi])
                patch_abs_x, patch_abs_y = patch_name.split('_')
                patch_now_x = int(np.round((int(patch_abs_x) - middle_ref_y) / 4))
                patch_now_y = int(np.round((int(patch_abs_y) - middle_ref_x) / 4))

                # batch_ind = np.where((closest_pairs_array[:,2].astype(int)==int(patch_abs_y)) & (closest_pairs_array[:,3].astype(int)==int(patch_abs_x)) & (closest_pairs_array[:,4].astype(int) < 1024 * 5))
                batch_ind = np.where((closest_pairs_array[:,2].astype(int)==int(patch_abs_y)) & (closest_pairs_array[:,3].astype(int)==int(patch_abs_x)))

                if len(batch_ind[0]) == 0:
                    continue

                print(closest_pairs_array[batch_ind[0],4].max())
                batch_array = np.zeros((len(batch_ind[0]), 1, 2)).astype(np.float32)
                batch_array[:,:,0] = np.expand_dims(np.array(data['now_y'][batch_ind[0]]),1)
                batch_array[:,:,1] = np.expand_dims(np.array(data['now_x'][batch_ind[0]]),1)

                'get reference location on the image'
                batch_array[:, :, 0] = batch_array[:,:,0] - (patch_now_y - 1024)
                batch_array[:, :, 1] = batch_array[:, :, 1] - (patch_now_x - 1024)

                has_negatives = np.any(batch_array < 0)
                has_large_values = np.any(batch_array > 2048)
                print(has_negatives, has_large_values)
                if has_negatives or has_large_values:
                    print('aaa')

                #assert has_negatives==0 and has_large_values==0

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

                    transformed_batch_array = cv2.perspectiveTransform(batch_array, M_new)

                    transformed_batch_array[:, :, 0] = transformed_batch_array[:, :, 0] + (patch_now_y - 1024)
                    transformed_batch_array[:, :, 1] = transformed_batch_array[:, :, 1] + (patch_now_x - 1024)

                    now_abs_x = transformed_batch_array[:,:,1] * 4 + middle_ref_y
                    now_abs_y = transformed_batch_array[:,:,0] * 4 + middle_ref_x

                    new_data.loc[batch_ind[0], 'abs_x'] = np.round((now_abs_x)).astype(int)[:, 0].tolist()
                    new_data.loc[batch_ind[0], 'abs_y'] = np.round((now_abs_y)).astype(int)[:, 0].tolist()
                    new_data.loc[batch_ind[0], 'now_x'] = np.round((transformed_batch_array[:, :, 1])).astype(int)[:, 0].tolist()
                    new_data.loc[batch_ind[0], 'now_y'] = np.round((transformed_batch_array[:, :, 0])).astype(int)[:, 0].tolist()
                    new_data.loc[batch_ind[0], 'affine_patch_abs_x'] = patch_abs_x
                    new_data.loc[batch_ind[0], 'affine_patch_abs_y'] = patch_abs_y

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


                    transformed_batch_array = cv2.perspectiveTransform(batch_array, M_new)
                    transformed_batch_array[:, :, 0] = transformed_batch_array[:, :, 0] + (patch_now_y - 1024)
                    transformed_batch_array[:, :, 1] = transformed_batch_array[:, :, 1] + (patch_now_x - 1024)

                    now_abs_x = transformed_batch_array[:,:,1] * 4 + middle_ref_y
                    now_abs_y = transformed_batch_array[:,:,0] * 4 + middle_ref_x

                    new_data.loc[batch_ind[0], 'abs_x'] = np.round((now_abs_x)).astype(int)[:, 0].tolist()
                    new_data.loc[batch_ind[0], 'abs_y'] = np.round((now_abs_y)).astype(int)[:, 0].tolist()
                    new_data.loc[batch_ind[0], 'now_x'] = np.round((transformed_batch_array[:, :, 1])).astype(int)[:, 0].tolist()
                    new_data.loc[batch_ind[0], 'now_y'] = np.round((transformed_batch_array[:, :, 0])).astype(int)[:, 0].tolist()
                    new_data.loc[batch_ind[0], 'affine_patch_abs_x'] = patch_abs_x
                    new_data.loc[batch_ind[0], 'affine_patch_abs_y'] = patch_abs_y

            new_data.to_csv(
                    os.path.join(reconstruction_dir, os.path.basename(images[now_idx]).replace('.png', '_center.csv')), index = False)
            #
            #
            #
            #
            # transformed_batch_array = cv2.perspectiveTransform(batch_array, M_new)
            #
            # now_abs_x = transformed_batch_array[:, :, 1] * 4 + middle_ref_y
            # now_abs_y = transformed_batch_array[:, :, 0] * 4 + middle_ref_x
            #
            # new_data['abs_x'] = (now_abs_x).astype(int)[:, 0].tolist()
            # new_data['abs_y'] = (now_abs_y).astype(int)[:, 0].tolist()
            # new_data['now_x'] = (transformed_batch_array[:, :, 1]).astype(int)[:, 0].tolist()
            # new_data['now_y'] = (transformed_batch_array[:, :, 0]).astype(int)[:, 0].tolist()
            #
            # new_data.to_csv(os.path.join(reconstruction_dir, os.path.basename(images[now_idx]).replace('.png', '_center.csv')))
            #
            #
            # if now_idx < middle_idx:
            # 	M_new = np.zeros((3, 3))
            # 	M_new[2, 2] = 1.
            #
            # 	for ri in range(now_idx, middle_idx):
            #
            # 		moving_index = int(os.path.basename(images[ri+1]).split('.')[0].split('_')[1])
            # 		fixed_index = int(os.path.basename(images[ri]).split('.')[0].split('_')[1])
            #
            # 		affine_root = os.path.join(image_output_dir, '%d_to_%d' % (moving_index, fixed_index), 'step2_run_ants_reg', 'output0GenericAffine.mat')
            # 		m = sitk.ReadTransform(affine_root)
            # 		FixP = m.GetFixedParameters()
            # 		M_vec = m.GetParameters()
            # 		M_inv = np.zeros((2, 3))
            #
            # 		M_inv[0, 0] = M_vec[0]
            # 		M_inv[0, 1] = M_vec[1]
            # 		M_inv[1, 0] = M_vec[2]
            # 		M_inv[1, 1] = M_vec[3]
            # 		M_inv[0, 2] = M_vec[4] - (FixP[0] * M_inv[0, 0] + FixP[1] * M_inv[0, 1] - FixP[0])
            # 		M_inv[1, 2] = M_vec[5] - (FixP[0] * M_inv[1, 0] + FixP[1] * M_inv[1, 1] - FixP[1])
            #
            # 		M1 = np.zeros((3, 3))
            # 		M1[2, 2] = 1.
            # 		M1[:2, :] = cv2.invertAffineTransform(M_inv)
            #
            # 		if ri == now_idx:
            # 			M_new = M1.copy()
            # 		else:
            # 			M_new = M_new.dot(M1)
            #
            # 	affine_matrix_inv = cv2.invertAffineTransform(M_new[:2, :])
            # 	M_new[:2, :] = affine_matrix_inv
            #
            # 	# now_image = plt.imread(images[now_idx])[:,:,:3]
            # 	# img1_affine = warp(now_image, M_new, output_shape=(middle_image.shape[0], middle_image.shape[1]))
            # 	# new_root = images[now_idx].replace(image_input_dir, reconstruction_dir)
            # 	# plt.imsave(new_root, img1_affine)
            #
            # 	data = pd.read_csv(images[now_idx].replace(case_folder, point_folder).replace('.png','_center.csv'))
            #
            # 	now_ref_x = int(os.path.basename(images[now_idx]).split('_')[2])
            # 	now_ref_y = int(os.path.basename(images[now_idx]).split('_')[3].replace('.png', ''))
            #
            # 	new_data = pd.DataFrame(columns = ['barcode','abs_x','abs_y','now_x','now_y'])
            # 	new_data['barcode'] = data['barcode']
            #
            # 	batch_array = np.zeros((len(new_data), 1, 2)).astype(np.float32)
            # 	batch_array[:,:,0] = np.expand_dims(np.array(data['now_y']),1)
            # 	batch_array[:,:,1] = np.expand_dims(np.array(data['now_x']),1)
            #
            # 	transformed_batch_array = cv2.perspectiveTransform(batch_array, M_new)
            #
            # 	now_abs_x = transformed_batch_array[:,:,1] * 4 + middle_ref_y
            # 	now_abs_y = transformed_batch_array[:,:,0] * 4 + middle_ref_x
            #
            # 	new_data['abs_x'] = (now_abs_x).astype(int)[:,0].tolist()
            # 	new_data['abs_y'] = (now_abs_y).astype(int)[:,0].tolist()
            # 	new_data['now_x'] = (transformed_batch_array[:,:,1]).astype(int)[:,0].tolist()
            # 	new_data['now_y'] = (transformed_batch_array[:,:,0]).astype(int)[:,0].tolist()
            #
            # 	new_data.to_csv(os.path.join(reconstruction_dir, os.path.basename(images[now_idx]).replace('.png', '_center.csv')))
            #
            # elif now_idx > middle_idx:
            # 	M_new = np.zeros((3, 3))
            # 	M_new[2, 2] = 1.
            #
            # 	for ri in range(middle_idx, now_idx):
            #
            # 		moving_index = int(os.path.basename(images[ri+1]).split('.')[0].split('_')[1])
            # 		fixed_index = int(os.path.basename(images[ri]).split('.')[0].split('_')[1])
            #
            # 		affine_root = os.path.join(image_output_dir, '%d_to_%d' % (moving_index, fixed_index),'step2_run_ants_reg', 'output0GenericAffine.mat')
            #
            # 		m = sitk.ReadTransform(affine_root)
            # 		FixP = m.GetFixedParameters()
            # 		M_vec = m.GetParameters()
            # 		M_inv = np.zeros((2, 3))
            #
            # 		M_inv[0, 0] = M_vec[0]
            # 		M_inv[0, 1] = M_vec[1]
            # 		M_inv[1, 0] = M_vec[2]
            # 		M_inv[1, 1] = M_vec[3]
            # 		M_inv[0, 2] = M_vec[4] - (FixP[0] * M_inv[0, 0] + FixP[1] * M_inv[0, 1] - FixP[0])
            # 		M_inv[1, 2] = M_vec[5] - (FixP[0] * M_inv[1, 0] + FixP[1] * M_inv[1, 1] - FixP[1])
            #
            # 		M1 = np.zeros((3, 3))
            # 		M1[2, 2] = 1.
            # 		M1[:2, :] = cv2.invertAffineTransform(M_inv)
            #
            # 		if ri == middle_idx:
            # 			M_new = M1.copy()
            # 		else:
            # 			M_new = M_new.dot(M1)
            #
            # 	# affine_matrix_inv = cv2.invertAffineTransform(M_new[:2, :])
            # 	# M_new[:2, :] = affine_matrix_inv
            #
            # 	# now_image = plt.imread(images[now_idx])[:, :, :3]
            # 	# img1_affine = warp(now_image, M_new, output_shape=(middle_image.shape[0], middle_image.shape[1]))
            # 	# new_root = images[now_idx].replace(image_input_dir, reconstruction_dir)
            # 	# plt.imsave(new_root, img1_affine)
            #
            # 	data = pd.read_csv(images[now_idx].replace(case_folder, point_folder).replace('.png', '_center.csv'))
            #
            # 	now_ref_x = int(os.path.basename(images[now_idx]).split('_')[2])
            # 	now_ref_y = int(os.path.basename(images[now_idx]).split('_')[3].replace('.png', ''))
            #
            # 	new_data = pd.DataFrame(columns=['barcode', 'abs_x', 'abs_y', 'now_x', 'now_y'])
            # 	new_data['barcode'] = data['barcode']
            #
            # 	batch_array = np.zeros((len(new_data), 1, 2)).astype(np.float32)
            # 	batch_array[:, :, 0] = np.expand_dims(np.array(data['now_y']), 1)
            # 	batch_array[:, :, 1] = np.expand_dims(np.array(data['now_x']), 1)
            #
            # 	transformed_batch_array = cv2.perspectiveTransform(batch_array, M_new)
            #
            # 	now_abs_x = transformed_batch_array[:, :, 1] * 4 + middle_ref_y
            # 	now_abs_y = transformed_batch_array[:, :, 0] * 4 + middle_ref_x
            #
            # 	new_data['abs_x'] = (now_abs_x).astype(int)[:, 0].tolist()
            # 	new_data['abs_y'] = (now_abs_y).astype(int)[:, 0].tolist()
            # 	new_data['now_x'] = (transformed_batch_array[:, :, 1]).astype(int)[:, 0].tolist()
            # 	new_data['now_y'] = (transformed_batch_array[:, :, 0]).astype(int)[:, 0].tolist()
            #
            # 	new_data.to_csv(
            # 		os.path.join(reconstruction_dir, os.path.basename(images[now_idx]).replace('.png', '_center.csv')))
            # else:
            # 	# new_root = images[middle_idx].replace(image_input_dir, reconstruction_dir)
            # 	# plt.imsave(new_root, middle_image)
            #
            # 	new_data = pd.read_csv(images[middle_idx].replace(case_folder, point_folder).replace('.png', '_center.csv'))
            # 	new_data.to_csv(os.path.join(reconstruction_dir, os.path.basename(images[middle_idx]).replace('.png', '_center.csv')))
