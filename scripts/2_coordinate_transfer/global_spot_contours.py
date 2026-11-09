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
"""Map every ST unit polygon into the reference frame using the global affine chain.

THiRST stage 2 - 3D coordinate transfer for ST spots and bins (paper Sec. 4.1.5-4.1.6, Algorithm 1).

Outputs: ``<contour_root>/<case>/registration_3D_ANTs/image_*_spot.csv``

Usage
-----
    python scripts/2_coordinate_transfer/global_spot_contours.py --config configs/example.yaml

Original script(s): Step3.5_registertoMiddle_ANTs_points_contour_fast.py
"""
import sys
import os
import glob
import re

import numpy as np
import cv2 as cv2
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from thirst.config import list_cases, parse_args, require_platform  # noqa: E402
from thirst.transforms import require_chain, require_transform  # noqa: E402
from thirst.st import is_circle_table  # noqa: E402


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
    point_folder = cfg['paths']['contour_root']

    cases = list_cases(cfg)
    for ci in range(len(cases)):
        case = cases[ci]
        now_case = os.path.basename(case)
        lbl_input_dir = case
        image_input_dir = case
        image_output_dir = os.path.join(case, 'registration_ANTs')
        # image_badcase_dir = os.path.join(case, 'registration_badcase_2048')
        reconstruction_dir = os.path.join(case.replace(case_folder,point_folder), 'registration_3D_ANTs')

        if not os.path.exists(reconstruction_dir):
            os.makedirs(reconstruction_dir)

        images = glob.glob(os.path.join(image_input_dir, 'image_*.png'))
        images.sort(key=natural_keys)
        require_chain(image_output_dir, images)  # every pair present and valid before anything is written

        middle_idx = int(len(images) / 2)

        # print(middle_idx)
        # middle_image = plt.imread(images[middle_idx])[:,:,:3]

        middle_ref_x = int(os.path.basename(images[middle_idx]).split('_')[2])
        middle_ref_y = int(os.path.basename(images[middle_idx]).split('_')[3].replace('.png', ''))

        for ii in range(len(images)):
            now_idx = ii

            if now_idx < middle_idx:
                M_new = np.zeros((3, 3))
                M_new[2, 2] = 1.

                for ri in range(now_idx, middle_idx):

                    moving_index = int(os.path.basename(images[ri+1]).split('.')[0].split('_')[1])
                    fixed_index = int(os.path.basename(images[ri]).split('.')[0].split('_')[1])

                    affine_root = os.path.join(image_output_dir, '%d_to_%d' % (moving_index, fixed_index), 'step2_run_ants_reg', 'output0GenericAffine.mat')
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

                data = pd.read_csv(images[now_idx].replace(case_folder, point_folder).replace('.png','_spot.csv'))

                now_ref_x = int(os.path.basename(images[now_idx]).split('_')[2])
                now_ref_y = int(os.path.basename(images[now_idx]).split('_')[3].replace('.png', ''))

                circle = is_circle_table(data, images[now_idx])  # 100-vertex V2 spots or 4-corner HD bins
                require_platform(cfg, images[now_idx], circle)   # must agree with st_sections

                if circle:
                    num_points = 100#int((len(data.columns) - 2) / 2)
                    columns = ['barcode'] + \
                              ['now_x_%d' % ki for ki in range(num_points)] + \
                              ['now_y_%d' % ki for ki in range(num_points)]

                    new_data = pd.DataFrame(columns = columns)
                    new_data['barcode'] = data['barcode']

                    batch_array = np.zeros((len(new_data), 100, 1, 2)).astype(np.float32)
                    for ni in range(num_points):
                        batch_array[:, ni, :, 0] = np.expand_dims(np.array(data['now_y_%d' % (ni)]), 1)
                        batch_array[:, ni, :, 1] = np.expand_dims(np.array(data['now_x_%d' % (ni)]), 1)

                    # batch_array = np.zeros((len(new_data), 1, 2)).astype(np.float32)
                    # batch_array[:,:,0] = np.expand_dims(np.array(data['now_y']),1)
                    # batch_array[:,:,1] = np.expand_dims(np.array(data['now_x']),1)

                    transformed_batch_array = cv2.perspectiveTransform(batch_array.reshape(batch_array.shape[0] * batch_array.shape[1], batch_array.shape[2], batch_array.shape[3]), M_new).reshape(batch_array.shape[0], batch_array.shape[1], batch_array.shape[2], batch_array.shape[3])
                    for ni in range(num_points):
                        new_data['now_x_%d' % (ni)] = np.round(transformed_batch_array[:, ni, 0, 1]).astype(int).tolist()
                        new_data['now_y_%d' % (ni)] = np.round(transformed_batch_array[:, ni, 0, 0]).astype(int).tolist()

                else:
                    new_data = pd.DataFrame(columns=['barcode', 'now_x_0', 'now_y_0', 'now_x_1', 'now_y_1', 'now_x_2', 'now_y_2', 'now_x_3','now_y_3'])
                    new_data['barcode'] = data['barcode']

                    batch_array = np.zeros((len(new_data), 4, 1, 2)).astype(np.float32)

                    batch_array[:, 0, :, 1] = np.expand_dims(np.array(data['now_x_0']), 1)
                    batch_array[:, 1, :, 1] = np.expand_dims(np.array(data['now_x_1']), 1)
                    batch_array[:, 2, :, 1] = np.expand_dims(np.array(data['now_x_2']), 1)
                    batch_array[:, 3, :, 1] = np.expand_dims(np.array(data['now_x_3']), 1)

                    batch_array[:, 0, :, 0] = np.expand_dims(np.array(data['now_y_0']), 1)
                    batch_array[:, 1, :, 0] = np.expand_dims(np.array(data['now_y_1']), 1)
                    batch_array[:, 2, :, 0] = np.expand_dims(np.array(data['now_y_2']), 1)
                    batch_array[:, 3, :, 0] = np.expand_dims(np.array(data['now_y_3']), 1)

                    transformed_batch_array = cv2.perspectiveTransform(
                            batch_array.reshape(batch_array.shape[0] * batch_array.shape[1], batch_array.shape[2],
                                                batch_array.shape[3]), M_new).reshape(batch_array.shape[0],
                                                                                  batch_array.shape[1],
                                                                                  batch_array.shape[2],
                                                                                  batch_array.shape[3])

                    new_data['now_x_0'] = np.round(transformed_batch_array[:, 0, 0, 1]).astype(int).tolist()
                    new_data['now_x_1'] = np.round(transformed_batch_array[:, 1, 0, 1]).astype(int).tolist()
                    new_data['now_x_2'] = np.round(transformed_batch_array[:, 2, 0, 1]).astype(int).tolist()
                    new_data['now_x_3'] = np.round(transformed_batch_array[:, 3, 0, 1]).astype(int).tolist()

                    new_data['now_y_0'] = np.round(transformed_batch_array[:, 0, 0, 0]).astype(int).tolist()
                    new_data['now_y_1'] = np.round(transformed_batch_array[:, 1, 0, 0]).astype(int).tolist()
                    new_data['now_y_2'] = np.round(transformed_batch_array[:, 2, 0, 0]).astype(int).tolist()
                    new_data['now_y_3'] = np.round(transformed_batch_array[:, 3, 0, 0]).astype(int).tolist()

                new_data.to_csv(os.path.join(reconstruction_dir, os.path.basename(images[now_idx]).replace('.png', '_spot.csv')), index = False)

            elif now_idx > middle_idx:
                M_new = np.zeros((3, 3))
                M_new[2, 2] = 1.

                for ri in range(middle_idx, now_idx):

                    moving_index = int(os.path.basename(images[ri+1]).split('.')[0].split('_')[1])
                    fixed_index = int(os.path.basename(images[ri]).split('.')[0].split('_')[1])

                    affine_root = os.path.join(image_output_dir, '%d_to_%d' % (moving_index, fixed_index),'step2_run_ants_reg', 'output0GenericAffine.mat')

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


                data = pd.read_csv(images[now_idx].replace(case_folder, point_folder).replace('.png', '_spot.csv'))

                now_ref_x = int(os.path.basename(images[now_idx]).split('_')[2])
                now_ref_y = int(os.path.basename(images[now_idx]).split('_')[3].replace('.png', ''))


                circle = is_circle_table(data, images[now_idx])  # 100-vertex V2 spots or 4-corner HD bins
                require_platform(cfg, images[now_idx], circle)   # must agree with st_sections

                if circle:
                    num_points = 100#int((len(data.columns) - 2) / 2)
                    columns = ['barcode'] + \
                              ['now_x_%d' % ki for ki in range(num_points)] + \
                              ['now_y_%d' % ki for ki in range(num_points)]

                    new_data = pd.DataFrame(columns = columns)
                    new_data['barcode'] = data['barcode']

                    batch_array = np.zeros((len(new_data), 100, 1, 2)).astype(np.float32)
                    for ni in range(num_points):
                        batch_array[:, ni, :, 0] = np.expand_dims(np.array(data['now_y_%d' % (ni)]), 1)
                        batch_array[:, ni, :, 1] = np.expand_dims(np.array(data['now_x_%d' % (ni)]), 1)

                    # batch_array = np.zeros((len(new_data), 1, 2)).astype(np.float32)
                    # batch_array[:,:,0] = np.expand_dims(np.array(data['now_y']),1)
                    # batch_array[:,:,1] = np.expand_dims(np.array(data['now_x']),1)

                    transformed_batch_array = cv2.perspectiveTransform(batch_array.reshape(batch_array.shape[0] * batch_array.shape[1], batch_array.shape[2], batch_array.shape[3]), M_new).reshape(batch_array.shape[0], batch_array.shape[1], batch_array.shape[2], batch_array.shape[3])
                    for ni in range(num_points):
                        new_data['now_x_%d' % (ni)] = np.round(transformed_batch_array[:, ni, 0, 1]).astype(int).tolist()
                        new_data['now_y_%d' % (ni)] = np.round(transformed_batch_array[:, ni, 0, 0]).astype(int).tolist()


                else:
                    new_data = pd.DataFrame(columns=['barcode', 'now_x_0', 'now_y_0', 'now_x_1', 'now_y_1', 'now_x_2', 'now_y_2', 'now_x_3','now_y_3'])
                    new_data['barcode'] = data['barcode']

                    batch_array = np.zeros((len(new_data), 4, 1, 2)).astype(np.float32)

                    batch_array[:, 0, :, 1] = np.expand_dims(np.array(data['now_x_0']), 1)
                    batch_array[:, 1, :, 1] = np.expand_dims(np.array(data['now_x_1']), 1)
                    batch_array[:, 2, :, 1] = np.expand_dims(np.array(data['now_x_2']), 1)
                    batch_array[:, 3, :, 1] = np.expand_dims(np.array(data['now_x_3']), 1)

                    batch_array[:, 0, :, 0] = np.expand_dims(np.array(data['now_y_0']), 1)
                    batch_array[:, 1, :, 0] = np.expand_dims(np.array(data['now_y_1']), 1)
                    batch_array[:, 2, :, 0] = np.expand_dims(np.array(data['now_y_2']), 1)
                    batch_array[:, 3, :, 0] = np.expand_dims(np.array(data['now_y_3']), 1)

                    transformed_batch_array = cv2.perspectiveTransform(
                            batch_array.reshape(batch_array.shape[0] * batch_array.shape[1], batch_array.shape[2],
                                                batch_array.shape[3]), M_new).reshape(batch_array.shape[0],
                                                                                  batch_array.shape[1],
                                                                                  batch_array.shape[2],
                                                                                  batch_array.shape[3])

                    new_data['now_x_0'] = np.round(transformed_batch_array[:, 0, 0, 1]).astype(int).tolist()
                    new_data['now_x_1'] = np.round(transformed_batch_array[:, 1, 0, 1]).astype(int).tolist()
                    new_data['now_x_2'] = np.round(transformed_batch_array[:, 2, 0, 1]).astype(int).tolist()
                    new_data['now_x_3'] = np.round(transformed_batch_array[:, 3, 0, 1]).astype(int).tolist()

                    new_data['now_y_0'] = np.round(transformed_batch_array[:, 0, 0, 0]).astype(int).tolist()
                    new_data['now_y_1'] = np.round(transformed_batch_array[:, 1, 0, 0]).astype(int).tolist()
                    new_data['now_y_2'] = np.round(transformed_batch_array[:, 2, 0, 0]).astype(int).tolist()
                    new_data['now_y_3'] = np.round(transformed_batch_array[:, 3, 0, 0]).astype(int).tolist()


                new_data.to_csv(
                        os.path.join(reconstruction_dir, os.path.basename(images[now_idx]).replace('.png', '_spot.csv')), index = False)

            else:
                # new_root = images[middle_idx].replace(image_input_dir, reconstruction_dir)
                # plt.imsave(new_root, middle_image)

                new_data = pd.read_csv(images[middle_idx].replace(case_folder, point_folder).replace('.png', '_spot.csv'))
                new_data.to_csv(os.path.join(reconstruction_dir, os.path.basename(images[middle_idx]).replace('.png', '_spot.csv')), index = False)
