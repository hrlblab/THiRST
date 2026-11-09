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
"""Quality control: draw the sampled patch centres and their ST units on the reference section.

THiRST stage 6 - quality control of ``sample_patch_centers.py``.

Outputs: ``<image_root>/<case>/Original_spot_matching_sampling/image_<k>_mask.png``, ``image_<k>_merge.png``,
         ``color_table_<k>.csv``

Usage
-----
    python scripts/6_visualization/qc_spot_sampling.py --config configs/example.yaml

Original script(s): Verify_pointlocation_local_contour_Ori_only_sample.py
"""
import sys
import os
import glob
import re

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
import pandas as pd
import openslide

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from thirst.config import list_cases, parse_args  # noqa: E402


def atoi(text):
    return int(text) if text.isdigit() else text


# Convert the image back to RGB before returning

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def number_to_rgb(number):
    """
    Convert an integer number to an RGB tuple.
    The number should be in the range 0 to 16,777,215.
    """
    if not (0 <= number <= 16777215):
        raise ValueError("Number out of range (must be between 0 and 16,777,215)")

    red = (number >> 16) & 255
    green = (number >> 8) & 255
    blue = number & 255
    return (red, green, blue)


if __name__ == '__main__':
    args, cfg = parse_args(__doc__)
    case_folder =  cfg['paths']['image_root']
    WSI_folder = cfg['paths']['wsi_dir']
    point_folder = cfg['paths']['point_root']
    contour_folder = cfg['paths']['contour_root']


    cases = list_cases(cfg)
    for ci in range(len(cases)):
        case = cases[ci]
        now_case = os.path.basename(case)
        lbl_input_dir = case
        image_input_dir = case
        reconstruction_dir = os.path.join(case, 'Original_spot_matching_sampling')

        if not os.path.exists(reconstruction_dir):
            os.makedirs(reconstruction_dir)

        images = glob.glob(os.path.join(image_input_dir, 'image_*.png'))
        images.sort(key=natural_keys)

        middle_idx = int(len(images) / 2)

        print(middle_idx)
        middle_image = plt.imread(images[middle_idx])[:,:,:3]
        middle_ref_x = int(os.path.basename(images[middle_idx]).split('_')[2])
        middle_ref_y = int(os.path.basename(images[middle_idx]).split('_')[3].replace('.png',''))

        patch_size = 1024

        real_ind_mid = int(os.path.basename(images[middle_idx]).split('_')[1])
        middle_image_svs = openslide.OpenSlide(glob.glob(os.path.join(WSI_folder, '%s-%d.svs' % (now_case, real_ind_mid)))[0])


        for ii in range(0,1):
            now_image = middle_image
            now_mask = np.zeros((now_image.shape))
            now_idx = ii
            color_table = pd.DataFrame(columns=['slide', 'barcode', 'R', 'G', 'B'])
            now_contour = pd.read_csv(images[middle_idx].replace(case_folder, contour_folder).replace('.png','_spot.csv'))
            # patch centres chosen by sample_patch_centers.py
            sample_barcode = pd.read_csv(os.path.join(point_folder, now_case, os.path.basename(images[middle_idx]).replace('.png', '_center_sample.csv')))['barcode'].tolist()

            if len(now_contour.columns) > 20:
                circle = 1
            else:
                circle = 0
                continue

            for ki in range(0, len(now_contour)):
                bi = ki
                barcode = now_contour.loc[bi]['barcode']

                if not barcode in sample_barcode:
                    continue
                'middle image'
                if circle == 1:
                    num_points = 100
                    affine_patch_contour = np.zeros((num_points, 1, 2))

                    for ni in range(num_points):
                        affine_patch_contour[ni, :, 1] = now_contour.loc[bi]['now_x_%d' % (ni)]
                        affine_patch_contour[ni, :, 0] = now_contour.loc[bi]['now_y_%d' % (ni)]

                else:
                    affine_patch_contour = np.zeros((4, 1, 2))
                    affine_patch_contour[0,:,1] = now_contour.loc[bi]['now_x_0']
                    affine_patch_contour[1,:,1] = now_contour.loc[bi]['now_x_1']
                    affine_patch_contour[2,:,1] = now_contour.loc[bi]['now_x_2']
                    affine_patch_contour[3,:,1] = now_contour.loc[bi]['now_x_3']

                    affine_patch_contour[0, :, 0] = now_contour.loc[bi]['now_y_0']
                    affine_patch_contour[1, :, 0] = now_contour.loc[bi]['now_y_1']
                    affine_patch_contour[2, :, 0] = now_contour.loc[bi]['now_y_2']
                    affine_patch_contour[3, :, 0] = now_contour.loc[bi]['now_y_3']

                row = len(color_table)
                cnt = int(16777215 / len(sample_barcode))
                color = number_to_rgb(row * cnt)
                now_mask = cv2.drawContours((now_mask*255).astype(np.uint8), [affine_patch_contour.astype(int)], -1, color, -1) / 255.
                color_table.loc[row] = [row,barcode,color[0],color[1],color[2]]

            save_dir = reconstruction_dir

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            plt.imsave(os.path.join(save_dir, 'image_%d_mask.png' % (now_idx)), now_mask)
            ori_image_merge = (now_mask * 0.4 + now_image * 0.6)
            ori_image_merge[ori_image_merge>1] = 1

            plt.imsave(os.path.join(save_dir, 'image_%d_merge.png' % (now_idx)), ori_image_merge)
            color_table.to_csv(os.path.join(save_dir, 'color_table_%d.csv' % (now_idx)), index=False)

