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
"""Merge the per-batch V2-HD IoU files into one JSON per section pair.

THiRST stage 3 - cross-resolution association (paper Sec. 4.1.7).

Outputs: ``<contour_root>/<case>/IoU_ST_to_HD_distance_<stage>reg_merge/row_<i>_col_<j>.json``

Usage
-----
    python scripts/3_st_association/merge_v2_hd_iou.py --config configs/example.yaml --registration patch

Original script(s): Verify_IoU_json_merge.py
"""
import sys
import os
import glob
import json
import re

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from thirst.config import add_registration_argument, list_cases, parse_args, require_alternating_platforms  # noqa: E402


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
        # per-batch outputs of v2_hd_iou.py for the same --registration stage
        IoU_dir = os.path.join(case.replace(case_folder,contour_folder), 'IoU_ST_to_HD_distance_%sreg' % args.registration.replace('none', 'no'))
        IoU_dir_merge = IoU_dir + '_merge'

        if not os.path.exists(IoU_dir_merge):
            os.makedirs(IoU_dir_merge)

        images = glob.glob(os.path.join(image_input_dir, 'image_*.png'))
        images.sort(key=natural_keys)
        require_alternating_platforms(cfg, images)  # V2 at even, HD at odd stack positions

        middle_idx = int(len(images) / 2)

        # print(middle_idx)
        middle_image = plt.imread(images[middle_idx])[:,:,:3]

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


        for ii in range(0, len(images), 2):
            now_idx = ii
            real_ind_ii = int(os.path.basename(images[ii]).split('_')[1])

            for jj in range(1,len(images), 2):   # HD sections (odd positions)
                now_idx = jj
                real_ind_jj = int(os.path.basename(images[jj]).split('_')[1])

                if ii == jj:
                    continue

                else:

                    # Create a dictionary to store IoU values
                    output_dict = {}

                    # batch files row_<i>_col_<j>_<batch>.json; the '_' keeps col 1 from also matching col 10, 11, ...
                    iou_file_path_list = glob.glob(os.path.join(IoU_dir, 'row_%d_col_%d_*.json' % (real_ind_ii, real_ind_jj)))
                    if len(iou_file_path_list) > 0:
                        # Load the IoU data from JSON
                        for filename in iou_file_path_list:
                            with open(filename, 'r') as f:
                                data = json.load(f)
                                output_dict.update(data)

                        with open(os.path.join(IoU_dir_merge, 'row_%d_col_%d.json' % (real_ind_ii, real_ind_jj)), 'w') as f:
                            json.dump(output_dict, f, indent=4)