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
"""Convert the XFeat affine matrices to ITK/ANTs transform files.

THiRST stage 1 - coarse-to-fine registration (paper Sec. 4.1.1 / 4.1.4).

Each ``sg_affine_init.npy`` (OpenCV, moving -> fixed) is inverted and written as an ITK
``AffineTransform_double_2_2`` centred on the image centre, the format expected by ANTs as initial transform.

Outputs: ``<stack>/registration_ANTs/<i+1>_to_<i>/sg_affine_init.mat``

Usage
-----
    python scripts/1_registration/affine_to_itk.py --config configs/example.yaml --level global

Original script(s): Step2_xFeat_matrix_nptomat.py (global), Step6_xFeat_matrix_nptomat_patchlevel.py (patch)
"""
import sys
import os
import glob
import re

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
import SimpleITK as sitk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from thirst.config import ITK_AFFINE_TEMPLATE, add_level_argument, list_stacks, parse_args  # noqa: E402


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
    args, cfg = parse_args(__doc__, add_level_argument)

    # global level: one stack per case; patch level: one stack per sampled patch
    cases = list_stacks(cfg, args.level)
    for ci in range(len(cases)):
        case = cases[ci]
        now_case = os.path.basename(case)
        lbl_input_dir = case
        image_input_dir = case
        sg_folder = os.path.join(case, 'registration_xfeat')
        ANTs_folder = os.path.join(case, 'registration_ANTs')

        images = glob.glob(os.path.join(image_input_dir, 'image_*.png'))
        images.sort(key=natural_keys)

        for ii in range(len(images)-1):
            moving_root = images[ii+1]
            moving_output_root = moving_root.replace(image_input_dir, sg_folder)
            displacement_root = moving_output_root.replace('.png', '.npy')

            fixed_root = images[ii]
            print('now case is %s to %s.' % (os.path.basename(moving_root), os.path.basename(fixed_root)))

            img_highres = plt.imread(fixed_root)[:,:,:3]

            overlay_dir = os.path.join(sg_folder,'%s_to_%s' % (os.path.basename(moving_root).split('.')[0].split('_')[1],os.path.basename(fixed_root).split('.')[0].split('_')[1]))
            affine_root = os.path.join(overlay_dir, 'sg_affine_init.npy')

            if not os.path.isfile(affine_root):
                raise FileNotFoundError(
                    '%s not found: xfeat_affine.py found no affine for this pair (see %s). Provide an '
                    'initialisation for it (e.g. np.eye(2, 3) to start ANTs from the identity) and re-run.'
                    % (affine_root, os.path.join(sg_folder, 'xfeat_status.csv')))
            affine = np.load(affine_root)
            M_inv = cv2.invertAffineTransform(affine)

            M_vector = np.zeros([6])

            # world coordinate to itk
            FixParameters = [img_highres.shape[1] / 2.0, img_highres.shape[0] / 2.0]
            M_vector[0] = M_inv[0, 0]
            M_vector[1] = M_inv[0, 1]
            M_vector[2] = M_inv[1, 0]
            M_vector[3] = M_inv[1, 1]
            M_vector[4] = (FixParameters[0] * M_inv[0, 0] + FixParameters[1] * M_inv[0, 1] - FixParameters[0]) + M_inv[
                0, 2]
            M_vector[5] = (FixParameters[0] * M_inv[1, 0] + FixParameters[1] * M_inv[1, 1] - FixParameters[1]) + M_inv[
                1, 2]

            m = sitk.ReadTransform(ITK_AFFINE_TEMPLATE)

            m.SetParameters(M_vector)
            m.SetFixedParameters(FixParameters)

            save_root = os.path.dirname(affine_root.replace(sg_folder, ANTs_folder))
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            sitk.WriteTransform(m, os.path.join(save_root, 'sg_affine_init.mat'))

