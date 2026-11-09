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
"""Crop a local 3D patch stack around every sampled centre from the globally registered sections.

THiRST stage 1 - local patch extraction (paper Sec. 4.1.3-4.1.4).

For each sampled centre a 2048 x 2048 px window of the 4x images (~2 x 2 mm) is cut from every section of
``registration_3D_ANTs``; these stacks are then registered again with ``--level patch``.

Inputs : ``<image_root>/<case>/registration_3D_ANTs/image_*.png``, sampled centres, WSI of the reference section
Outputs: ``<image_root>/<case>/3D_Patch_crop/<abs_x>_<abs_y>/image_<section>.png``

Usage
-----
    python scripts/1_registration/crop_patch_stacks.py --config configs/example.yaml

Original script(s): Step4_get_3D_Patch_with_spotlocation_crop3Dimages.py
"""
import sys
import os
import glob
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openslide

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from thirst.config import list_cases, parse_args  # noqa: E402


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
    WSI_folder = cfg['paths']['wsi_dir']

    cases = list_cases(cfg)
    for ci in range(len(cases)):
        case = cases[ci]
        now_case = os.path.basename(case)
        lbl_input_dir = case
        image_input_dir = case
        image_output_dir = os.path.join(case, 'registration_ANTs')
        # image_badcase_dir = os.path.join(case, 'registration_badcase_2048')
        reconstruction_dir_3D = os.path.join(case, 'registration_3D_ANTs')
        reconstruction_dir = os.path.join(case, '3D_Patch_crop')

        if not os.path.exists(reconstruction_dir):
            os.makedirs(reconstruction_dir)

        images = glob.glob(os.path.join(reconstruction_dir_3D, 'image_*.png'))
        images.sort(key=natural_keys)

        middle_idx = int(len(images) / 2)
        print(middle_idx)
        middle_image = plt.imread(images[middle_idx])[:,:,:3]
        ref_x = int(os.path.basename(images[middle_idx]).split('_')[2])
        ref_y = int(os.path.basename(images[middle_idx]).split('_')[3].replace('.png',''))

        patch_size = 1024

        reference_name = os.path.basename(images[middle_idx]).replace('.png', '_center.csv')
        reference_csv = pd.read_csv(os.path.join(cfg['paths']['point_root'], now_case, reference_name))
        barcodes = list(reference_csv['barcode'])

        # patch centres chosen by sample_patch_centers.py
        sample_barcode = pd.read_csv(os.path.join(cfg['paths']['point_root'], now_case, reference_name.replace('_center.csv', '_center_sample.csv')))[
            'barcode'].tolist()

        real_ind_mid = int(os.path.basename(images[middle_idx]).split('_')[1])
        middle_image_svs = openslide.OpenSlide(glob.glob(os.path.join(WSI_folder, '%s-%d.svs' % (now_case, real_ind_mid)))[0])

        for bi in range(0, len(barcodes)):
        # for bi in range(0, 1000, 100):
            now_barcode = barcodes[bi]
            barcode_str = now_barcode
            if not barcode_str in sample_barcode:
                continue

            abs_x = int(reference_csv[reference_csv['barcode'] == barcode_str].iloc[0]['abs_x'])
            abs_y = int(reference_csv[reference_csv['barcode'] == barcode_str].iloc[0]['abs_y'])

            now_x = int(reference_csv[reference_csv['barcode'] == barcode_str].iloc[0]['now_x'])
            now_y = int(reference_csv[reference_csv['barcode'] == barcode_str].iloc[0]['now_y'])

            now_middle_img_low = middle_image[now_x-patch_size:now_x+patch_size, now_y-patch_size:now_y+patch_size,:]
            now_middle_img_high = np.array(middle_image_svs.read_region((int(abs_y-patch_size * 4),int(abs_x-patch_size * 4)), 2, (patch_size*2, patch_size*2)))[:,:,:3]
            # now_middle_ref = reference_png[jpg_x-300:jpg_x+300, jpg_y-300:jpg_y+300,:]

            # low_root = os.path.join(reconstruction_dir, '%d_%d' % (abs_x, abs_y),'low', 'image_%d.png' % (middle_idx))
            # high_root = os.path.join(reconstruction_dir, '%d_%d' % (abs_x, abs_y),'high', 'image_%d.png' % (middle_idx))

            low_root = os.path.join(reconstruction_dir, '%d_%d' % (abs_x, abs_y), 'image_%d.png' % (real_ind_mid))
            high_root = os.path.join(reconstruction_dir, '%d_%d' % (abs_x, abs_y), 'svs_image_%d.png' % (real_ind_mid))
            # ref_root = os.path.join(reconstruction_dir, '%d_%d' % (abs_x, abs_y), 'image_%d_fromtiff.png' % (real_ind_mid))


            if not os.path.exists(os.path.dirname(low_root)):
                os.makedirs(os.path.dirname(low_root))

            if not os.path.exists(os.path.dirname(high_root)):
                os.makedirs(os.path.dirname(high_root))

            # now_middle_img_low = draw_line(now_middle_img_low)
            plt.imsave(low_root, now_middle_img_low)
            #plt.imsave(high_root, now_middle_img_high)

            start_x = now_y
            start_y = now_x

            for ii in range(0, len(images)):
                now_idx = ii

                if now_idx != middle_idx:
                    now_image = plt.imread(images[now_idx])[:,:,:3]
                    real_idx = int(os.path.basename(images[now_idx]).split('.')[0].split('_')[1])

                    now_patch = now_image[now_x-patch_size:now_x+patch_size, now_y-patch_size:now_y+patch_size,:]
                    low_root = os.path.join(reconstruction_dir, '%d_%d' % (abs_x, abs_y),'image_%d.png' % (real_idx))
                    plt.imsave(low_root, now_patch)
