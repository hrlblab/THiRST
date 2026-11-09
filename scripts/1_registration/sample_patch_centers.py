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
"""Choose the centres of the local patches used for patch-level refinement.

THiRST stage 1 - region sampling based on the ST spot distribution (paper Sec. 4.1.3).

The ST unit centres of the reference (middle) section are visited in order and a centre is kept only
if no previously kept centre lies within ``--min-distance`` (KD-tree query; default 512 px of the 4x
image, i.e. half a patch).

Inputs : ``<point_root>/<case>/<reference image>_center.csv``
Outputs: ``<point_root>/<case>/<reference image>_center_sample.csv``

Usage
-----
    python scripts/1_registration/sample_patch_centers.py --config configs/example.yaml

Original script(s): sampling_spot.py
"""
import sys
import os
import glob
import re

import pandas as pd
from scipy.spatial import KDTree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from thirst.config import list_cases, parse_args  # noqa: E402


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''Sort file names in human order (image_2 before image_10).'''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def is_within_distance_kdtree(current_point, tree, threshold=512):
    """
    Check if any point in the KD-Tree is within the threshold distance from current_point.

    :param current_point: Tuple (x, y) of the current coordinate.
    :param tree: KDTree built from the point list.
    :param threshold: Distance threshold (default is 512).
    :return: True if at least one point is within the threshold, else False.
    """
    # Query for points within the threshold distance
    indices = tree.query_ball_point(current_point, r=threshold)
    return len(indices) > 0


if __name__ == '__main__':
    args, cfg = parse_args(__doc__, lambda p: p.add_argument(
        '--min-distance', type=float, default=512,
        help='minimum distance (pixels of the 4x-downsampled image) between two patch centres'))

    for case in list_cases(cfg):
        now_case = os.path.basename(case)
        images = glob.glob(os.path.join(case, 'image_*.png'))
        images.sort(key=natural_keys)
        middle_idx = int(len(images) / 2)
        # reference (middle) section: ST unit centres in its own frame = the common 3D frame
        reference_name = os.path.basename(images[middle_idx]).replace('.png', '_center.csv')
        reference_csv = pd.read_csv(os.path.join(cfg['paths']['point_root'], now_case, reference_name))
        barcodes = list(reference_csv['barcode'])
        sample_csv = pd.DataFrame(columns = reference_csv.columns.tolist())

        exists_point = []

        for ki in range(len(reference_csv)):
            now_line = reference_csv.loc[ki]
            current_point = (int(now_line['now_x']), int(now_line['now_y']))

            if ki != 0:
                tree = KDTree(exists_point)
                result = is_within_distance_kdtree(current_point, tree, threshold=args.min_distance)

                if result == False:
                    now_row = len(sample_csv)
                    sample_csv.loc[now_row] = now_line
                    exists_point.append((current_point))
                else:
                    print('less than %g' % args.min_distance)

            else:
                now_row = len(sample_csv)
                sample_csv.loc[now_row] = now_line
                exists_point.append(current_point)

        sample_path = os.path.join(cfg['paths']['point_root'], now_case, reference_name.replace('_center.csv', '_center_sample.csv'))
        sample_csv.to_csv(sample_path, index = False)
        print('%s: %d patch centres -> %s' % (now_case, len(sample_csv), sample_path))
