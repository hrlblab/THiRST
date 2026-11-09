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
"""Refine every adjacent pair with ANTs, initialised by the XFeat affine.

THiRST stage 1 - coarse-to-fine registration (paper Sec. 4.1.1 / 4.1.4).

Runs ``antsRegistrationSyN.sh -t s`` (rigid + affine + SyN, mutual information, multi-resolution) for
each pair; the pipeline uses the refined affine (the SyN warp fields are written but not applied
downstream). Pairs that this script completed successfully earlier (``ants_completed.json`` next to a
matching affine) are skipped, so the script can be restarted; ``--overwrite`` recomputes them. Any
other existing output - left by a failed or interrupted run, or written by an earlier version without a
completion record - is recomputed; ``--trust-existing`` keeps such affines instead (with a warning),
e.g. results of the original lab scripts. If ANTs fails, its partial outputs are moved to
``step2_run_ants_reg.failed/`` and never used. Pairs for which ANTs fails are listed at the end and the script exits
with status 1, so later steps never run on an incomplete set of transforms. Requires ANTs
(``paths.ants_bin``, ``$ANTSPATH`` or ``PATH``).

Outputs: ``<stack>/registration_ANTs/<i+1>_to_<i>/step2_run_ants_reg/output0GenericAffine.mat`` (+ warps)

Usage
-----
    python scripts/1_registration/ants_affine.py --config configs/example.yaml --level global [--overwrite | --trust-existing]

Original script(s): Step2_superglue+ANTs.py (global), Step7_xfeat+ANTs_patchlevel.py (patch); ANTs call from superglue_ants_registration_onpair_new_RGB.py
"""
import sys
import os
import glob
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from thirst.config import add_level_argument, list_stacks, parse_args  # noqa: E402
from thirst.ants import (ANTsRegistrationError, affine_path, ants_version, find_ants_bin,  # noqa: E402
                         is_completed, register_a_pair)
from thirst.transforms import is_valid_transform  # noqa: E402


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def _add_arguments(parser):
    add_level_argument(parser)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--overwrite', action='store_true',
                       help='recompute every pair, including pairs completed earlier (default: keep those)')
    group.add_argument('--trust-existing', action='store_true',
                       help='also keep valid affines that have no completion record (e.g. results of the original '
                            'lab scripts); by default they are recomputed')


if __name__ == '__main__':
    args, cfg = parse_args(__doc__, _add_arguments)
    ants_bin = find_ants_bin(cfg['paths'].get('ants_bin'))  # fail now if ANTs is not installed
    version = ants_version(ants_bin)
    print('ANTs: %s (%s)' % (version, ants_bin))
    failed, trusted = [], []

    # global level: one stack per case; patch level: one stack per sampled patch
    cases = list_stacks(cfg, args.level)
    for ci in range(len(cases)):
        case = cases[ci]
        now_case = os.path.basename(case)
        lbl_input_dir = case
        image_input_dir = case
        ANTs_folder = os.path.join(case, 'registration_ANTs')

        images = glob.glob(os.path.join(image_input_dir, 'image_*.png'))
        images.sort(key=natural_keys)

        for ii in range(len(images)-1):
            moving_root = images[ii+1]
            fixed_root = images[ii]
            print('now case is %s to %s.' % (os.path.basename(moving_root), os.path.basename(fixed_root)))

            working_dir = os.path.join(ANTs_folder,'%s_to_%s' % (os.path.basename(moving_root).split('.')[0].split('_')[1],os.path.basename(fixed_root).split('.')[0].split('_')[1]))
            affine_file = affine_path(working_dir)
            if not args.overwrite and is_completed(working_dir):
                print('  existing result kept (completed earlier; use --overwrite to recompute): %s' % affine_file)
                continue
            if args.trust_existing and is_valid_transform(affine_file):
                print('  [WARN] existing result kept without a completion record (--trust-existing): %s' % affine_file)
                trusted.append(working_dir)
                continue
            if os.path.exists(affine_file) and not args.overwrite:
                print('  %s has no completion record (failed or interrupted run, or an earlier version): '
                      'recomputing' % affine_file)

            try:
                register_a_pair(moving_root, fixed_root, working_dir, ants_bin, version)
            except (ANTsRegistrationError, FileNotFoundError) as exc:
                print('[ERROR] %s: %s' % (working_dir, exc))
                failed.append(working_dir)

    if trusted:
        print('\n[WARN] %d pair(s) kept without a completion record (--trust-existing).' % len(trusted))
    if failed:
        print('\n[ERROR] ANTs registration failed for %d pair(s); no transform was kept (partial outputs are in '
              'step2_run_ants_reg.failed/):' % len(failed))
        for w in failed:
            print('  ' + w)
        sys.exit(1)
