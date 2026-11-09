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
"""Pairwise XFeat affine initialisation between adjacent sections (global or patch level).

THiRST stage 1 - coarse-to-fine registration (paper Sec. 4.1.1 / 4.1.4).

For every adjacent pair (moving = section i+1, fixed = section i) of an ordered ``image_*.png`` stack,
XFeat keypoints (top-k 4096, images resized to 1000 px) are matched, outliers are removed with
USAC-MAGSAC + RANSAC, and an affine transform is estimated at full PNG resolution.

``--level global`` : stacks ``<image_root>/<case>/``
``--level patch``  : local stacks ``<image_root>/<case>/3D_Patch_crop/<x>_<y>/`` (crop_patch_stacks.py)

A pair with <= 10 inliers gets an identity initialisation (ANTs then starts from the unaligned pair);
a pair with fewer than 20 matches gets none. (The retry loop over the ratio threshold ``t1`` is kept
from the SuperGlue-based original; ``t1`` no longer affects the XFeat matches.) The outcome of every
pair is written to ``registration_xfeat/xfeat_status.csv`` (``ok`` / ``identity_fallback`` /
``failed``) and summarised at the end; the script exits with status 1 if any pair failed.

Outputs: ``<stack>/registration_xfeat/<i+1>_to_<i>/sg_affine_init.npy`` (+ match_raw.jpg, match_inliers.jpg,
         keypoints.csv for quality control), ``<stack>/registration_xfeat/xfeat_status.csv``,
         ``<image_root>/xfeat_provenance_<level>.json``: XFeat source and ref, SHA-256 of the code files
         that were run (and their git commit when known), SHA-256 of the weights, torch version.
         XFeat comes from torch.hub: ``xfeat_hub`` in the config is ``owner/repo[:branch|tag|commit]``
         (default ``verlab/accelerated_features``, the unpinned default branch) or a local clone;
         ``xfeat_weights_sha256`` makes the run stop unless the weights have that SHA-256.

Usage
-----
    python scripts/1_registration/xfeat_affine.py --config configs/example.yaml --level global

Original script(s): Step1_affine_registration_2d_PAS_2048_xfeat.py (global), Step5_affine_registration_2d_PAS_2048_xfeat_patchlevel.py (patch)
"""
import sys
import os
import glob
import re

import numpy as np
import pandas as pd
import torch
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from thirst.config import add_level_argument, list_stacks, parse_args  # noqa: E402
from thirst.provenance import COMMIT_RE, file_sha256, find_code_root, git_commit, pinned_ref, tree_sha256  # noqa: E402


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def frame2tensor(frame, device):
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)


def saveKpt(overlay_dir, src_kpt, dst_kpt):
    df = pd.DataFrame(columns=['src_x', 'src_y', 'dst_x', 'dst_y'])
    for ki in range(len(src_kpt)):
        df.loc[ki] = [src_kpt[ki, 0, 0], src_kpt[ki, 0, 1], dst_kpt[ki, 0, 0], dst_kpt[ki, 0, 1]]

    csv_root = os.path.join(overlay_dir, 'keypoints.csv')
    df.to_csv(csv_root, index=False)


def sg_affine(img1_file, img2_file, overlay_dir, t1, small_image_res=1000):
    # use 255 - since the protein background is black

    img1 = cv2.imread(img1_file, 0)
    img2 = cv2.imread(img2_file, 0)

    img1_highres = img1
    img2_highres = img2

    ratio_img1 = max(img1_highres.shape[0], img1_highres.shape[1]) / float(small_image_res)
    width1 = int(img1.shape[1] / ratio_img1)
    height1 = int(img1.shape[0] / ratio_img1)
    dim1 = (width1, height1)
    # resize image
    img1 = cv2.resize(img1, dim1, interpolation=cv2.INTER_AREA)

    ratio_img2 = max(img2_highres.shape[0], img2_highres.shape[1]) / float(small_image_res)
    width2 = int(img2.shape[1] / ratio_img2)
    height2 = int(img2.shape[0] / ratio_img2)
    dim2 = (width2, height2)
    # resize image

    img2 = cv2.resize(img2, dim2, interpolation=cv2.INTER_AREA)

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))


    inp1 = frame2tensor(img1, device)
    inp2 = frame2tensor(img2, device)


    # timer = AverageTimer(newline=True)

    kpts1, kpts2 = xfeat.match_xfeat(inp1, inp2, top_k = 4096)

    # Perform the matching.

    # matches, conf = pred['matches0'], pred['matching_scores0']
    # timer.update('matcher')

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # kp1 = [cv2.KeyPoint(x[0], x[1], 1) for x in kp1]
    # kp2 = [cv2.KeyPoint(x[0], x[1], 1) for x in kp2]

    # des1 = pred['des0'].transpose()
    # des2 = pred['des1'].transpose()

    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    # good_matches = []
    # for m, n in matches:
    #     # print(m.distance,n.distance)
    #     # if m.distance < t1 * n.distance:
    #     if m.distance < t1 * n.distance:
    #         good_matches.append(m)

    good_matches = []

    H, mask = cv2.findHomography(kpts1, kpts2, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = inp1.shape[:2]
    corners_inp1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_inp1, H)

    # Draw the warped corners in image2
    inp2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(inp2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    kp1 = [cv2.KeyPoint(p[0], p[1], 5) for p in kpts1]
    kp2 = [cv2.KeyPoint(p[0], p[1], 5) for p in kpts2]
    good_matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    # print(t1, len(good_matches))

    good_matches_show = [good_matches]
    # Draw matches
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches_show, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if not os.path.exists(overlay_dir):
        os.makedirs(overlay_dir)
    # Image.fromarray(img3).show()
    cv2.imwrite(os.path.join(overlay_dir, 'match_raw.jpg'), img3)

    MIN_MATCH_COUNT = 20
    if len(good_matches) >= MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M0, mask0 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask0.ravel().tolist()

    else:
        # print ("Not enough matches are found - %d/%d") % (len(good_matches),MIN_MATCH_COUNT)
        matchesMask = None
        # src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        # M_0, _ = cv2.estimateAffine2D(src_pts, src_pts)
        # np.save(os.path.join(overlay_dir, 'sg_affine_init.npy'), M_0)
        return 0

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img1 = cv2.imread(img1_file)
    img2 = cv2.imread(img2_file)

    img1_highres = img1
    img2_highres = img2

    ratio_img1 = max(img1_highres.shape[0], img1_highres.shape[1]) / float(small_image_res)
    width1 = int(img1.shape[1] / ratio_img1)
    height1 = int(img1.shape[0] / ratio_img1)
    dim1 = (width1, height1)
    # resize image
    img1 = cv2.resize(img1, dim1, interpolation=cv2.INTER_AREA)

    ratio_img2 = max(img2_highres.shape[0], img2_highres.shape[1]) / float(small_image_res)
    width2 = int(img2.shape[1] / ratio_img2)
    height2 = int(img2.shape[0] / ratio_img2)
    dim2 = (width2, height2)
    # resize image
    img2 = cv2.resize(img2, dim2, interpolation=cv2.INTER_AREA)
    img4 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
    cv2.imwrite(os.path.join(overlay_dir, 'match_inliers.jpg'), img4)
    # Image.fromarray(img4).show()

    # get inlier
    inlier_matches = []
    for mm in range(len(matchesMask)):
        if matchesMask[mm] == 1:
            inlier_matches.append(good_matches[mm])
    good_matches = inlier_matches

    print(good_matches)

    if len(good_matches) <= 10:
        M_0, _ = cv2.estimateAffine2D(src_pts, src_pts)
        np.save(os.path.join(overlay_dir, 'sg_affine_init.npy'), M_0)
        return 0

    # # Select good matched keypoints
    source_matched_kpts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    target_matched_kpts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #

    # cv2.drawContours(img1, [source_matched_kpts.astype(int)], -1, (0, 255, 0), 3)
    # Image.fromarray(img1).show()

    source_matched_kpts_new = np.copy(source_matched_kpts)
    target_matched_kpts_new = np.copy(target_matched_kpts)

    x_scale = (img1_highres.shape[0] / float(img1.shape[0]))
    y_scale = (img1_highres.shape[1] / float(img1.shape[1]))
    for si in range(len(source_matched_kpts_new)):
        for sj in range(len(source_matched_kpts_new[si])):
            source_matched_kpts_new[si][sj][0] = source_matched_kpts_new[si][sj][0] * x_scale
            source_matched_kpts_new[si][sj][1] = source_matched_kpts_new[si][sj][1] * y_scale

    x_scale = (img2_highres.shape[0] / float(img2.shape[0]))
    y_scale = (img2_highres.shape[1] / float(img2.shape[1]))
    for si in range(len(target_matched_kpts_new)):
        for sj in range(len(target_matched_kpts_new[si])):
            target_matched_kpts_new[si][sj][0] = target_matched_kpts_new[si][sj][0] * x_scale
            target_matched_kpts_new[si][sj][1] = target_matched_kpts_new[si][sj][1] * y_scale

    # cv2.drawContours(img2_highres, [target_matched_kpts.astype(int)], -1, (0, 255, 0), 3)
    # Image.fromarray(img2_highres).show()

    M_inv, rigid_mask_inv = cv2.estimateAffine2D(target_matched_kpts_new, source_matched_kpts_new)
    M_0, _ = cv2.estimateAffine2D(source_matched_kpts_new, target_matched_kpts_new)
    np.save(os.path.join(overlay_dir, 'sg_affine_init.npy'), M_0)

    warped_image = cv2.warpAffine(img1_highres, M_0[:2, :3], (img2_highres.shape[1], img2_highres.shape[0]))

    cv2.imwrite(os.path.join(overlay_dir, 'swift_affine.jpg'), warped_image)
    cv2.imwrite(os.path.join(overlay_dir, 'original_affine.jpg'), img2_highres)

    saveKpt(overlay_dir, source_matched_kpts_new, target_matched_kpts_new)
    return 1

def load_xfeat(cfg):
    """Load XFeat with torch.hub from ``xfeat_hub`` (``owner/repo[:ref]`` on GitHub, or a local clone)."""
    source = cfg.get('xfeat_hub') or 'verlab/accelerated_features'
    kwargs = dict(pretrained=True, top_k=4096)
    if os.path.isdir(source):
        return torch.hub.load(source, 'XFeat', source='local', **kwargs), source
    if COMMIT_RE.fullmatch(pinned_ref(source) or ''):
        kwargs['skip_validation'] = True   # torch.hub only accepts the head commits of branches and tags otherwise
    try:
        return torch.hub.load(source, 'XFeat', **kwargs), source
    except EOFError as exc:                # first use: torch.hub asks whether to trust the repository
        raise RuntimeError(
            "torch.hub asks you to confirm that you trust %s before it runs its code. Run xfeat_affine.py once "
            "in an interactive terminal and answer 'y', or set xfeat_hub to a local clone of the repository."
            % source) from exc


def record_xfeat_provenance(model, source, cfg, out_path):
    """Record which XFeat code and weights this run used; check the weights against xfeat_weights_sha256."""
    import inspect
    import json
    import platform
    import time
    ref = pinned_ref(source)
    info = {"hub_source": source,
            "requested_ref": "local folder" if os.path.isdir(source) else (ref or "default branch (not pinned)"),
            "torch": getattr(torch, "__version__", "unknown"), "python": platform.python_version(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    code_root = None
    for obj in (type(model), getattr(type(model), 'match_xfeat', None)):   # where the model's code lives
        try:
            code_root = find_code_root(inspect.getfile(obj))
            break
        except TypeError:
            continue
    commit = ref if COMMIT_RE.fullmatch(ref or '') else None   # pinned in xfeat_hub
    if code_root:
        info["code_dir"] = code_root
        info["code_sha256"] = tree_sha256(code_root)   # identifies the source files that were run
        commit = git_commit(code_root) or commit
    info["git_commit"] = commit
    if not commit:
        info["note"] = ("commit not known: torch.hub downloads a branch without recording its commit. code_sha256 "
                        "identifies the files that were run; pin a commit in xfeat_hub or use a local git clone "
                        "to record it.")
    try:
        weights = os.path.join(torch.hub.get_dir(), "checkpoints", "xfeat.pt")
    except AttributeError:
        weights = None
    if weights and os.path.isfile(weights):
        info["weights_path"] = weights
        info["weights_sha256"] = file_sha256(weights)
    expected = cfg.get('xfeat_weights_sha256')
    if expected:
        info["weights_expected_sha256"] = str(expected).lower()
        info["weights_verified"] = info.get("weights_sha256") == str(expected).lower()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fo:
        json.dump(info, fo, indent=2)
    print("XFeat provenance:", json.dumps(info, indent=2))
    if expected and not info["weights_verified"]:
        raise ValueError("XFeat weights %s have SHA-256 %s, but the config requires xfeat_weights_sha256 = %s"
                         % (weights, info.get("weights_sha256"), expected))


if __name__ == '__main__':
    args, cfg = parse_args(__doc__, add_level_argument)
    xfeat, xfeat_source = load_xfeat(cfg)
    record_xfeat_provenance(xfeat, xfeat_source, cfg,
                            os.path.join(cfg['paths']['image_root'], 'xfeat_provenance_%s.json' % args.level))

    # global level: one stack per case; patch level: one stack per sampled patch
    cases = list_stacks(cfg, args.level)
    not_ok = []  # (pair folder, status) of every pair without a proper XFeat affine
    for ci in range(len(cases)):
        case = cases[ci]
        now_case = os.path.basename(case)
        lbl_input_dir = case
        image_input_dir = case
        image_output_dir = os.path.join(case, 'registration_xfeat')
        # image_badcase_dir = os.path.join(case, 'registration_badcase_2048')


        images = glob.glob(os.path.join(image_input_dir, 'image_*.png'))
        images.sort(key=natural_keys)
        status_rows = []

        for ii in range(len(images)-1):
            moving_root = images[ii+1]
            moving_output_root = moving_root.replace(image_input_dir, image_output_dir)
            displacement_root = moving_output_root.replace('.png', '.npy')

            fixed_root = images[ii]
            print('now case is %s to %s.' % (os.path.basename(moving_root), os.path.basename(fixed_root)))

            t1 = 0.7
            overlay_dir = os.path.join(image_output_dir,'%s_to_%s' % (os.path.basename(moving_root).split('.')[0].split('_')[1],os.path.basename(fixed_root).split('.')[0].split('_')[1]))
            if not os.path.exists(overlay_dir):
                os.makedirs(overlay_dir)


            init_file = os.path.join(overlay_dir, 'sg_affine_init.npy')
            if os.path.exists(init_file):
                os.remove(init_file)  # never fall back on an initialisation left by an earlier run

            A = sg_affine(moving_root, fixed_root, overlay_dir, t1)
            while A == 0 and t1 <= 1.:
                print('A', A)
                t1 = t1 + 0.01 * (5 ** (t1 <= 0.9))
                A = sg_affine(moving_root, fixed_root, overlay_dir, t1)

            if A == 1:
                status = 'ok'
            elif os.path.exists(init_file):
                status = 'identity_fallback'  # <= 10 inlier matches: identity initialisation
            else:
                status = 'failed'             # too few matches at every ratio threshold: no initialisation
            status_rows.append({'pair': os.path.basename(overlay_dir), 'status': status})
            if status != 'ok':
                print('[WARN] %s: %s' % (overlay_dir, status))
                not_ok.append((overlay_dir, status))

        if status_rows:
            pd.DataFrame(status_rows).to_csv(os.path.join(image_output_dir, 'xfeat_status.csv'), index=False)

    if not_ok:
        print('\nXFeat summary: %d pair(s) without a feature-based affine' % len(not_ok))
        for folder, status in not_ok:
            print('  %-17s %s' % (status, folder))
        print('identity_fallback: ANTs starts from the unaligned pair - check its result. '
              'failed: no sg_affine_init.npy was written; affine_to_itk.py and ants_affine.py cannot run '
              'for these pairs until one is provided (e.g. np.save(<pair>/sg_affine_init.npy, np.eye(2, 3)) '
              'to start ANTs from the identity).')
        if any(status == 'failed' for _, status in not_ok):
            sys.exit(1)
