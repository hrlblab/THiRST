"""Map the segmented objects (Omni-Seg, QA'd in QuPath) into the reference frame (global affine chain).

THiRST stage 2 - 3D object reconstruction, global projection (paper Sec. 4.2.5).

Inputs : ``<object_annotation_dir>/<case>-<section>.svs.geojson`` (classes listed in ``object_classes``)
Outputs: ``<contour_root>/<case>/registration_3D_ANTs/annotation_QA/<case>-<section>.svs.geojson`` (4x reference frame)
         ``.../annotation_QA/contours_before_after.csv`` (every vertex before/after, used by track_objects.py
         to map tracks back to WSI coordinates)

Usage
-----
    python scripts/2_coordinate_transfer/global_objects.py --config configs/example.yaml

Original script(s): Step3.5_registertoMiddle_ANTs_points_object_fast_ori.py (superset of ..._object_fast.py)
"""
import sys
import os
import glob
import json
import re
import csv

import cv2
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from thirst.config import list_cases, parse_args  # noqa: E402
from thirst.transforms import require_chain, require_transform  # noqa: E402
from thirst.objects import object_feature, read_object_contours  # noqa: E402


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """Sort filenames in human order."""
    return [atoi(c) for c in re.split(r'(\d+)', text)]

if __name__ == '__main__':
    args, cfg = parse_args(__doc__)
    case_folder = cfg['paths']['image_root']
    point_folder = cfg['paths']['contour_root']
    Json_folder = cfg['paths']['object_annotation_dir']

    cases = list_cases(cfg)

    for case in cases:
        now_case = os.path.basename(case)
        image_input_dir = case
        image_output_dir = os.path.join(case, 'registration_ANTs')
        reconstruction_dir = os.path.join(case.replace(case_folder, point_folder), 'registration_3D_ANTs', 'annotation_QA')

        if not os.path.exists(reconstruction_dir):
            os.makedirs(reconstruction_dir)

        # single merged CSV collector
        rows_merged = []  # columns: case,image_name,slice_idx,read_index,class,contour_id,point_idx,x_orig,y_orig,x_new,y_new

        images = glob.glob(os.path.join(image_input_dir, 'image_*.png'))
        images.sort(key=natural_keys)
        require_chain(image_output_dir, images)  # every pair present and valid before anything is written

        middle_idx = int(len(images) / 2)

        # Process each image
        for ii in range(len(images)):
            now_idx = ii
            image_name = os.path.basename(images[now_idx])

            read_index = image_name.split('_')[1]

            mask_json_path = os.path.join(Json_folder, '%s-%s.svs.geojson' % (now_case, read_index))

            # Load GeoJSON
            if not os.path.exists(mask_json_path):
                print(f"[WARN] Skipping {mask_json_path}, file not found: this section will have no objects "
                      f"and track_objects.py will stop until it has a GeoJSON")
                continue

            mask_label = list(cfg['object_classes'])  # Omni-Seg classes reconstructed in 3D
            # one Polygon feature per object (exterior ring), float coordinates
            contours_by_class = read_object_contours(mask_json_path, mask_label)
            transformed_annotations = {"type": "FeatureCollection", "features": []}

            # Extract real_x, real_y from filename (assuming format: "image_X_Y.png")
            parts = image_name.split('_')
            real_x, real_y = int(parts[2]), int(parts[3].replace('.png', ''))

            if now_idx < middle_idx:
                M_new = np.eye(3)

                # Accumulate transformations
                for ri in range(now_idx, middle_idx):
                    moving_index = int(os.path.basename(images[ri + 1]).split('.')[0].split('_')[1])
                    fixed_index = int(os.path.basename(images[ri]).split('.')[0].split('_')[1])

                    affine_root = os.path.join(image_output_dir, f'{moving_index}_to_{fixed_index}',
                                               'step2_run_ants_reg', 'output0GenericAffine.mat')
                    m = sitk.ReadTransform(require_transform(affine_root))  # a missing pair stops the run (no partial chain)
                    FixP = m.GetFixedParameters()
                    M_vec = m.GetParameters()
                    M_inv = np.zeros((2, 3))

                    M_inv[0, 0], M_inv[0, 1] = M_vec[0], M_vec[1]
                    M_inv[1, 0], M_inv[1, 1] = M_vec[2], M_vec[3]
                    M_inv[0, 2] = M_vec[4] - (FixP[0] * M_inv[0, 0] + FixP[1] * M_inv[0, 1] - FixP[0])
                    M_inv[1, 2] = M_vec[5] - (FixP[0] * M_inv[1, 0] + FixP[1] * M_inv[1, 1] - FixP[1])

                    M1 = np.eye(3)
                    M1[:2, :] = cv2.invertAffineTransform(M_inv)
                    M_new = M_new @ M1  # Matrix multiplication

                # Apply transformation to contours
                for obj in mask_label:
                    contours_obj = contours_by_class[obj]

                    transformed_contours = []
                    for c_idx, contour in enumerate(contours_obj):
                        contour = np.array(contour)  # Convert to numpy array
                        if contour.ndim == 3:  # If it's (1, N, 2), squeeze it
                            contour = np.squeeze(contour, axis=0)

                        if contour.shape[-1] != 2:  # Ensure it's (N,2) format
                            print(f"Unexpected contour shape: {contour.shape}")
                            continue  # Skip this contour if incorrectly formatted

                        # capture RAW full-res coords AFTER squeezing, BEFORE normalization
                        orig_raw = contour.astype(np.float32).copy()

                        # Normalize contour coordinates
                        contour[:, 0] = (contour[:, 0] - real_x) / 4
                        contour[:, 1] = (contour[:, 1] - real_y) / 4

                        # Apply affine transformation
                        points = contour.reshape(-1, 1, 2).astype(np.float32)
                        M_new_reverse = np.linalg.inv(M_new)

                        transformed_points = cv2.perspectiveTransform(points, M_new_reverse).reshape(-1, 2).tolist()
                        transformed_contours.append(transformed_points)
                        transformed_annotations["features"].append(object_feature(obj, transformed_points, contour_id=c_idx))

                        # log merged (orig_raw vs transformed_points)
                        n_pairs = min(len(orig_raw), len(transformed_points))
                        for p_idx in range(n_pairs):
                            pt0 = orig_raw[p_idx]
                            x0 = float(pt0[0]); y0 = float(pt0[1])
                            pt1 = transformed_points[p_idx]
                            x1 = float(pt1[0]); y1 = float(pt1[1])
                            rows_merged.append([
                                now_case, image_name, now_idx, read_index, obj, c_idx, p_idx, x0, y0, x1, y1
                            ])


            elif now_idx > middle_idx:
                M_new = np.zeros((3, 3))
                M_new[2, 2] = 1.

                for ri in range(middle_idx, now_idx):

                    moving_index = int(os.path.basename(images[ri + 1]).split('.')[0].split('_')[1])
                    fixed_index = int(os.path.basename(images[ri]).split('.')[0].split('_')[1])

                    affine_root = os.path.join(image_output_dir, '%d_to_%d' % (moving_index, fixed_index),
                                               'step2_run_ants_reg', 'output0GenericAffine.mat')

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

                # Apply transformation to contours
                for obj in mask_label:
                    contours_obj = contours_by_class[obj]

                    transformed_contours = []
                    for c_idx, contour in enumerate(contours_obj):
                        contour = np.array(contour)  # Convert to numpy array
                        if contour.ndim == 3:  # If it's (1, N, 2), squeeze it
                            contour = np.squeeze(contour, axis=0)

                        if contour.shape[-1] != 2:  # Ensure it's (N,2) format
                            print(f"Unexpected contour shape: {contour.shape}")
                            continue  # Skip this contour if incorrectly formatted

                        # capture RAW full-res coords AFTER squeezing, BEFORE normalization
                        orig_raw = contour.astype(np.float32).copy()

                        # Normalize contour coordinates
                        contour[:, 0] = (contour[:, 0] - real_x) / 4
                        contour[:, 1] = (contour[:, 1] - real_y) / 4

                        # Apply affine transformation
                        points = contour.reshape(-1, 1, 2).astype(np.float32)

                        transformed_points = cv2.perspectiveTransform(points, M_new).reshape(-1, 2).tolist()
                        transformed_contours.append(transformed_points)
                        transformed_annotations["features"].append(object_feature(obj, transformed_points, contour_id=c_idx))

                        # log merged (orig_raw vs transformed_points)
                        n_pairs = min(len(orig_raw), len(transformed_points))
                        for p_idx in range(n_pairs):
                            pt0 = orig_raw[p_idx]
                            x0 = float(pt0[0]); y0 = float(pt0[1])
                            pt1 = transformed_points[p_idx]
                            x1 = float(pt1[0]); y1 = float(pt1[1])
                            rows_merged.append([
                                now_case, image_name, now_idx, read_index, obj, c_idx, p_idx, x0, y0, x1, y1
                            ])


            else:
                M_new = np.eye(3)
                # Apply transformation to contours
                for obj in mask_label:
                    contours_obj = contours_by_class[obj]

                    transformed_contours = []
                    for c_idx, contour in enumerate(contours_obj):
                        contour = np.array(contour)  # Convert to numpy array
                        if contour.ndim == 3:  # If it's (1, N, 2), squeeze it
                            contour = np.squeeze(contour, axis=0)

                        if contour.shape[-1] != 2:  # Ensure it's (N,2) format
                            print(f"Unexpected contour shape: {contour.shape}")
                            continue  # Skip this contour if incorrectly formatted

                        # capture RAW full-res coords AFTER squeezing, BEFORE normalization
                        orig_raw = contour.astype(np.float32).copy()

                        # Normalize contour coordinates
                        contour[:, 0] = (contour[:, 0] - real_x) / 4
                        contour[:, 1] = (contour[:, 1] - real_y) / 4

                        # Apply affine transformation
                        points = contour.reshape(-1, 1, 2).astype(np.float32)

                        transformed_points = cv2.perspectiveTransform(points, M_new).reshape(-1, 2).tolist()
                        transformed_contours.append(transformed_points)
                        transformed_annotations["features"].append(object_feature(obj, transformed_points, contour_id=c_idx))

                        # log merged (orig_raw vs transformed_points)
                        n_pairs = min(len(orig_raw), len(transformed_points))
                        for p_idx in range(n_pairs):
                            pt0 = orig_raw[p_idx]
                            x0 = float(pt0[0]); y0 = float(pt0[1])
                            pt1 = transformed_points[p_idx]
                            x1 = float(pt1[0]); y1 = float(pt1[1])
                            rows_merged.append([
                                now_case, image_name, now_idx, read_index, obj, c_idx, p_idx, x0, y0, x1, y1
                            ])


            # Save transformed GeoJSON
            output_json_path = os.path.join(reconstruction_dir, os.path.basename(mask_json_path))
            with open(output_json_path, 'w') as f:
                json.dump(transformed_annotations, f, indent=4)

        # WRITE SINGLE MERGED CSV
        merged_csv = os.path.join(reconstruction_dir, 'contours_before_after.csv')
        try:
            with open(merged_csv, 'w', newline='') as fm:
                w = csv.writer(fm)
                w.writerow([
                    'case','image_name','slice_idx','read_index','class','contour_id','point_idx',
                    'x_orig','y_orig','x_new','y_new'
                ])
                w.writerows(rows_merged)
            print(f"Saved {len(rows_merged)} point pairs to {merged_csv}")
        except Exception as e:  # track_objects.py needs this file: stop here
            raise RuntimeError(f"Writing {merged_csv} failed: {e}") from e
