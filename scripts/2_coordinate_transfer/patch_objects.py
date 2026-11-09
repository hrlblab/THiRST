"""Refine the object contours with the nearest patch-level registration.

THiRST stage 2 - 3D object reconstruction, nearest-patch refinement (paper Sec. 4.2.5).

Each contour is assigned to the patch closest to its centroid; contours that leave that patch are skipped.

Inputs : outputs of ``global_objects.py``
Outputs: ``<contour_root>/<case>/registration_3D_patch_ANTs/annotation_QA_generegions/<case>-<section>.svs.geojson``
         ``.../annotation_QA_generegions/patch_contours_before_after.csv``

Usage
-----
    python scripts/2_coordinate_transfer/patch_objects.py --config configs/example.yaml

Original script(s): Step8.5_registertoMiddle_ANTs_points_object_patchlevel_fast_new_generegions_ori.py (superset of ..._generegions.py; ..._fast_new.py stopped on contours outside a patch)
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
from scipy.spatial import distance

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from thirst.config import list_cases, parse_args  # noqa: E402
from thirst.transforms import require_patch_chains, require_transform  # noqa: E402
from thirst.objects import object_feature, read_object_contours  # noqa: E402


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """Sort filenames in human order."""
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def get_nearest_patch_index(contour: np.ndarray, patch_array: np.ndarray) -> int:
    """
    Given a contour (an array of shape (N,2)) and a patch_array of shape (num_patches, 1, 2),
    compute the geometric center of the contour, then use distance.cdist to compute the Euclidean
    distances between this center and each patch center. Return the index of the nearest patch.
    """
    contour_center = np.mean(contour, axis=0, keepdims=True)  # shape: (1,2)
    patch_centers = patch_array[:, 0, :]  # shape: (num_patches, 2)
    dist_matrix = distance.cdist(contour_center, patch_centers)
    return int(np.argmin(dist_matrix))

if __name__ == '__main__':
    args, cfg = parse_args(__doc__)
    case_folder = cfg['paths']['image_root']
    point_folder = cfg['paths']['contour_root']

    cases = list_cases(cfg)

    for case in cases:
        now_case = os.path.basename(case)
        image_input_dir = case
        image_output_dir = os.path.join(case, 'registration_ANTs')
        reconstruction_dir = os.path.join(case.replace(case_folder, point_folder), 'registration_3D_ANTs', 'annotation_QA')
        patch_reconstruction_dir = os.path.join(case.replace(case_folder, point_folder), 'registration_3D_patch_ANTs', 'annotation_QA_generegions')

        if not os.path.exists(patch_reconstruction_dir):
            os.makedirs(patch_reconstruction_dir)

        # <-- added: single merged CSV collector
        rows_merged = []  # columns: case,image_name,slice_idx,read_index,class,contour_id,point_idx,x_orig,y_orig,x_new,y_new

        images = glob.glob(os.path.join(image_input_dir, 'image_*.png'))
        images.sort(key=natural_keys)
        require_patch_chains(case, images)  # every pair of every patch stack, before anything is written

        middle_idx = int(len(images) / 2)

        middle_ref_x = int(os.path.basename(images[middle_idx]).split('_')[2])
        middle_ref_y = int(os.path.basename(images[middle_idx]).split('_')[3].replace('.png', ''))

        patch_level_folder = os.path.join(case, '3D_Patch_crop')
        patch_list = glob.glob(os.path.join(patch_level_folder, '*'))
        patch_array = np.zeros((len(patch_list), 1, 2)).astype(np.float32)
        for pi in range(len(patch_list)):
            patch_name = os.path.basename(patch_list[pi])
            patch_abs_x, patch_abs_y = patch_name.split('_')
            patch_array[pi, :, 0] = int(patch_abs_x)
            patch_array[pi, :, 1] = int(patch_abs_y)

        # Process each image
        for ii in range(len(images)):
            now_idx = ii
            image_name = os.path.basename(images[now_idx])

            read_index = image_name.split('_')[1]

            mask_json_path = os.path.join(reconstruction_dir, '%s-%s.svs.geojson' % (now_case, read_index))

            # Load GeoJSON
            if not os.path.exists(mask_json_path):
                print(f"[WARN] Skipping {mask_json_path}, file not found: this section will have no objects "
                      f"and track_objects.py will stop until it has a GeoJSON")
                continue

            mask_label = list(cfg['object_classes'])  # Omni-Seg classes reconstructed in 3D
            contours_by_class = read_object_contours(mask_json_path, mask_label)  # output of global_objects.py
            transformed_annotations = {"type": "FeatureCollection", "features": []}

            # Extract real_x, real_y from filename (assuming format: "image_X_Y.png")
            parts = image_name.split('_')
            real_x, real_y = int(parts[2]), int(parts[3].replace('.png', ''))

            # Apply transformation to contours
            n_outside = 0  # contours left out because they fall outside their nearest patch window
            for obj in mask_label:
                contours_obj = contours_by_class[obj]

                transformed_contours = []
                contour_id_counter = 0  # <-- added: to keep a stable contour_id across nested loops

                for contour in contours_obj:
                    contour = np.array(contour)  # Convert to numpy array
                    if contour.ndim == 3:  # If it's (1, N, 2), squeeze it
                        contour = np.squeeze(contour, axis=0)

                    if contour.shape[-1] != 2:  # Ensure it's (N,2) format
                        print(f"Unexpected contour shape: {contour.shape}")
                        continue  # Skip this contour if incorrectly formatted

                    # <-- added: capture RAW original points AFTER squeeze, BEFORE any offsets/affine
                    orig_raw = contour.astype(np.float32).copy()  # shape (N,2)

                    new_patch_array = np.zeros((patch_array.shape))
                    new_patch_array[:, :, 0] = (patch_array[:, :, 0] - middle_ref_y) / 4
                    new_patch_array[:, :, 1] = (patch_array[:, :, 1] - middle_ref_x) / 4

                    # Find the index of the nearest patch using the contour center.
                    best_index = get_nearest_patch_index(contour[:, ::-1], new_patch_array)

                    if best_index is None:
                        print("No suitable patch found for this contour!")
                        contour_id_counter += 1
                        continue  # Skip if no patch is found

                    affine_matrix_folder = os.path.join(patch_list[best_index], 'registration_ANTs')

                    abs_patch_x = patch_array[best_index, 0, 1]
                    abs_patch_y = patch_array[best_index, 0, 0]

                    now_patch_y = int((abs_patch_x - middle_ref_x) / 4)
                    now_patch_x = int((abs_patch_y - middle_ref_y) / 4)

                    if now_idx < middle_idx:
                        M_new = np.eye(3)

                        # Accumulate transformations
                        for ri in range(now_idx, middle_idx):
                            moving_index = int(os.path.basename(images[ri + 1]).split('.')[0].split('_')[1])
                            fixed_index = int(os.path.basename(images[ri]).split('.')[0].split('_')[1])

                            affine_root = os.path.join(affine_matrix_folder, f'{moving_index}_to_{fixed_index}',
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

                        if contour.ndim == 2:
                            contour = contour.reshape((-1, 1, 2))

                        # get reference location on the image
                        patch_size = 1024
                        contour[:, :, 0] = contour[:, :, 0] - (now_patch_y - patch_size)
                        contour[:, :, 1] = contour[:, :, 1] - (now_patch_x - patch_size)

                        has_negatives = np.any(contour < 0)
                        has_large_values = np.any(contour > 2048)
                        if not (has_negatives == 0 and has_large_values == 0):
                            n_outside += 1  # not refined: outside the window of its nearest patch stack
                            contour_id_counter += 1
                            continue
                        else:
                            # Apply affine transformation
                            M_new_reverse = np.linalg.inv(M_new)

                            points = contour.reshape(-1, 1, 2).astype(np.float32)

                            transformed_points = cv2.perspectiveTransform(points, M_new_reverse).reshape(-1, 2)

                            # add offsets back to global coords
                            transformed_points[:, 0] = transformed_points[:, 0] + (now_patch_y - patch_size)
                            transformed_points[:, 1] = transformed_points[:, 1] + (now_patch_x - patch_size)

                            # for GeoJSON
                            transformed_contours.append(transformed_points.tolist())

                            transformed_annotations["features"].append(object_feature(obj, transformed_contours[-1], contour_id=contour_id_counter))

                            # <-- added: log merged pairs (orig_raw vs final transformed_points)
                            new_pts = transformed_points.tolist()
                            n_pairs = min(len(orig_raw), len(new_pts))
                            for p_idx in range(n_pairs):
                                x0, y0 = float(orig_raw[p_idx, 0]), float(orig_raw[p_idx, 1])
                                x1, y1 = float(new_pts[p_idx][0]), float(new_pts[p_idx][1])
                                rows_merged.append([
                                    now_case, image_name, now_idx, read_index, obj, contour_id_counter, p_idx,
                                    x0, y0, x1, y1
                                ])

                    elif now_idx > middle_idx:
                        M_new = np.zeros((3, 3))
                        M_new[2, 2] = 1.

                        for ri in range(middle_idx, now_idx):

                            moving_index = int(os.path.basename(images[ri + 1]).split('.')[0].split('_')[1])
                            fixed_index = int(os.path.basename(images[ri]).split('.')[0].split('_')[1])

                            affine_root = os.path.join(affine_matrix_folder, '%d_to_%d' % (moving_index, fixed_index),
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

                        if contour.ndim == 2:
                            contour = contour.reshape((-1, 1, 2))

                        # get reference location on the image
                        patch_size = 1024
                        contour[:, :, 0] = contour[:, :, 0] - (now_patch_y - patch_size)
                        contour[:, :, 1] = contour[:, :, 1] - (now_patch_x - patch_size)

                        has_negatives = np.any(contour < 0)
                        has_large_values = np.any(contour > 2048)
                        if not (has_negatives == 0 and has_large_values == 0):
                            n_outside += 1  # not refined: outside the window of its nearest patch stack
                            contour_id_counter += 1
                            continue
                        else:
                            # Apply affine transformation
                            points = contour.reshape(-1, 1, 2).astype(np.float32)

                            transformed_points = cv2.perspectiveTransform(points, M_new).reshape(-1, 2)

                            # add offsets back to global coords
                            transformed_points[:, 0] = transformed_points[:, 0] + (now_patch_y - patch_size)
                            transformed_points[:, 1] = transformed_points[:, 1] + (now_patch_x - patch_size)

                            # for GeoJSON
                            transformed_contours.append(transformed_points.tolist())

                            transformed_annotations["features"].append(object_feature(obj, transformed_contours[-1], contour_id=contour_id_counter))

                            # <-- added: log merged pairs
                            new_pts = transformed_points.tolist()
                            n_pairs = min(len(orig_raw), len(new_pts))
                            for p_idx in range(n_pairs):
                                x0, y0 = float(orig_raw[p_idx, 0]), float(orig_raw[p_idx, 1])
                                x1, y1 = float(new_pts[p_idx][0]), float(new_pts[p_idx][1])
                                rows_merged.append([
                                    now_case, image_name, now_idx, read_index, obj, contour_id_counter, p_idx,
                                    x0, y0, x1, y1
                                ])

                    else:
                        M_new = np.eye(3)

                        # Apply affine transformation (no offsets)
                        points = contour.reshape(-1, 1, 2).astype(np.float32)
                        transformed_points = cv2.perspectiveTransform(points, M_new).reshape(-1, 2).tolist()

                        transformed_contours.append(transformed_points)

                        transformed_annotations["features"].append(object_feature(obj, transformed_contours[-1], contour_id=contour_id_counter))

                        # <-- added: log merged pairs (identity)
                        n_pairs = min(len(orig_raw), len(transformed_points))
                        for p_idx in range(n_pairs):
                            x0, y0 = float(orig_raw[p_idx, 0]), float(orig_raw[p_idx, 1])
                            x1, y1 = float(transformed_points[p_idx][0]), float(transformed_points[p_idx][1])
                            rows_merged.append([
                                now_case, image_name, now_idx, read_index, obj, contour_id_counter, p_idx,
                                x0, y0, x1, y1
                            ])

                    contour_id_counter += 1  # advance id for the next contour

            if n_outside:
                print(f"[WARN] {image_name}: {n_outside} contour(s) extend beyond the 2048 px window of their nearest "
                      f"patch stack; they are not refined and are left out of the patch-level GeoJSON")

            # Save transformed GeoJSON
            output_json_path = os.path.join(patch_reconstruction_dir, os.path.basename(mask_json_path))
            with open(output_json_path, 'w') as f:
                json.dump(transformed_annotations, f, indent=4)

        # <-- added: write single merged CSV for this case
        merged_csv = os.path.join(patch_reconstruction_dir, 'patch_contours_before_after.csv')
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
