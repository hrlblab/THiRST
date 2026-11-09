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
"""Per-object 3D gene-expression maps of marker genes on Visium HD bins (paper Fig. 3a).

THiRST stage 5 - object-level 3D ST visualisation.

For every gene in the marker list and every track, HD bins inside the object are coloured by expression
(shared colour scale) on the locally registered patch of each HD section.

External inputs (``analysis:`` in the config): hd_expression_mtx (Matrix Market, genes x bins),
hd_barcodes_csv, hd_genes_csv (no header) and marker_gene_list_csv (column ``marker_gene_column``).
Section-to-sample names come from ``st_sections[*].sample`` (barcodes are ``<sample>-8-<barcode>``).

Outputs: ``<contour_root>/<case>/Object_tracking_IoU_affine_patch_gene_highselect_colorbar_yanglist<tag>/<track_id>/<gene>/``
(``<tag>`` = empty for the greedy tracks, ``_connectglom`` for the Hungarian tracks, see ``--tracks-subdir``)

Usage
-----
    python scripts/5_object_analysis/gene_maps_per_track.py --config configs/example.yaml

Original script(s): Step10_Match_Object_gene_patchregistration_using_patch_gene_highselect_colorbar_withYanglist.py
"""
import sys
import os
import glob
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import cv2 as cv2
import pandas as pd
import matplotlib.cm as cm
from scipy.spatial import distance
import scipy.io as spio
from matplotlib.colors import LinearSegmentedColormap, Normalize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from thirst.config import add_tracks_argument, analysis_path, list_cases, parse_args, require_alternating_platforms, section_object_geojsons, single_object_class, stack_sample_names, tracks_json, tracks_output_dir  # noqa: E402
from thirst.objects import read_object_contours  # noqa: E402


def atoi(text):
    return int(text) if text.isdigit() else text


# Step 3: Compute IoU between each circle and each bin
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


# The original intensity_to_bgr function is replaced with a custom version.
def intensity_to_bgr_custom(intensity, max_intensity):
    """
    Map an intensity in [0, max_intensity] to a BGR color.
    The mapping uses a custom colormap:
      - 0       -> Blue   (BGR: (255, 0, 0))
      - mid-low -> Grey-blue
      - mid     -> Grey
      - mid-high-> Grey-red
      - max     -> Red    (BGR: (0, 0, 255))
    """
    # Normalize intensity between 0 and 1.
    norm_val = intensity / max_intensity if max_intensity != 0 else 0.0

    # Define a custom colormap with colors in RGB.
    colors = [
            (0, 0, 1),  # Blue
            (0.5, 0.5, 1),  # Grey-blue
            (0.8, 0.8, 0.8),  # Grey
            (1, 0.5, 0.5),  # Grey-red
            (1, 0, 0)  # Red
    ]
    cmap_custom = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    color_rgba = cmap_custom(norm_val)
    # Convert RGBA (0-1 range) to BGR (0-255 range).
    r, g, b, _ = color_rgba
    return (int(b * 255), int(g * 255), int(r * 255))


def load_annotations(image_name, mask_json_path, mask_labels):
    """
    Loads a GeoJSON file, extracts contours for given object labels,
    applies a translation transformation based on the image filename,
    and returns a dictionary mapping object type to a list of numpy arrays (contours).
    """
    if not os.path.exists(mask_json_path):
        print(f"Skipping {mask_json_path}, file not found")
        return None

    # {class: [contour, ...]}; reads standard GeoJSON and the earlier per-class layout
    return read_object_contours(mask_json_path, mask_labels)


def get_nearest_patch_index(contour: np.ndarray, patch_array: np.ndarray) -> int:
    """
    Given a contour (an array of shape (N,2)) and a patch_array of shape (num_patches, 1, 2),
    compute the geometric center of the contour, then use distance.cdist to compute the Euclidean
    distances between this center and each patch center. Return the index of the nearest patch.
    """
    # Compute the contour's geometric center; keep dims to get shape (1,2)
    contour_center = np.mean(contour, axis=0, keepdims=True)  # shape: (1,2)

    # Extract patch centers from patch_array (from shape (num_patches,1,2) to (num_patches,2))
    patch_centers = patch_array[:, 0, :]  # shape: (num_patches, 2)

    # Compute the distance matrix between the contour center and all patch centers.
    dist_matrix = distance.cdist(contour_center, patch_centers)
    return int(np.argmin(dist_matrix))


def contours_to_polygons(annotations_by_object):
    """
    Converts each contour (numpy array) into a Shapely Polygon.
    Returns a list of dictionaries for the slice,
    each with keys: 'object' (type), 'index' (contour index) and 'polygon' (Shapely Polygon).
    """
    slice_contours = []
    for obj, contour_list in annotations_by_object.items():
        for idx, pts in enumerate(contour_list):
            # Here, pts remains as a numpy array.
            poly = pts  # You may convert to a proper Polygon if needed: Polygon(pts)
            slice_contours.append({'object': obj, 'index': idx, 'polygon': poly})
    return slice_contours


# New function to save images with a color bar.
def save_image_with_colorbar(image, filename, max_intensity):
    """
    Save the given image with an appended color bar.
    The image is assumed to be in BGR format, so it is converted to RGB before plotting.
    """
    # Convert from BGR to RGB.
    image_rgb = image[..., ::-1]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(image_rgb)
    ax.axis('off')

    # Create the same custom colormap.
    colors = [
            (0, 0, 1),
            (0.5, 0.5, 1),
            (0.8, 0.8, 0.8),
            (1, 0.5, 0.5),
            (1, 0, 0)
    ]
    cmap_custom = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    norm = Normalize(vmin=0, vmax=max_intensity)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap_custom)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Intensity", rotation=270, labelpad=15)

    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    args, cfg = parse_args(__doc__, lambda p: add_tracks_argument(p, default='Object_IoU_patch'))
    # Directories and folder setup.
    case_folder = cfg['paths']['image_root']
    point_folder = cfg['paths']['point_root']
    contour_folder = cfg['paths']['contour_root']

    cases = list_cases(cfg)
    for ci in range(len(cases)):
        case = cases[ci]
        now_case = os.path.basename(case)
        lbl_input_dir = case
        image_input_dir = case
        image_output_dir = os.path.join(case, 'registration_ANTs')
        reconstruction_dir = os.path.join(case.replace(case_folder, point_folder), 'registration_3D_patch_ANTs')
        reconstruction_contour_dir = os.path.join(case.replace(case_folder, contour_folder),
                                                  'registration_3D_patch_ANTs')
        IoU_dir = os.path.join(case.replace(case_folder, contour_folder), args.tracks_subdir)
        global_reconstruction_dir = os.path.join(case.replace(case_folder, point_folder), 'registration_3D_ANTs')
        global_reconstruction_contour_dir = os.path.join(case.replace(case_folder, contour_folder),
                                                         'registration_3D_ANTs')
        after_annotation_dir = os.path.join(reconstruction_contour_dir, 'annotation_QA_generegions')
        tracking_output = tracks_output_dir(case.replace(case_folder, contour_folder),
                                            'Object_tracking_IoU_affine_patch_gene_highselect_colorbar_yanglist',
                                            args.tracks_subdir, args.tracks_suffix)

        if not os.path.exists(tracking_output):
            os.makedirs(tracking_output)

        big_images = glob.glob(
                os.path.join(os.path.join(case, 'registration_3D_ANTs'),
                             'image_*.png'))
        big_images.sort(key=natural_keys)

        images = glob.glob(os.path.join(image_input_dir, 'image_*.png'))
        images.sort(key=natural_keys)

        jsons = section_object_geojsons(images, after_annotation_dir, now_case)  # one per section, in stack order
        require_alternating_platforms(cfg, images)  # V2 at even, HD at odd stack positions

        csv_after = glob.glob(os.path.join(reconstruction_contour_dir, '*.csv'))
        csv_after.sort(key=natural_keys)

        middle_idx = int(len(images) / 2)
        middle_ref_x = int(os.path.basename(images[middle_idx]).split('_')[2])
        middle_ref_y = int(os.path.basename(images[middle_idx]).split('_')[3].replace('.png', ''))

        patch_level_folder = os.path.join(case, '3D_Patch_crop')
        patch_list = glob.glob(os.path.join(patch_level_folder, '*'))
        patch_array = np.zeros((len(patch_list), 1, 2)).astype(np.float32)
        for pi in range(len(patch_list)):
            if '3D' in os.path.basename(patch_list[pi]):
                continue

            patch_name = os.path.basename(patch_list[pi])
            patch_abs_x, patch_abs_y = patch_name.split('_')
            patch_array[pi, :, 0] = int(patch_abs_x)
            patch_array[pi, :, 1] = int(patch_abs_y)

        mask_labels = single_object_class(cfg)  # the Omni-Seg class reconstructed in 3D (one class)
        # Build a dictionary that holds all contours (as polygons) for each slice.
        all_slices = {}  # key: slice index, value: list of dicts with keys 'object' and 'polygon'
        for idx, (img_name, json_path) in enumerate(zip(images, jsons)):
            ann = load_annotations(img_name, json_path, mask_labels)
            if ann is None:
                continue
            all_slices[idx] = contours_to_polygons(ann)

        images_slice = {}
        for xi, img in enumerate(images):
            real_x = int(os.path.basename(images[xi]).split('_')[2])
            real_y = int(os.path.basename(images[xi]).split('_')[3].replace('.png', ''))
            now_img = plt.imread(images[xi])[:, :, :3]
            images_slice[xi] = {'img': now_img, 'real_x': real_x, 'real_y': real_y}

        # Load the feature data.
        matrix = spio.mmread(analysis_path(cfg, 'hd_expression_mtx'))  # genes x bins (Matrix Market)
        feature_HD = matrix.toarray().T
        feature_HD_barcode = pd.read_csv(analysis_path(cfg, 'hd_barcodes_csv'),
                                         header=None)
        feature_HD_gene = pd.read_csv(analysis_path(cfg, 'hd_genes_csv'),
                                      header=None)

        # Instead of normalizing the features, we use the raw values.
        max_intensity = np.max(feature_HD)

        spot_list_after = {}

        # section index -> sample name used in the merged expression matrix (barcodes '<sample>-8-<barcode>')
        name_match = stack_sample_names(cfg, images)  # sample of every section, in stack order

        for si, csv in enumerate(csv_after):
            now_df = pd.read_csv(csv)
            spot_list_after[si] = {'spot': now_df}

        trakcing_file = tracks_json(IoU_dir, args.tracks_suffix)
        with open(trakcing_file, 'r') as f:
            tracking_results = json.load(f)

        # Load the marker data.
        # df_cluster_4 = df[df["cluster"] == 4]
        # high_select_gene = df_cluster_4['gene']
        # common_genes_list = list(set(high_select_gene) & set(feature_HD_gene[0]))

        common_genes_list = pd.read_csv(analysis_path(cfg, 'marker_gene_list_csv'))[cfg.get('marker_gene_column', 'suggested glom enricheed genes')].tolist()

        for fi in range(len(common_genes_list)):
            print_gene = common_genes_list[fi]
            gene_index = np.where(np.array(feature_HD_gene[0]) == print_gene)[0][0]

            # Prepare a DataFrame to record spot information.
            spot_list = pd.DataFrame(columns=['track_id', 'slice_num', 'barcode', 'IoU', 'annotation_index'])

            for ti in range(len(tracking_results)):
                now_tracking = tracking_results[ti]
                track_id = now_tracking['track_id']
                for si in range(len(now_tracking['contours'])):
                    now_obj = now_tracking['contours'][si]
                    now_idx = now_obj['index']
                    now_slice = now_obj['slice_index']

                    if now_slice % 2 == 0:
                        continue

                    now_contour = all_slices[now_slice][now_idx]['polygon']
                    now_image = images_slice[now_slice]['img']

                    new_patch_array = np.zeros((patch_array.shape))
                    new_patch_array[:, :, 0] = (patch_array[:, :, 0] - middle_ref_y) / 4
                    new_patch_array[:, :, 1] = (patch_array[:, :, 1] - middle_ref_x) / 4

                    best_index = get_nearest_patch_index(now_contour[:, ::-1], new_patch_array)
                    print(best_index)
                    if best_index is None:
                        print("No suitable patch found for this contour!")
                        continue

                    now_affine_img_folder = os.path.join(patch_list[best_index], 'registration_3D_ANTs')
                    region_list = glob.glob(os.path.join(now_affine_img_folder, 'image_*.png'))
                    region_list.sort(key=natural_keys)

                    now_region = plt.imread(region_list[now_slice])[:, :, :3]

                    abs_patch_x = patch_array[best_index, 0, 1]
                    abs_patch_y = patch_array[best_index, 0, 0]

                    now_patch_y = int((abs_patch_x - middle_ref_x) / 4)
                    now_patch_x = int((abs_patch_y - middle_ref_y) / 4)

                    center_y = int(now_contour[:, 0].mean() - (now_patch_x - now_region.shape[0]))
                    center_x = int(now_contour[:, 1].mean() - (now_patch_y - now_region.shape[1]))

                    patch_size = 1024
                    now_patch = now_region  # Alternatively, you could crop the patch as needed.

                    now_df_after = spot_list_after[now_slice]['spot']

                    num_points = 4
                    batch_array_i = np.zeros((len(now_df_after), num_points, 1, 2)).astype(np.float32)
                    for ni in range(num_points):
                        batch_array_i[:, ni, :, 0] = np.expand_dims(np.array(now_df_after['now_y_%d' % (ni)]), 1)
                        batch_array_i[:, ni, :, 1] = np.expand_dims(np.array(now_df_after['now_x_%d' % (ni)]), 1)

                    try:
                        polygon_now = Polygon(now_contour)
                    except Exception as e:
                        continue

                    import shapely  # contains_xy replaces the deprecated shapely.vectorized.contains

                    coords = batch_array_i[:, :, 0, :]  # shape: (N, 4, 2)
                    x_coords = coords[:, :, 0]
                    y_coords = coords[:, :, 1]
                    N, n_points = x_coords.shape
                    contained = shapely.contains_xy(polygon_now, x_coords.ravel(), y_coords.ravel())
                    contained = contained.reshape(N, n_points)
                    inside_mask = np.all(contained, axis=1)
                    inside_indices = np.nonzero(inside_mask)[0]

                    if now_contour.ndim == 2:
                        now_contour = now_contour.reshape((-1, 1, 2))

                    now_contour_patch = np.zeros((now_contour.shape))
                    now_contour_patch[:, :, 0] = now_contour[:, :, 0] - (now_patch_y - patch_size)
                    now_contour_patch[:, :, 1] = now_contour[:, :, 1] - (now_patch_x - patch_size)
                    contour_int = np.ascontiguousarray(now_contour_patch.astype(np.int32))

                    now_patch = (now_patch * 255).astype(np.uint8)
                    # Draw the main contour.
                    # cv2.drawContours(now_patch, [contour_int], -1, (255, 255, 0), 20)
                    print("draw the contour")
                    print(contour_int[:, :, 0].mean(), contour_int[:, :, 1].mean())

                    now_mask = np.zeros((now_patch.shape)).astype(np.uint8)

                    for i in range(len(inside_indices)):
                        now_pos_barcode = now_df_after.iloc[inside_indices[i]]['barcode']
                        values_list = list(name_match.values())
                        slice_name = values_list[now_slice]
                        # Use the raw feature values instead of normalized values.
                        gene_feature = feature_HD[feature_HD_barcode[
                                feature_HD_barcode[0] == '%s-8-%s' % (slice_name, now_pos_barcode)].index]
                        if len(gene_feature) == 0:
                            continue
                        intensity = gene_feature[0][gene_index]
                        color = intensity_to_bgr_custom(intensity, max_intensity)

                        now_spot = batch_array_i[inside_indices[i]]
                        if now_spot.ndim == 2:
                            now_spot = now_spot.reshape((-1, 1, 2))
                        now_spot[:, :, 0] = now_spot[:, :, 0] - (now_patch_y - patch_size)
                        now_spot[:, :, 1] = now_spot[:, :, 1] - (now_patch_x - patch_size)
                        contour_int = np.ascontiguousarray(now_spot.astype(np.int32))

                        # Record the spot information.
                        row = len(spot_list)
                        spot_list.loc[row] = [track_id, now_slice, now_df_after['barcode'][inside_indices[i]], 1,
                                              now_idx]
                        cv2.drawContours(now_patch, [contour_int], -1, color, -1)
                        cv2.drawContours(now_mask, [contour_int], -1, color, -1)

                        # # Optionally, add the raw intensity value as text near the spot.
                        # text_x = int(contour_int[:, :, 0].mean())
                        # text_y = int(contour_int[:, :, 1].mean())
                        # cv2.putText(now_patch, f"{intensity:.2f}", (text_x, text_y),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                    save_folder = os.path.join(tracking_output, str(track_id), print_gene)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    # Save the main image with the color bar using the custom function.
                    save_image_with_colorbar(now_patch, os.path.join(save_folder, f'Glo_slice_{now_slice}.png'),
                                             max_intensity)
                    # Save the mask image as before.
                    cv2.imwrite(os.path.join(save_folder, f'Glo_slice_{now_slice}_mask.png'), now_mask)

            csv_root = os.path.join(tracking_output, 'IoU_withspot.csv')
            spot_list.to_csv(csv_root)