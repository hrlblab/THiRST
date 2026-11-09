# Copyright 2018 University of Basel, Center for medical Image Analysis
# and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
# -----------------------------------------------------------------------------
"""QuPath annotations of Visium HD bins of selected Seurat clusters inside every 3D object.

THiRST stage 5 - object-level 3D ST visualisation.

Bins whose projected Seurat cluster belongs to a class in ``cell_clusters_per_track`` (e.g. podocyte,
mesangial cell) and that lie inside a tracked object are drawn as 32 px boxes, one GeoJSON per track and section.

External inputs: ``analysis.clustering_meta_csv`` (columns Sample, Barcode, seurat_cluster.projected),
Space Ranger positions of the HD sections, ``st_sections[*].sample``.
Outputs: ``<contour_root>/<case>/qupath_annotations_by_track_clusters<tag>/track_<id>/<case>_<sample>_track<id>_cells.geojson``
(``<tag>`` = empty for the greedy tracks, ``_connectglom`` for the Hungarian tracks, see ``--tracks-subdir``)

Usage
-----
    python scripts/5_object_analysis/qupath_cells_by_cluster.py --config configs/example.yaml

Original script(s): Step10_Match_Object_gene_patchregistration_using_patch_gene_colorbar_withYuclustering_cell_bbox_QuPath_0813.py
"""
import sys
import os
import glob
import re
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import distance
from shapely.geometry import Polygon
import shapely  # contains_xy replaces the deprecated shapely.vectorized.contains

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from thirst.config import add_tracks_argument, analysis_path, list_cases, parse_args, read_positions, section_object_geojsons, section_sample_names, single_object_class, st_section, stack_sample_names, tracks_json, tracks_output_dir  # noqa: E402
from thirst.st import short_hd_barcode  # noqa: E402
from thirst.objects import read_object_contours  # noqa: E402


BOX_SIDE = 32  # side length in pixels for annotation boxes centered at (x,y)

# ------------------------------ Slice mapping ------------------------------ #
# Map sample name (as used in the merged expression matrix, e.g. "<sample>-8-<barcode>") to its
# section index; filled in __main__ from the 'sample' fields of 'st_sections' in the config.
_SLICE_TO_A = {}

# Config loaded in __main__ (used by get_ori_files)
_CFG = None

# ============================ Utility functions ============================ #

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def load_annotations(image_name, mask_json_path, mask_labels):
    if not os.path.exists(mask_json_path):
        print(f"Skipping {mask_json_path}, file not found")
        return None
    # {class: [contour, ...]}; reads standard GeoJSON and the earlier per-class layout
    return read_object_contours(mask_json_path, mask_labels)


def contours_to_polygons(annotations_by_object):
    slice_contours = []
    for obj, contour_list in annotations_by_object.items():
        for idx, pts in enumerate(contour_list):
            slice_contours.append({'object': obj, 'index': idx, 'polygon': pts})
    return slice_contours

def get_nearest_patch_index(contour: np.ndarray, patch_array: np.ndarray) -> int:
    contour_center = np.mean(contour, axis=0, keepdims=True)
    patch_centers = patch_array[:, 0, :]
    dist_matrix = distance.cdist(contour_center, patch_centers)
    return int(np.argmin(dist_matrix))

# ------------------------------ GeoJSON helpers ---------------------------- #

def box32_polygon_at_center(cx, cy, side=32):
    half = side / 2.0
    x0, y0 = float(cx - half), float(cy - half)
    x1, y1 = float(cx + half), float(cy + half)
    return [[
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
        [x0, y0],
    ]]

def make_qupath_feature(poly_coords, classification_name, props):
    """
    QuPath-friendly Feature with classification + measurements.
    classification_name: e.g., "podocyte" or "mesangial cell"
    """
    feature = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": poly_coords},
        "properties": {
            "objectType": "annotation",
            "isLocked": False,
            "classification": {"name": classification_name},
            "measurements": [
                {"name": "slice_index", "value": int(props.get("slice_index", -1))},
                {"name": "track_id", "value": int(props.get("track_id", -1))},
                {"name": "cluster_projected", "value": float(props.get("cluster_projected", -1))},
            ],
            "barcode": str(props.get("barcode", "")),
        }
    }
    return feature

def write_geojson_featurecollection(features, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fc = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w") as f:
        json.dump(fc, f, indent=2)
    print(f"[GeoJSON] Wrote {len(features)} features -> {out_path}")

# ------------------------- Barcode & positions helpers --------------------- #

def parse_full_barcode(full_barcode):
    """Split 'S10-8-<barcode>' into ('S10', '<barcode>')."""
    parts = full_barcode.split("-8-", 1)
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]

_POSITIONS_CACHE = {}

def _read_parquet_any(path):
    """
    Try pyarrow first (recommended), then fastparquet.
    """
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception as e_py:
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except Exception as e_fp:
            raise ImportError(
                "Neither 'pyarrow' nor 'fastparquet' is available for reading parquet.\n"
                "Install one of them, e.g.:\n"
                "  conda install -c conda-forge pyarrow\n"
                "  # or\n"
                "  pip install pyarrow\n"
                f"\npyarrow error: {e_py}\nfastparquet error: {e_fp}"
            )

def lookup_xy_from_positions(full_barcode, get_ori_files_fn):
    """
    Use get_ori_files(a) to load positions and find the (x,y) for this barcode.
    Supports both (barcode,x,y) and 10x-style pxl_col_in_fullres/pxl_row_in_fullres.
    """
    slice_name, bare_barcode = parse_full_barcode(full_barcode)
    if slice_name is None:
        return None
    a = _SLICE_TO_A.get(slice_name)
    if a is None:
        return None

    if a not in _POSITIONS_CACHE:
        pos_df = get_ori_files_fn(a)
        _POSITIONS_CACHE[a] = pos_df
    else:
        pos_df = _POSITIONS_CACHE[a]

    cols = {c.lower(): c for c in pos_df.columns}

    # Primary schema: columns named 'barcode', 'x', 'y'
    if "barcode" in cols and "x" in cols and "y" in cols:
        row = pos_df.loc[pos_df[cols["barcode"]] == bare_barcode]
        if len(row):
            return float(row.iloc[0][cols["x"]]), float(row.iloc[0][cols["y"]])

    # Fallback: 10x Visium-style columns
    cand_x = None
    cand_y = None
    for name in ("pxl_col_in_fullres", "x", "col", "px", "pixel_x"):
        if name in cols:
            cand_x = cols[name]; break
    for name in ("pxl_row_in_fullres", "y", "row", "py", "pixel_y"):
        if name in cols:
            cand_y = cols[name]; break
    if "barcode" in cols and cand_x and cand_y:
        row = pos_df.loc[pos_df[cols["barcode"]] == bare_barcode]
        if len(row):
            return float(row.iloc[0][cand_x]), float(row.iloc[0][cand_y])

    return None

# ------------------------------ Cluster logic ------------------------------ #

def build_barcode_to_cluster_map(clustering_csv_path):
    """
    Reads the clustering CSV and constructs:
        full_barcode = '{Sample}-8-{short_barcode}' -> seurat_cluster.projected (int)
    """
    df = pd.read_csv(clustering_csv_path)


    barcode2cluster = {}
    bad = 0
    for _, row in df.iterrows():
        sample = str(row.get('Sample', '')).strip()
        bc_raw = str(row.get('Barcode', '')).strip()
        if not sample or not bc_raw:
            bad += 1
            continue
        bc_short = short_hd_barcode(bc_raw)
        full_bc = f"{sample}-8-{bc_short}"
        try:
            cl = int(row['seurat_cluster.projected'])
        except Exception:
            bad += 1
            continue
        barcode2cluster[full_bc] = cl

    print(f"[Clusters] Loaded {len(barcode2cluster)} barcode→cluster labels from CSV "
          f"(skipped {bad} rows with missing/invalid data).")
    return barcode2cluster

def collect_cell_boxes_for_track_by_cluster(
        track,
        all_slices,
        images_slice,
        patch_array,
        patch_list,
        name_match,
        middle_ref_x,
        middle_ref_y,
        spot_list_after,
        get_ori_files_fn,
        barcode2cluster,     # dict: full_barcode -> int
        cell_clusters,       # dict: class -> list[int]
        box_side=BOX_SIDE,
        skip_even_slices=False):

    """
    For one glomerulus (track):
      • Find all spots whose quadrilateral is fully inside the glomerulus contour on that slice
      • Build full_barcode = "{slice_name}-8-{barcode_short}"
      • Look up seurat_cluster.projected from barcode2cluster
      • If the cluster is in any target class's ID list, draw a 32×32 box at that spot's XY
    Returns: {slice_idx: [GeoJSON features]}
    """
    track_id = track['track_id']
    values_list = list(name_match.values())
    track_slice_features = defaultdict(list)

    for obj in track['contours']:
        now_slice = obj['slice_index']
        if skip_even_slices and (now_slice % 2 == 0):
            continue

        now_idx = obj['index']
        now_contour = all_slices[now_slice][now_idx]['polygon']

        # keep alignment steps intact (even if not used downstream)
        new_patch_array = np.zeros(patch_array.shape, dtype=np.float32)
        new_patch_array[:, :, 0] = (patch_array[:, :, 0] - middle_ref_y) / 4
        new_patch_array[:, :, 1] = (patch_array[:, :, 1] - middle_ref_x) / 4
        _ = get_nearest_patch_index(np.array(now_contour)[:, ::-1], new_patch_array)

        # spots for this slice
        now_df_after = spot_list_after[now_slice]['spot']
        if len(now_df_after) == 0:
            continue

        # Build Nx4x2 array of quad corners (y,x order preserved from your CSV)
        num_points = 4
        batch_array_i = np.zeros((len(now_df_after), num_points, 1, 2), dtype=np.float32)
        for ni in range(num_points):
            batch_array_i[:, ni, :, 0] = np.expand_dims(now_df_after[f'now_y_{ni}'].values, 1)
            batch_array_i[:, ni, :, 1] = np.expand_dims(now_df_after[f'now_x_{ni}'].values, 1)

        polygon_now = Polygon(now_contour)
        coords = batch_array_i[:, :, 0, :]
        contained = shapely.contains_xy(
            polygon_now,
            coords[:, :, 0].ravel(),
            coords[:, :, 1].ravel()
        ).reshape(len(now_df_after), num_points)
        inside_indices = np.nonzero(np.all(contained, axis=1))[0]
        if inside_indices.size == 0:
            continue

        # slice->name (e.g., S10)
        slice_name = values_list[now_slice] if 0 <= now_slice < len(values_list) else f"slice{now_slice}"

        for ii in inside_indices:
            barcode = now_df_after.iloc[ii]['barcode']        # as in the spot table, e.g. 's_008um_00341_00498-1'
            full_barcode = f"{slice_name}-8-{barcode}"        # used to find the bin in the positions table

            # Look up projected cluster (keys use the short form '<sample>-8-00341_00498-1')
            cl = barcode2cluster.get(f"{slice_name}-8-{short_hd_barcode(barcode)}", None)
            if cl is None:
                print(f"[WARN] No cluster for {full_barcode}")
                continue

            # Find which class (if any) this cluster belongs to
            matched_classes = [cls for cls, ids in cell_clusters.items() if cl in ids]
            if not matched_classes:
                continue  # not a target cluster

            # Get XY from positions parquet
            xy = lookup_xy_from_positions(full_barcode, get_ori_files_fn)
            if xy is None:
                continue
            cx, cy = xy

            # Emit one annotation per matched class
            for cls in matched_classes:
                poly_coords = box32_polygon_at_center(cx, cy, side=box_side)
                props = {
                    "slice_index": int(now_slice),
                    "track_id": int(track_id),
                    "barcode": str(barcode),
                    "cluster_projected": int(cl),
                }
                feat = make_qupath_feature(poly_coords, cls, props)
                track_slice_features[now_slice].append(feat)

    return track_slice_features

# ------------------------------ Data access -------------------------------- #

def get_ori_files(a):
    """Space Ranger tissue_positions of section ``a`` (``st_sections`` in the config)."""
    return read_positions(st_section(_CFG, a)['positions'])

# ------------------------------ Main script -------------------------------- #

if __name__ == '__main__':

    args, cfg = parse_args(__doc__, lambda p: add_tracks_argument(p, default='Object_IoU_patch'))
    _CFG = cfg
    _SLICE_TO_A.update({name: idx for idx, name in section_sample_names(cfg).items()})

    # ----------- I/O roots ----------------------------------------------------
    case_folder    = cfg['paths']['image_root']
    contour_folder = cfg['paths']['contour_root']

    # merged Visium HD meta table with columns 'Sample', 'Barcode', 'seurat_cluster.projected'
    clustering_csv_path = analysis_path(cfg, 'clustering_meta_csv')

    # Map class names to the Seurat projected cluster IDs (config key 'cell_clusters_per_track')
    cell_clusters = cfg.get('cell_clusters_per_track') or {
        "podocyte": [35,36,37],
        "mesangial cell": [7],
    }

    # ----------- Build barcode → projected cluster map from CSV --------------
    barcode2cluster = build_barcode_to_cluster_map(clustering_csv_path)

    # ----------- Iterate through cases ---------------------------------------
    cases = list_cases(cfg)

    for ci, case in enumerate(cases):
        now_case = os.path.basename(case)

        reconstruction_contour_dir = os.path.join(
            case.replace(case_folder, contour_folder), 'registration_3D_patch_ANTs')
        IoU_dir = os.path.join(case.replace(case_folder, contour_folder), args.tracks_subdir)
        after_annotation_dir = os.path.join(
            reconstruction_contour_dir, 'annotation_QA_generegions')

        images = glob.glob(os.path.join(case, 'image_*.png'))
        images.sort(key=natural_keys)

        jsons = section_object_geojsons(images, after_annotation_dir, now_case)  # one per section, in stack order

        csv_after = glob.glob(os.path.join(reconstruction_contour_dir, '*.csv'))
        csv_after.sort(key=natural_keys)

        if len(images) == 0 or len(jsons) == 0 or len(csv_after) == 0:
            print(f"[WARN] Missing inputs for case {now_case}, skipping.")
            continue

        middle_idx   = int(len(images) / 2)
        middle_ref_x = int(os.path.basename(images[middle_idx]).split('_')[2])
        middle_ref_y = int(os.path.basename(images[middle_idx]).split('_')[3].replace('.png', ''))

        patch_level_folder = os.path.join(case, '3D_Patch_crop')
        patch_list = glob.glob(os.path.join(patch_level_folder, '*'))
        patch_array = np.zeros((len(patch_list), 1, 2), dtype=np.float32)
        for pi in range(len(patch_list)):
            if '3D' in os.path.basename(patch_list[pi]):
                continue
            patch_name = os.path.basename(patch_list[pi])
            patch_abs_x, patch_abs_y = patch_name.split('_')
            patch_array[pi, :, 0] = int(patch_abs_x)
            patch_array[pi, :, 1] = int(patch_abs_y)

        mask_labels = single_object_class(cfg)  # the Omni-Seg class reconstructed in 3D (one class)
        # Contours per slice
        all_slices = {}
        for idx, (img_name, json_path) in enumerate(zip(images, jsons)):
            ann = load_annotations(img_name, json_path, mask_labels)
            if ann is None:
                continue
            all_slices[idx] = contours_to_polygons(ann)

        # slice image meta (not strictly needed for boxes, but preserved)
        images_slice = {}
        for xi, img in enumerate(images):
            real_x = int(os.path.basename(images[xi]).split('_')[2])
            real_y = int(os.path.basename(images[xi]).split('_')[3].replace('.png', ''))
            now_img = plt.imread(images[xi])[:, :, :3]
            images_slice[xi] = {'img': now_img, 'real_x': real_x, 'real_y': real_y}

        # Spots per slice
        spot_list_after = {}
        for si, csv_f in enumerate(csv_after):
            now_df = pd.read_csv(csv_f)
            spot_list_after[si] = {'spot': now_df}

        # Tracking results
        trakcing_file = tracks_json(IoU_dir, args.tracks_suffix)
        if not os.path.exists(trakcing_file):
            print(f"[WARN] Missing tracking file for case {now_case}: {trakcing_file}")
            continue
        with open(trakcing_file, 'r') as f:
            tracking_results = json.load(f)

        # section index -> sample name (from 'st_sections' in the config)
        name_match = stack_sample_names(cfg, images)  # sample of every section, in stack order
        values_list = list(name_match.values())

        # === Write GeoJSON per slide IMMEDIATELY for each track ===
        per_track_root = tracks_output_dir(case.replace(case_folder, contour_folder),
                                           'qupath_annotations_by_track_clusters', args.tracks_subdir,
                                           args.tracks_suffix)
        os.makedirs(per_track_root, exist_ok=True)

        total_boxes_case = 0
        for tr in tracking_results:
            track_id = tr['track_id']
            track_slice_features = collect_cell_boxes_for_track_by_cluster(
                tr,
                all_slices,
                images_slice,
                patch_array,
                patch_list,
                name_match,
                middle_ref_x,
                middle_ref_y,
                spot_list_after,
                get_ori_files_fn=get_ori_files,
                barcode2cluster=barcode2cluster,
                cell_clusters=cell_clusters,
                box_side=BOX_SIDE,
                skip_even_slices=False  # set True if you want to skip even slices
            )

            out_dir_track = os.path.join(per_track_root, f"track_{track_id}")
            os.makedirs(out_dir_track, exist_ok=True)

            count_this_track = 0
            for slice_idx, feats in track_slice_features.items():
                slice_name = values_list[slice_idx] if 0 <= slice_idx < len(values_list) else f"slice{slice_idx}"
                fname = f"{now_case}_{slice_name}_track{track_id}_cells.geojson"
                write_geojson_featurecollection(feats, os.path.join(out_dir_track, fname))
                count_this_track += len(feats)
            total_boxes_case += count_this_track
            print(f"[Track] {now_case} track {track_id}: {count_this_track} boxes")

        print(f"[DONE] Case {now_case} — total boxes: {total_boxes_case}")
