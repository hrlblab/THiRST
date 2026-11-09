"""QuPath annotations of all Visium HD bins of selected Seurat clusters, one GeoJSON per section.

THiRST stage 5 - visualisation.

Draws a 32 px box for every bin whose projected cluster belongs to ``cell_clusters_all``, using the bin
position from Space Ranger.

External inputs: ``analysis.clustering_meta_csv``, Space Ranger positions, ``st_sections[*].sample``.
Outputs: ``<clustering CSV dir>/GeoJSON_allcells/<sample>_allcells.geojson`` (or ``--out-dir``)

Usage
-----
    python scripts/5_object_analysis/qupath_all_cells_by_cluster.py --config configs/example.yaml

Original script(s): Step10_Match_Object_gene_patchregistration_using_patch_gene_colorbar_withYuclustering_allcell_bbox_QuPath_0911.py
"""
import sys
import os
import re
import json
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from thirst.config import analysis_path, parse_args, read_positions, section_sample_names, st_section  # noqa: E402
from thirst.st import short_hd_barcode  # noqa: E402


BOX_SIDE = 32  # pixel side length for annotation boxes

# Default target clusters: class name -> projected cluster IDs (override with 'cell_clusters_all' in the config)
DEFAULT_CELL_CLUSTERS = {
    "podocyte": [2],
    "mesangial cell": [7],
}

# --------------------------- I/O helpers -------------------------------- #

def _read_parquet_any(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception as e_py:
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except Exception as e_fp:
            raise ImportError(
                "Neither 'pyarrow' nor 'fastparquet' is available for reading parquet.\n"
                "Install one of them, e.g.: `conda install -c conda-forge pyarrow` or `pip install pyarrow`.\n"
                f"pyarrow error: {e_py}\nfastparquet error: {e_fp}"
            )

def get_ori_files(cfg, a: str) -> pd.DataFrame:
    """Space Ranger tissue_positions of section ``a`` (``st_sections`` in the config)."""
    return read_positions(st_section(cfg, a)['positions'])

# --------------------------- CSV / barcode utils ------------------------- #

def detect_cluster_column(df: pd.DataFrame) -> str:
    candidates = [
        'seurat_cluster.projected',
        'seurat_cluster_projected',
        'seurat_clusterProjected',
        'cluster_projected',
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Cluster column not found. Tried: {candidates}")


extract_short_barcode = short_hd_barcode  # 's_008um_00341_00498-1' / 'S07-8-00341_00498-1' -> '00341_00498-1'

def cluster_to_classes_map(cell_clusters: dict) -> dict:
    """Invert mapping to: cluster_id -> [class names]."""
    c2c = defaultdict(list)
    for cls, ids in cell_clusters.items():
        for cid in ids:
            c2c[int(cid)].append(cls)
    return dict(c2c)

# ---------------------------- GeoJSON builders --------------------------- #

def box32_polygon_at_center(cx: float, cy: float, side: int = BOX_SIDE):
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
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": poly_coords},
        "properties": {
            "objectType": "annotation",
            "isLocked": False,
            "classification": {"name": classification_name},
            "measurements": [
                {"name": "cluster_projected", "value": float(props.get("cluster_projected", -1))},
            ],
            "barcode": str(props.get("barcode", "")),
            "sample": str(props.get("sample", "")),
        }
    }

def write_geojson(features: list, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fc = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w") as f:
        json.dump(fc, f, indent=2)
    print(f"[GeoJSON] {len(features):6d} features -> {out_path}")

# ------------------------------- Main ------------------------------------ #

def main(cfg, out_dir=None):
    cell_clusters = cfg.get('cell_clusters_all') or DEFAULT_CELL_CLUSTERS
    clustering_csv_path = analysis_path(cfg, 'clustering_meta_csv')
    # sample name -> section index (from the 'sample' fields of 'st_sections')
    _SLICE_TO_A = {name: idx for idx, name in section_sample_names(cfg).items()}

    # Guards
    if not any(len(v) for v in cell_clusters.values()):
        raise ValueError("cell_clusters is empty; add cluster IDs to draw.")

    # Read clustering CSV
    df = pd.read_csv(clustering_csv_path)
    required_cols = {'Sample', 'Barcode'}
    if not required_cols.issubset(set(df.columns)):
        raise KeyError(f"CSV must contain columns {required_cols}, got {list(df.columns)}")

    cluster_col = detect_cluster_column(df)
    df['_cluster'] = df[cluster_col].astype(int)
    df['_barcode_short'] = df['Barcode'].astype(str).map(extract_short_barcode)
    df['_sample'] = df['Sample'].astype(str)

    # Keep only rows whose sample we can map to a positions parquet
    df = df[df['_sample'].isin(_SLICE_TO_A.keys())].copy()
    if df.empty:
        raise RuntimeError("After filtering by _SLICE_TO_A, no rows remain. Check sample names.")

    # Invert cluster map: cluster_id -> [class names]
    c2classes = cluster_to_classes_map(cell_clusters)

    # Output directory (default: next to the clustering CSV)
    out_root = out_dir or os.path.join(os.path.dirname(clustering_csv_path), 'GeoJSON_allcells')
    os.makedirs(out_root, exist_ok=True)

    # Process per WSI (per Sample)
    for sample in sorted(df['_sample'].unique()):
        print(f"\n[Sample] {sample}")

        # Load positions for this sample
        a = _SLICE_TO_A.get(sample)
        if a is None:
            print(f"  [SKIP] No positions mapping for sample {sample}")
            continue
        pos_df = get_ori_files(cfg, a)

        # Normalize positions columns
        cols = {c.lower(): c for c in pos_df.columns}
        if {'barcode', 'x', 'y'}.issubset(cols.keys()):
            x_col, y_col, bc_col = cols['x'], cols['y'], cols['barcode']
        else:
            bc_col = cols.get('barcode')
            x_col = cols.get('pxl_col_in_fullres') or cols.get('col') or cols.get('pixel_x') or cols.get('x')
            y_col = cols.get('pxl_row_in_fullres') or cols.get('row') or cols.get('pixel_y') or cols.get('y')
            missing = [n for n, v in [('barcode', bc_col), ('x', x_col), ('y', y_col)] if v is None]
            if missing:
                print(f"  [SKIP] Positions parquet missing {missing} for {sample}")
                continue

        # Normalize parquet barcodes to short form (handles 's_008um_XXXXX_YYYYY-1')
        pos_df['_barcode_short'] = pos_df[bc_col].astype(str).map(extract_short_barcode)

        # Build barcode_short -> (x,y) map
        xy_map = dict(zip(
            pos_df['_barcode_short'],
            zip(pos_df[x_col].astype(float), pos_df[y_col].astype(float))
        ))
        print(f"  [Positions] {sample}: {len(xy_map)} XY entries (from {len(pos_df)} rows)")

        # Filter clustering rows for this sample + target clusters
        df_s = df[(df['_sample'] == sample) & (df['_cluster'].isin(c2classes.keys()))].copy()
        if df_s.empty:
            print(f"  [INFO] No target-cluster rows for {sample}")
            # Emit an empty GeoJSON to keep outputs predictable
            out_path = os.path.join(out_root, f"{sample}_allcells.geojson")
            write_geojson([], out_path)
            continue

        features = []
        miss_xy = 0

        # Emit a 32×32 box per matching class
        for _, row in df_s.iterrows():
            bc_short = row['_barcode_short']
            cl = int(row['_cluster'])
            classes = c2classes.get(cl, [])
            xy = xy_map.get(bc_short)
            if xy is None:
                miss_xy += 1
                continue
            cx, cy = xy
            poly = box32_polygon_at_center(cx, cy, side=BOX_SIDE)
            for cls in classes:
                props = {
                    "cluster_projected": cl,
                    "barcode": bc_short,
                    "sample": sample,
                }
                features.append(make_qupath_feature(poly, cls, props))

        # Write per-sample GeoJSON
        out_path = os.path.join(out_root, f"{sample}_allcells.geojson")
        write_geojson(features, out_path)

        if miss_xy:
            print(f"  [WARN] {miss_xy} target barcodes missing XY in positions parquet for {sample}")

    print("\n[Done] Wrote per-WSI GeoJSON files based on projected clusters.")

if __name__ == "__main__":
    args, cfg = parse_args(__doc__, lambda p: p.add_argument(
        "--out-dir", default=None, help="output folder (default: <clustering CSV dir>/GeoJSON_allcells)"))
    main(cfg, args.out_dir)
