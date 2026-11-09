"""Cross-sectional area of every 3D object on every section.

THiRST stage 4 - object morphometry.

Outputs (next to IoU_Object.json): cross_section_areas_per_contour.csv, cross_section_areas_per_slice.csv
(areas in px^2 of the 4x images; add ``--pixel-spacing`` for physical units).

Usage
-----
    python scripts/4_object_tracking/cross_section_area.py --config configs/example.yaml

Original script(s): Area_cal.py
"""
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from thirst.config import add_tracks_suffix_argument, list_cases, parse_args, tracks_json  # noqa: E402


def polygon_area(coords: List[List[float]]) -> float:
    """
    Compute area of a simple polygon from its vertices using the shoelace formula.
    `coords` is a list of [x, y] points. It does not need to repeat the first point.
    Returns a non-negative area in pixel^2.
    """
    arr = np.asarray(coords, dtype=float)
    if arr.shape[0] < 3:
        return 0.0
    x = arr[:, 0]
    y = arr[:, 1]
    x2 = np.roll(x, -1)  # successor x (wraps around)
    y2 = np.roll(y, -1)  # successor y (wraps around)
    return float(0.5 * np.abs(np.sum(x * y2 - x2 * y)))


def compute_areas(data: List[Dict[str, Any]]) -> (pd.DataFrame, pd.DataFrame):
    """
    Given the parsed JSON list of objects, compute:
      - per-contour areas
      - per-slice aggregated areas (sum of contours for each object × slice)
    Expects each object like:
      {
        "track_id": ...,
        "contours": [
            {"slice_index": int, "object": str, "index": int, "coordinates": [[x, y], ...]},
            ...
        ]
      }
    """
    rows = []
    for obj in data:
        track_id = obj.get("track_id")
        for c in obj.get("contours", []):
            coords = c.get("coordinates", []) or []
            rows.append(
                {
                    "track_id": track_id,
                    "object": c.get("object"),
                    "slice_index": c.get("slice_index"),
                    "contour_index": c.get("index"),
                    "num_points": len(coords),
                    "area_px2": polygon_area(coords),
                }
            )

    # explicit columns, so that a track file without tracks (e.g. none reached --min-track-len) gives
    # empty tables with headers instead of an error
    columns = ["track_id", "object", "slice_index", "contour_index", "num_points", "area_px2"]
    per_contour = (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["track_id", "object", "slice_index", "contour_index"])
        .reset_index(drop=True)
    )

    per_slice = (
        per_contour.groupby(["track_id", "object", "slice_index"], as_index=False)
        .agg(
            total_area_px2=("area_px2", "sum"),
            contours=("contour_index", "count"),
            total_points=("num_points", "sum"),
        )
        .sort_values(["track_id", "object", "slice_index"])
        .reset_index(drop=True)
    )

    return per_contour, per_slice


def main(
    input_json: str,
    per_contour_csv: str,
    per_slice_csv: str,
    # Set pixel_spacing to convert px^2 → (unit)^2. Example: 0.5 mm/px → area scale 0.25 mm^2/px^2
    pixel_spacing: float = None,  # e.g., 0.5 for 0.5 mm/px
    unit: str = "mm",
):
    input_path = Path(input_json)
    with input_path.open("r") as f:
        data = json.load(f)

    per_contour, per_slice = compute_areas(data)
    if per_contour.empty:
        print(f"[INFO] {input_json} contains no tracks: writing empty tables")

    # Optional: convert to physical units if pixel spacing provided
    if pixel_spacing is not None:
        scale = (float(pixel_spacing) ** 2)
        per_contour[f"area_{unit}2"] = per_contour["area_px2"] * scale
        per_slice[f"total_area_{unit}2"] = per_slice["total_area_px2"] * scale

    # Save
    Path(per_contour_csv).parent.mkdir(parents=True, exist_ok=True)
    per_contour.to_csv(per_contour_csv, index=False)
    per_slice.to_csv(per_slice_csv, index=False)

    print(f"Saved per-contour areas → {per_contour_csv}")
    print(f"Saved per-slice areas   → {per_slice_csv}")


def _add_arguments(parser):
    parser.add_argument("--tracks-subdir", default="Object_IoU_patch",
                        help="folder under paths.contour_root/<case> holding IoU_Object.json "
                             "(Object_IoU_patch = track_objects_greedy.py, "
                             "Object_IoU_patch_connectglom = track_objects.py)")
    add_tracks_suffix_argument(parser)
    parser.add_argument("--pixel-spacing", type=float, default=None,
                        help="physical size of one pixel of the tracked contours (optional)")
    parser.add_argument("--unit", default="mm", help="unit of --pixel-spacing")


if __name__ == "__main__":
    args, cfg = parse_args(__doc__, _add_arguments)
    for case in list_cases(cfg):
        iou_dir = os.path.join(cfg['paths']['contour_root'], os.path.basename(case), args.tracks_subdir)
        main(input_json=tracks_json(iou_dir, args.tracks_suffix),
             per_contour_csv=os.path.join(iou_dir, "cross_section_areas_per_contour%s.csv" % args.tracks_suffix),
             per_slice_csv=os.path.join(iou_dir, "cross_section_areas_per_slice%s.csv" % args.tracks_suffix),
             pixel_spacing=args.pixel_spacing, unit=args.unit)
