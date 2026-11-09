"""Small builders shared by the tests: tiny section stacks, ST unit tables and configs."""
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

CASE = "CASE-01"


def load_script(relpath):
    """Import ``scripts/<relpath>`` as a module; its ``if __name__ == '__main__'`` block does not run."""
    path = os.path.join(REPO, "scripts", relpath)
    name = "thirst_test_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(relpath, *args):
    """Run ``scripts/<relpath>`` with the current interpreter; returns the CompletedProcess."""
    env = dict(os.environ, MPLBACKEND="Agg")
    return subprocess.run([sys.executable, os.path.join(REPO, "scripts", relpath), *map(str, args)],
                          capture_output=True, text=True, env=env, timeout=600)


def write_config(tmp_path, platforms, extra=None, **paths):
    """Config with the three working roots under ``tmp_path``; ``platforms`` = {section: 'visium_v2'|'visium_hd'};
    ``extra``: further top-level keys."""
    cfg = {
        "paths": {"image_root": str(tmp_path / "png"), "point_root": str(tmp_path / "point"),
                  "contour_root": str(tmp_path / "contour"), **{k: str(v) for k, v in paths.items()}},
        "object_classes": ["capsule"],
        "st_sections": {int(s): {"platform": p} for s, p in platforms.items()},
        **(extra or {}),
    }
    path = tmp_path / "test.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def make_images(tmp_path, sections, x0=0, y0=0):
    """Tiny ``image_<section>_<x0>_<y0>.png`` files (the stage 2-4 scripts mostly read only their names)."""
    import cv2
    case_dir = tmp_path / "png" / CASE
    case_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for s in sections:
        p = case_dir / f"image_{s}_{x0}_{y0}.png"
        cv2.imwrite(str(p), np.full((8, 8, 3), 255, np.uint8))
        names.append(str(p))
    return names


def circle(cx, cy, r=10.0, n=100):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


def square(cx, cy, half=2.0):
    return np.array([[cx - half, cy - half], [cx + half, cy - half], [cx + half, cy + half], [cx - half, cy + half]])


def write_unit_tables(tmp_path, section, units, x0=0, y0=0):
    """Write ``*_center.csv`` and ``*_spot.csv`` of one section into point_root / contour_root.

    ``units``: list of (barcode, (now_x, now_y), polygon Nx2 in (now_x, now_y) order). A unit with
    now_x == 0 is one that received no coordinates.
    """
    stem = f"image_{section}_{x0}_{y0}"
    centers = pd.DataFrame({"barcode": [u[0] for u in units],
                            "abs_x": [int(u[1][0]) * 4 for u in units], "abs_y": [int(u[1][1]) * 4 for u in units],
                            "now_x": [int(u[1][0]) for u in units], "now_y": [int(u[1][1]) for u in units]})
    n = len(units[0][2])
    cols = {"barcode": [u[0] for u in units]}
    for k in range(n):
        cols[f"now_x_{k}"] = [float(u[2][k, 0]) for u in units]
        cols[f"now_y_{k}"] = [float(u[2][k, 1]) for u in units]
    polygons = pd.DataFrame(cols)
    for root, table, suffix in (("point", centers, "_center.csv"), ("contour", polygons, "_spot.csv")):
        d = tmp_path / root / CASE
        d.mkdir(parents=True, exist_ok=True)
        table.to_csv(d / (stem + suffix), index=False)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_identity_transform(path):
    import SimpleITK as sitk
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sitk.WriteTransform(sitk.AffineTransform(2), str(path))


def write_geojson(path, contours, cls="capsule", layout="standard"):
    """Object GeoJSON of one section.

    ``layout="standard"``: one ``Polygon`` feature per object (QuPath export, and what the stage 2
    scripts write now). ``layout="legacy"``: the earlier stage 2 layout, one feature per class with
    ``coordinates = [list of contours]``, repeated once per contour as the old patch-level writer did.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rings = [np.asarray(c).tolist() for c in contours]
    props = {"classification": {"name": cls}}
    if layout == "standard":
        features = [{"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": [r]}}
                    for r in rings]
    else:
        features = [{"type": "Feature", "properties": props,
                     "geometry": {"type": "Polygon", "coordinates": [rings]}}] * max(len(rings), 1) if rings else []
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
