"""Synthetic exemplar dataset for testing the THiRST pipeline end to end (no patient data).

Creates one tissue block of 12 serial sections (Visium V2 and Visium HD alternating, V2 first) whose
sections are displaced from the middle section by known rigid transforms:

* ``WSI/DEMO-01-<k>.svs``          pyramidal TIFF (level 2 = 4x downsample) readable by OpenSlide
* ``WSI_json/``                    QuPath region annotation per section (class ``normal``)
* ``Omni-Seg/geojson/``            "segmentation": 6 glomerulus-like objects (class ``capsule``) + a distractor
                                   class; one object is present on sections 2-11 only
* ``SpaceRanger/<sample>/...``     tissue_positions.csv (V2, 100 um grid) / .parquet (HD, 8 um bins)
* ``demo.yaml``                    ready-to-use configuration (absolute paths)

Sub-commands
------------
    python demo/make_synthetic_data.py create --out demo_data [--first-section 1]
    python demo/make_synthetic_data.py simulate-registration --config demo_data/demo.yaml --level global
    python demo/make_synthetic_data.py simulate-registration --config demo_data/demo.yaml --level patch

``simulate-registration`` writes known transforms in exactly the files that XFeat + ANTs would produce
(``registration_xfeat/*/sg_affine_init.npy`` and
``registration_ANTs/*/step2_run_ants_reg/output0GenericAffine.mat``), so the rest of the pipeline can
be run without a GPU or an ANTs installation. The *global* transforms are deliberately off by a small
known rotation + shift per section (:func:`residual_transform`, a few pixels), as an imperfect global
registration would be; the *patch-level* transforms are the exact correction of that residual in every
patch frame. The demo therefore checks that the global stage leaves the expected residual and that the
patch-level stage removes it. ``bash demo/run_demo.sh --real`` uses XFeat and ANTs instead.

``--first-section`` numbers the sections from that value (default 1), e.g. 2..13.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CASE = "DEMO-01"
N = 12                           # sections per block; V2 at 0-based positions 0, 2, ..., HD at 1, 3, ...
REF_POS = N // 2                 # 0-based position of the middle (reference) section
REGION = 4000                    # tissue region, full-resolution pixels (1 mm at 0.25 um/px)
XMIN, YMIN = 6000, 5000          # region origin in the WSI (full resolution)
X0, Y0 = XMIN - 4096, YMIN - 4096  # origin of the padded 4x section images
IMG = REGION // 4 + 2048         # 4x section image size
CENTER = IMG / 2.0
OBJECTS = [(1200, 1200), (1500, 1300), (1800, 1250), (1300, 1700), (1650, 1650), (1900, 1850)]


def _rotation_about_center(theta, shift):
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    m = np.eye(3)
    m[:2, :2] = rot
    m[:2, 2] = CENTER - rot @ np.array([CENTER, CENTER]) + shift
    return m


def section_transform(pos):
    """3x3 affine mapping reference-section 4x image coordinates (x=col, y=row) to the section at 0-based
    position ``pos`` of the stack (identity for the reference)."""
    if pos == REF_POS:
        return np.eye(3)
    rng = np.random.default_rng(101 + pos)
    return _rotation_about_center(np.deg2rad(rng.uniform(-2, 2)), rng.uniform(-15, 15, 2))


def residual_transform(pos):
    """Known error of the simulated global registration of the section at position ``pos``, in reference
    4x coordinates (identity for the reference): globally registered, a tissue point x of that section
    lands at ``residual_transform(pos) @ x`` instead of x."""
    if pos == REF_POS:
        return np.eye(3)
    rng = np.random.default_rng(701 + pos)
    theta = np.deg2rad(rng.choice([-1, 1]) * rng.uniform(0.2, 0.5))
    # whole-pixel shifts: the pipeline stores coordinates as whole 4x pixels, and a residual whose
    # fractional part sits at 0.5 px would add a systematic rounding error of up to 0.5 px per stage
    shift = rng.choice([-1, 1], 2) * rng.integers(2, 6, 2)
    return _rotation_about_center(theta, shift.astype(float))


def _apply(m, pts):
    return np.asarray(pts, float) @ m[:2, :2].T + m[:2, 2]


def _to_full(pts4x):
    pts4x = np.asarray(pts4x, float)
    return np.stack([X0 + 4 * pts4x[:, 0], Y0 + 4 * pts4x[:, 1]], 1)


def _circle(cx, cy, r, n=48):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)


def _base_image():
    import cv2
    rng = np.random.default_rng(0)
    img = np.full((IMG, IMG, 3), (235, 225, 235), np.uint8)
    for _ in range(400):
        c = tuple(int(v) for v in rng.integers(1024, 1024 + REGION // 4, 2))
        cv2.circle(img, c, int(rng.integers(8, 40)), tuple(int(v) for v in rng.integers(60, 200, 3)), -1)
    for cx, cy in OBJECTS:                     # glomerulus-like structures
        cv2.circle(img, (cx, cy), 40, (140, 60, 150), -1)
        cv2.circle(img, (cx, cy), 20, (90, 30, 110), -1)
    return img


def _write_wsi(path, section_img):
    """Pyramidal TIFF with levels 1x/2x/4x; the 4x level holds the section image."""
    import tifffile
    ts = 512
    w2 = int(np.ceil((XMIN + REGION + 8192) / 4 / ts)) * ts
    h2 = int(np.ceil((YMIN + REGION + 8192) / 4 / ts)) * ts
    lvl2 = np.full((h2, w2, 3), 235, np.uint8)
    ox, oy = X0 // 4, Y0 // 4
    lvl2[oy:oy + IMG, ox:ox + IMG] = section_img

    def flat(w, h):
        for _ in range(0, h, ts):
            for _ in range(0, w, ts):
                yield np.full((ts, ts, 3), 235, np.uint8)

    def tiles(a):
        for y in range(0, a.shape[0], ts):
            for x in range(0, a.shape[1], ts):
                yield a[y:y + ts, x:x + ts]

    with tifffile.TiffWriter(path) as tw:
        for lvl, ds in enumerate((1, 2, 4)):
            w, h = w2 * 4 // ds, h2 * 4 // ds
            tw.write(tiles(lvl2) if ds == 4 else flat(w, h), shape=(h, w, 3), dtype=np.uint8,
                     tile=(ts, ts), photometric="rgb", compression="zlib", subfiletype=(1 if lvl else 0))


def create(out, first_section=1):
    import cv2
    import pandas as pd
    out = os.path.abspath(out)
    for sub in ("WSI", "WSI_json", "Omni-Seg/geojson"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    base = _base_image()
    n_grid = REGION // 4 // 100
    v2_grid = np.array([(1024 + 50 + 100 * i, 1024 + 50 + 100 * j) for j in range(n_grid) for i in range(n_grid)], float)
    n_hd = REGION // 4 // 8
    hd_grid = np.array([(1024 + 4 + 8 * i, 1024 + 4 + 8 * j) for j in range(n_hd) for i in range(n_hd)], float)
    region = [[XMIN, YMIN], [XMIN + REGION, YMIN], [XMIN + REGION, YMIN + REGION], [XMIN, YMIN + REGION], [XMIN, YMIN]]
    sections = {}
    for pos, k in enumerate(range(first_section, first_section + N)):
        m = section_transform(pos)
        section_img = cv2.warpAffine(base, m[:2], (IMG, IMG), borderValue=(235, 235, 235))
        _write_wsi(os.path.join(out, "WSI", f"{CASE}-{k}.svs"), section_img)
        json.dump({"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [region]},
             "properties": {"objectType": "annotation", "classification": {"name": "normal"}}}]},
            open(os.path.join(out, "WSI_json", f"{CASE}-{k}.svs.geojson"), "w"))
        feats = []
        for oi, (cx, cy) in enumerate(OBJECTS):
            if pos in (0, N - 1) and oi == len(OBJECTS) - 1:   # this object misses the first and last section
                continue
            for cls, r in (("capsule", 40), ("tuft", 20)):
                ring = _to_full(_apply(m, _circle(cx, cy, r)))
                feats.append({"type": "Feature", "properties": {"classification": {"name": cls}},
                              "geometry": {"type": "Polygon", "coordinates": [np.vstack([ring, ring[:1]]).tolist()]}})
        json.dump({"type": "FeatureCollection", "features": feats},
                  open(os.path.join(out, "Omni-Seg/geojson", f"{CASE}-{k}.svs.geojson"), "w"))
        sample = "S%02d" % k
        if pos % 2 == 0:   # Visium V2 spots
            p = _to_full(_apply(m, v2_grid))
            rng = np.random.default_rng(k)
            df = pd.DataFrame({"barcode": ["%s-1" % "".join(rng.choice(list("ACGT"), 16)) for _ in range(len(p))],
                               "in_tissue": [0 if i % 37 == 5 else 1 for i in range(len(p))],
                               "array_row": np.arange(len(p)) // n_grid, "array_col": np.arange(len(p)) % n_grid,
                               "pxl_row_in_fullres": p[:, 1], "pxl_col_in_fullres": p[:, 0]})
            pos = os.path.join(out, "SpaceRanger", sample, "outs", "spatial", "tissue_positions.csv")
            os.makedirs(os.path.dirname(pos), exist_ok=True)
            df.to_csv(pos, index=False)
            sections[k] = {"platform": "visium_v2", "sample": sample, "positions": pos}
        else:            # Visium HD 8 um bins
            p = _to_full(_apply(m, hd_grid))
            ij = [(i, j) for j in range(n_hd) for i in range(n_hd)]
            df = pd.DataFrame({"barcode": ["s_008um_%05d_%05d-1" % (j, i) for i, j in ij], "in_tissue": 1,
                               "array_row": [j for _, j in ij], "array_col": [i for i, _ in ij],
                               "pxl_row_in_fullres": p[:, 1], "pxl_col_in_fullres": p[:, 0]})
            pos = os.path.join(out, "SpaceRanger", sample, "outs", "binned_outputs", "square_008um", "spatial",
                               "tissue_positions.parquet")
            os.makedirs(os.path.dirname(pos), exist_ok=True)
            df.to_parquet(pos, index=False)
            sections[k] = {"platform": "visium_hd", "sample": sample, "positions": pos}
        print(f"section {k:2d}: {sections[k]['platform']}")

    import yaml
    cfg = {"paths": {"wsi_dir": f"{out}/WSI", "region_annotation_dir": f"{out}/WSI_json",
                     "object_annotation_dir": f"{out}/Omni-Seg/geojson", "image_root": f"{out}/WSI_png",
                     "point_root": f"{out}/WSI_png_point", "contour_root": f"{out}/WSI_png_contour"},
           "region_class": "normal", "object_classes": ["capsule"], "st_sections": sections}
    cfg.update(xfeat_settings())       # used by --real runs
    with open(os.path.join(out, "demo.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("wrote", os.path.join(out, "demo.yaml"))


def xfeat_settings(environ=None):
    """XFeat entries of demo.yaml: the release configuration (thirst/provenance.py). The environment
    variable THIRST_XFEAT_HUB may name a local clone of the XFeat repository instead of the GitHub ref."""
    from thirst.provenance import XFEAT_COMMIT, XFEAT_REPO, XFEAT_WEIGHTS_SHA256
    environ = os.environ if environ is None else environ
    hub = environ.get("THIRST_XFEAT_HUB") or f"{XFEAT_REPO}:{XFEAT_COMMIT}"
    return {"xfeat_hub": os.path.abspath(hub) if os.path.isdir(hub) else hub,
            "xfeat_weights_sha256": XFEAT_WEIGHTS_SHA256}


def _write_itk(path, m_fixed_to_moving):
    import SimpleITK as sitk
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = sitk.AffineTransform(2)
    t.SetMatrix(m_fixed_to_moving[:2, :2].ravel().tolist())
    t.SetTranslation(m_fixed_to_moving[:2, 2].tolist())
    sitk.WriteTransform(t, path)


def _translation(v):
    m = np.eye(3)
    m[:2, 2] = v
    return m


def patch_origin(stack):
    """(col, row) of the patch window's top-left corner in the reference 4x frame, computed from the
    patch folder name exactly as the stage 2 patch scripts do."""
    case = os.path.dirname(os.path.dirname(stack))
    ref = sorted(glob.glob(os.path.join(case, "image_*.png")),
                 key=lambda p: int(os.path.basename(p).split("_")[1]))[N // 2]
    x0, y0 = (int(v) for v in os.path.basename(ref)[:-4].split("_")[2:4])
    abs_x, abs_y = (int(v) for v in os.path.basename(stack).split("_"))       # full-res row, column
    return np.array([round((abs_y - x0) / 4) - 1024, round((abs_x - y0) / 4) - 1024], float)


def simulate_registration(config, level):
    """Write known pairwise transforms where XFeat (npy) and ANTs (mat) would put them.

    Global level: true section transforms combined with :func:`residual_transform`, i.e. a slightly
    wrong registration. Patch level: the residual expressed in each patch frame, i.e. the exact local
    correction (patch stacks are cut from the globally registered sections).
    """
    from thirst.config import list_stacks, load_config
    cfg = load_config(config)
    for stack in list_stacks(cfg, level):
        images = sorted(glob.glob(os.path.join(stack, "image_*.png")),
                        key=lambda p: int(os.path.basename(p).split(".")[0].split("_")[1]))
        idx = [int(os.path.basename(p).split(".")[0].split("_")[1]) for p in images]
        if level == "global":
            def to_section(pos):           # reference 4x frame -> section at pos, as "registered"
                return section_transform(pos) @ np.linalg.inv(residual_transform(pos))
        else:
            o = patch_origin(stack)

            def to_section(pos):           # patch frame of the reference -> patch frame of section pos
                return _translation(-o) @ residual_transform(pos) @ _translation(o)
        for pos in range(len(idx) - 1):
            fixed, moving = idx[pos], idx[pos + 1]
            # ANTs convention: the affine maps fixed-image coordinates to moving-image coordinates.
            m = to_section(pos + 1) @ np.linalg.inv(to_section(pos))
            pair = f"{moving}_to_{fixed}"
            _write_itk(os.path.join(stack, "registration_ANTs", pair, "step2_run_ants_reg", "output0GenericAffine.mat"), m)
            d = os.path.join(stack, "registration_xfeat", pair)
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, "sg_affine_init.npy"), np.linalg.inv(m)[:2])   # XFeat: moving -> fixed
        print(f"{level}: simulated transforms for {len(idx) - 1} pairs in {stack}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--out", required=True)
    c.add_argument("--first-section", type=int, default=1, help="number of the first section (default 1)")
    s = sub.add_parser("simulate-registration")
    s.add_argument("--config", required=True)
    s.add_argument("--level", choices=("global", "patch"), required=True)
    a = ap.parse_args()
    if a.cmd == "create":
        create(a.out, a.first_section)
    else:
        simulate_registration(a.config, a.level)
