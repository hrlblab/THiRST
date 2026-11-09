"""Configuration helpers shared by every script in ``scripts/``.

Each THiRST run is described by one YAML file (see ``configs/example.yaml``). The
file lists where the data of one serial-section stack live, the Space Ranger output
of every section, and a few dataset-specific names. Relative paths are resolved
against the current working directory, so run the scripts from the repository root.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: ITK affine transform used as a template when converting XFeat matrices to ANTs format.
ITK_AFFINE_TEMPLATE = os.path.join(REPO_ROOT, "thirst", "assets", "itk_affine_template.mat")

#: Patch-stack folders are named ``<abs_x>_<abs_y>`` (full-resolution coordinates).
_PATCH_DIR_RE = re.compile(r"^-?\d+_-?\d+$")

_REQUIRED_PATHS = ("image_root", "point_root", "contour_root")


def _abspath(p: Optional[str]) -> Optional[str]:
    if p is None or p == "":
        return p
    return os.path.abspath(os.path.expanduser(str(p)))


def load_config(path: str) -> Dict[str, Any]:
    """Read a YAML config and normalise every path it contains (absolute, no trailing '/')."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    paths = cfg.setdefault("paths", {})
    for key in list(paths):
        paths[key] = _abspath(paths[key])
    missing = [k for k in _REQUIRED_PATHS if not paths.get(k)]
    if missing:
        raise KeyError(f"{path}: missing required entries under 'paths': {missing}")

    sections = {}
    for idx, sec in (cfg.get("st_sections") or {}).items():
        sec = dict(sec or {})
        for key in ("positions", "in_tissue_positions"):
            if sec.get(key):
                sec[key] = _abspath(sec[key])
        sections[int(idx)] = sec
    cfg["st_sections"] = sections

    # Every string under 'analysis:' is a file path (external inputs of scripts/5_object_analysis).
    analysis = cfg.get("analysis") or {}
    for key, value in list(analysis.items()):
        if isinstance(value, str) and value:
            analysis[key] = _abspath(value)
    cfg["analysis"] = analysis

    cfg.setdefault("cases", [])
    cfg.setdefault("region_class", "normal")
    cfg.setdefault("object_classes", ["capsule"])
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


def parse_args(description: Optional[str] = None,
               add_arguments: Optional[Callable[[argparse.ArgumentParser], None]] = None):
    """Standard command line: ``--config <yaml>`` plus script-specific options."""
    parser = argparse.ArgumentParser(
        description=description.strip() if description else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True,
                        help="YAML configuration file (see configs/example.yaml)")
    if add_arguments is not None:
        add_arguments(parser)
    args = parser.parse_args()
    return args, load_config(args.config)


def add_level_argument(parser: argparse.ArgumentParser) -> None:
    """``--level global`` registers whole sections; ``--level patch`` registers the local patch stacks."""
    parser.add_argument("--level", choices=("global", "patch"), required=True,
                        help="'global': whole-section stacks in paths.image_root/<case>; "
                             "'patch': local stacks in paths.image_root/<case>/3D_Patch_crop/<x>_<y>")


def list_cases(cfg: Dict[str, Any]) -> List[str]:
    """Case (tissue block) folders inside ``paths.image_root``, optionally restricted by ``cases``."""
    root = cfg["paths"]["image_root"]
    wanted = {str(c) for c in (cfg.get("cases") or [])}
    cases = sorted(p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p))
    if wanted:
        cases = [p for p in cases if os.path.basename(p) in wanted]
    return cases


def list_patch_stacks(case_dir: str) -> List[str]:
    """Patch-stack folders (``<abs_x>_<abs_y>``) created by ``crop_patch_stacks.py`` for one case."""
    root = os.path.join(case_dir, "3D_Patch_crop")
    return sorted(p for p in glob.glob(os.path.join(root, "*"))
                  if os.path.isdir(p) and _PATCH_DIR_RE.match(os.path.basename(p)))


def list_stacks(cfg: Dict[str, Any], level: str) -> List[str]:
    """Folders holding an ordered ``image_*.png`` stack to register at the requested level."""
    if level == "global":
        return list_cases(cfg)
    if level == "patch":
        stacks: List[str] = []
        for case_dir in list_cases(cfg):
            stacks.extend(list_patch_stacks(case_dir))
        return stacks
    raise ValueError(f"unknown level: {level!r}")


# --------------------------------------------------------------------------------------
# Spatial transcriptomics helpers
# --------------------------------------------------------------------------------------

def st_section(cfg: Dict[str, Any], section: Any) -> Dict[str, Any]:
    """Entry of ``st_sections`` for a section index (int or str)."""
    try:
        return cfg["st_sections"][int(section)]
    except KeyError as exc:
        raise KeyError(f"section {section!r} is not listed under 'st_sections' in "
                       f"{cfg.get('_config_path')}") from exc


def read_positions(path: str) -> pd.DataFrame:
    """Read a Space Ranger ``tissue_positions`` table (``.csv`` for Visium V2, ``.parquet`` for Visium HD)."""
    if path.endswith(".parquet"):
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except (ImportError, ValueError):
            return pd.read_parquet(path)
    return pd.read_csv(path)


def read_in_tissue_positions(section_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Rows of the section's ``positions`` table whose barcode has ``in_tissue == 1``.

    The flag is read from ``in_tissue_positions`` when given (default: ``positions`` itself) and is
    matched by barcode, so the two tables do not need to list the barcodes in the same order.
    """
    positions_path = section_cfg["positions"]
    flags_path = section_cfg.get("in_tissue_positions") or positions_path
    position = read_positions(positions_path)
    flags = read_positions(flags_path)
    if flags["barcode"].duplicated().any():
        raise ValueError(f"{flags_path}: duplicated barcodes")
    in_tissue = position["barcode"].map(flags.set_index("barcode")["in_tissue"])
    if in_tissue.isna().any():
        raise ValueError(f"{int(in_tissue.isna().sum())} barcodes of {positions_path} are not listed in "
                         f"{flags_path}; both tables must describe the same capture area")
    return position[in_tissue.to_numpy() == 1]


def is_circular_spot(cfg: Dict[str, Any], section: Any) -> int:
    """1 for Visium V2 (55 um circular spots), 0 for Visium HD (square bins)."""
    platform = str(st_section(cfg, section).get("platform", "")).lower()
    if platform in ("visium_v2", "v2", "visium"):
        return 1
    if platform in ("visium_hd", "hd"):
        return 0
    raise ValueError(f"section {section}: platform must be 'visium_v2' or 'visium_hd', got {platform!r}")


def section_index(image_path: str) -> str:
    """Section number of a section image ``image_<section>_<x0>_<y0>.png``."""
    return os.path.basename(image_path).split("_")[1]


def require_platform(cfg: Dict[str, Any], image_path: str, circle: int) -> None:
    """Raise if a section's ST unit polygons (``circle``: 1 = V2 spots, 0 = HD bins) contradict ``st_sections``."""
    sec = section_index(image_path)
    if is_circular_spot(cfg, sec) != int(circle):
        found = "100-vertex Visium V2 spots" if circle else "4-corner Visium HD bins"
        raise ValueError(f"section {sec}: the ST unit polygons are {found}, but st_sections gives platform "
                         f"{st_section(cfg, sec).get('platform')!r} ({cfg.get('_config_path')})")


def require_alternating_platforms(cfg: Dict[str, Any], images: List[str]) -> None:
    """Check the section layout assumed by the V2/HD association and object-analysis scripts.

    Those scripts treat the sections at even 0-based positions of the stack (``images``, sorted) as
    Visium V2 and the sections at odd positions as Visium HD, as in the paper. Raise ``ValueError``
    when the ``platform`` entries of ``st_sections`` do not follow that pattern, instead of silently
    pairing the wrong sections.
    """
    wrong = []
    for pos, img in enumerate(images):
        sec = section_index(img)
        expected = 1 if pos % 2 == 0 else 0
        if is_circular_spot(cfg, sec) != expected:
            wrong.append(f"section {sec} at position {pos} (expected "
                         f"{'visium_v2' if expected else 'visium_hd'})")
    if wrong:
        raise ValueError(
            "this script assumes Visium V2 and Visium HD sections alternate, starting with V2 "
            "(V2 at positions 0, 2, 4, ... of the sorted stack). Not satisfied for: " + "; ".join(wrong)
            + f". Check 'st_sections' in {cfg.get('_config_path')}.")


def section_object_geojsons(images: List[str], folder: str, case_name: str) -> List[str]:
    """Object GeoJSON (``<case>-<section>.svs.geojson`` in ``folder``) of every section, in ``images`` order.

    Positions in the returned list are the slice indices used by the tracker and by the
    before/after CSVs, so a missing file must not shift the sections after it: a
    ``FileNotFoundError`` naming the missing sections is raised instead. A section without objects
    still needs a GeoJSON (an empty FeatureCollection).
    """
    paths, missing = [], []
    for img in images:
        sec = section_index(img)
        path = os.path.join(folder, f"{case_name}-{sec}.svs.geojson")
        paths.append(path)
        if not os.path.isfile(path):
            missing.append(sec)
    if missing:
        raise FileNotFoundError(
            f"{folder}: no object GeoJSON for section(s) {missing} of case {case_name}. Every section "
            "needs one (an empty FeatureCollection if it has no objects); check "
            "paths.object_annotation_dir and the output of global_objects.py / patch_objects.py.")
    return paths


def single_object_class(cfg: Dict[str, Any]) -> List[str]:
    """``object_classes`` for the scripts that support exactly one object class (stages 5 and 6).

    They look a tracked contour up by its index within its class, which identifies the contour only
    when a single class is reconstructed (the paper reconstructs glomeruli through ``capsule``).
    """
    classes = list(cfg["object_classes"])
    if len(classes) != 1:
        raise ValueError(f"this script supports exactly one entry in 'object_classes', got {classes} "
                         f"({cfg.get('_config_path')})")
    return classes


def stack_sample_names(cfg: Dict[str, Any], images: List[str]) -> Dict[str, str]:
    """``{'<section>': '<sample>'}`` for the sections of a stack, in stack order (position = slice index)."""
    names = section_sample_names(cfg)
    missing = [section_index(i) for i in images if section_index(i) not in names]
    if missing:
        raise KeyError(f"st_sections gives no 'sample' for section(s) {missing} ({cfg.get('_config_path')})")
    return {section_index(i): names[section_index(i)] for i in images}


def section_sample_names(cfg: Dict[str, Any]) -> Dict[str, str]:
    """``{'<section index>': '<sample name>'}`` from the optional ``sample`` field of ``st_sections``."""
    return {str(i): str(sec["sample"]) for i, sec in sorted(cfg["st_sections"].items()) if sec.get("sample")}


def analysis_path(cfg: Dict[str, Any], key: str) -> str:
    """Path of an external analysis input listed under ``analysis:`` in the config."""
    value = (cfg.get("analysis") or {}).get(key)
    if not value:
        raise KeyError(f"'analysis.{key}' is required by this script; add it to {cfg.get('_config_path')}")
    return value


# --------------------------------------------------------------------------------------
# Registration stages
# --------------------------------------------------------------------------------------

#: Folder (inside ``<point_root|contour_root>/<case>``) with the coordinates produced by each stage.
REGISTRATION_SUBDIRS = {"none": None, "global": "registration_3D_ANTs", "patch": "registration_3D_patch_ANTs"}


def add_registration_argument(parser: argparse.ArgumentParser, default: str = "patch") -> None:
    parser.add_argument("--registration", choices=tuple(REGISTRATION_SUBDIRS), default=default,
                        help="coordinates to use: 'patch' = global + patch-level refinement (default), "
                             "'global' = global affine chain only, 'none' = original section coordinates")


def registered_dir(case_dir: str, registration: str) -> str:
    """``<case_dir>/<stage sub-folder>`` for the requested registration stage."""
    sub = REGISTRATION_SUBDIRS[registration]
    return os.path.join(case_dir, sub) if sub else case_dir


#: Track folders written by the two trackers (patch-level registration).
HUNGARIAN_TRACKS = "Object_IoU_patch_connectglom"   # track_objects.py
GREEDY_TRACKS = "Object_IoU_patch"                  # track_objects_greedy.py


def add_tracks_argument(parser: argparse.ArgumentParser, default: str) -> None:
    """``--tracks-subdir`` / ``--tracks-suffix``: which 3D object tracking result to read."""
    parser.add_argument("--tracks-subdir", default=default,
                        help="folder under paths.contour_root/<case> containing IoU_Object.json: "
                             f"'{HUNGARIAN_TRACKS}' (track_objects.py, Hungarian) or "
                             f"'{GREEDY_TRACKS}' (track_objects_greedy.py). The output folder names say "
                             "which one was read ('_connectglom' = Hungarian). Default: %(default)s")
    add_tracks_suffix_argument(parser)


def tracks_tag(tracks_subdir: str) -> str:
    """Part of an output folder name that says which tracks were read, as in the original scripts:
    ``_connectglom`` for the Hungarian tracks, nothing for the greedy tracks of the patch-level
    registration, ``_<folder>`` for any other track folder."""
    name = os.path.basename(os.path.normpath(tracks_subdir))
    if name == HUNGARIAN_TRACKS:
        return "_connectglom"
    if name == GREEDY_TRACKS:
        return ""
    return "_" + name


def tracks_output_dir(case_contour_dir: str, name: str, tracks_subdir: str, tracks_suffix: str = "",
                      tail: str = "") -> str:
    """``<case_contour_dir>/<name><tracks_tag><tail><tracks_suffix>``: output folder of a script that reads
    tracks, so that results computed from different track files never share (or misname) a folder."""
    return os.path.join(case_contour_dir, name + tracks_tag(tracks_subdir) + tail + tracks_suffix)


def add_tracks_suffix_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tracks-suffix", default="",
                        help="read IoU_Object<suffix>.json, e.g. '_longglom' for the tracks written by "
                             "track_objects.py --min-track-len N (N > 1). The outputs of this script go to "
                             "folders / files carrying the same suffix, so results for all tracks and for "
                             "filtered tracks do not overwrite each other. Default: no suffix")


def tracks_json(iou_dir: str, suffix: str = "") -> str:
    """Path of ``IoU_Object<suffix>.json`` in ``iou_dir``; a clear error if it has not been written."""
    path = os.path.join(iou_dir, f"IoU_Object{suffix}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Run scripts/4_object_tracking/track_objects.py first. Tracks filtered with "
            "--min-track-len N (N > 1) are written as IoU_Object_longglom.json: read them with "
            "--tracks-suffix _longglom.")
    with open(path) as f:
        head = f.read(64).strip()
    if head.replace(" ", "").replace("\n", "") == "[]":
        print(f"[INFO] {path} contains no tracks (e.g. none reached --min-track-len): outputs will be empty")
    return path
