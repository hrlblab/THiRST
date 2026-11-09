"""Object contours (segmented structures such as glomeruli) of one section, as GeoJSON.

The segmentation inputs (Omni-Seg after QuPath review) and the files that stage 2 writes are standard
GeoJSON FeatureCollections: one ``Polygon`` feature per object, the class in
``properties.classification.name``; only the exterior ring is used (holes are ignored), and a
``MultiPolygon`` counts as one object per polygon.

Earlier versions of the stage 2 scripts wrote one feature per class whose ``coordinates[0]`` held the
list of all contours (not valid GeoJSON); :func:`read_object_contours` still reads such files.
"""
from __future__ import annotations

import json
from typing import Dict, Iterable, List

import numpy as np


def _norm(name) -> str:
    return str(name).strip().lower()


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _rings(geometry: dict):
    """(exterior rings as float arrays, earlier-layout flag) of a Polygon / MultiPolygon geometry."""
    gtype = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if gtype == "Polygon":
        if not coords or not coords[0]:
            return [], False
        first = coords[0]
        probe = next((p for p in first if p), None)            # first non-empty entry
        if probe is None:
            return [], False
        if not _is_number(probe[0]):                           # [[ring, ring, ...]]: earlier THiRST layout
            return [np.asarray(r, dtype=float) for r in first if len(r)], True
        return [np.asarray(first, dtype=float)], False
    if gtype == "MultiPolygon":
        return [np.asarray(p[0], dtype=float) for p in coords if p and p[0]], False
    return [], False


def read_object_contours(path: str, classes: Iterable[str]) -> Dict[str, List[np.ndarray]]:
    """``{class: [contour (N x 2 float array, x = column, y = row), ...]}`` in file order.

    Class names are compared case-insensitively. In files of the earlier layout every feature of a
    class repeated the complete contour list, so only the first such feature per class is used.
    """
    classes = list(classes)
    by_norm = {_norm(c): c for c in classes}
    out: Dict[str, List[np.ndarray]] = {c: [] for c in classes}
    earlier_seen = set()
    with open(path, "r") as f:
        collection = json.load(f)
    for feature in collection.get("features", []):
        name = ((feature.get("properties") or {}).get("classification") or {}).get("name")
        cls = by_norm.get(_norm(name))
        if cls is None:
            continue
        rings, earlier = _rings(feature.get("geometry") or {})
        if earlier:
            if cls in earlier_seen:
                continue
            earlier_seen.add(cls)
        out[cls].extend(rings)
    return out


def object_feature(cls: str, ring, contour_id=None) -> dict:
    """Standard GeoJSON feature (QuPath-compatible) for one object contour."""
    props = {"objectType": "annotation", "classification": {"name": cls}}
    if contour_id is not None:
        props["contour_id"] = int(contour_id)
    ring = np.asarray(ring, dtype=float).tolist()
    return {"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": [ring]}}
