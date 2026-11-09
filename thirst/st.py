"""Helpers for the ST unit tables written by stages 0-2 and read by stage 3.

Every section has two tables with one row per Visium V2 spot / Visium HD bin, in the same order:

* ``image_<section>_<x0>_<y0>_center.csv``: ``barcode, abs_x, abs_y, now_x, now_y`` (unit centres);
* ``image_<section>_<x0>_<y0>_spot.csv``: ``barcode, now_x_0, now_y_0, ..., now_x_<n-1>, now_y_<n-1>``
  (unit polygons: 100 vertices for a V2 spot, 4 corners for an HD bin).
"""
from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence

import pandas as pd

#: Vertices per polygon written by ``export_st_contours.py``.
V2_SPOT_VERTICES = 100
HD_BIN_VERTICES = 4


def contour_vertex_count(table: pd.DataFrame) -> int:
    """Number of polygon vertices stored per row (columns ``now_x_0`` ... ``now_x_<n-1>``)."""
    n = 0
    while f"now_x_{n}" in table.columns:
        n += 1
    return n


def is_circle_table(table: pd.DataFrame, what: str = "polygon table") -> int:
    """1 for a table of Visium V2 spots (100-vertex circles), 0 for Visium HD bins (4-vertex squares).

    Decided from the vertex columns of the table itself, not from its number of rows.
    """
    n = contour_vertex_count(table)
    if n == V2_SPOT_VERTICES:
        return 1
    if n == HD_BIN_VERTICES:
        return 0
    raise ValueError(f"{what}: {n} vertices per ST unit; expected {V2_SPOT_VERTICES} (Visium V2 spot) "
                     f"or {HD_BIN_VERTICES} (Visium HD bin)")


def require_same_barcodes(centers: pd.DataFrame, polygons: pd.DataFrame, what: str) -> None:
    """Raise unless the centre and polygon tables of a section list the same barcodes in the same order.

    Stage 3 selects units on the centre table and reads their polygons from the polygon table by
    row, so both tables must stay row-aligned.
    """
    a = centers["barcode"].astype(str).to_numpy()
    b = polygons["barcode"].astype(str).to_numpy()
    if len(a) != len(b) or (a != b).any():
        raise ValueError(f"{what}: *_center.csv ({len(a)} rows) and *_spot.csv ({len(b)} rows) do not list "
                         "the same barcodes in the same order; re-export both tables with stage 0 and "
                         "re-run stage 2")


def registered_rows(centers: pd.DataFrame) -> List[int]:
    """Row positions of the units that received coordinates (units outside every patch keep now_x = 0)."""
    return [i for i, v in enumerate(centers["now_x"].to_numpy()) if v != 0]


def selected_barcodes(polygons: pd.DataFrame, rows: Sequence[int]) -> List[str]:
    """Barcodes of ``rows`` (row positions), in that order: entry k belongs to the k-th selected unit."""
    return polygons["barcode"].iloc[list(rows)].tolist()


def n_batches(n_rows: int, batch_size: int) -> int:
    """Number of batches of ``batch_size`` needed for ``n_rows`` rows (0 for an empty table)."""
    return math.ceil(n_rows / batch_size)


def short_hd_barcode(barcode) -> Optional[str]:
    """Normalise a Visium HD barcode to its short form ``XXXXX_YYYYY-1``.

    ``s_008um_00341_00498-1``, ``S07-8-00341_00498-1``, ``S07-8s_008um_00341_00498-1`` and
    ``00341_00498-1`` all give ``00341_00498-1``; other strings are returned unchanged.
    """
    if not isinstance(barcode, str):
        return None
    m = re.search(r"(\d{5}_\d{5}-\d+)$", barcode)
    return m.group(1) if m else barcode
