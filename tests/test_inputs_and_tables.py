"""Shared helpers: in-tissue filtering, section layout, ST unit tables."""
import pandas as pd
import pytest

from thirst.config import read_in_tissue_positions, require_alternating_platforms, single_object_class
from thirst.st import is_circle_table, n_batches, registered_rows, require_same_barcodes, selected_barcodes

from _helpers import circle, square


def _positions(path, barcodes, in_tissue=None):
    df = pd.DataFrame({"barcode": barcodes, "pxl_row_in_fullres": range(len(barcodes)),
                       "pxl_col_in_fullres": range(len(barcodes))})
    if in_tissue is not None:
        df["in_tissue"] = in_tissue
    df.to_csv(path, index=False)
    return str(path)


def test_in_tissue_flag_is_matched_by_barcode(tmp_path):
    positions = _positions(tmp_path / "wsi.csv", ["a", "b", "c"])                # re-mapped table, no flag
    flags = _positions(tmp_path / "orig.csv", ["c", "a", "b"], in_tissue=[1, 0, 1])  # other row order
    kept = read_in_tissue_positions({"positions": positions, "in_tissue_positions": flags})
    assert kept["barcode"].tolist() == ["b", "c"]


def test_in_tissue_from_the_positions_table_itself(tmp_path):
    positions = _positions(tmp_path / "p.csv", ["a", "b", "c"], in_tissue=[1, 0, 1])
    assert read_in_tissue_positions({"positions": positions})["barcode"].tolist() == ["a", "c"]


def test_in_tissue_table_must_cover_every_barcode(tmp_path):
    positions = _positions(tmp_path / "wsi.csv", ["a", "b", "z"])
    flags = _positions(tmp_path / "orig.csv", ["a", "b"], in_tissue=[1, 1])
    with pytest.raises(ValueError, match="not listed"):
        read_in_tissue_positions({"positions": positions, "in_tissue_positions": flags})


def _cfg(platforms):
    return {"st_sections": {s: {"platform": p} for s, p in platforms.items()}, "_config_path": "test.yaml",
            "object_classes": ["capsule"]}


def test_alternating_layout():
    images = [f"image_{s}_0_0.png" for s in (1, 2, 3, 4)]
    require_alternating_platforms(_cfg({1: "visium_v2", 2: "visium_hd", 3: "visium_v2", 4: "visium_hd"}), images)
    with pytest.raises(ValueError, match="section 3 at position 2"):
        require_alternating_platforms(_cfg({1: "visium_v2", 2: "visium_hd", 3: "visium_hd", 4: "visium_v2"}), images)


def test_single_object_class():
    assert single_object_class(_cfg({})) == ["capsule"]
    with pytest.raises(ValueError):
        single_object_class(dict(_cfg({}), object_classes=["capsule", "tuft"]))


def _polygon_table(n_rows, poly):
    cols = {"barcode": [f"u{i}" for i in range(n_rows)]}
    for k in range(len(poly)):
        cols[f"now_x_{k}"] = [poly[k, 0]] * n_rows
        cols[f"now_y_{k}"] = [poly[k, 1]] * n_rows
    return pd.DataFrame(cols)


def test_unit_shape_comes_from_the_vertex_columns():
    # a small HD region (<= 10,000 bins) and a large V2 capture area (> 10,000 spots)
    assert is_circle_table(_polygon_table(500, square(0, 0))) == 0
    assert is_circle_table(_polygon_table(12000, circle(0, 0))) == 1
    with pytest.raises(ValueError):
        is_circle_table(_polygon_table(3, circle(0, 0, n=7)))


def test_registered_rows_and_their_barcodes():
    centers = pd.DataFrame({"barcode": ["z", "p", "a"], "now_x": [0, 5, 7]})
    rows = registered_rows(centers)
    assert rows == [1, 2]
    assert selected_barcodes(centers, rows) == ["p", "a"]


def test_center_and_polygon_tables_must_be_row_aligned():
    a = pd.DataFrame({"barcode": ["x", "y"]})
    require_same_barcodes(a, a.copy(), "s")
    with pytest.raises(ValueError):
        require_same_barcodes(a, pd.DataFrame({"barcode": ["y", "x"]}), "s")
    with pytest.raises(ValueError):
        require_same_barcodes(a, pd.DataFrame({"barcode": ["x"]}), "s")


def test_batch_count():
    assert [n_batches(n, 1000) for n in (0, 1, 1000, 1001)] == [0, 1, 1, 2]
