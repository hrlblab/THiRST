"""Stage 3: overlaps must be reported under the barcodes of the overlapping units.

Before the fix the scripts looked barcodes up by label in a filtered Series: when some units had no
coordinates (now_x == 0) the IoUs were filed under the barcode of a different unit (or raised KeyError).
"""
import pytest

from _helpers import (CASE, circle, make_images, read_json, run_script, square, write_config,
                      write_unit_tables)

V2_V2 = "3_st_association/v2_v2_iou.py"
V2_HD = "3_st_association/v2_hd_iou.py"
MERGE = "3_st_association/merge_v2_hd_iou.py"


def _v2(barcode, x, y):
    return (barcode, (x, y), circle(x, y))


def _hd(barcode, x, y):
    return (barcode, (x, y), square(x, y))


@pytest.fixture
def stack(tmp_path):
    """Sections 1 (V2), 2 (HD), 3 (V2); on section 1 the first unit has no coordinates (now_x = 0)
    and the second overlaps nothing, so a label-based lookup is off by one."""
    make_images(tmp_path, [1, 2, 3])
    write_unit_tables(tmp_path, 1, [("Z-1", (0, 0), circle(0, 0)),     # filtered out
                                    _v2("P-1", 500, 500),                # no overlap
                                    _v2("A-1", 100, 100),
                                    _v2("C-1", 300, 300)])
    write_unit_tables(tmp_path, 2, [_hd("H0-1", 700, 700),               # no overlap
                                    _hd("H1-1", 100, 100),
                                    _hd("H2-1", 300, 302)])
    write_unit_tables(tmp_path, 3, [_v2("X-1", 100, 100), _v2("Y-1", 300, 300)])
    cfg = write_config(tmp_path, {1: "visium_v2", 2: "visium_hd", 3: "visium_v2"})
    return tmp_path, cfg


def test_v2_v2_barcodes_follow_filtered_rows(stack):
    tmp_path, cfg = stack
    r = run_script(V2_V2, "--config", cfg, "--registration", "none")
    assert r.returncode == 0, r.stdout + r.stderr
    out = read_json(tmp_path / "contour" / CASE / "IoU_noreg" / "row_1_col_3.json")
    assert sorted(out) == ["A-1", "C-1"]
    assert [m["barcode_j"] for m in out["A-1"]] == ["X-1"]
    assert [m["barcode_j"] for m in out["C-1"]] == ["Y-1"]
    assert out["A-1"][0]["iou"] == pytest.approx(1.0)


def test_v2_hd_barcodes_follow_filtered_rows_and_merge(stack):
    tmp_path, cfg = stack
    r = run_script(V2_HD, "--config", cfg, "--registration", "none")
    assert r.returncode == 0, r.stdout + r.stderr
    iou_dir = tmp_path / "contour" / CASE / "IoU_ST_to_HD_distance_noreg"
    out = read_json(iou_dir / "row_1_col_2_0.json")
    assert sorted(out) == ["A-1", "C-1"]
    assert [m["barcode_j"] for m in out["A-1"]] == ["H1-1"]
    assert [m["barcode_j"] for m in out["C-1"]] == ["H2-1"]
    expected_iou = 16.0 / (3.14159 * 10 ** 2)  # 4x4 bin inside a radius-10 spot
    assert out["A-1"][0]["iou"] == pytest.approx(expected_iou, rel=0.01)

    # a batch file of another column whose index starts with the same digit must not be merged in
    (iou_dir / "row_1_col_20_0.json").write_text('{"STRAY-1": [{"barcode_j": "S-1", "iou": 1.0}]}')
    r = run_script(MERGE, "--config", cfg, "--registration", "none")
    assert r.returncode == 0, r.stdout + r.stderr
    merged = read_json(tmp_path / "contour" / CASE / "IoU_ST_to_HD_distance_noreg_merge" / "row_1_col_2.json")
    assert sorted(merged) == ["A-1", "C-1"]


def test_misaligned_center_and_polygon_tables_stop(stack):
    tmp_path, cfg = stack
    # polygon table of section 3 in a different row order than its centre table
    write_unit_tables(tmp_path, 3, [_v2("X-1", 100, 100), _v2("Y-1", 300, 300)])
    spot = tmp_path / "contour" / CASE / "image_3_0_0_spot.csv"
    lines = spot.read_text().splitlines()
    spot.write_text("\n".join([lines[0], lines[2], lines[1]]) + "\n")
    r = run_script(V2_V2, "--config", cfg, "--registration", "none")
    assert r.returncode != 0
    assert "same barcodes in the same order" in r.stderr


def test_platform_layout_is_checked(tmp_path):
    make_images(tmp_path, [1, 2, 3])
    for s in (1, 2, 3):
        write_unit_tables(tmp_path, s, [_v2(f"A{s}-1", 100, 100)])
    cfg = write_config(tmp_path, {1: "visium_v2", 2: "visium_v2", 3: "visium_hd"})
    r = run_script(V2_V2, "--config", cfg, "--registration", "none")
    assert r.returncode != 0
    assert "alternate" in r.stderr


def test_odd_number_of_sections(tmp_path):
    """V2-HD association with an odd number of sections (the HD loop used to run past the last one)."""
    make_images(tmp_path, [1, 2, 3, 4, 5])
    for s in (1, 3, 5):
        write_unit_tables(tmp_path, s, [_v2(f"A{s}-1", 100, 100)])
    for s in (2, 4):
        write_unit_tables(tmp_path, s, [_hd(f"H{s}-1", 100, 100)])
    cfg = write_config(tmp_path, {1: "visium_v2", 2: "visium_hd", 3: "visium_v2", 4: "visium_hd", 5: "visium_v2"})
    r = run_script(V2_HD, "--config", cfg, "--registration", "none")
    assert r.returncode == 0, r.stdout + r.stderr
    r = run_script(MERGE, "--config", cfg, "--registration", "none")
    assert r.returncode == 0, r.stdout + r.stderr
    merged = tmp_path / "contour" / CASE / "IoU_ST_to_HD_distance_noreg_merge"
    assert sorted(p.name for p in merged.iterdir()) == [
        f"row_{i}_col_{j}.json" for i in (1, 3, 5) for j in (2, 4)]


def test_unit_shape_must_match_the_configured_platform(tmp_path):
    """A section configured as HD whose polygons are 100-vertex V2 spots stops the contour transfer."""
    from _helpers import write_identity_transform
    from thirst.transforms import pair_transform_path
    make_images(tmp_path, [1, 2, 3, 4, 5])                  # reference: section 3
    for s in (1, 2, 3, 5):
        write_unit_tables(tmp_path, s, [_v2(f"A{s}-1", 100, 100)])   # section 2: V2-like polygons
    write_unit_tables(tmp_path, 4, [_hd("H4-1", 100, 100)])
    # global_spot_contours.py reads the unregistered polygon tables from contour_root
    for moving, fixed in ((2, 1), (3, 2), (4, 3), (5, 4)):
        write_identity_transform(pair_transform_path(str(tmp_path / "png" / CASE / "registration_ANTs"), moving, fixed))
    cfg = write_config(tmp_path, {1: "visium_v2", 2: "visium_hd", 3: "visium_v2", 4: "visium_hd", 5: "visium_v2"})
    r = run_script("2_coordinate_transfer/global_spot_contours.py", "--config", cfg)
    assert r.returncode != 0
    assert "section 2: the ST unit polygons are 100-vertex Visium V2 spots" in r.stderr
