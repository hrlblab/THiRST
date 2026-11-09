"""Object GeoJSON layout, class-restricted tracking, track-file suffix, QuPath barcode keys."""
import json
import os

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon, shape

from thirst.config import stack_sample_names
from thirst.objects import object_feature, read_object_contours
from thirst.st import short_hd_barcode

from _helpers import CASE, load_script, run_script, write_config, write_geojson

BOX = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
BOX2 = [[50, 50], [60, 50], [60, 60], [50, 60], [50, 50]]


def test_reader_standard_multipolygon_and_earlier_layout(tmp_path):
    std = tmp_path / "std.geojson"
    write_geojson(std, [BOX, BOX2])
    old = tmp_path / "old.geojson"
    write_geojson(old, [BOX, BOX2], layout="legacy")          # every feature repeats the whole list
    multi = tmp_path / "multi.geojson"
    with open(multi, "w") as f:
        json.dump({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"classification": {"name": "Capsule"}},
             "geometry": {"type": "MultiPolygon", "coordinates": [[BOX], [BOX2]]}}]}, f)
    for path in (std, old, multi):
        got = read_object_contours(str(path), ["capsule"])["capsule"]
        assert len(got) == 2, path
        np.testing.assert_array_equal(got[1], np.array(BOX2, float))


def test_written_features_are_standard_geojson():
    feat = object_feature("capsule", [[0.5, 0.5], [4.5, 0.5], [4.5, 4.5], [0.5, 0.5]], contour_id=3)
    geom = shape(feat["geometry"])          # failed for the earlier layout (extra nesting level)
    assert geom.geom_type == "Polygon" and geom.is_valid
    assert feat["properties"]["contour_id"] == 3


def test_hungarian_tracker_does_not_link_different_classes():
    tracker = load_script("4_object_tracking/track_objects.py")
    sq = Polygon(BOX)
    slices = [[{"object": "capsule", "index": 0, "polygon": sq, "slice_index": 0}],
              [{"object": "tuft", "index": 0, "polygon": sq, "slice_index": 1}]]
    tracks, _ = tracker.track_sections(slices, 0.01)
    assert sorted([c["object"] for c in t] for t in tracks.values()) == [["capsule"], ["tuft"]]


def test_greedy_tracker_does_not_link_different_classes():
    greedy = load_script("4_object_tracking/track_objects_greedy.py")
    sq = Polygon(BOX)
    prev = {0: [{"object": "capsule", "index": 0, "polygon": sq, "slice_index": 0}]}
    _, unmatched = greedy.match_tracks(prev, [{"object": "tuft", "index": 0, "polygon": sq}], iou_threshold=0.01)
    assert len(prev[0]) == 1 and len(unmatched) == 1


def test_consumers_read_the_requested_track_file(tmp_path):
    iou_dir = tmp_path / "contour" / CASE / "Object_IoU_patch_connectglom"
    iou_dir.mkdir(parents=True)
    (tmp_path / "png" / CASE).mkdir(parents=True)
    tracks = [{"track_id": 0, "contours": [{"slice_index": 0, "object": "capsule", "index": 0, "coordinates": BOX}]}]
    (iou_dir / "IoU_Object_longglom.json").write_text(json.dumps(tracks))   # only the filtered tracks exist
    cfg = write_config(tmp_path, {1: "visium_v2"})
    args = ["--config", cfg, "--tracks-subdir", "Object_IoU_patch_connectglom"]
    r = run_script("4_object_tracking/cross_section_area.py", *args)
    assert r.returncode != 0 and "--tracks-suffix _longglom" in r.stderr
    r = run_script("4_object_tracking/cross_section_area.py", *args, "--tracks-suffix", "_longglom")
    assert r.returncode == 0, r.stdout + r.stderr
    per_contour = pd.read_csv(iou_dir / "cross_section_areas_per_contour_longglom.csv")
    assert per_contour.shape[0] == 1


def test_hd_barcode_keys_match_in_both_forms(tmp_path):
    qupath = load_script("5_object_analysis/qupath_cells_by_cluster.py")
    clusters = tmp_path / "clusters.csv"
    pd.DataFrame({"Sample": ["S02"], "Barcode": ["S02-8-s_008um_00001_00001-1"],
                  "seurat_cluster.projected": [4]}).to_csv(clusters, index=False)
    mapping = qupath.build_barcode_to_cluster_map(str(clusters))
    for table_barcode in ("s_008um_00001_00001-1", "00001_00001-1"):     # Space Ranger HD / short form
        assert mapping.get(f"S02-8-{short_hd_barcode(table_barcode)}") == 4


def test_sample_names_follow_the_stack():
    cfg = {"st_sections": {s: {"platform": "visium_v2", "sample": f"S{s:02d}"} for s in (1, 2, 3, 4)},
           "_config_path": "x.yaml"}
    images = [f"image_{s}_0_0.png" for s in (2, 3, 4)]       # stack without section 1
    assert list(stack_sample_names(cfg, images).values()) == ["S02", "S03", "S04"]
    with pytest.raises(KeyError):
        stack_sample_names(cfg, ["image_5_0_0.png"])


def test_demo_checker_fails_without_outputs(tmp_path):
    data = tmp_path / "demo_dir"
    for sub in ("png", "point", "contour"):
        (data / sub).mkdir(parents=True)
    (data / "demo.yaml").write_text(
        f"paths: {{image_root: {data}/png, point_root: {data}/point, contour_root: {data}/contour}}\n")
    (data / ".demo_mode").write_text("simulated\n")
    import subprocess
    import sys
    from _helpers import REPO
    r = subprocess.run([sys.executable, os.path.join(REPO, "demo", "check_demo.py"), "--config", str(data / "demo.yaml")],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "DEMO FAILED" in r.stdout


TRACK_CONSUMERS = ["5_object_analysis/match_objects_to_spots.py", "5_object_analysis/export_track_barcodes.py",
                   "5_object_analysis/gene_maps_per_track.py", "5_object_analysis/qupath_cells_by_cluster.py",
                   "6_visualization/tracking_figure.py", "6_visualization/render_3d_objects.py"]


def test_output_folders_name_the_tracks_they_read(tmp_path):
    from thirst.config import tracks_output_dir, tracks_tag
    assert tracks_tag("Object_IoU_patch_connectglom") == "_connectglom"        # Hungarian
    assert tracks_tag("Object_IoU_patch") == ""                                # greedy
    assert tracks_tag("Object_IoU_global/") == "_Object_IoU_global"           # anything else
    d = str(tmp_path)
    base = "Object_tracking_IoU_affine_patch"
    # the default track folders keep the folder names of the original scripts ...
    assert tracks_output_dir(d, base, "Object_IoU_patch_connectglom") == os.path.join(d, base + "_connectglom")
    assert tracks_output_dir(d, base, "Object_IoU_patch_connectglom", "_longglom", tail="_global_vis") == \
        os.path.join(d, base + "_connectglom_global_vis_longglom")
    # ... and results of the other tracker never land in (or are named like) the same folder
    assert tracks_output_dir(d, base, "Object_IoU_patch") == os.path.join(d, base)
    assert tracks_output_dir(d, "qupath_annotations_by_track_clusters", "Object_IoU_patch_connectglom") == \
        os.path.join(d, "qupath_annotations_by_track_clusters_connectglom")
    for script in TRACK_CONSUMERS:      # every script that reads tracks names its output folder this way
        from _helpers import REPO
        src = open(os.path.join(REPO, "scripts", script)).read().replace("Object_IoU_patch_connectglom", "")
        assert "tracks_output_dir(" in src, script
        assert "_connectglom'" not in src and '_connectglom"' not in src, script
