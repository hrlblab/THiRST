"""Stage 4: an empty section must not hide the sections after it; GeoJSONs stay aligned with sections."""
import json

import pytest
from shapely.geometry import Polygon

from _helpers import CASE, load_script, make_images, run_script, write_config, write_geojson

from thirst.config import section_object_geojsons


def sq(x, y, s=10):
    return Polygon([(x, y), (x + s, y), (x + s, y + s), (x, y + s)])


def contour(slice_index, poly, obj="capsule", index=0):
    return {"object": obj, "index": index, "polygon": poly, "slice_index": slice_index}


def test_hungarian_tracking_visits_sections_after_an_empty_one():
    tracker = load_script("4_object_tracking/track_objects.py")
    slices = [[contour(0, sq(0, 0))], [], [contour(2, sq(0, 0))], [contour(3, sq(1, 1))]]
    tracks, _ = tracker.track_sections(slices, 0.01)
    assert sorted(c["slice_index"] for tr in tracks.values() for c in tr) == [0, 2, 3]
    assert sorted(len(tr) for tr in tracks.values()) == [1, 2]   # no link across the empty section


def test_greedy_matcher_handles_several_classes():
    greedy = load_script("4_object_tracking/track_objects_greedy.py")
    prev = {0: [contour(0, sq(0, 0), "capsule")], 1: [contour(0, sq(50, 50), "tuft")]}
    current = [{"object": "capsule", "index": 0, "polygon": sq(0, 0)},
               {"object": "tuft", "index": 0, "polygon": sq(50, 50)}]   # same per-class index
    _, unmatched = greedy.match_tracks(prev, current, iou_threshold=0.01)
    assert unmatched == []
    assert len(prev[0]) == 2 and len(prev[1]) == 2


def test_greedy_loader_accepts_a_section_without_the_class(tmp_path):
    greedy = load_script("4_object_tracking/track_objects_greedy.py")
    path = tmp_path / "s.geojson"
    write_geojson(path, [[[0, 0], [5, 0], [5, 5]]], cls="tuft")
    assert greedy.load_annotations("image_1_0_0.png", str(path), ["capsule"]) == {"capsule": []}


def test_section_geojsons_are_aligned_with_sections(tmp_path):
    images = [f"/x/image_{s}_0_0.png" for s in (1, 2, 10)]
    for s in (1, 10):
        write_geojson(tmp_path / f"{CASE}-{s}.svs.geojson", [])
    with pytest.raises(FileNotFoundError, match=r"\['2'\]"):
        section_object_geojsons(images, str(tmp_path), CASE)
    write_geojson(tmp_path / f"{CASE}-2.svs.geojson", [])
    paths = section_object_geojsons(images, str(tmp_path), CASE)
    assert [p.rsplit("-", 1)[1] for p in paths] == ["1.svs.geojson", "2.svs.geojson", "10.svs.geojson"]


def test_track_objects_script_with_an_empty_section(tmp_path):
    """End to end: section 2 of 4 has no object; the object on sections 3-4 must still be tracked."""
    make_images(tmp_path, [1, 2, 3, 4])
    obj_dir = tmp_path / "contour" / CASE / "registration_3D_patch_ANTs" / "annotation_QA_generegions"
    box = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    write_geojson(obj_dir / f"{CASE}-1.svs.geojson", [box])
    write_geojson(obj_dir / f"{CASE}-2.svs.geojson", [])
    write_geojson(obj_dir / f"{CASE}-3.svs.geojson", [box])
    write_geojson(obj_dir / f"{CASE}-4.svs.geojson", [box])
    cfg = write_config(tmp_path, {1: "visium_v2", 2: "visium_hd", 3: "visium_v2", 4: "visium_hd"})
    r = run_script("4_object_tracking/track_objects.py", "--config", cfg)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "no ['capsule'] objects on section(s) ['2']" in r.stdout
    with open(tmp_path / "contour" / CASE / "Object_IoU_patch_connectglom" / "IoU_Object.json") as f:
        tracks = json.load(f)
    seen = sorted(c["slice_index"] for t in tracks for c in t["contours"])
    assert seen == [0, 2, 3]


def test_track_objects_script_stops_on_a_missing_section_file(tmp_path):
    make_images(tmp_path, [1, 2, 3])
    obj_dir = tmp_path / "contour" / CASE / "registration_3D_patch_ANTs" / "annotation_QA_generegions"
    box = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    write_geojson(obj_dir / f"{CASE}-1.svs.geojson", [box])
    write_geojson(obj_dir / f"{CASE}-3.svs.geojson", [box])
    cfg = write_config(tmp_path, {1: "visium_v2", 2: "visium_hd", 3: "visium_v2"})
    r = run_script("4_object_tracking/track_objects.py", "--config", cfg)
    assert r.returncode != 0
    assert "no object GeoJSON for section(s) ['2']" in r.stderr


def test_no_track_reaches_min_track_len(tmp_path):
    """--min-track-len that no object reaches gives an empty track file, which the area analysis accepts."""
    make_images(tmp_path, [1, 2, 3])
    obj_dir = tmp_path / "contour" / CASE / "registration_3D_patch_ANTs" / "annotation_QA_generegions"
    box = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    for s in (1, 2, 3):
        write_geojson(obj_dir / f"{CASE}-{s}.svs.geojson", [box])
    cfg = write_config(tmp_path, {1: "visium_v2", 2: "visium_hd", 3: "visium_v2"})
    r = run_script("4_object_tracking/track_objects.py", "--config", cfg, "--min-track-len", "10")
    assert r.returncode == 0, r.stdout + r.stderr
    iou_dir = tmp_path / "contour" / CASE / "Object_IoU_patch_connectglom"
    assert json.loads((iou_dir / "IoU_Object_longglom.json").read_text()) == []
    r = run_script("4_object_tracking/cross_section_area.py", "--config", cfg,
                   "--tracks-subdir", "Object_IoU_patch_connectglom", "--tracks-suffix", "_longglom")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "contains no tracks" in r.stdout
    lines = (iou_dir / "cross_section_areas_per_slice_longglom.csv").read_text().splitlines()
    assert lines == ["track_id,object,slice_index,total_area_px2,contours,total_points"]
