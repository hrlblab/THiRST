"""Stages 1-2: failed or missing registrations must stop the run instead of yielding partial chains."""
import json
import os
import stat
import subprocess

import numpy as np
import pytest
from shapely.geometry import shape

from thirst.ants import ANTsRegistrationError, is_completed, register_a_pair
from thirst.transforms import (MissingTransformError, is_valid_transform, pair_transform_path, require_chain,
                               require_transform)

from _helpers import CASE, REPO, make_images, run_script, write_config, write_geojson, write_identity_transform

CONVERT = '#!/bin/sh\ntouch "$2"\n'
SYN_FAIL = "#!/bin/sh\nexit 3\n"
SYN_NO_OUTPUT = "#!/bin/sh\nexit 0\n"
# stand-in for a successful run: writes the initial transform as the result (a valid ITK affine)
SYN_OK = ('#!/bin/sh\nwhile [ $# -gt 0 ]; do [ "$1" = "-o" ] && out="$2"; [ "$1" = "-i" ] && init="$2"; shift; done\n'
          'cp "$init" "${out}0GenericAffine.mat"\n')
# writes a valid affine and then fails (e.g. a later SyN stage crashes)
SYN_WRITE_THEN_FAIL = ('#!/bin/sh\nwhile [ $# -gt 0 ]; do [ "$1" = "-o" ] && out="$2"; [ "$1" = "-i" ] && init="$2"; shift; done\n'
                       'cp "$init" "${out}0GenericAffine.mat"\nexit 42\n')
SYN_GARBAGE = '#!/bin/sh\nwhile [ $# -gt 0 ]; do [ "$1" = "-o" ] && out="$2"; shift; done\necho garbage > "${out}0GenericAffine.mat"\n'


def fake_ants(tmp_path, syn):
    d = tmp_path / "ants_bin"
    d.mkdir(exist_ok=True)
    for name, body in (("ConvertImagePixelType", CONVERT), ("antsRegistrationSyN.sh", syn)):
        p = d / name
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return d


def pair(tmp_path):
    work = tmp_path / "png" / CASE / "registration_ANTs" / "2_to_1"
    work.mkdir(parents=True)
    write_identity_transform(work / "sg_affine_init.mat")
    moving, fixed = tmp_path / "image_2_0_0.png", tmp_path / "image_1_0_0.png"
    moving.write_bytes(b"")
    fixed.write_bytes(b"")
    return str(moving), str(fixed), work


def test_require_transform(tmp_path):
    path = pair_transform_path(str(tmp_path), 2, 1)
    with pytest.raises(MissingTransformError):
        require_transform(path)
    write_identity_transform(path)
    assert require_transform(path) == path


@pytest.mark.parametrize("syn", [SYN_FAIL, SYN_NO_OUTPUT, SYN_GARBAGE, SYN_WRITE_THEN_FAIL])
def test_register_a_pair_raises_when_ants_fails(tmp_path, syn):
    moving, fixed, work = pair(tmp_path)
    with pytest.raises(ANTsRegistrationError):
        register_a_pair(moving, fixed, str(work), str(fake_ants(tmp_path, syn)))
    assert not (work / "step2_run_ants_reg").exists()            # nothing left where results are read
    assert (work / "step2_run_ants_reg.failed").is_dir()          # partial outputs kept for inspection
    assert not is_completed(str(work))


def test_register_a_pair_needs_the_xfeat_initialisation(tmp_path):
    moving, fixed, work = pair(tmp_path)
    (work / "sg_affine_init.mat").unlink()
    with pytest.raises(FileNotFoundError, match="affine_to_itk"):
        register_a_pair(moving, fixed, str(work), str(fake_ants(tmp_path, SYN_OK)))


def test_register_a_pair_returns_the_affine(tmp_path):
    moving, fixed, work = pair(tmp_path)
    out = register_a_pair(moving, fixed, str(work), str(fake_ants(tmp_path, SYN_OK)))
    assert out == str(work / "step2_run_ants_reg" / "output0GenericAffine.mat") and os.path.isfile(out)
    assert is_completed(str(work))
    record = json.loads((work / "step2_run_ants_reg" / "ants_completed.json").read_text())
    assert record["exit_status"] == 0 and len(record["affine_sha256"]) == 64
    with open(out, "ab") as f:                                       # result changed after completion
        f.write(b"\n")
    assert not is_completed(str(work))


def _ants_stack(tmp_path, syn):
    make_images(tmp_path, [1, 2])
    work = tmp_path / "png" / CASE / "registration_ANTs" / "2_to_1"
    work.mkdir(parents=True)
    write_identity_transform(work / "sg_affine_init.mat")
    cfg = write_config(tmp_path, {1: "visium_v2", 2: "visium_hd"}, ants_bin=fake_ants(tmp_path, syn))
    return cfg, work / "step2_run_ants_reg" / "output0GenericAffine.mat"


def test_ants_affine_exits_with_error_and_lists_failed_pairs(tmp_path):
    cfg, _ = _ants_stack(tmp_path, SYN_FAIL)
    r = run_script("1_registration/ants_affine.py", "--config", cfg, "--level", "global")
    assert r.returncode == 1
    assert "ANTs registration failed for 1 pair(s)" in r.stdout


def _ants(cfg, *extra):
    return run_script("1_registration/ants_affine.py", "--config", cfg, "--level", "global", *extra)


def test_failed_run_is_not_taken_as_a_result_on_restart(tmp_path):
    """ANTs writes a valid affine, then exits with an error: the restart must run ANTs again."""
    cfg, affine = _ants_stack(tmp_path, SYN_WRITE_THEN_FAIL)
    r = _ants(cfg)
    assert r.returncode == 1 and not affine.exists()
    r = _ants(cfg)                                                    # restart
    assert r.returncode == 1 and "existing result kept" not in r.stdout
    assert "antsRegistrationSyN.sh" in r.stdout                       # ANTs was run again


def test_completed_results_are_kept_and_recomputed_on_request(tmp_path):
    cfg, affine = _ants_stack(tmp_path, SYN_OK)
    assert _ants(cfg).returncode == 0 and is_valid_transform(str(affine))
    fake_ants(tmp_path, SYN_FAIL)                                     # any later ANTs call would fail
    r = _ants(cfg)
    assert r.returncode == 0 and "existing result kept" in r.stdout
    r = _ants(cfg, "--overwrite")
    assert r.returncode == 1 and not affine.exists()                  # a failed re-run leaves no result


def test_results_without_completion_record(tmp_path):
    cfg, affine = _ants_stack(tmp_path, SYN_OK)
    write_identity_transform(affine)        # valid, but not written by a completed run of this script
    r = _ants(cfg, "--trust-existing")
    assert r.returncode == 0 and "without a completion record" in r.stdout
    r = _ants(cfg)
    assert r.returncode == 0 and "no completion record" in r.stdout and "existing result kept" not in r.stdout
    affine.write_text("truncated")                                    # unreadable: always recomputed
    r = _ants(cfg, "--trust-existing")
    assert r.returncode == 0 and is_valid_transform(str(affine))


def _object_stack(tmp_path, pairs):
    make_images(tmp_path, [1, 2, 3])                    # middle (reference) section: 2
    seg = tmp_path / "seg"
    box = [[100, 100], [140, 100], [140, 140], [100, 140], [100, 100]]
    other = [[300, 300], [340, 300], [340, 340], [300, 340], [300, 300]]
    for s in (1, 2, 3):
        write_geojson(seg / f"{CASE}-{s}.svs.geojson", [box, other])   # QuPath layout
    for moving, fixed in pairs:
        write_identity_transform(pair_transform_path(str(tmp_path / "png" / CASE / "registration_ANTs"), moving, fixed))
    return write_config(tmp_path, {1: "visium_v2", 2: "visium_hd", 3: "visium_v2"}, object_annotation_dir=seg), box


def test_global_objects_stops_on_a_missing_pair(tmp_path):
    # 2_to_1 (needed to bring section 1 to the reference) is missing; this used to be skipped silently
    cfg, _ = _object_stack(tmp_path, [(3, 2)])
    r = run_script("2_coordinate_transfer/global_objects.py", "--config", cfg)
    assert r.returncode != 0
    assert "missing or invalid ANTs transform" in r.stderr and "2_to_1" in r.stderr
    out_dir = tmp_path / "contour" / CASE / "registration_3D_ANTs" / "annotation_QA"
    assert not list(out_dir.glob("*.geojson"))          # checked before any section is written


def test_require_chain_lists_every_bad_pair(tmp_path):
    images = [f"image_{s}_0_0.png" for s in (1, 2, 3, 4)]
    reg = tmp_path / "registration_ANTs"
    write_identity_transform(pair_transform_path(str(reg), 3, 2))
    garbage = pair_transform_path(str(reg), 4, 3)
    os.makedirs(os.path.dirname(garbage))
    open(garbage, "w").write("not a transform")
    with pytest.raises(MissingTransformError, match=r"\['2_to_1', '4_to_3'\]"):
        require_chain(str(reg), images)


def test_global_objects_with_identity_transforms(tmp_path):
    cfg, box = _object_stack(tmp_path, [(2, 1), (3, 2)])
    r = run_script("2_coordinate_transfer/global_objects.py", "--config", cfg)
    assert r.returncode == 0, r.stdout + r.stderr
    out_dir = tmp_path / "contour" / CASE / "registration_3D_ANTs" / "annotation_QA"
    for s in (1, 2, 3):
        with open(out_dir / f"{CASE}-{s}.svs.geojson") as f:
            features = json.load(f)["features"]
        # standard GeoJSON: one Polygon per object, in the 4x frame of the reference section (origin 0 here)
        assert len(features) == 2
        geoms = [shape(feat["geometry"]) for feat in features]
        assert all(g.is_valid for g in geoms)
        np.testing.assert_allclose(np.array(geoms[0].exterior.coords), np.array(box) / 4, atol=1e-3)


def test_integer_input_coordinates_are_not_truncated(tmp_path):
    make_images(tmp_path, [1, 2, 3])
    seg = tmp_path / "seg"
    ring = [[101, 103], [141, 103], [141, 147], [101, 147], [101, 103]]   # integers, not multiples of 4
    for s in (1, 2, 3):
        write_geojson(seg / f"{CASE}-{s}.svs.geojson", [ring])
    for moving, fixed in ((2, 1), (3, 2)):
        write_identity_transform(pair_transform_path(str(tmp_path / "png" / CASE / "registration_ANTs"), moving, fixed))
    cfg = write_config(tmp_path, {1: "visium_v2", 2: "visium_hd", 3: "visium_v2"}, object_annotation_dir=seg)
    r = run_script("2_coordinate_transfer/global_objects.py", "--config", cfg)
    assert r.returncode == 0, r.stdout + r.stderr
    with open(tmp_path / "contour" / CASE / "registration_3D_ANTs" / "annotation_QA" / f"{CASE}-1.svs.geojson") as f:
        out = json.load(f)["features"][0]["geometry"]["coordinates"][0]
    np.testing.assert_allclose(np.array(out), np.array(ring) / 4, atol=1e-3)   # 25.25, not 25


def _demo_dir(tmp_path, mode):
    d = tmp_path / "demo_dir"
    d.mkdir()
    (d / "demo.yaml").write_text("paths: {}\n")
    if mode:
        (d / ".demo_mode").write_text(mode + "\n")
    return d


@pytest.mark.parametrize("mode,args", [("simulated", ["--real"]), ("real", []), (None, ["--real"])])
def test_demo_refuses_to_mix_registration_modes(tmp_path, mode, args):
    d = _demo_dir(tmp_path, mode)
    r = subprocess.run(["bash", os.path.join(REPO, "demo", "run_demo.sh"), str(d), *args],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1
    assert "error:" in r.stderr
