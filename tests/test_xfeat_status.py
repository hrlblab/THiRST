"""Stage 1: XFeat outcomes are recorded, failed pairs stop the run, stale initialisations are not reused."""
import hashlib
import json
import os
import shutil
import subprocess

import cv2
import numpy as np
import pandas as pd

from _helpers import CASE, REPO, run_script, write_config

STUBS = os.path.join(REPO, "tests", "stubs")


def test_xfeat_status_and_failed_pairs(tmp_path, monkeypatch):
    stack = tmp_path / "png" / CASE
    stack.mkdir(parents=True)
    rng = np.random.default_rng(1)
    textured = (rng.uniform(0, 255, (64, 64, 3))).astype(np.uint8)
    cv2.imwrite(str(stack / "image_1_0_0.png"), textured)
    cv2.imwrite(str(stack / "image_2_0_0.png"), textured)
    cv2.imwrite(str(stack / "image_3_0_0.png"), np.full((64, 64, 3), 200, np.uint8))   # blank section
    stale = stack / "registration_xfeat" / "3_to_2" / "sg_affine_init.npy"
    stale.parent.mkdir(parents=True)
    np.save(stale, np.eye(2, 3))                                                       # left by an earlier run
    cfg = write_config(tmp_path, {1: "visium_v2", 2: "visium_hd", 3: "visium_v2"})

    monkeypatch.setenv("PYTHONPATH", STUBS)
    r = run_script("1_registration/xfeat_affine.py", "--config", cfg, "--level", "global")
    assert r.returncode == 1, r.stdout + r.stderr
    status = pd.read_csv(stack / "registration_xfeat" / "xfeat_status.csv")
    assert status.to_dict("list") == {"pair": ["2_to_1", "3_to_2"], "status": ["ok", "failed"]}
    assert (stack / "registration_xfeat" / "2_to_1" / "sg_affine_init.npy").exists()
    assert not stale.exists()
    provenance = json.loads((tmp_path / "png" / "xfeat_provenance_global.json").read_text())
    assert provenance["hub_source"] == "verlab/accelerated_features"
    assert provenance["requested_ref"] == "default branch (not pinned)" and provenance["git_commit"] is None
    assert "commit not known" in provenance["note"]

    r = run_script("1_registration/affine_to_itk.py", "--config", cfg, "--level", "global")
    assert r.returncode != 0
    assert "xfeat_status.csv" in r.stderr


# a stand-in repository laid out like verlab/accelerated_features: hubconf.py + modules/xfeat.py
HUBCONF = """
from modules.xfeat import XFeat as _XFeat


def XFeat(pretrained=True, top_k=4096):
    return _XFeat()
"""
XFEAT_MODULE = """
import torch


class XFeat:
    def match_xfeat(self, a, b, top_k=4096):
        return torch._Matcher().match_xfeat(a, b, top_k)
"""


def _two_sections(tmp_path):
    stack = tmp_path / "png" / CASE
    stack.mkdir(parents=True)
    img = np.random.default_rng(1).uniform(0, 255, (64, 64, 3)).astype(np.uint8)
    for s in (1, 2):
        cv2.imwrite(str(stack / f"image_{s}_0_0.png"), img)


def test_xfeat_provenance_of_a_local_clone_and_weights(tmp_path, monkeypatch):
    from thirst.provenance import tree_sha256
    repo = tmp_path / "accelerated_features"
    (repo / "modules").mkdir(parents=True)
    (repo / "hubconf.py").write_text(HUBCONF)
    (repo / "modules" / "__init__.py").write_text("")
    (repo / "modules" / "xfeat.py").write_text(XFEAT_MODULE)
    commit = None
    if shutil.which("git"):
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@t")
        for cmd in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "x"]):
            subprocess.run(["git", "-C", str(repo), *cmd], check=True, env=env)
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
    hub = tmp_path / "hub"
    (hub / "checkpoints").mkdir(parents=True)
    (hub / "checkpoints" / "xfeat.pt").write_bytes(b"weights")
    good = hashlib.sha256(b"weights").hexdigest()
    _two_sections(tmp_path)
    monkeypatch.setenv("PYTHONPATH", STUBS)
    monkeypatch.setenv("STUB_TORCH_HUB", str(hub))

    cfg = write_config(tmp_path, {1: "visium_v2", 2: "visium_hd"},
                       extra={"xfeat_hub": str(repo), "xfeat_weights_sha256": good})
    r = run_script("1_registration/xfeat_affine.py", "--config", cfg, "--level", "global")
    assert r.returncode == 0, r.stdout + r.stderr
    p = json.loads((tmp_path / "png" / "xfeat_provenance_global.json").read_text())
    assert p["requested_ref"] == "local folder" and p["code_dir"] == str(repo)
    shutil.rmtree(repo / "modules" / "__pycache__", ignore_errors=True)
    assert p["code_sha256"] == tree_sha256(str(repo))
    assert p["git_commit"] == commit
    assert p["weights_sha256"] == good and p["weights_verified"] is True

    cfg = write_config(tmp_path, {1: "visium_v2", 2: "visium_hd"},
                       extra={"xfeat_hub": str(repo), "xfeat_weights_sha256": "0" * 64})
    r = run_script("1_registration/xfeat_affine.py", "--config", cfg, "--level", "global")
    assert r.returncode != 0 and "xfeat_weights_sha256" in r.stderr


def test_tree_digest_ignores_git_and_caches(tmp_path):
    from thirst.provenance import tree_sha256
    (tmp_path / "a.py").write_text("x = 1\n")
    first = tree_sha256(str(tmp_path))
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"cache")
    assert tree_sha256(str(tmp_path)) == first
    (tmp_path / "a.py").write_text("x = 2\n")
    assert tree_sha256(str(tmp_path)) != first


def test_xfeat_pinned_commit_is_recorded(tmp_path, monkeypatch):
    _two_sections(tmp_path)
    monkeypatch.setenv("PYTHONPATH", STUBS)
    monkeypatch.setenv("STUB_TORCH_HUB", str(tmp_path / "hub"))
    cfg = write_config(tmp_path, {1: "visium_v2", 2: "visium_hd"},
                       extra={"xfeat_hub": "verlab/accelerated_features:0123abcd"})
    r = run_script("1_registration/xfeat_affine.py", "--config", cfg, "--level", "global")
    assert r.returncode == 0, r.stdout + r.stderr
    p = json.loads((tmp_path / "png" / "xfeat_provenance_global.json").read_text())
    assert p["requested_ref"] == "0123abcd" and p["git_commit"] == "0123abcd"


def test_demo_and_example_use_the_xfeat_release_configuration(tmp_path):
    import importlib.util
    import yaml
    from thirst.provenance import XFEAT_COMMIT, XFEAT_REPO, XFEAT_WEIGHTS_SHA256
    spec = importlib.util.spec_from_file_location("make_synthetic_data", os.path.join(REPO, "demo", "make_synthetic_data.py"))
    gt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gt)
    assert gt.xfeat_settings({}) == {"xfeat_hub": f"{XFEAT_REPO}:{XFEAT_COMMIT}",
                                     "xfeat_weights_sha256": XFEAT_WEIGHTS_SHA256}
    clone = tmp_path / "accelerated_features"
    clone.mkdir()
    assert gt.xfeat_settings({"THIRST_XFEAT_HUB": str(clone)})["xfeat_hub"] == str(clone)
    example = yaml.safe_load(open(os.path.join(REPO, "configs", "example.yaml")))
    assert example["xfeat_hub"] == f"{XFEAT_REPO}:{XFEAT_COMMIT}"
    assert example["xfeat_weights_sha256"] == XFEAT_WEIGHTS_SHA256


def _real_demo_dir(tmp_path, name, provenance):
    data = tmp_path / name
    for sub in ("png", "point", "contour"):
        (data / sub).mkdir(parents=True)
    (data / "demo.yaml").write_text(
        f"paths: {{image_root: {data}/png, point_root: {data}/point, contour_root: {data}/contour}}\n")
    (data / ".demo_mode").write_text("real\n")
    for level in ("global", "patch"):
        (data / "png" / f"xfeat_provenance_{level}.json").write_text(json.dumps(provenance))
    r = subprocess.run([os.sys.executable, os.path.join(REPO, "demo", "check_demo.py"), "--config", str(data / "demo.yaml")],
                       capture_output=True, text=True)
    return r


def test_real_demo_check_requires_verified_xfeat_weights(tmp_path):
    from thirst.provenance import XFEAT_COMMIT
    r = _real_demo_dir(tmp_path, "unverified", {"git_commit": XFEAT_COMMIT, "weights_sha256": "ab" * 32, "weights_verified": False})
    assert r.returncode == 1 and "weights were not verified" in r.stdout
    r = _real_demo_dir(tmp_path, "other_commit", {"git_commit": "1234567", "weights_sha256": "ab" * 32, "weights_verified": True})
    assert "weights were not verified" not in r.stdout
    assert "is not the release configuration" in r.stdout    # other XFeat commits are reported
