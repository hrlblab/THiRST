"""Verify a demo run against the known ground truth of the synthetic dataset.

    python demo/check_demo.py --config demo_data/demo.yaml

Fails unless every expected output is present and correct:

* the run used the registration mode recorded in ``<data dir>/.demo_mode``; for ``real`` the XFeat
  status files, the ANTs warp fields and completion records must exist (proof that the tools ran and
  finished) and the XFeat weights must have been verified against ``xfeat_weights_sha256`` (the XFeat
  commit is reported; a commit other than the release configuration is noted), for ``simulated`` the
  global stage must show the known residual of the simulated registration;
* V2 spot centres of every V2 section exist at every stage and are finite; after the global stage they
  sit where the simulated residual puts them (< 0.8 px, simulated) or within 1 px of the reference grid
  (real); after patch-level refinement every section is within 0.8 px of the grid (1 px real) and,
  simulated, the residual is at least halved (coordinates are whole 4x pixels, hence the tolerances);
* ``track_objects.py`` reconstructs the six objects with the expected number of sections;
* V2 spots are matched across sections (median best IoU > 0.9).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from thirst.config import is_circular_spot, list_cases, list_patch_stacks, load_config, section_index  # noqa: E402
from thirst.provenance import XFEAT_COMMIT  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_synthetic_data as gt  # noqa: E402


def _grid():
    n = gt.REGION // 4 // 100
    return np.array([(1024 + 50 + 100 * j, 1024 + 50 + 100 * i) for j in range(n) for i in range(n)], float)


def _match(csv):
    """V2 spot centres (row, col) in ``csv`` with registered coordinates, and their reference grid point."""
    ref = _grid()
    pts = pd.read_csv(csv)[["now_x", "now_y"]].to_numpy(float)
    pts = pts[(pts != 0).any(1)]
    if len(pts) == 0:
        return pts, pts
    nearest = np.argmin(((pts[:, None, :] - ref[None, :, :]) ** 2).sum(-1), axis=1)
    return pts, ref[nearest]


def _apply_rc(m, rc):
    """Apply a 3x3 affine acting on (x = column, y = row) to (row, col) points."""
    xy = rc[:, ::-1] @ m[:2, :2].T + m[:2, 2]
    return xy[:, ::-1]


class Check:
    def __init__(self):
        self.ok = True

    def __call__(self, condition, message):
        if not condition:
            self.ok = False
            print(f"  FAIL: {message}")
        return condition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    config = ap.parse_args().config
    cfg = load_config(config)
    check = Check()

    mode_file = os.path.join(os.path.dirname(os.path.abspath(config)), ".demo_mode")
    mode = open(mode_file).read().strip() if os.path.isfile(mode_file) else None
    print(f"registration mode: {mode or 'unknown'}"
          + {"simulated": " (known transforms written by make_synthetic_data.py; no XFeat/ANTs)",
             "real": " (XFeat + ANTs)"}.get(mode, ""))
    check(mode in ("simulated", "real"), f"{mode_file} missing or unreadable: cannot tell which registration ran")

    if mode == "real":       # which XFeat code and weights ran (xfeat_affine.py writes one record per level)
        for level in ("global", "patch"):
            prov = os.path.join(cfg["paths"]["image_root"], f"xfeat_provenance_{level}.json")
            if not check(os.path.isfile(prov), f"{prov} missing: no record of the XFeat code and weights"):
                continue
            p = json.load(open(prov))
            check(p.get("weights_verified") is True,
                  f"{prov}: XFeat weights were not verified against xfeat_weights_sha256")
            commit = p.get("git_commit") or "unknown"
            print(f"  XFeat ({level}): commit {commit}, code SHA-256 {str(p.get('code_sha256'))[:16]}..., "
                  f"weights SHA-256 {str(p.get('weights_sha256'))[:16]}..."
                  + (" (verified)" if p.get("weights_verified") is True else ""))
            if not (commit != "unknown" and XFEAT_COMMIT.startswith(commit.lower())):
                print(f"  NOTE: XFeat commit {commit} is not the release configuration ({XFEAT_COMMIT})")

    cases = list_cases(cfg)
    check([os.path.basename(c) for c in cases] == [gt.CASE], f"expected exactly one case {gt.CASE}, found {cases}")
    for case in cases:
        name = os.path.basename(case)
        point = os.path.join(cfg["paths"]["point_root"], name)
        contour = os.path.join(cfg["paths"]["contour_root"], name)
        images = sorted(glob.glob(os.path.join(case, "image_*.png")), key=lambda p: int(section_index(p)))
        print(f"case {name}: {len(images)} sections")
        check(len(images) == gt.N, f"expected {gt.N} section images, found {len(images)}")
        v2 = [(pos, img) for pos, img in enumerate(images) if is_circular_spot(cfg, section_index(img))]
        check(len(v2) == gt.N // 2, f"expected {gt.N // 2} Visium V2 sections, found {len(v2)}")

        if mode == "real":
            stacks = [case] + list_patch_stacks(case)
            for stack in stacks:
                status = os.path.join(stack, "registration_xfeat", "xfeat_status.csv")
                if check(os.path.isfile(status), f"{status} missing: XFeat did not run"):
                    s = pd.read_csv(status)
                    check((s["status"] == "ok").all(), f"{status}: pairs without a feature-based affine")
                warps = glob.glob(os.path.join(stack, "registration_ANTs", "*", "step2_run_ants_reg", "output1Warp.nii.gz"))
                check(len(warps) == gt.N - 1, f"{stack}: {len(warps)} of {gt.N - 1} ANTs warp fields: ANTs did not run for every pair")
                done = glob.glob(os.path.join(stack, "registration_ANTs", "*", "step2_run_ants_reg", "ants_completed.json"))
                check(len(done) == gt.N - 1, f"{stack}: {len(done)} of {gt.N - 1} ANTs completion records")
            print(f"  XFeat status and ANTs outputs present for {len(stacks)} stacks")

        # Coordinates are stored as whole 4x pixels at every stage, so each stage adds up to ~0.7 px of
        # rounding; the tolerances below allow for it.
        means = {}
        for stage, sub in (("unregistered", ""), ("global", "registration_3D_ANTs"),
                           ("patch-refined", "registration_3D_patch_ANTs")):
            per_section = []                    # (pos, mean |p - g|, mean |p - expected|, mean |expected - g|)
            for pos, img in v2:
                csv = os.path.join(point, sub, os.path.basename(img).replace(".png", "_center.csv"))
                if not check(os.path.isfile(csv), f"{stage}: {csv} missing"):
                    continue
                pts, g = _match(csv)
                if not check(len(pts) > 0 and np.all(np.isfinite(pts)), f"{stage}: no registered spots in {csv}"):
                    continue
                expected = _apply_rc(gt.residual_transform(pos), g) if (stage == "global" and mode == "simulated") else g
                per_section.append((pos, float(np.mean(np.hypot(*(pts - g).T))),
                                    float(np.mean(np.hypot(*(pts - expected).T))),
                                    float(np.mean(np.hypot(*(expected - g).T)))))
            if not per_section:
                continue
            means[stage] = float(np.mean([e for _, e, _, _ in per_section]))
            line = f"  V2 spot centres, {stage:13s}: mean distance to the reference grid {means[stage]:6.2f} px"
            if stage == "global" and mode == "simulated":
                line += (f" (simulated residual {np.mean([r for *_, r in per_section]):.2f} px; "
                         f"distance to the expected position {np.mean([d for _, _, d, _ in per_section]):.2f} px)")
            print(line)
            if stage == "global" and mode == "simulated":
                check(np.mean([r for *_, r in per_section]) >= 2.0, "simulated residual too small to test the patch level")
                check(all(d < 0.8 for _, _, d, _ in per_section),
                      "global stage does not reproduce the simulated residual: " +
                      ", ".join(f"pos {p}: {d:.2f} px" for p, _, d, _ in per_section))
            elif stage == "global":
                check(means[stage] < 1.0, f"global registration error {means[stage]:.2f} px >= 1 px")
            elif stage == "patch-refined":
                check(all(e < (0.8 if mode == "simulated" else 1.0) for _, e, _, _ in per_section),
                      "patch-refined spots off the reference grid: " +
                      ", ".join(f"pos {p}: {e:.2f} px" for p, e, _, _ in per_section))
                if mode == "simulated" and "global" in means:
                    check(means[stage] < 0.5 * means["global"], "patch-level refinement did not remove the residual")

        tracks_file = os.path.join(contour, "Object_IoU_patch_connectglom", "IoU_Object.json")
        if check(os.path.isfile(tracks_file), f"{tracks_file} missing"):
            tracks = json.load(open(tracks_file))
            lengths = sorted(len(t["contours"]) for t in tracks)
            expected_lengths = sorted([gt.N] * (len(gt.OBJECTS) - 1) + [gt.N - 2])
            print(f"  3D objects: {len(tracks)} tracks, lengths {lengths} (expected {expected_lengths})")
            check(lengths == expected_lengths, "3D objects differ from the ground truth")

        best = []
        pairs = glob.glob(os.path.join(contour, "IoU_patchreg", "row_*_col_*.json"))
        for f in pairs:
            best += [max(m["iou"] for m in v) for v in json.load(open(f)).values()]
        expected_pairs = len(v2) * (len(v2) - 1) // 2
        check(len(pairs) == expected_pairs, f"{len(pairs)} of {expected_pairs} V2-V2 IoU tables")
        if check(bool(best), "no V2-V2 matches"):
            print(f"  V2-V2 spot matching: {len(best)} matched spots, median best IoU {np.median(best):.3f}")
            check(np.median(best) > 0.9, "median best IoU <= 0.9")

    print("\nDEMO", "PASSED" if check.ok else "FAILED")
    sys.exit(0 if check.ok else 1)


if __name__ == "__main__":
    main()
