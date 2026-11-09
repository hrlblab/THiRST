"""Checks on the ANTs pairwise transforms that the coordinate-transfer scripts chain together.

A section is mapped to the middle (reference) section by composing the affine of every adjacent pair
in between: ``<stack>/registration_ANTs/<moving>_to_<fixed>/step2_run_ants_reg/output0GenericAffine.mat``.
Leaving out a missing pair would still produce coordinates, just wrong ones, so the scripts stop
instead.
"""
from __future__ import annotations

import os
from typing import List, Sequence

TRANSFORM_NAME = os.path.join("step2_run_ants_reg", "output0GenericAffine.mat")


class MissingTransformError(FileNotFoundError):
    """An ANTs pairwise transform needed by a section-to-reference chain does not exist."""


def pair_transform_path(registration_dir: str, moving_index, fixed_index) -> str:
    """``<registration_dir>/<moving>_to_<fixed>/step2_run_ants_reg/output0GenericAffine.mat``."""
    return os.path.join(registration_dir, f"{moving_index}_to_{fixed_index}", TRANSFORM_NAME)


def require_transform(path: str) -> str:
    """Return ``path`` if the ANTs affine exists, otherwise raise :class:`MissingTransformError`."""
    if not os.path.isfile(path):
        raise MissingTransformError(
            f"missing ANTs transform {path}. Every adjacent section pair between a section and the middle "
            "(reference) section needs one. Re-run scripts/1_registration (xfeat_affine.py, "
            "affine_to_itk.py, ants_affine.py) for this stack at the same --level and check their "
            "summaries for failed pairs.")
    return path


def is_valid_transform(path: str) -> bool:
    """True if ``path`` is a readable 2D affine with finite parameters and an invertible matrix."""
    if not os.path.isfile(path):
        return False
    try:
        import numpy as np
        import SimpleITK as sitk
        t = sitk.ReadTransform(path)
        params = np.asarray(t.GetParameters(), dtype=float)
        fixed = np.asarray(t.GetFixedParameters(), dtype=float)
    except Exception:
        return False
    if params.size != 6 or not np.all(np.isfinite(params)) or not np.all(np.isfinite(fixed)):
        return False
    return abs(float(np.linalg.det(params[:4].reshape(2, 2)))) > 1e-8


def section_number(image_path: str) -> int:
    """Section number of ``image_<section>[_<x0>_<y0>].png``."""
    return int(os.path.basename(image_path).split(".")[0].split("_")[1])


def chain_pairs(images: Sequence[str]) -> List[str]:
    """``<moving>_to_<fixed>`` names of all adjacent pairs of a sorted section stack."""
    idx = [section_number(p) for p in images]
    return [f"{m}_to_{f}" for f, m in zip(idx[:-1], idx[1:])]


def bad_pairs(registration_dir: str, images: Sequence[str]) -> List[str]:
    """Adjacent pairs of the stack whose transform is missing or invalid in ``registration_dir``."""
    return [p for p in chain_pairs(images)
            if not is_valid_transform(os.path.join(registration_dir, p, TRANSFORM_NAME))]


def require_chain(registration_dir: str, images: Sequence[str]) -> None:
    """Check, before anything is written, that every adjacent pair of the stack has a valid transform.

    Raises :class:`MissingTransformError` naming all missing or unreadable pairs.
    """
    bad = bad_pairs(registration_dir, images)
    if bad:
        raise MissingTransformError(
            f"{registration_dir}: missing or invalid ANTs transform for pair(s) {bad}. Every adjacent "
            "section pair needs one before coordinates can be mapped to the reference section; re-run "
            "scripts/1_registration (xfeat_affine.py, affine_to_itk.py, ants_affine.py) for this stack "
            "and check their summaries for failed pairs.")


def require_patch_chains(case_dir: str, images: Sequence[str]) -> None:
    """:func:`require_chain` for every patch stack of a case (``<case>/3D_Patch_crop/<x>_<y>/``)."""
    from thirst.config import list_patch_stacks
    stacks = list_patch_stacks(case_dir)
    if not stacks:
        raise MissingTransformError(f"{case_dir}: no patch stacks in 3D_Patch_crop/; run "
                                    "sample_patch_centers.py, crop_patch_stacks.py and the patch-level "
                                    "registration first")
    problems = [f"{os.path.basename(st)}: {bad}" for st in stacks
                for bad in [bad_pairs(os.path.join(st, "registration_ANTs"), images)] if bad]
    if problems:
        raise MissingTransformError(
            f"{case_dir}: {len(problems)} of {len(stacks)} patch stack(s) lack valid transforms for "
            "pairs (re-run the patch-level registration and check its summaries):\n  " + "\n  ".join(problems))
