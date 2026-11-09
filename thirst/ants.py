"""ANTs wrapper used for pairwise section registration (global and patch level).

This is ``register_a_pair`` from the original ``superglue_ants_registration_onpair_new_RGB.py``
(the only function of that module used by the pipeline). For every moving->fixed pair it:

1. ``step1_prepare_data/``  copies both PNGs and converts them to NIfTI (``ConvertImagePixelType``);
2. ``step2_run_ants_reg/``  runs ``antsRegistrationSyN.sh -t s`` (rigid + affine + SyN, mutual
   information), initialised with the XFeat affine ``<working_dir>/sg_affine_init.mat``.

Downstream scripts use only ``step2_run_ants_reg/output0GenericAffine.mat`` (the affine part); the SyN
warp fields are written but not applied by any script in this repository.

A failed ANTs command or a missing output raises :class:`ANTsRegistrationError`. A pair that is
(re)computed starts from an empty ``step2_run_ants_reg/``; if ANTs fails, whatever it wrote is moved
to ``step2_run_ants_reg.failed/`` (kept for inspection, never read by the pipeline), so a failed run
cannot leave a transform behind. Only a successful run writes ``step2_run_ants_reg/ants_completed.json``
(command, ANTs version, SHA-256 of the affine); :func:`is_completed` requires it, so a restart never
mistakes the output of a failed or interrupted run for a result.

The ANTs binaries are located from (in order) the ``ants_bin_dir`` argument
(``paths.ants_bin`` in the config), the ``ANTSPATH`` environment variable, or ``PATH``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from typing import Optional

from thirst.transforms import is_valid_transform


def find_ants_bin(ants_bin_dir: Optional[str] = None) -> str:
    """Return the folder that contains ``antsRegistrationSyN.sh`` and ``ConvertImagePixelType``."""
    for candidate in (ants_bin_dir, os.environ.get("ANTSPATH")):
        if candidate and os.path.isfile(os.path.join(candidate, "antsRegistrationSyN.sh")):
            return candidate.rstrip("/")
    found = shutil.which("antsRegistrationSyN.sh")
    if found:
        return os.path.dirname(found)
    raise FileNotFoundError(
        "ANTs not found. Install ANTs (https://github.com/ANTsX/ANTs) and set 'paths.ants_bin' in the "
        "config, export ANTSPATH, or put antsRegistrationSyN.sh on PATH.")


class ANTsRegistrationError(RuntimeError):
    """ANTs failed for a section pair (non-zero exit status or missing output)."""


COMPLETION_RECORD = "ants_completed.json"


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def ants_version(ants_bin_dir: str) -> str:
    """First line of ``antsRegistration --version`` (``'unknown'`` if it cannot be run)."""
    try:
        out = subprocess.run([os.path.join(ants_bin_dir, "antsRegistration"), "--version"],
                             capture_output=True, text=True, timeout=60)
        lines = (out.stdout or out.stderr).strip().splitlines()
        return lines[0].strip() if lines else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def affine_path(working_dir: str) -> str:
    return os.path.join(working_dir, "step2_run_ants_reg", "output0GenericAffine.mat")


def is_completed(working_dir: str) -> bool:
    """True if :func:`register_a_pair` finished this pair successfully: its completion record exists and
    matches the (valid) affine that is there now."""
    affine = affine_path(working_dir)
    record = os.path.join(working_dir, "step2_run_ants_reg", COMPLETION_RECORD)
    if not (os.path.isfile(record) and is_valid_transform(affine)):
        return False
    try:
        with open(record) as f:
            info = json.load(f)
    except (OSError, ValueError):
        return False
    return info.get("exit_status") == 0 and info.get("affine_sha256") == _sha256(affine)


def _run(cmd, cwd=None):
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise ANTsRegistrationError(f"exit status {result.returncode}: {' '.join(cmd)}")
    return result.returncode


def _convert(convert_cmd: str, src: str, dst: str) -> None:
    """ConvertImagePixelType into a temporary file, renamed only on success (no partial file is reused)."""
    tmp = dst.replace(".nii.gz", ".partial.nii.gz")
    if os.path.exists(tmp):
        os.remove(tmp)
    try:
        _run([convert_cmd, src, tmp, '1'])
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, dst)


def register_a_pair(moving_jpg: str, fixed_jpg: str, working_dir: str,
                    ants_bin_dir: Optional[str] = None, version: Optional[str] = None) -> str:
    """Register ``moving_jpg`` to ``fixed_jpg`` with ANTs SyN, initialised by the XFeat affine.

    Returns the path of ``output0GenericAffine.mat``; raises :class:`ANTsRegistrationError` if ANTs fails
    (the partial outputs are then in ``step2_run_ants_reg.failed/``).
    """
    ants_bin_dir = find_ants_bin(ants_bin_dir)
    os.environ["ANTSPATH"] = ants_bin_dir

    output_name = 'output'

    # step 1 ==============================================================================
    step_1_dir = os.path.join(working_dir, 'step1_prepare_data')
    os.makedirs(step_1_dir, exist_ok=True)

    # copy raw data (kept for quality control)
    moving_jpg_file = os.path.join(step_1_dir, os.path.basename(moving_jpg))
    fix_jpg_file = os.path.join(step_1_dir, os.path.basename(fixed_jpg))
    print(moving_jpg, fixed_jpg)
    shutil.copyfile(moving_jpg, moving_jpg_file)
    shutil.copyfile(fixed_jpg, fix_jpg_file)

    # convert png to nifti (written under a temporary name and renamed on success)
    convert_cmd = os.path.join(ants_bin_dir, 'ConvertImagePixelType')
    moving_nii = os.path.join(step_1_dir, 'moving.nii.gz')  # moving image
    fix_nii = os.path.join(step_1_dir, 'fix.nii.gz')        # fixed image
    if not os.path.exists(moving_nii):
        _convert(convert_cmd, moving_jpg, moving_nii)
    if not os.path.exists(fix_nii):
        _convert(convert_cmd, fixed_jpg, fix_nii)

    sg_affine_file = os.path.join(working_dir, 'sg_affine_init.mat')
    if not os.path.isfile(sg_affine_file):
        raise FileNotFoundError(f"{sg_affine_file} not found: run xfeat_affine.py and affine_to_itk.py first")

    # step 2 ==============================================================================
    # The pair is (re)computed from scratch: no output of an earlier or failed run may survive.
    step_2_dir = os.path.join(working_dir, 'step2_run_ants_reg')
    failed_dir = step_2_dir + '.failed'
    for d in (step_2_dir, failed_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)
    os.makedirs(step_2_dir)
    syn_cmd = os.path.join(ants_bin_dir, 'antsRegistrationSyN.sh')
    output_file = os.path.join(step_2_dir, output_name)

    print(moving_nii)
    print(fix_nii)
    print(sg_affine_file)
    print(os.path.join(step_2_dir, '%sWarped.nii.gz' % output_name))

    # Outputs: output0GenericAffine.mat, output1Warp.nii.gz, output1InverseWarp.nii.gz,
    #          outputWarped.nii.gz, outputInverseWarped.nii.gz (all in step_2_dir)
    cmd = [syn_cmd, '-d', '2', '-m', moving_nii, '-f', fix_nii, '-j', '1', '-t', 's', '-n', '8',
           '-i', sg_affine_file, '-o', output_file]
    affine = os.path.join(step_2_dir, '%s0GenericAffine.mat' % output_name)
    try:
        _run(cmd, cwd=step_2_dir)
        if not is_valid_transform(affine):
            raise ANTsRegistrationError(f"antsRegistrationSyN.sh finished without a valid {affine}")
    except BaseException:
        os.rename(step_2_dir, failed_dir)   # quarantined: never read by the pipeline or taken as a result
        raise

    with open(os.path.join(step_2_dir, COMPLETION_RECORD), 'w') as f:
        json.dump({"command": [os.path.basename(cmd[0])] + cmd[1:], "ants_bin": ants_bin_dir,
                   "ants_version": version or ants_version(ants_bin_dir), "exit_status": 0,
                   "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "affine_sha256": _sha256(affine)}, f, indent=2)
    return affine
