"""Provenance of external code and weights (XFeat via torch.hub): what exactly was used in a run."""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from typing import Optional

COMMIT_RE = re.compile(r"[0-9a-fA-F]{7,40}")

# XFeat release configuration of THiRST (configs/example.yaml, demo; see docs/PROVENANCE.md, "XFeat
# version"): the head of verlab/accelerated_features on 2026-11-09 and the SHA-256 of the weights that
# its torch.hub entry point downloads (weights/xfeat.pt, unchanged since the release commit 9539f1f).
XFEAT_REPO = "verlab/accelerated_features"
XFEAT_COMMIT = "e92685f57f8318b18725c5c8c0bd28c7fe188d9a"
XFEAT_WEIGHTS_SHA256 = "0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b"


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha256(root: str) -> str:
    """SHA-256 over the relative path and content of every file under ``root``.

    ``.git``, ``__pycache__`` and ``*.pyc`` are skipped, so the digest identifies the source files
    themselves (the same files give the same digest wherever and however they were obtained).
    """
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in (".git", "__pycache__"))
        for name in sorted(filenames):
            if name.endswith(".pyc"):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            h.update(rel.encode() + b"\0" + file_sha256(path).encode() + b"\n")
    return h.hexdigest()


def git_commit(root: str) -> Optional[str]:
    """``git rev-parse HEAD`` of ``root`` if it is a git checkout, else None."""
    if not os.path.exists(os.path.join(root, ".git")):
        return None
    try:
        out = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def find_code_root(path: Optional[str], marker: str = "hubconf.py") -> Optional[str]:
    """Closest folder above ``path`` that contains ``marker`` (the root of a torch.hub repository)."""
    if not path:
        return None
    d = os.path.dirname(os.path.abspath(path))
    while True:
        if os.path.isfile(os.path.join(d, marker)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def pinned_ref(hub_source: str) -> Optional[str]:
    """The ``<ref>`` of ``owner/repo:<ref>`` (None for a local folder or an unpinned source)."""
    if os.path.isdir(hub_source):
        return None
    return hub_source.partition(":")[2] or None
