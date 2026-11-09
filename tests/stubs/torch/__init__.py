"""Stand-in for ``torch``, used only by tests/test_xfeat_status.py (put on PYTHONPATH of a subprocess).

``torch.hub.load(...)`` returns a matcher that gives 50 exact correspondences (a shift of 3, -2 px)
between two textured images and only 5 when either image is blank, so that ``xfeat_affine.py``
finds an affine for the first kind of pair and none for the second. With ``source="local"`` it runs
the entry point of ``<dir>/hubconf.py``; ``hub.get_dir()`` is ``$STUB_TORCH_HUB``.
"""
import importlib.util
import os
import sys

import numpy as np


class _T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def float(self):
        return _T(self.a.astype(np.float32))

    def __getitem__(self, k):
        return _T(self.a[k])

    def to(self, device):
        return self

    @property
    def shape(self):
        return self.a.shape


def from_numpy(a):
    return _T(a)


class cuda:
    @staticmethod
    def is_available():
        return False


class _Matcher:
    def match_xfeat(self, a, b, top_k=4096):
        n = 5 if (a.a.std() < 1e-6 or b.a.std() < 1e-6) else 50
        h, w = a.a.shape[-2:]
        rng = np.random.default_rng(0)
        pts = np.float32(np.stack([rng.uniform(20, w - 20, n), rng.uniform(20, h - 20, n)], axis=1))
        return pts, pts + np.float32([3.0, -2.0])


class hub:
    @staticmethod
    def get_dir():
        return os.environ.get("STUB_TORCH_HUB", os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub"))

    @staticmethod
    def load(repo_or_dir, model, *args, source="github", **kwargs):
        if source == "local":                      # like torch.hub: run the entry point of <dir>/hubconf.py
            sys.path.insert(0, repo_or_dir)
            try:
                spec = importlib.util.spec_from_file_location("hubconf", os.path.join(repo_or_dir, "hubconf.py"))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return getattr(module, model)(*args, **kwargs)
            finally:
                sys.path.remove(repo_or_dir)
        return _Matcher()


class Tensor:
    pass
