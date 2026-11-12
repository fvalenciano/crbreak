"""Read the ``checkpoint.pkl.gz`` written by a run (:func:`crbreak.fitting.pipeline.run`).

A checkpoint is a gzip-compressed pickle of a ``dict``: the merged config, a few flat
keys (``glon_edges``, ``kinds``, ...) and ``runs``, one bundle per dataset with the
:class:`~crbreak.fitting.main.Fitting` (per-window fits), the global fits and the scan.

Checkpoints written before the public release stored the classes under ``src.crbreak``
and the pipeline module as ``fitting.main_test``; both are mapped here, so the paper
checkpoints in ``results/paper1`` load with the released package.
"""
from __future__ import annotations

import gzip
import pickle
from pathlib import Path

_RENAMED_MODULES = {
    "crbreak.fitting.main_test": "crbreak.fitting.pipeline",
    "crbreak.data_extract.main_test": "crbreak.data_extract.diagnostics",
}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("src.crbreak"):
            module = module[len("src."):]
        return super().find_class(_RENAMED_MODULES.get(module, module), name)


def load_pickle(f):
    """Unpickle an open (binary) checkpoint file."""
    return _Unpickler(f).load()


def load_checkpoint(path: str | Path) -> dict:
    """State ``dict`` of a run; ``path`` is the ``checkpoint.pkl.gz`` or its folder."""
    path = Path(path)
    if path.is_dir():
        path = path / "checkpoint.pkl.gz"
    with gzip.open(path, "rb") as f:
        return load_pickle(f)
