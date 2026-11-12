"""Replot the single-panel SED for a checkpoint with envelope, profiled solid
curves, and the blue Eb sys. bands removed. Saved as '..._v1.png' alongside
the checkpoint.

Usage (from the repo root):
    python scripts/regen_sed_v1.py <checkpoint_dir_or_pkl>
"""
from __future__ import annotations

import gzip
import pickle
import sys
from pathlib import Path

# Works without installing the package: put <repo>/src on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")

import numpy as np

from crbreak.checkpoint import load_pickle
from crbreak.config.analysis import config_from_checkpoint
from crbreak.plotting.plot_sed import SedStyle, plot_sed

_KIND_DEFAULTS = {
    "PASS8": dict(color="cornflowerblue", marker="o"),
    "IEM": dict(color="tab:orange", marker="^"),
}


def _load(path: Path):
    with gzip.open(path, "rb") as f:
        return load_pickle(f)


def regen(ckpt_path: Path) -> None:
    s = _load(ckpt_path)
    outdir = ckpt_path.parent

    runs = s["runs"]
    cfg = config_from_checkpoint(s)
    prior_kind = s.get("prior_kind") if cfg.plots.stamp_prior_nuisance else None
    nuisance_mode = s.get("nuisance_mode") or s.get("stamp_nuisance_mode")
    stamp_nuisance_mode = s.get("stamp_nuisance_mode") or nuisance_mode

    glon_edges = np.asarray(s["glon_edges"], dtype=float)
    glat_lims = tuple(float(x) for x in s["glat_lims"])
    n_glon = int(glon_edges.size - 1)
    glon_tag = f"{int(glon_edges[0])}-{int(glon_edges[-1])}l_{n_glon}win"
    glat_tag = f"{int(glat_lims[0])}b{int(glat_lims[1])}"

    for r in runs:
        kind_u = r.kind.upper()
        is_pass8 = kind_u == "PASS8"
        data_label = "LAT subs." if is_pass8 else kind_u
        style = SedStyle(
            data_color=_KIND_DEFAULTS.get(kind_u, {}).get("color", "tab:gray"),
            data_label=data_label,
            ebreak_color=("cornflowerblue" if is_pass8 else "tab:orange"),
        )
        sed_path = outdir / f"{r.kind.lower()}_fit_sanity_{glat_tag}_{glon_tag}_v1.png"
        plot_sed(
            r.fitting,
            g=(r.g_prof or r.g),
            style=style,
            pl_prof_name=r.pl_prof_name,
            sbpl_prof_name=r.sbpl_prof_name,
            sed_scaling=cfg.data.sed_scaling,
            sed_band=cfg.plots.sed_band,
            cr_ebr_band=cfg.cr_reference.ebreak_band,
            prior_kind=prior_kind,
            nuisance_mode=stamp_nuisance_mode,
            show_envelope=False,
            show_profiled_curves=False,
            show_eb_uncorr_band=True,
            show_eb_corr_band=False,
            save=sed_path,
            show=False,
        )
        print(f"Saved: {sed_path.resolve()}")


if __name__ == "__main__":
    arg = Path(sys.argv[1])
    ckpt = arg if arg.suffix in {".gz", ".pkl"} else arg / "checkpoint.pkl.gz"
    regen(ckpt)
