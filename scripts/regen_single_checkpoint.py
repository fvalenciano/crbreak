"""Replot params-vs-glon + contour from a single checkpoint.

Usage (from the repo root):
    python scripts/regen_single_checkpoint.py <checkpoint_dir_or_pkl>

If a directory is given, looks for checkpoint.pkl.gz inside it.
Output is saved alongside the checkpoint.
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

from crbreak.checkpoint import load_pickle
from crbreak.config.analysis import config_from_checkpoint
from crbreak.plotting.params_vs_glon_plus_contour import (
    plot_params_vs_glon_plus_contour,
)
from crbreak.plotting.plot_contour_plot import ContourDataset
from crbreak.plotting.plot_params_vs_glon import ParamsDataset
from crbreak.plotting.plot_sed import SedStyle
from crbreak.plotting.plot_sed_plus_nuisances import plot_sed_plus_nuisances

FERMI_LAT             = r"$\mathit{Fermi}$-LAT"
FERMI_LAT_UNSUBTRACTED = r"$\mathit{Fermi}$-LAT unsubtracted"


def _load(path: Path):
    with gzip.open(path, "rb") as f:
        return load_pickle(f)


def _kind_color(kind: str) -> str:
    return {"PASS8": "cornflowerblue", "IEM": "tab:orange"}.get(
        str(kind).upper(), "grey"
    )


def _kind_marker(kind: str) -> str:
    return {"PASS8": "o", "IEM": "s"}.get(str(kind).upper(), "^")


def replot(ckpt_path: Path) -> None:
    state = _load(ckpt_path)
    outdir = ckpt_path.parent

    runs         = state["runs"]
    cfg          = config_from_checkpoint(state)
    cr           = cfg.cr_reference
    prior_kind   = state.get("prior_kind", "box_gauss") if cfg.plots.stamp_prior_nuisance else None
    nuisance_mode = state.get("stamp_nuisance_mode", state.get("nuisance_mode", "indep"))
    glat_lims    = tuple(float(x) for x in state["glat_lims"])
    glon_edges   = list(state["glon_edges"])
    smooth_sigma = float(state.get("scan_smooth_sigma", 1.0))

    glat_tag = f"0b{int(glat_lims[1])}"
    glon_tag = f"0-{int(glon_edges[-1])}l"
    n_wins   = len(glon_edges) - 1

    def _kind_label(kind: str) -> str:
        return {
            "PASS8": FERMI_LAT,
            "IEM":   "IEM",
        }.get(str(kind).upper(), str(kind).upper())

    params_specs = [
        ParamsDataset(
            label=_kind_label(r.kind),
            fitting=r.fitting,
            sbpl_prof_name=r.sbpl_prof_name,
            color=_kind_color(r.kind),
            marker=_kind_marker(r.kind),
            glon_offset_deg=(0.0 if i == 0 else (1.0 if i == 1 else -1.0)),
        )
        for i, r in enumerate(runs)
    ]

    raw_specs = []
    for r in runs:
        if getattr(r, "raw_fitting", None) is None or getattr(r, "raw_sbpl_name", None) is None:
            continue
        if str(r.kind).upper() != "PASS8":
            continue
        raw_specs.append(
            ParamsDataset(
                label=FERMI_LAT_UNSUBTRACTED,
                fitting=r.raw_fitting,
                sbpl_name="SBPL",
                sbpl_prof_name=(
                    r.raw_sbpl_name if "prof." in str(r.raw_sbpl_name) else None
                ),
                color="cornflowerblue",
                marker="o",
                glon_offset_deg=-2.0,
                edge_color="cornflowerblue",
                fill=False,
                error_ls=":",
            )
        )

    contour_specs = []
    for r in runs:
        if r.scan is None:
            continue
        g = r.g_prof if r.g_prof is not None else r.g
        kind = str(r.kind).upper()
        if kind == "PASS8":
            contour_specs.append(ContourDataset.pass8(r.scan, g=g, label=FERMI_LAT))
        elif kind == "IEM":
            contour_specs.append(ContourDataset.iem(r.scan, g=g, label="IEM"))
        else:
            contour_specs.append(ContourDataset(label=kind, scan=r.scan, g=g))

    if len(runs) == 1:
        stem = f"{runs[0].kind.lower()}_fit_sanity_{glat_tag}_{glon_tag}_{n_wins}win"
    else:
        stem = f"overlay_{'_vs_'.join(r.kind.lower() for r in runs)}_{glat_tag}_{glon_tag}_{n_wins}win"

    # SED + nuisance combined plot
    for r in runs:
        kind = str(r.kind).upper()
        sed_style = SedStyle(
            data_color=_kind_color(r.kind),
            data_label=FERMI_LAT if kind == "PASS8" else ("IEM" if kind == "IEM" else kind),
            ebreak_color=("cornflowerblue" if kind == "PASS8" else "tab:orange"),
        )
        sed_nuis_path = outdir / f"{r.kind.lower()}_sed_nuisance_{glat_tag}_{glon_tag}_{n_wins}win.png"
        plot_sed_plus_nuisances(
            r.fitting,
            g=r.g_prof if r.g_prof is not None else r.g,
            sed_style=sed_style,
            pl_prof_name=r.pl_prof_name,
            sbpl_prof_name=r.sbpl_prof_name,
            sed_scaling=cfg.data.sed_scaling,
            sed_band=cfg.plots.sed_band,
            nuisance_band=cfg.plots.nuisance_band,
            cr_ebr_band=cr.ebreak_band,
            cr_ebr_fixed=cr.ebreak_gev,
            prior_kind=prior_kind,
            nuisance_mode=nuisance_mode,
            panel_size=(7.0, 5.4),
            save=sed_nuis_path,
            show=False,
        )
        print(f"Saved: {sed_nuis_path.resolve()}")

    out_path = outdir / f"{stem}_params_plus_contour.png"
    plot_params_vs_glon_plus_contour(
        params_datasets=(raw_specs + params_specs),
        contour_datasets=contour_specs,
        smooth_sigma=smooth_sigma,
        levels=cfg.plots.contour_levels,
        cr_ref=(cr.ebreak_gev, cr.dgamma),
        cr_ebr_band=cr.ebreak_band,
        cr_dgamma_band=cr.dgamma_band,
        prior_kind=prior_kind,
        nuisance_mode=nuisance_mode,
        layout="vertical",
        figsize=(7.0, 10.0),
        cr_label="Local CR Break",
        save=out_path,
        show=False,
        highb_text=glat_lims[1],
    )
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "results/paper1/baseline"
    )
    ckpt = arg if arg.suffix in {".gz", ".pkl"} else arg / "checkpoint.pkl.gz"
    replot(ckpt)
