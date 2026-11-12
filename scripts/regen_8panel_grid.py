"""Combined 4-row x 2-column figure: multiwindow (9l/4l/18l) + model comparison
(x2 PS / raw / IEM) + prior comparison (piecewise smoothness / gaussian), all PASS8.

Each cell = stacked Ebreak(top)+Delta-gamma(bottom) vs |l| (left) + Delta-chi^2
contour (right).  Columns share the |l| x-axis (only the bottom row shows tick
labels).  The case title lives as text inside the contour panel (upper area),
with the legend placed just below it.

Usage (from the repo root):
    python scripts/regen_8panel_grid.py
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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FERMI_LAT            = "LAT"
FERMI_LAT_SUBTRACTED = "LAT subs."

# Row-major: [ (row0 left, row0 right), (row1 left, row1 right), ... ]
CASES = [
    dict(ckpt="results/paper1/b0-2/checkpoint.pkl.gz",
         title=r"Latitude cut $|b|<2^\circ$",
         label=FERMI_LAT_SUBTRACTED, color="cornflowerblue", fill=True),
    dict(ckpt="results/paper1/windows4/checkpoint.pkl.gz",
         title="Galactic plane: 4 windows",
         label=FERMI_LAT_SUBTRACTED, color="cornflowerblue", fill=True),
    dict(ckpt="results/paper1/windows18/checkpoint.pkl.gz",
         title="Galactic plane: 18 windows",
         label=FERMI_LAT_SUBTRACTED, color="cornflowerblue", fill=True),
    dict(ckpt="results/paper1/ps_x2/checkpoint.pkl.gz",
         title=r"$\times$2 Point-source contribution",
         label=FERMI_LAT_SUBTRACTED, color="cornflowerblue", fill=True),
    dict(ckpt="results/paper1/unsubtracted/checkpoint.pkl.gz",
         title=FERMI_LAT,
         label=FERMI_LAT, color="cornflowerblue", fill=False),
    dict(ckpt="results/paper1/iem/checkpoint.pkl.gz",
         title="IEM template",
         label="IEM", color="tab:orange", fill=True),
    dict(ckpt="results/paper1/piecewise/checkpoint.pkl.gz",
         title="Piecewise function",
         label=FERMI_LAT_SUBTRACTED, color="cornflowerblue", fill=True),
    dict(ckpt="results/paper1/gaussian_prior/checkpoint.pkl.gz",
         title="Gaussian prior",
         label=FERMI_LAT_SUBTRACTED, color="cornflowerblue", fill=True),
]
N_ROWS, N_COLS = 4, 2
OUTDIR = Path("results/paper1/figures")

EBREAK_YLIM   = (0.0, 70.0)
DGAMMA_YLIM   = (0.0, 0.49)
CONTOUR_XLIM  = (0.0, 70.0)

LEGEND_FS = 24
TEXT_FS   = 27
TITLE_FS  = 27
AXIS_FS   = 30
TICK_FS   = 25

from crbreak.checkpoint import load_pickle
from crbreak.config.analysis import config_from_checkpoint
from crbreak.plotting.plot_contour_plot import (
    ContourDataset,
    _DEFAULT_LEVEL_LABELS,
    _draw_one_dataset,
    _normalise_datasets as _norm_contour,
    _asym_err,
)
from crbreak.plotting.plot_params_vs_glon import (
    ParamsDataset,
    _gather,
    _normalise_datasets as _norm_params,
    _plot_one,
)
from crbreak.plotting._run_stamp import stamp_prior_nuisance


def _load(path):
    with gzip.open(path, "rb") as f:
        return load_pickle(f)


def _params_ds(r, case):
    return [ParamsDataset(
        label=case["label"],
        fitting=r.fitting,
        sbpl_prof_name=r.sbpl_prof_name,
        color=case["color"],
        marker="o",
        glon_offset_deg=0.0,
        fill=case["fill"],
    )]


def _contour_ds(r):
    g = r.g_prof if r.g_prof is not None else r.g
    kind = str(r.kind).upper()
    if kind == "IEM":
        return ContourDataset.iem(r.scan, g=g, label="IEM")
    return ContourDataset.pass8(r.scan, g=g, label=FERMI_LAT_SUBTRACTED)


def _z_values(r):
    g = r.g_prof if r.g_prof is not None else r.g
    z_w = float("nan")
    try:
        wilks = getattr(g, "wilks_results", {}) or {}
        wk = next((k for k in wilks if str(k).startswith("PL|prof.")), None)
        if wk is None and "PL" in wilks:
            wk = "PL"
        if wk is not None:
            z_w = float(wilks[wk].get("Z_wilks", float("nan")))
    except Exception:
        pass
    wilks_txt = rf"$Z_{{\rm Wilks}}={z_w:.2f}\,\sigma$" if np.isfinite(z_w) \
        else r"$Z_{\rm Wilks}=\mathrm{---}$"

    z_mc = float("nan")
    z_mc_ul = False
    try:
        mc_res = getattr(g, "mc_results", {}) or {}
        mk = next((k for k in mc_res if str(k).startswith("PL|prof.")), None)
        if mk is None and "PL" in mc_res:
            mk = "PL"
        if mk is not None:
            mc_obj = mc_res[mk]
            z_mc = float(getattr(mc_obj, "Z", float("nan")))
            z_mc_ul = bool(getattr(mc_obj, "p_is_upper_limit", False))
    except Exception:
        pass
    if np.isfinite(z_mc):
        op = ">" if z_mc_ul else "="
        mc_txt = rf"$Z_{{\rm MC}}{op}{z_mc:.2f}\,\sigma$"
    else:
        mc_txt = r"$Z_{\rm MC}=\mathrm{---}$"
    return wilks_txt, mc_txt


# ---------------------------------------------------------------------------
# Phase 1: load everything + per-case glon range
# ---------------------------------------------------------------------------
cells = []
for case in CASES:
    state = _load(case["ckpt"])
    r = state["runs"][0]
    pds = _norm_params(_params_ds(r, case))
    cds = _norm_contour(_contour_ds(r))

    glon_min, glon_max = +np.inf, -np.inf
    gathered = []
    for ds in pds:
        x, xerr, Eb, Eb_err, dg, dg_err = _gather(ds)
        gathered.append((x, xerr, Eb, Eb_err, dg, dg_err, ds))
        if x.size:
            glon_min = min(glon_min, float(x.min() - xerr.max() - 1.0))
            glon_max = max(glon_max, float(x.max() + xerr.max() + 1.0))

    cells.append(dict(case=case, state=state, r=r, pds=gathered, cds=cds,
                       glon_min=glon_min, glon_max=glon_max))

# Per-column shared |l| range (union over the column's 4 rows).
col_xlim = []
for c in range(N_COLS):
    idxs = [c + N_COLS * row for row in range(N_ROWS)]
    lo = min(cells[i]["glon_min"] for i in idxs)
    hi = max(cells[i]["glon_max"] for i in idxs)
    col_xlim.append((lo if np.isfinite(lo) else 0.0, hi if np.isfinite(hi) else 90.0))

# ---------------------------------------------------------------------------
# Phase 2: figure + axes grid
# ---------------------------------------------------------------------------
ROW_GAP = 0.030
COL_GAP = 0.075

fig = plt.figure(figsize=(25.0, 26.0))

left_fig, right_fig = 0.040, 0.975
bottom_fig, top_fig = 0.040, 0.975

total_w  = right_fig - left_fig
col_w    = (total_w - COL_GAP) / N_COLS
params_w = col_w * 0.50
contour_w = col_w - params_w

total_h = top_fig - bottom_fig
row_h   = (total_h - (N_ROWS - 1) * ROW_GAP) / N_ROWS
half_h  = row_h / 2.0

col_dg_ref: list = [None, None]
col_c_ref: list = [None, None]
axes_grid = []  # [(ax_e, ax_dg, ax_c, cell, is_bottom, col)]

for row in range(N_ROWS):
    y0 = top_fig - (row + 1) * row_h - row * ROW_GAP
    is_bottom = (row == N_ROWS - 1)
    for col in range(N_COLS):
        idx = col + N_COLS * row
        cell = cells[idx]
        x_left = left_fig if col == 0 else (left_fig + col_w + COL_GAP)

        dg_kwargs = {} if col_dg_ref[col] is None else {"sharex": col_dg_ref[col]}
        ax_dg = fig.add_axes([x_left, y0, params_w, half_h], **dg_kwargs)
        ax_e  = fig.add_axes([x_left, y0 + half_h, params_w, half_h], sharex=ax_dg)
        if col_dg_ref[col] is None:
            col_dg_ref[col] = ax_dg

        c_kwargs = {} if col_c_ref[col] is None else {"sharex": col_c_ref[col]}
        ax_c = fig.add_axes([x_left + params_w, y0, contour_w, row_h], **c_kwargs)
        if col_c_ref[col] is None:
            col_c_ref[col] = ax_c

        axes_grid.append((ax_e, ax_dg, ax_c, cell, is_bottom, col))

# Apply the shared |l| xlim once per column (propagates to every linked axis).
for col in range(N_COLS):
    col_dg_ref[col].set_xlim(*col_xlim[col])
    col_c_ref[col].set_xlim(*CONTOUR_XLIM)

# ---------------------------------------------------------------------------
# Phase 3: fill each cell
# ---------------------------------------------------------------------------
for ax_e, ax_dg, ax_c, cell, is_bottom, col in axes_grid:
    case, state, r = cell["case"], cell["state"], cell["r"]
    glon_xlim = (cell["glon_min"], cell["glon_max"])
    cfg = config_from_checkpoint(state)  # plot references (CR bands, contour levels, stamp)
    cr = cfg.cr_reference

    for x, xerr, Eb, Eb_err, dg, dg_err, ds in cell["pds"]:
        _plot_one(ax_e,  x, xerr, Eb, Eb_err, ds)
        _plot_one(ax_dg, x, xerr, dg, dg_err, ds)

    ax_e.fill_between(list(glon_xlim), *cr.ebreak_band,
                       color="grey", alpha=0.30, linewidth=0)
    ax_dg.fill_between(list(glon_xlim), *cr.dgamma_band,
                        color="grey", alpha=0.30, linewidth=0)

    ax_e.set_ylim(*EBREAK_YLIM)
    ax_dg.set_ylim(*DGAMMA_YLIM)

    for ax in (ax_e, ax_dg):
        ax.tick_params(axis="both", which="major", direction="in",
                        labelsize=TICK_FS, length=6, width=1.8)

    ax_e.tick_params(axis="x", labelbottom=False)
    ax_e.spines["bottom"].set_linewidth(1.4)
    ax_dg.spines["top"].set_linewidth(1.4)
    ax_e.spines["right"].set_visible(False)
    ax_dg.spines["right"].set_visible(False)

    ax_e.set_ylabel(r"$E_{\rm break}\,[\mathrm{GeV}]$", fontsize=AXIS_FS)
    ax_dg.set_ylabel(r"$\Delta\gamma$", fontsize=AXIS_FS)

    if is_bottom:
        ax_dg.set_xlabel(r"$|\ell|\,[^{\circ}]$", fontsize=AXIS_FS, labelpad=8)
    else:
        ax_dg.tick_params(axis="x", labelbottom=False)

    handles, labels = ax_e.get_legend_handles_labels()
    if handles:
        ax_e.legend(handles, labels, fontsize=LEGEND_FS, frameon=False,
                    loc="upper left", ncol=1,
                    handlelength=1.4, handletextpad=0.4, labelspacing=0.25,
                    borderaxespad=0.2)

    stamp_prior_nuisance(ax_e, prior_kind=(state["prior_kind"] if cfg.plots.stamp_prior_nuisance else None),
                          nuisance_mode=state["stamp_nuisance_mode"], color="0.25")

    # ---- contour panel ----
    cds = cell["cds"]
    xs = np.asarray(cds[0].scan.Ebreak_grid, float)
    ys = np.asarray(cds[0].scan.dgamma_grid, float)
    ylim_c = (float(ys.min()), float(ys.max()))

    _draw_one_dataset(ax_c, cds[0],
                       levels=cfg.plots.contour_levels,
                       level_labels=_DEFAULT_LEVEL_LABELS,
                       smooth_sigma=float(state["scan_smooth_sigma"]),
                       label_contours=True)

    ax_c.errorbar(
        [cr.ebreak_gev], [cr.dgamma],
        xerr=_asym_err(cr.ebreak_gev, cr.ebreak_band),
        yerr=_asym_err(cr.dgamma, cr.dgamma_band),
        fmt="o", color="grey", ecolor="grey", mec="black", mew=0.8,
        ms=9, elinewidth=1.2, capsize=3, linestyle="none", zorder=5,
    )

    ax_c.set_ylim(*ylim_c)
    ax_c.tick_params(axis="both", which="major", direction="in",
                      labelsize=TICK_FS, length=6, width=1.8)

    ax_c.yaxis.set_label_position("right")
    ax_c.yaxis.tick_right()
    ax_c.set_ylabel(r"$\Delta\gamma$", fontsize=AXIS_FS, labelpad=10)

    if is_bottom:
        ax_c.set_xlabel(r"$E_{\rm break}\,[\mathrm{GeV}]$", fontsize=AXIS_FS)
    else:
        ax_c.tick_params(axis="x", labelbottom=False)

    # Title + Z stats in the upper area of the contour panel; legend at the bottom.
    ax_c.text(0.04, 0.97, case["title"], transform=ax_c.transAxes,
               va="top", ha="left", fontsize=TITLE_FS, wrap=True)

    wilks_txt, mc_txt = _z_values(r)
    ax_c.text(0.04, 0.87, wilks_txt + "\n" + mc_txt,
              transform=ax_c.transAxes, va="top", ha="left",
              fontsize=TEXT_FS, color="0.25")

    face = cds[0].band_rgba[0] if cds[0].band_rgba else cds[0].line_color
    leg_handles = [
        Patch(facecolor=face, edgecolor="none", label=case["label"]),
        Line2D([0], [0], marker="o", color="grey", mec="black", lw=0, ms=9,
               label="Local CR Break"),
    ]
    leg_ncol = 1 if case["label"] == FERMI_LAT else 2
    ax_c.legend(handles=leg_handles, frameon=False,
                loc="lower left", bbox_to_anchor=(0.0, 0.02), bbox_transform=ax_c.transAxes,
                fontsize=LEGEND_FS, ncol=leg_ncol,
                handlelength=1.4, handletextpad=0.4, columnspacing=0.8,
                borderaxespad=0.2)

    highb = float(state["glat_lims"][1])
    ax_c.text(0.96, 0.88, rf"$|b|<{highb:g}^\circ$",
              transform=ax_c.transAxes, va="top", ha="right", fontsize=TEXT_FS,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="none"))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)
out = OUTDIR / "pass8_8panel_grid_params_plus_contour.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved: {out.resolve()}")
plt.close(fig)
