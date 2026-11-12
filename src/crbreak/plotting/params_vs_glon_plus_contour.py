"""Combined figure: params vs |l| (Ebreak, Δγ) + contour plot.

This module is a convenience wrapper that lays out:

1) Top panel:    Ebreak vs |l|
2) Middle panel: Δγ vs |l| (shares x with top)
3) Bottom panel: Δχ²(Ebreak, Δγ) contour overlay

It reuses the same dataset specs as the standalone plotting helpers:

* :class:`~crbreak.plotting.plot_params_vs_glon.ParamsDataset`
* :class:`~crbreak.plotting.plot_contour_plot.ContourDataset`
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ._run_stamp import stamp_prior_nuisance
from .plot_contour_plot import (
    ContourDataset,
    _DEFAULT_LEVEL_LABELS,
    _draw_one_dataset,
    _normalise_datasets as _normalise_contour_datasets,
    _asym_err,
)
from .plot_params_vs_glon import (
    ParamsDataset,
    _gather,
    _normalise_datasets as _normalise_param_datasets,
    _plot_one,
)


def plot_params_vs_glon_plus_contour(
    *,
    params_datasets: Sequence[ParamsDataset] | ParamsDataset,
    contour_datasets: Sequence[ContourDataset] | ContourDataset,
    prior_kind: str | None = None,
    nuisance_mode: str | None = None,
    layout: str = "horizontal",  # "horizontal" (default) or "vertical" (legacy)
    # Top panels (params vs |l|) config
    cr_ebr_band: tuple[float, float] | None,
    cr_dgamma_band: tuple[float, float] | None,
    glon_xlim: tuple[float, float] | None = None,
    ebreak_ylim: tuple[float, float] | None = None,
    dgamma_ylim: tuple[float, float] = (0.0, 0.50),
    highb_text: float | None = None,
    # Bottom panel (contours) config
    smooth_sigma: float,
    levels: Sequence[float],
    level_labels: dict | None = None,
    label_contours: bool = True,
    cr_ref: tuple[float, float] | None,
    cr_color: str = "grey",
    cr_marker: str = "o",
    cr_marker_size: float = 9.0,
    cr_label: str = "Local CR Break",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    # Figure / output
    # Slightly shorter default to avoid a "tall" look in the horizontal layout.
    figsize: tuple[float, float] = (12.0, 4.8),
    # Legacy-like manual layout parameters (see legacy/make_final_plots.py).
    left: float = 0.14,
    right: float = 0.97,
    bottom: float = 0.08,
    gap_ec: float = 0.08,
    top_margin: float = 0.03,
    col_gap: float = 0.06,
    save: str | Path | None = None,
    show: bool = False,
    dpi: int = 200,
):
    """Combined params-vs-glon + contour figure (overlay-capable).

    Parameters
    ----------
    params_datasets
        One or more :class:`ParamsDataset` instances (PASS8/IEM overlay).
    contour_datasets
        One or more :class:`ContourDataset` instances for the bottom contour panel.
    """
    pds = _normalise_param_datasets(params_datasets)
    cds = _normalise_contour_datasets(contour_datasets)

    layout_n = str(layout or "horizontal").strip().lower()
    if layout_n not in {"horizontal", "vertical"}:
        raise ValueError(f"layout must be 'horizontal' or 'vertical', got {layout!r}.")

    fig = plt.figure(figsize=figsize)

    if layout_n == "vertical":
        # Legacy spacing: top block (two half-height axes) + bottom contour axis.
        width = float(right) - float(left)
        block_h = (1.0 - float(bottom) - float(top_margin) - float(gap_ec)) / 2.0
        contour_h = block_h
        top_block_h = block_h
        top_block_bottom = float(bottom) + contour_h + float(gap_ec)
        half_top_h = top_block_h / 2.0
        ax_dg = fig.add_axes([float(left), top_block_bottom, width, half_top_h])
        ax_e = fig.add_axes(
            [float(left), top_block_bottom + half_top_h, width, half_top_h], sharex=ax_dg
        )
        ax_c = fig.add_axes([float(left), float(bottom), width, contour_h])
    else:
        # Horizontal layout: params (two stacked axes) on the left, contour on the right.
        full_w = float(right) - float(left)
        full_h = 1.0 - float(bottom) - float(top_margin)
        # Make left (params) and right (contours) columns equal width.
        col_w = (full_w - float(col_gap)) / 2.0
        left_w = col_w
        right_w = col_w

        x0 = float(left)
        y0 = float(bottom)
        half_h = full_h / 2.0

        ax_dg = fig.add_axes([x0, y0, left_w, half_h])
        ax_e = fig.add_axes([x0, y0 + half_h, left_w, half_h], sharex=ax_dg)
        ax_c = fig.add_axes([x0 + left_w + float(col_gap), y0, right_w, full_h])

    # ------------------------------------------------------------------
    # Top two panels: Ebreak and Δγ vs |l|
    # ------------------------------------------------------------------
    all_eb_top: list[float] = []
    glon_min, glon_max = +np.inf, -np.inf
    for ds in pds:
        x, xerr, Eb, Eb_err, dg, dg_err = _gather(ds)
        _plot_one(ax_e, x, xerr, Eb, Eb_err, ds)
        _plot_one(ax_dg, x, xerr, dg, dg_err, ds)
        finite_eb = Eb[np.isfinite(Eb)]
        if finite_eb.size:
            all_eb_top.append(float(np.nanmax(Eb)))
        if x.size:
            glon_min = min(glon_min, float(x.min() - xerr.max() - 1.0))
            glon_max = max(glon_max, float(x.max() + xerr.max() + 1.0))

    if glon_xlim is None:
        if not np.isfinite(glon_min) or not np.isfinite(glon_max):
            glon_xlim = (0.0, 90.0)
        else:
            glon_xlim = (glon_min, glon_max)

    if cr_ebr_band is not None:
        lo, hi = cr_ebr_band
        ax_e.fill_between(list(glon_xlim), lo, hi, color="grey", alpha=0.30, linewidth=0)
    if cr_dgamma_band is not None:
        lo, hi = cr_dgamma_band
        ax_dg.fill_between(list(glon_xlim), lo, hi, color="grey", alpha=0.30, linewidth=0)

    ax_e.set_ylabel(r"$E_{\rm break}\,[\mathrm{GeV}]$", fontsize=20)
    if ebreak_ylim is None:
        top = max(80.0, 1.1 * (max(all_eb_top) if all_eb_top else 80.0))
        ax_e.set_ylim(0.0, top)
    else:
        ax_e.set_ylim(*ebreak_ylim)

    ax_dg.set_ylim(*dgamma_ylim)
    ax_dg.set_ylabel(r"$\Delta\gamma$", fontsize=20)
    ax_dg.set_xlabel(r"$|\ell|\,[^{\circ}]$", fontsize=20, labelpad=10)

    for ax in (ax_e, ax_dg):
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            labelsize=18,
            length=6,
            width=1.8,
        )
        ax.set_xlim(*glon_xlim)
    # Avoid duplicate |l| x tick labels on the top axis (legacy behaviour).
    ax_e.tick_params(axis="x", labelbottom=False)
    ax_e.spines["bottom"].set_linewidth(1.4)
    ax_dg.spines["top"].set_linewidth(1.4)

    # Top legend (datasets). Keep PASS8-raw just under PASS8 and avoid spilling
    # into the contour column in horizontal mode.
    handles, labels = ax_e.get_legend_handles_labels()
    if handles and labels:
        def _lab_key(lab: str) -> tuple[int, str]:
            s = str(lab).upper()
            if s == "PASS8":
                return (0, s)
            if "PASS8-RAW" in s or ("RAW" in s and "PASS8" in s):
                return (1, s)
            if s == "IEM":
                return (2, s)
            return (9, s)

        order = sorted(range(len(labels)), key=lambda i: _lab_key(labels[i]))
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]

        legend_ax = ax_e
        legend_ax.legend(
            handles,
            labels,
            fontsize=19,
            frameon=False,
            loc="upper left",
            # In the typical overlay (PASS8, PASS8-raw, IEM) keep everything on one row.
            ncol=(min(len(labels), 3) if layout_n == "horizontal" else max(1, len(pds))),
            handlelength=1.6,
            handletextpad=0.5,
            columnspacing=1.1,
            labelspacing=0.30,
            borderaxespad=0.25,
        )

    stamp_prior_nuisance(
        ax_e,
        prior_kind=prior_kind,
        nuisance_mode=nuisance_mode,
        color="0.25",
    )

    if highb_text is not None:
        ax_dg.text(
            0.95,
            0.05,
            rf"$|b|<{highb_text:g}^\circ$",
            transform=ax_dg.transAxes,
            va="bottom",
            ha="right",
            fontsize=20,
        )

    # ------------------------------------------------------------------
    # Bottom panel: contour overlay
    # ------------------------------------------------------------------
    # Determine axis ranges from the union of grids if not given.
    if xlim is None:
        xs = np.concatenate([np.asarray(d.scan.Ebreak_grid, float) for d in cds])
        xlim = (float(xs.min()), float(xs.max()))
    if ylim is None:
        ys = np.concatenate([np.asarray(d.scan.dgamma_grid, float) for d in cds])
        ylim = (float(ys.min()), float(ys.max()))

    for ds in cds:
        _draw_one_dataset(
            ax_c,
            ds,
            levels=levels,
            level_labels=level_labels or _DEFAULT_LEVEL_LABELS,
            smooth_sigma=smooth_sigma,
            label_contours=label_contours,
        )

    if cr_ref is not None:
        Eb_ref, dg_ref = cr_ref
        ax_c.errorbar(
            [Eb_ref],
            [dg_ref],
            xerr=_asym_err(Eb_ref, cr_ebr_band),
            yerr=_asym_err(dg_ref, cr_dgamma_band),
            fmt=cr_marker,
            color=cr_color,
            ecolor=cr_color,
            mec="black",
            mew=0.8,
            ms=cr_marker_size,
            elinewidth=1.2,
            capsize=3,
            linestyle="none",
            zorder=5,
        )

    # Legend for contour panel: one Patch per dataset + CR marker.
    handles = []
    for ds in cds:
        face = ds.band_rgba[0] if ds.band_rgba else ds.line_color
        handles.append(Patch(facecolor=face, edgecolor="none", label=ds.label))
    if cr_ref is not None:
        handles.append(
            Line2D([0], [0], marker=cr_marker, color=cr_color, mec="black", lw=0, ms=9, label=cr_label)
        )
    ax_c.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        fontsize=19,
        ncol=max(1, min(len(handles), 2)),
    )
    if highb_text is not None:
        ax_c.text(
            0.95,
            0.05,
            rf"$|b|<{highb_text:g}^\circ$",
            transform=ax_c.transAxes,
            va="bottom",
            ha="right",
            fontsize=20,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="none"),
        )

    ax_c.set_xlabel(r"$E_{\rm break}\,[\mathrm{GeV}]$", fontsize=20)
    ax_c.set_ylabel(r"$\Delta\gamma$", fontsize=20)
    ax_c.tick_params(
        axis="both", which="major", direction="in", labelsize=18, length=6, width=1.8
    )
    ax_c.set_xlim(*xlim)
    ax_c.set_ylim(*ylim)

    # Z summary (Wilks + MC) per dataset, colored as requested.
    # Classical globals are keyed by "PL"; profiled globals are keyed by
    # names such as "PL|prof.box" / "PL|prof.gauss".
    def _first_result_key(d: dict, preferred: str = "PL") -> str | None:
        for key in d:
            if str(key).startswith("PL|prof."):
                return key
        if preferred in d:
            return preferred
        if len(d) == 1:
            return next(iter(d))
        return None

    def _z_vals(ds: ContourDataset) -> tuple[float, float, str]:
        z_w = float("nan")
        z_mc = float("nan")
        z_mc_op = ""
        g = getattr(ds, "g", None)
        if g is None:
            return z_w, z_mc, z_mc_op
        try:
            wilks = getattr(g, "wilks_results", {}) or {}
            w_key = _first_result_key(wilks)
            w = wilks.get(w_key, {}) if w_key is not None else {}
            z_w = float(w.get("Z_wilks", float("nan")))
        except Exception:
            pass
        try:
            mc_results = getattr(g, "mc_results", {}) or {}
            mc_key = _first_result_key(mc_results)
            mc = mc_results.get(mc_key) if mc_key is not None else None
            if mc is not None:
                z_mc = float(getattr(mc, "Z", float("nan")))
                z_mc_op = ">" if bool(getattr(mc, "p_is_upper_limit", False)) else ""
        except Exception:
            pass
        return z_w, z_mc, z_mc_op

    # Place PASS8 / IEM just under the legend; user-tuned y-position.
    extras = []
    z_fs = 19
    z_y = 0.82
    for ds in cds:
        lab = str(getattr(ds, "label", "")).upper()
        if "PASS8" in lab or "FERMI" in lab or "LAT" in lab:
            slot = "left"
        elif "IEM" in lab:
            slot = "right"
        else:
            slot = "extra"
        z_w, z_mc, z_mc_op = _z_vals(ds)
        # Same numbers as printed tables: joint global Wilks Z and MC Z from
        # ``g_prof.mc_results`` / ``g.mc_results`` (profile key preferred).
        if np.isfinite(z_w):
            wilks_txt = rf"$Z_{{\rm Wilks}}={z_w:.2f}\,\sigma$"
        else:
            wilks_txt = rf"$Z_{{\rm Wilks}}=\mathrm{{---}}$"
        if np.isfinite(z_mc):
            if str(z_mc_op) in {">", r"\geq"}:
                op = ">" if str(z_mc_op) == ">" else r"\geq"
                mc_txt = rf"$Z_{{\rm MC}}{op} {z_mc:.2f}\,\sigma$"
            else:
                mc_txt = rf"$Z_{{\rm MC}}={z_mc:.2f}\,\sigma$"
        else:
            mc_txt = rf"$Z_{{\rm MC}}=\mathrm{{---}}$"
        txt = wilks_txt + "\n" + mc_txt
        color = str(getattr(ds, "marker_color", "0.25"))
        if slot == "left":
            ax_c.text(
                0.05,
                z_y,
                txt,
                transform=ax_c.transAxes,
                ha="left",
                va="top",
                fontsize=z_fs,
                color=color,
            )
        elif slot == "right":
            ax_c.text(
                0.95,
                z_y,
                txt,
                transform=ax_c.transAxes,
                ha="right",
                va="top",
                fontsize=z_fs,
                color=color,
            )
        else:
            extras.append((txt, color))
    if extras:
        y0 = 0.06
        for j, (txt, color) in enumerate(extras):
            ax_c.text(
                0.50,
                y0 + 0.08 * j,
                txt,
                transform=ax_c.transAxes,
                ha="center",
                va="bottom",
                fontsize=z_fs,
                color=color,
            )

    if save is not None:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig
