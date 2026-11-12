"""Stacked SED + nuisance panels per window.

This helper is a "long figure" variant of :func:`crbreak.plotting.plot_sed`
where each window gets:

  - a main SED panel (data + PL/SBPL curves, Eb bands, CR band, etc.)
  - a smaller nuisance panel directly below it (posterior of s(E; w) + w_j)

Both panels share the same x-axis in **Energy [GeV]** (log scale) so they are
"glued" together visually, as requested for quick per-window diagnostics.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from ._run_stamp import stamp_prior_nuisance
from .plot_nuisance import NuisanceStyle, _masked_window_arrays, _plot_one_model
from .plot_sed import SedStyle, _draw_one_panel as _draw_sed_one_panel, _resolve_window_keys


def _draw_nuisance_panel_energy_x(
    ax,
    wf,
    *,
    style: NuisanceStyle,
    pl_prof_name: str | None,
    sbpl_prof_name: str | None,
    cr_ebr_band: tuple[float, float] | None,
    cr_ebr_fixed: float | None,
    annotate_roi: bool,
    legend: bool,
    nsamp: int,
    seed: int,
    eps_hess: float,
    ridge_hess: float,
    n_dense: int,
    dense_pad: float,
) -> None:
    """Nuisance panel drawn against energy (GeV) with log x-scale."""
    pl = wf[pl_prof_name] if (pl_prof_name and pl_prof_name in wf) else None
    sbpl = wf[sbpl_prof_name] if (sbpl_prof_name and sbpl_prof_name in wf) else None

    if pl is None and sbpl is None:
        ax.text(
            0.5,
            0.5,
            "no profiled fit",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
        )
        ax.set_axis_off()
        return

    E_m, y_m, sigma_m = _masked_window_arrays(wf)
    if E_m.size == 0:
        ax.set_axis_off()
        return

    E_dense = np.logspace(
        np.log10(float(E_m.min()) / dense_pad),
        np.log10(float(E_m.max()) * dense_pad),
        n_dense,
    )
    logE_dense = np.log10(E_dense)
    rng = np.random.default_rng(seed)

    # Plot the posterior bands / median + w_j errorbars for each profiled model.
    # Re-use the legacy internal helper (works in logE); we pass E-derived logE.
    if pl is not None:
        _plot_one_model(
            ax,
            pl,
            color=style.pl_color,
            band_color=style.pl_color,
            ls=style.pl_ls,
            label="PL",
            marker="o",
            x_offset=-float(style.w_offset_x),
            band_alpha_68=style.pl_band_alpha_68,
            band_alpha_90=style.pl_band_alpha_90,
            line_alpha=style.line_alpha,
            logE_dense=logE_dense,
            y_m=y_m,
            sigma_m=sigma_m,
            nsamp=nsamp,
            rng=rng,
            eps_hess=eps_hess,
            ridge_hess=ridge_hess,
            legend=legend,
        )
    if sbpl is not None:
        _plot_one_model(
            ax,
            sbpl,
            color=style.sbpl_color,
            band_color=style.sbpl_band_color,
            ls=style.sbpl_ls,
            label="SBPL",
            marker="^",
            x_offset=+float(style.w_offset_x),
            band_alpha_68=style.band_alpha_68,
            band_alpha_90=style.band_alpha_90,
            line_alpha=style.line_alpha,
            logE_dense=logE_dense,
            y_m=y_m,
            sigma_m=sigma_m,
            nsamp=nsamp,
            rng=rng,
            eps_hess=eps_hess,
            ridge_hess=ridge_hess,
            legend=legend,
        )

    # Convert x-axis from log10(E) to E by relabelling ticks and setting log-scale.
    # The artists were drawn against logE values, so we remap by changing the
    # x-data of all Line2D/PolyCollections from logE -> E.
    for line in ax.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        line.set_xdata(np.power(10.0, x))
    for coll in ax.collections:
        # Fill-between creates PolyCollections; errorbars include LineCollections.
        # Only remap objects that expose offsets or paths in x-y data space.
        if hasattr(coll, "get_offsets") and hasattr(coll, "set_offsets"):
            offs = coll.get_offsets()
            if offs is not None and len(offs) > 0:
                offs = np.asarray(offs, dtype=float)
                offs[:, 0] = np.power(10.0, offs[:, 0])
                coll.set_offsets(offs)
        if hasattr(coll, "get_paths"):
            try:
                for p in coll.get_paths() or []:
                    v = p.vertices
                    if v is not None and v.size and np.isfinite(v[:, 0]).any():
                        v[:, 0] = np.power(10.0, v[:, 0])
            except Exception:
                pass

    ax.set_xscale("log")
    ax.set_xlim(E_dense.min(), E_dense.max())
    ax.axhline(0.0, color="black", ls="--", lw=0.8, alpha=0.6, zorder=0)

    if cr_ebr_band is not None:
        lo_b, hi_b = cr_ebr_band
        ax.axvspan(lo_b, hi_b, color=style.cr_band_color, alpha=0.20, linewidth=0, zorder=0)
    if cr_ebr_fixed is not None:
        ax.axvline(cr_ebr_fixed, color=style.cr_band_color, ls="--", lw=1.0, alpha=0.9, zorder=1)

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=6,
        width=1.4,
        labelsize=style.tick_fs,
    )
    ax.tick_params(axis="both", which="minor", direction="in", length=3.5, width=1.1)
    ax.grid(False)
    ax.set_ylim(-2.5, 2.5)
    # Keep nuisance y ticks clean (integers) for publication plots.
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))

    if annotate_roi:
        bbox = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="none")
        ax.text(
            0.95,
            0.16,
            f"{wf.lowb:g}<|$b$|<{wf.highb:g}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=style.roi_fs,
            bbox=bbox,
        )
        ax.text(
            0.95,
            0.05,
            f"{wf.lowl:g}<|$l$|<{wf.highl:g}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=style.roi_fs,
            bbox=bbox,
        )

    if legend:
        ax.legend(frameon=False, fontsize=style.legend_fs, loc="upper left")


def plot_sed_plus_nuisances(
    fitting,
    *,
    g=None,
    window_keys: Sequence[str] | str | None = None,
    sed_style: SedStyle | None = None,
    nuisance_style: NuisanceStyle | None = None,
    pl_name: str = "PL",
    sbpl_name: str = "SBPL",
    pl_prof_name: str | None = None,
    sbpl_prof_name: str | None = None,
    sed_scaling: float,
    sed_band,
    nuisance_band,
    cr_ebr_band: tuple[float, float] | None,
    cr_ebr_fixed: float | None,
    annotate_roi: bool = True,
    show_envelope: bool = True,
    show_global: bool = False,
    # Layout (grid, like plot_sed but with a nuisance row under each panel)
    n_cols: int = 3,
    panel_size: tuple[float, float] = (6.0, 5.4),
    nuisance_frac: float = 0.45,
    hspace: float = 0.05,
    wspace: float = 0.30,
    # Typography (applied to both stacked axes in each cell)
    tick_labelsize: int = 18,
    axis_labelsize: int = 24,
    nuisance_labelsize: int = 20,
    legend_fontsize: int = 22,
    n_dense_sed: int = 500,
    n_dense_nuis: int = 320,
    dense_pad: float = 1.15,
    save: str | Path | None = None,
    show: bool = False,
    dpi: int = 240,
    prior_kind: str | None = None,
    nuisance_mode: str | None = None,
):
    """Plot a long figure with SED + nuisance panel per window (stacked).

    Notes
    -----
    - The nuisance panel uses the *profiled* fits (`pl_prof_name` /
      `sbpl_prof_name`). If neither exists for a window, the nuisance panel is
      annotated and hidden.
    - x-axis is always Energy [GeV] (log), shared between the two panels.
    """
    sed_style = sed_style or SedStyle()
    nuisance_style = nuisance_style or NuisanceStyle(
        data_color=sed_style.data_color,
        data_label=sed_style.data_label,
    )
    # Band settings from the analysis config (``plots.sed_band`` / ``plots.nuisance_band``).
    nsamp_band = nuisance_band.n_samples_sed_plus_nuisances
    seed = nuisance_band.seed
    eps_hess, ridge_hess = nuisance_band.hessian_eps, nuisance_band.hessian_ridge

    keys = _resolve_window_keys(fitting, window_keys)
    if not keys:
        raise ValueError("No windows selected for plot_sed_plus_nuisances.")

    sed_scaling_ = float(sed_scaling)

    n_panels = len(keys)
    n_cols = max(1, min(int(n_cols), n_panels))
    n_rows = int(np.ceil(n_panels / n_cols))
    # Slightly smaller SED tick labels when there is only one window (matches plot_sed).
    sed_tick_ls = max(12, int(tick_labelsize) - (1 if n_panels == 1 else 0))

    # Size: mimic plot_sed sizing per SED panel, but add nuisance height.
    w0, h0 = float(panel_size[0]), float(panel_size[1])
    cell_h = h0 * (1.0 + float(nuisance_frac))
    fig = plt.figure(figsize=(w0 * n_cols, cell_h * n_rows))
    sed_ratio = 1.0
    nuis_ratio = max(0.20, float(nuisance_frac))
    gs = fig.add_gridspec(
        nrows=2 * n_rows,
        ncols=n_cols,
        height_ratios=[sed_ratio, nuis_ratio] * n_rows,
        hspace=float(hspace),
        wspace=float(wspace),
    )

    axes_sed: list = []
    axes_nuis: list = []
    for i, key in enumerate(keys):
        r = i // n_cols
        c = i % n_cols
        ax_s = fig.add_subplot(gs[2 * r, c])
        ax_n = fig.add_subplot(gs[2 * r + 1, c], sharex=ax_s)
        axes_sed.append(ax_s)
        axes_nuis.append(ax_n)

        wf = fitting[key]
        _draw_sed_one_panel(
            ax_s,
            wf,
            key=key,
            g=g,
            style=sed_style,
            sbpl_name=sbpl_name,
            pl_name=pl_name,
            pl_prof_name=pl_prof_name,
            sbpl_prof_name=sbpl_prof_name,
            sed_scaling=sed_scaling_,
            sed_band=sed_band,
            show_envelope=show_envelope,
            show_global=show_global,
            show_profiled_curves=True,
            show_eb_uncorr_band=True,
            show_eb_corr_band=True,
            cr_ebr_band=cr_ebr_band,
            cr_ebr_fixed=cr_ebr_fixed,
            annotate_roi=annotate_roi,
            annotate_fit=False,
            annotate_band_labels=(i == 0),  # keep labels only once for readability
            e_break_legend_as_vertical_text=(n_panels == 1 and i == 0),
            extra_annotations=None,
            legend=(i == 0),
            n_dense=n_dense_sed,
            dense_pad=dense_pad,
            tick_labelsize=sed_tick_ls,
        )

        # Remove x tick labels on SED panel (shared x).
        ax_s.tick_params(axis="x", which="both", labelbottom=False)
        ax_s.tick_params(axis="both", which="major", labelsize=sed_tick_ls)
        ax_s.tick_params(axis="both", which="minor", labelsize=sed_tick_ls)

        # Nuisance panel in energy-x.
        small_style = replace(
            nuisance_style,
            label_fs=min(nuisance_style.label_fs, 16),
            legend_fs=min(nuisance_style.legend_fs, 12),
            tick_fs=min(nuisance_style.tick_fs, 14),
            title_fs=min(nuisance_style.title_fs, 15),
            roi_fs=min(nuisance_style.roi_fs, 12),
        )
        _draw_nuisance_panel_energy_x(
            ax_n,
            wf,
            style=small_style,
            pl_prof_name=pl_prof_name,
            sbpl_prof_name=sbpl_prof_name,
            cr_ebr_band=cr_ebr_band,
            cr_ebr_fixed=cr_ebr_fixed,
            annotate_roi=False,  # ROI already shown on SED panel
            legend=False,
            nsamp=nsamp_band,
            seed=seed + i,
            eps_hess=eps_hess,
            ridge_hess=ridge_hess,
            n_dense=n_dense_nuis,
            dense_pad=dense_pad,
        )
        # Only label the nuisance y-axis on the left column to reduce clutter.
        if c == 0:
            ax_n.set_ylabel(
                r"$s(E;\,\mathbf{w})$",
                fontsize=int(nuisance_labelsize),
            )
        else:
            ax_n.set_ylabel("")
        ax_n.tick_params(axis="both", which="major", labelsize=int(tick_labelsize))
        ax_n.tick_params(axis="both", which="minor", labelsize=int(tick_labelsize))

    # ------------------------------------------------------------------
    # Requested legend for Eb systematic bands (subplot 2)
    # ------------------------------------------------------------------
    if len(axes_sed) >= 2:
        from matplotlib.patches import Patch

        ebreak_color = sed_style.ebreak_color or sed_style.data_color
        axes_sed[1].legend(
            handles=[
                Patch(
                    facecolor=ebreak_color,
                    alpha=sed_style.eb_uncorr_alpha,
                    edgecolor="none",
                    label="Uncorrelated sys.",
                ),
                Patch(
                    facecolor=ebreak_color,
                    alpha=sed_style.eb_corr_alpha,
                    edgecolor="none",
                    label="Correlated sys.",
                ),
            ],
            frameon=False,
            fontsize=22,
            loc="upper left",
        )

    # Add the physical SED ylabel on the left column (like plot_sed does).
    s = f"{sed_scaling_:g}"
    s_minus_1 = f"{(float(sed_scaling_) - 1.0):g}"
    sed_ylabel = (
        rf"$E^{{{s}}}\,J\;[\mathrm{{GeV^{{{s_minus_1}}}\,cm^{{-2}}\,s^{{-1}}\,sr^{{-1}}}}]$"
    )
    for i, ax_s in enumerate(axes_sed):
        c = i % n_cols
        if c == 0:
            ax_s.set_ylabel(sed_ylabel, fontsize=int(axis_labelsize))
        else:
            ax_s.set_ylabel("")

    # Label x only on the bottom nuisance row.
    for i, ax_n in enumerate(axes_nuis):
        r = i // n_cols
        if r == (n_rows - 1):
            ax_n.set_xlabel("Energy [GeV]", fontsize=int(axis_labelsize))
        else:
            ax_n.tick_params(axis="x", which="both", labelbottom=False)

    if prior_kind is not None and nuisance_mode is not None:
        target = axes_sed[0]
        stamp_prior_nuisance(
            target,
            prior_kind=prior_kind,
            nuisance_mode=nuisance_mode,
            color="0.25",
            fontsize=12,
        )

    # Enlarge the SED legend (only exists on the first SED panel).
    leg = axes_sed[0].get_legend()
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(int(legend_fontsize))

    # Keep explicit spacing between subplots; tight_layout tends to squeeze
    # columns and re-introduce label overlaps in this dense grid.
    fig.subplots_adjust(hspace=float(hspace), wspace=float(wspace))

    if save is not None:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig

