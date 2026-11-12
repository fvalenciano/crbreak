"""Plot ``E_break`` and ``Δγ`` vs longitude.

Mirrors the legacy ``_plot_break_eb_dg`` / ``plot_pass8_vs_model`` figure:
two stacked panels, one per parameter, with grey reference bands for the
local-CR break.  Overlay-capable via small :class:`ParamsDataset` specs --
each one drives one set of points, so PASS8 + IEM (or DRAGON, or anything
else) coexist with their own colors / markers / longitude offsets.
:func:`plot_params_vs_glat` draws the same parameters vs ``|b|`` for a run
with one |l| window in several |b| bands.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from ._run_stamp import stamp_prior_nuisance

# When profile/Hessian one-sided errors are missing (NaN), ``errorbar`` uses
# these reference bounds (matplotlib: lower row = down from y, upper row = up).
EBREAK_LOWER_ERROR_BOUND_GEV: float = 0.0
EBREAK_UPPER_ERROR_BOUND_GEV: float = 80.0
DGAMMA_LOWER_ERROR_BOUND: float = 0.0
DGAMMA_UPPER_ERROR_BOUND: float = 0.4


# ----------------------------------------------------------------------
# Dataset spec
# ----------------------------------------------------------------------


@dataclass
class ParamsDataset:
    """One curve in the ``Ebreak`` / ``Δγ`` vs ``|l|`` figure.

    Parameters
    ----------
    label : str
        Legend label (e.g. ``"PASS8"``, ``"IEM"``).
    fitting : :class:`Fitting`
        Already populated with at least the SBPL fit named ``sbpl_name``.
    sbpl_name : str
        Name under which the *classical* SBPL :class:`FitResult` was
        stored on each :class:`WindowFitResult`.
    sbpl_prof_name : str | None
        Optional name of the profiled SBPL :class:`FitResult` stored on
        each :class:`WindowFitResult`.  When given (i.e. profiled flow,
        ``USE_CLASSICAL=False``), the per-window points and error bars
        are sourced from the profiled fit instead of the classical one
        and use the asymmetric ``Δχ² = 1`` profile-likelihood errors
        cached in ``wf.profile_errors`` when available (falling back to
        the symmetric Hessian errors otherwise).
    color, marker
        Matplotlib styling.
    glon_offset_deg : float
        Visual offset on the x-axis to keep overlapping markers separable
        (the legacy uses ``+1``/``-1``).
    edge_color
        Marker edge color (defaults to black for filled markers).
    fill : bool
        If False, the marker is open (no facecolor) -- handy for "raw" or
        "alt" variants overlaid on top of a filled main dataset.
    """

    label: str
    fitting: object
    sbpl_name: str = "SBPL"
    sbpl_prof_name: str | None = None
    color: str = "cornflowerblue"
    marker: str = "o"
    glon_offset_deg: float = 0.0
    edge_color: str = "black"
    fill: bool = True
    capsize: float = 4.0
    markersize: float = 8.0
    # Optional linestyle for the errorbar segments (legacy uses ':' for "raw").
    error_ls: str | None = None
    # Line through the window values (:func:`plot_gammas_vs_glon` only).
    linestyle: str = "-"
    extra_kwargs: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _normalise_datasets(datasets) -> list[ParamsDataset]:
    if isinstance(datasets, ParamsDataset):
        return [datasets]
    out = list(datasets)
    if not out:
        raise ValueError("plot_params_vs_glon: at least one dataset required.")
    if not all(isinstance(d, ParamsDataset) for d in out):
        raise TypeError(
            "plot_params_vs_glon expects ParamsDataset instances "
            "(or a single one)."
        )
    return out


def _gather(
    ds: ParamsDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect ``(centers, xerr, Eb, Eb_yerr, dgamma, dgamma_yerr)``
    arrays for one dataset, sorted by ``|l|``.

    ``Eb_yerr`` / ``dgamma_yerr`` are always returned as a ``(2, N)``
    array so :func:`matplotlib.axes.Axes.errorbar` can render
    asymmetric bars natively when profile-likelihood errors are
    available; for symmetric (Hessian / classical) errors the two rows
    are equal.

    Missing **lower** errors are filled so bars extend down to the lower
    bound (``E_break\\rightarrow`` :data:`EBREAK_LOWER_ERROR_BOUND_GEV`,
    ``\\Delta\\gamma\\rightarrow`` :data:`DGAMMA_LOWER_ERROR_BOUND`).
    Missing **upper** errors are filled toward the upper bound (``E_break
    \\rightarrow`` :data:`EBREAK_UPPER_ERROR_BOUND_GEV` GeV,
    ``\\Delta\\gamma\\rightarrow`` :data:`DGAMMA_UPPER_ERROR_BOUND`).
    """
    fitting = ds.fitting
    keys = sorted(fitting.window_fits.keys(), key=lambda k: fitting[k].lowl)
    centers = np.array([0.5 * (fitting[k].lowl + fitting[k].highl) for k in keys])
    xerr = np.array([0.5 * (fitting[k].highl - fitting[k].lowl) for k in keys])
    Eb, Eb_lo, Eb_hi = [], [], []
    dg, dg_lo, dg_hi = [], [], []

    def _push_nan() -> None:
        Eb.append(np.nan); Eb_lo.append(np.nan); Eb_hi.append(np.nan)
        dg.append(np.nan); dg_lo.append(np.nan); dg_hi.append(np.nan)

    for k in keys:
        wf = fitting[k]
        # Pick the *primary* SBPL fit name: profiled when both available
        # (consistent with the SED panels); fall back to the classical
        # one (or NaN if neither is fitted).
        if ds.sbpl_prof_name and ds.sbpl_prof_name in wf:
            primary = ds.sbpl_prof_name
            prof_err = getattr(wf, "profile_errors", {}).get(primary)
        elif ds.sbpl_name in wf:
            primary = ds.sbpl_name
            prof_err = None
        else:
            _push_nan()
            continue

        params = wf[primary].params
        e_v, e_e = params.get("Ebreak", (np.nan, np.nan))
        d_v, d_e = params.get("dgamma", (np.nan, np.nan))

        # Δχ²=1 asymmetric profile errors take priority when available.
        if prof_err and "Ebreak" in prof_err:
            _, lo_e, hi_e = prof_err["Ebreak"]
            Eb.append(e_v)
            Eb_lo.append(e_v - lo_e if np.isfinite(lo_e) else np.nan)
            Eb_hi.append(hi_e - e_v if np.isfinite(hi_e) else np.nan)
        else:
            Eb.append(e_v)
            Eb_lo.append(e_e)
            Eb_hi.append(e_e)

        if prof_err and "dgamma" in prof_err:
            _, lo_d, hi_d = prof_err["dgamma"]
            dg.append(d_v)
            dg_lo.append(d_v - lo_d if np.isfinite(lo_d) else np.nan)
            dg_hi.append(hi_d - d_v if np.isfinite(hi_d) else np.nan)
        else:
            dg.append(d_v); dg_lo.append(d_e); dg_hi.append(d_e)

    Eb_arr = np.asarray(Eb, dtype=float)
    Eb_lo_a = np.asarray(Eb_lo, dtype=float)
    Eb_hi_a = np.asarray(Eb_hi, dtype=float)
    eb_lo_b = float(EBREAK_LOWER_ERROR_BOUND_GEV)
    eb_hi_b = float(EBREAK_UPPER_ERROR_BOUND_GEV)
    for i in range(Eb_arr.size):
        ev = Eb_arr[i]
        if not np.isfinite(ev):
            continue
        if not np.isfinite(Eb_lo_a[i]) or Eb_lo_a[i] < 0:
            Eb_lo_a[i] = max(0.0, ev - eb_lo_b)
        if not np.isfinite(Eb_hi_a[i]) or Eb_hi_a[i] < 0:
            Eb_hi_a[i] = max(0.0, eb_hi_b - ev)

    Eb_yerr = np.vstack([Eb_lo_a, Eb_hi_a]).astype(float)

    dg_arr = np.asarray(dg, dtype=float)
    dg_lo_a = np.asarray(dg_lo, dtype=float)
    dg_hi_a = np.asarray(dg_hi, dtype=float)
    dg_lo_b = float(DGAMMA_LOWER_ERROR_BOUND)
    dg_hi_b = float(DGAMMA_UPPER_ERROR_BOUND)
    for i in range(dg_arr.size):
        dv = dg_arr[i]
        if not np.isfinite(dv):
            continue
        if not np.isfinite(dg_lo_a[i]) or dg_lo_a[i] < 0:
            dg_lo_a[i] = max(0.0, dv - dg_lo_b)
        if not np.isfinite(dg_hi_a[i]) or dg_hi_a[i] < 0:
            dg_hi_a[i] = max(0.0, dg_hi_b - dv)

    dg_yerr = np.vstack([dg_lo_a, dg_hi_a]).astype(float)
    return (
        centers, xerr,
        Eb_arr, Eb_yerr,
        dg_arr, dg_yerr,
    )


def _plot_one(ax, x, xerr, y, yerr, ds: ParamsDataset) -> None:
    mfc = ds.color if ds.fill else "white"
    eb = ax.errorbar(
        x + ds.glon_offset_deg,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt=ds.marker,
        markeredgecolor=ds.edge_color if ds.fill else ds.color,
        markeredgewidth=0.5 if ds.fill else 1.0,
        markerfacecolor=mfc,
        ecolor=ds.color,
        capsize=ds.capsize,
        markersize=ds.markersize,
        label=ds.label,
        **ds.extra_kwargs,
    )
    if ds.error_ls:
        try:
            for barlinecol in eb[2]:
                barlinecol.set_linestyle(str(ds.error_ls))
        except Exception:
            pass


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def plot_params_vs_glon(
    datasets: Sequence[ParamsDataset] | ParamsDataset,
    *,
    cr_ebr_band: tuple[float, float] | None,
    cr_dgamma_band: tuple[float, float] | None,
    glon_xlim: tuple[float, float] | None = None,
    ebreak_ylim: tuple[float, float] | None = None,
    dgamma_ylim: tuple[float, float] = (0.0, 0.50),
    highb_text: float | None = None,
    legend_ncol: int | None = None,
    figsize: tuple[float, float] = (6.0, 9.5),
    save: str | Path | None = None,
    show: bool = False,
    dpi: int = 200,
    prior_kind: str | None = None,
    nuisance_mode: str | None = None,
):
    """``E_break`` and ``Δγ`` vs ``|l|`` for one or more datasets (overlay).

    Parameters
    ----------
    datasets : :class:`ParamsDataset` or sequence of them
        Each dataset draws its own pair of points (and is allowed an x-axis
        offset to keep overlapping markers separable).
    cr_ebr_band, cr_dgamma_band
        Reference grey bands; pass ``None`` to suppress.
    glon_xlim
        ``(xmin, xmax)`` for the longitude axis.  If ``None``, it is taken
        from the first dataset's ROI extent (``[lowl-2, highl+2]``).
    ebreak_ylim, dgamma_ylim
        Vertical limits.  ``ebreak_ylim`` defaults to ``(0, max(70, 1.1*max))``
        as in the legacy.
    highb_text : float | None
        If given, draw a small ``|b|<{highb}`` text box in the top panel.
    legend_ncol : int | None
        Legend columns (default: one per dataset).
    save : path-like, optional
    show : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    dss = _normalise_datasets(datasets)

    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.08)

    _draw_params_vs_glon(
        axs[0], axs[1], dss,
        cr_ebr_band=cr_ebr_band, cr_dgamma_band=cr_dgamma_band, glon_xlim=glon_xlim,
        ebreak_ylim=ebreak_ylim, dgamma_ylim=dgamma_ylim, highb_text=highb_text,
        legend_ncol=legend_ncol,
    )

    # Run configuration stamp (use top panel).
    stamp_prior_nuisance(
        axs[0],
        prior_kind=prior_kind,
        nuisance_mode=nuisance_mode,
        color="0.25",
    )

    fig.tight_layout()

    if save is not None:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _draw_params_vs_glon(
    ax_e, ax_dg, dss: list[ParamsDataset], *,
    cr_ebr_band, cr_dgamma_band, glon_xlim, ebreak_ylim, dgamma_ylim, highb_text, legend_ncol,
) -> None:
    """The ``E_break`` (``ax_e``) and ``Δγ`` (``ax_dg``) panels of :func:`plot_params_vs_glon`."""
    all_eb_top: list[float] = []
    glon_min, glon_max = +np.inf, -np.inf
    for ds in dss:
        x, xerr, Eb, Eb_err, dg, dg_err = _gather(ds)
        _plot_one(ax_e, x, xerr, Eb, Eb_err, ds)
        _plot_one(ax_dg, x, xerr, dg, dg_err, ds)
        finite_eb = Eb[np.isfinite(Eb)]
        if finite_eb.size:
            all_eb_top.append(float(np.nanmax(Eb)))
        if x.size:
            glon_min = min(glon_min, float(x.min() - xerr.max() - 1.0))
            glon_max = max(glon_max, float(x.max() + xerr.max() + 1.0))

    # Reference bands (drawn on top of an interval that spans current x-axis).
    if glon_xlim is None:
        if not np.isfinite(glon_min) or not np.isfinite(glon_max):
            glon_xlim = (0.0, 90.0)
        else:
            glon_xlim = (glon_min, glon_max)

    if cr_ebr_band is not None:
        lo, hi = cr_ebr_band
        ax_e.fill_between(
            list(glon_xlim), lo, hi,
            color="grey", alpha=0.30, linewidth=0,
            label="CR $E_b$ band",
        )
    if cr_dgamma_band is not None:
        lo, hi = cr_dgamma_band
        ax_dg.fill_between(
            list(glon_xlim), lo, hi,
            color="grey", alpha=0.30, linewidth=0,
            label=r"CR $\Delta\gamma$ band",
        )

    # Cosmetics: top panel (Ebreak)
    ax_e.set_ylabel(r"$E_{\rm break}\,[\mathrm{GeV}]$", fontsize=18)
    if ebreak_ylim is None:
        top = max(80.0, 1.1 * (max(all_eb_top) if all_eb_top else 80.0))
        ax_e.set_ylim(0.0, top)
    else:
        ax_e.set_ylim(*ebreak_ylim)
    ax_e.legend(fontsize=14, frameon=False, loc="upper left", ncol=legend_ncol or max(1, len(dss)))
    ax_e.tick_params(axis="both", which="major", direction="in",
                     labelsize=15, length=6, width=1.6)
    # Cosmetics: bottom panel (Δγ)
    ax_dg.set_ylim(*dgamma_ylim)
    ax_dg.set_xlabel(r"$|\ell|\,[^{\circ}]$", fontsize=18)
    ax_dg.set_ylabel(r"$\Delta\gamma$", fontsize=18)
    ax_dg.tick_params(axis="both", which="major", direction="in",
                      labelsize=15, length=6, width=1.6)
    ax_dg.set_xlim(*glon_xlim)
    if highb_text is not None:
        ax_dg.text(
            0.95, 0.05, f"|b|<{highb_text:g}",
            transform=ax_dg.transAxes, va="bottom", ha="right", fontsize=16,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      alpha=0.8, edgecolor="none"),
        )


def _gamma_series(ds: ParamsDataset, offset_deg: float, *, shade: bool) -> tuple[tuple, tuple]:
    """``(param, label, linestyle, colour, marker, filled, |l| shift)`` of the γ1 and γ2 series of ``ds``.

    γ1 takes the dataset colour and marker.  γ2 is a darker shade with squares
    when ``shade`` (a single dataset); with several datasets, whose colours and
    markers already tell them apart, it keeps them and is drawn open.
    """
    g1 = ("gamma1", r"$\gamma_1$", "-", ds.color, ds.marker, ds.fill, -offset_deg)
    if shade:
        dark = tuple(0.55 * c for c in to_rgb(ds.color))
        g2 = ("gamma2", r"$\gamma_2$", "--", dark, "s" if ds.marker != "s" else "o", ds.fill, offset_deg)
    else:
        g2 = ("gamma2", r"$\gamma_2$", "--", ds.color, ds.marker, False, offset_deg)
    return g1, g2


def _draw_gammas_vs_glon(ax, ds: ParamsDataset, series, gamma_radius: float) -> None:
    """γ1 and γ2 of every window of ``ds`` in the styles of ``series`` (:func:`_gamma_series`)."""
    fitting = ds.fitting
    keys = sorted(fitting.window_fits.keys(), key=lambda k: fitting[k].lowl)
    x = np.array([0.5 * (fitting[k].lowl + fitting[k].highl) for k in keys]) + ds.glon_offset_deg
    for pname, _, ls, color, marker, filled, dx in series:
        v, lo, hi = np.array([window_interval(fitting[k], ds, pname) for k in keys]).T
        lo = np.where(np.isfinite(lo), lo, v - gamma_radius)
        hi = np.where(np.isfinite(hi), hi, v + gamma_radius)
        ax.errorbar(
            x + dx, v, yerr=np.clip(np.vstack([v - lo, hi - v]), 0.0, None),
            fmt=marker, ls=ls, lw=1.5, color=color,
            markerfacecolor=color if filled else "white",
            markeredgecolor=ds.edge_color if filled else color,
            markeredgewidth=0.5 if filled else 1.0,
            ecolor=color, capsize=ds.capsize, markersize=ds.markersize,
        )


def _gamma_handles(ref: ParamsDataset, series, prefix: str = "") -> list[Line2D]:
    """Legend handles of the γ1 / γ2 ``series`` (marker size and edge of ``ref``)."""
    return [Line2D([], [], color=color, ls=ls, lw=1.5, marker=marker, markersize=ref.markersize,
                   markerfacecolor=color if filled else "white",
                   markeredgecolor=ref.edge_color if filled else color,
                   markeredgewidth=0.5 if filled else 1.0, label=prefix + label)
            for _, label, ls, color, marker, filled, _ in series]


def _primary_fit(wf, ds: ParamsDataset):
    """The SBPL fit :func:`window_interval` reads (profiled when present), or None."""
    if ds.sbpl_prof_name and ds.sbpl_prof_name in wf:
        return wf[ds.sbpl_prof_name]
    return wf[ds.sbpl_name] if ds.sbpl_name in wf else None


def _has_free_s(ds: ParamsDataset) -> bool:
    """True when the SBPL smoothness ``s`` was fitted (``model.sbpl_s_fixed: null``)."""
    fits = (_primary_fit(wf, ds) for wf in ds.fitting.window_fits.values())
    return any(f is not None and "s" in f.params for f in fits)


def _draw_s_vs_glon(ax, dss: list[ParamsDataset], *, s_ref: float | None) -> None:
    """SBPL smoothness ``s`` per window (±1σ from the fit covariance, clipped to
    the fit bounds of ``s``), in the style of the ``E_break`` / ``Δγ`` panels,
    with a grey line at ``s_ref`` (the fixed ``s`` of the other runs)."""
    s_hi = 0.0
    for ds in dss:
        fitting = ds.fitting
        keys = sorted(fitting.window_fits.keys(), key=lambda k: fitting[k].lowl)
        x = np.array([0.5 * (fitting[k].lowl + fitting[k].highl) for k in keys])
        xerr = np.array([0.5 * (fitting[k].highl - fitting[k].lowl) for k in keys])
        rows = []
        for k in keys:
            fit = _primary_fit(fitting[k], ds)
            if fit is None or "s" not in fit.params:
                rows.append((np.nan, np.nan, np.nan, -np.inf, np.inf))
                continue
            lo_b, hi_b = getattr(fit.model, "spectral_model", fit.model).bounds["s"]
            rows.append((*window_interval(fitting[k], ds, "s"), lo_b, hi_b))
        v, lo, hi, lo_b, hi_b = np.array(rows, dtype=float).T
        lo = np.where(np.isfinite(lo), np.maximum(lo, lo_b), lo_b)
        hi = np.where(np.isfinite(hi), np.minimum(hi, hi_b), hi_b)
        _plot_one(ax, x, xerr, v, np.clip(np.vstack([v - lo, hi - v]), 0.0, None), ds)
        s_hi = max(s_hi, float(np.nanmax(np.where(np.isfinite(hi_b), hi_b, hi))))
    if s_ref is not None:
        ax.axhline(s_ref, color="grey", lw=1.5, zorder=0, label=rf"$s={s_ref:g}$")
        ax.legend(fontsize=14, frameon=False, loc="upper left")
    ax.set_ylim(0.0, 1.05 * s_hi if s_hi > 0 else None)
    ax.set_ylabel(r"$s$", fontsize=18)
    ax.set_xlabel(r"$|\ell|\,[^{\circ}]$", fontsize=18)
    ax.tick_params(axis="both", which="major", direction="in",
                   labelsize=15, length=6, width=1.6, labelbottom=True)


def plot_params_gammas_vs_glon(
    datasets: Sequence[ParamsDataset] | ParamsDataset,
    *,
    cr_ebr_band: tuple[float, float] | None,
    cr_dgamma_band: tuple[float, float] | None,
    gamma_radius: float,
    s_ref: float | None = None,
    glon_xlim: tuple[float, float] | None = None,
    ebreak_ylim: tuple[float, float] | None = None,
    dgamma_ylim: tuple[float, float] = (0.0, 0.50),
    highb_text: float | None = None,
    legend_ncol: int | None = None,
    gamma_offset_deg: float = 0.75,
    gamma_overlay: Sequence[ParamsDataset] = (),
    figsize: tuple[float, float] = (12.0, 9.5),
    save: str | Path | None = None,
    show: bool = False,
    dpi: int = 200,
    prior_kind: str | None = None,
    nuisance_mode: str | None = None,
):
    """:func:`plot_params_vs_glon` (left column) plus ``γ1`` and ``γ2`` vs ``|l|`` (top right).

    The left column is drawn exactly as in :func:`plot_params_vs_glon`.  In
    the top-right panel each window gives a ``γ1`` point (dataset colour and
    marker, solid line through the windows) and a ``γ2 = γ1 - Δγ`` point
    (dashed line; darker shade and squares for one dataset, open markers for
    several, see :func:`_gamma_series`), with the interval of
    :func:`window_interval`.  A missing
    side (Δχ² = 1 crossing outside the profile grid, best ± ``gamma_radius``
    = ``cfg.profile_errors.gamma_radius``) is drawn to the grid edge.  ``γ1``
    / ``γ2`` are shifted by ∓ ``gamma_offset_deg`` so both bars stay visible.
    ``gamma_overlay`` datasets (e.g. a DRAGON mock of the same windows) are
    drawn only in the γ panel, each in its own colour and marker with a darker
    γ2; the legend then names every series and moves to the bottom-right cell.
    When the SBPL smoothness ``s`` was fitted (``model.sbpl_s_fixed: null``),
    the bottom-right cell shows ``s`` vs ``|l|`` instead (:func:`_draw_s_vs_glon`,
    grey line at ``s_ref``) and the γ legend stays in the γ panel.
    """
    dss = _normalise_datasets(datasets)
    s_dss = [ds for ds in dss if _has_free_s(ds)]

    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.08)

    _draw_params_vs_glon(
        axs[0, 0], axs[1, 0], dss,
        cr_ebr_band=cr_ebr_band, cr_dgamma_band=cr_dgamma_band, glon_xlim=glon_xlim,
        ebreak_ylim=ebreak_ylim, dgamma_ylim=dgamma_ylim, highb_text=highb_text,
        legend_ncol=legend_ncol,
    )

    ax_g = axs[0, 1]
    if s_dss:
        _draw_s_vs_glon(axs[1, 1], s_dss, s_ref=s_ref)
    elif not gamma_overlay:
        axs[1, 1].remove()
    shade = len(dss) == 1
    for ds in dss:
        _draw_gammas_vs_glon(ax_g, ds, _gamma_series(ds, gamma_offset_deg, shade=shade), gamma_radius)
    # Legend: the dataset's own styles for one dataset, neutral grey for several.
    ref = dss[0] if shade else replace(dss[0], color="0.35", marker="o", fill=True)
    handles = _gamma_handles(ref, _gamma_series(ref, gamma_offset_deg, shade=shade))
    if gamma_overlay:
        prefix = f"{dss[0].label} " if shade else ""
        for h in handles:
            h.set_label(prefix + h.get_label())
        for ds in gamma_overlay:
            series = _gamma_series(ds, gamma_offset_deg, shade=True)
            _draw_gammas_vs_glon(ax_g, ds, series, gamma_radius)
            handles += _gamma_handles(ds, series, prefix=f"{ds.label} ")
        if s_dss:
            ax_g.legend(handles=handles, fontsize=11, frameon=False, loc="best")
        else:
            # Several series: the legend goes in the empty bottom-right cell.
            axs[1, 1].axis("off")
            axs[1, 1].legend(handles=handles, fontsize=16, frameon=False, loc="upper left")
    else:
        ax_g.legend(handles=handles, fontsize=14, frameon=False, loc="best")
    ax_g.set_ylabel(r"$\gamma$", fontsize=18)
    ax_g.set_xlabel(r"$|\ell|\,[^{\circ}]$", fontsize=18)
    ax_g.tick_params(axis="both", which="major", direction="in",
                     labelsize=15, length=6, width=1.6, labelbottom=True)

    stamp_prior_nuisance(axs[0, 0], prior_kind=prior_kind, nuisance_mode=nuisance_mode, color="0.25")

    fig.tight_layout()

    if save is not None:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def window_interval(wf, ds: ParamsDataset, pname: str) -> tuple[float, float, float]:
    """``(value, lo, hi)`` of ``pname`` in one window, as absolute values.

    ``[lo, hi]`` is the Δχ² = 1 profile interval when it is cached in
    ``wf.profile_errors`` (profiled fit), else ``value ± 1σ`` from the fit
    covariance.  ``gamma2`` is the derived slope ``gamma1 - dgamma``.
    """
    if ds.sbpl_prof_name and ds.sbpl_prof_name in wf:
        name = ds.sbpl_prof_name
        prof = (getattr(wf, "profile_errors", {}).get(name) or {}).get(pname)
    elif ds.sbpl_name in wf:
        name, prof = ds.sbpl_name, None
    else:
        return (np.nan, np.nan, np.nan)
    fit = wf[name]
    if pname == "gamma2":
        names = list(fit.model.param_names)
        i1, i2 = names.index("gamma1"), names.index("dgamma")
        C = np.asarray(fit.pcov, dtype=float)
        v = float(fit.popt[i1] - fit.popt[i2])
        e = float(np.sqrt(C[i1, i1] + C[i2, i2] - 2.0 * C[i1, i2]))
    else:
        v, e = fit.params[pname]
    if prof is not None:
        return (v, float(prof[1]), float(prof[2]))
    return (v, v - e, v + e)


def plot_gammas_vs_glon(
    datasets: Sequence[ParamsDataset] | ParamsDataset,
    *,
    glon_xlim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (6.0, 9.5),
    save: str | Path | None = None,
    show: bool = False,
    dpi: int = 200,
    prior_kind: str | None = None,
    nuisance_mode: str | None = None,
):
    """``γ1`` (top) and ``γ2 = γ1 - Δγ`` (bottom) vs ``|l|``, overlay of datasets.

    Same layout as :func:`plot_params_vs_glon`, but each dataset is a line
    (``ds.linestyle``) through the window centres with vertical error bars
    from :func:`window_interval`.
    """
    dss = _normalise_datasets(datasets)

    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.08)

    glon_min, glon_max = +np.inf, -np.inf
    for ds in dss:
        fitting = ds.fitting
        keys = sorted(fitting.window_fits.keys(), key=lambda k: fitting[k].lowl)
        x = np.array([0.5 * (fitting[k].lowl + fitting[k].highl) for k in keys])
        xerr = np.array([0.5 * (fitting[k].highl - fitting[k].lowl) for k in keys])
        for ax, pname in zip(axs, ("gamma1", "gamma2")):
            v, lo, hi = np.array([window_interval(fitting[k], ds, pname) for k in keys]).T
            ax.errorbar(x + ds.glon_offset_deg, v, yerr=np.vstack([v - lo, hi - v]),
                        color=ds.color, ls=ds.linestyle, lw=2.0, elinewidth=1.2,
                        capsize=ds.capsize, label=ds.label)
        if x.size:
            glon_min = min(glon_min, float(x.min() - xerr.max() - 1.0))
            glon_max = max(glon_max, float(x.max() + xerr.max() + 1.0))

    if glon_xlim is None:
        glon_xlim = (glon_min, glon_max) if np.isfinite(glon_min) else (0.0, 90.0)

    for ax, label in zip(axs, (r"$\gamma_1$", r"$\gamma_2$")):
        ax.set_ylabel(label, fontsize=18)
        ax.tick_params(axis="both", which="major", direction="in",
                       labelsize=15, length=6, width=1.6)
    axs[0].legend(fontsize=12, frameon=False, loc="upper left", ncol=2)
    axs[1].set_xlabel(r"$|\ell|\,[^{\circ}]$", fontsize=18)
    axs[1].set_xlim(*glon_xlim)

    stamp_prior_nuisance(axs[0], prior_kind=prior_kind, nuisance_mode=nuisance_mode, color="0.25")

    fig.tight_layout()

    if save is not None:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig


_PARAM_LABELS = {
    "Ebreak": r"$E_{\rm break}\,[\mathrm{GeV}]$",
    "dgamma": r"$\Delta\gamma$",
    "gamma1": r"$\gamma_1$",
    "gamma2": r"$\gamma_2$",
}
_REF_BAND_LABELS = {"Ebreak": "CR $E_b$ band", "dgamma": r"CR $\Delta\gamma$ band"}
# A missing Δχ² = 1 side is drawn to these bounds, as in :func:`_gather`.
_MISSING_BOUNDS = {
    "Ebreak": (EBREAK_LOWER_ERROR_BOUND_GEV, EBREAK_UPPER_ERROR_BOUND_GEV),
    "dgamma": (DGAMMA_LOWER_ERROR_BOUND, DGAMMA_UPPER_ERROR_BOUND),
}


def _glat_points(datasets, bands):
    """Datasets, their single |l| windows, and the band points ``(x, xerr, edges, log)`` of a vs-|b| figure."""
    dss = _normalise_datasets(datasets)
    if len(dss) != len(bands):
        raise ValueError(f"plot_params_vs_glat: {len(dss)} datasets for {len(bands)} |b| bands.")
    wfs = []
    for ds in dss:
        keys = list(ds.fitting.window_fits)
        if len(keys) != 1:
            raise ValueError(f"plot_params_vs_glat: each dataset needs a single |l| window, got {len(keys)}.")
        wfs.append(ds.fitting[keys[0]])

    edges = np.asarray(bands, dtype=float)
    log = bool(edges.min() > 0)
    x = np.sqrt(edges[:, 0] * edges[:, 1]) if log else edges.mean(axis=1)
    xerr = np.vstack([x - edges[:, 0], edges[:, 1] - x])
    return dss, wfs, x, xerr, edges, log


def _draw_param_vs_glat(ax, pname, dss, wfs, x, xerr, *, ref_bands, ylims) -> None:
    """One panel of :func:`plot_params_vs_glat`: ``pname`` per band, joined by a grey line."""
    v, lo, hi = np.array([window_interval(wf, ds, pname) for wf, ds in zip(wfs, dss)]).T
    if pname in _MISSING_BOUNDS:
        b_lo, b_hi = _MISSING_BOUNDS[pname]
        lo = np.where(np.isfinite(lo), lo, b_lo)
        hi = np.where(np.isfinite(hi), hi, b_hi)
    yerr = np.clip(np.vstack([v - lo, hi - v]), 0.0, None)
    ax.plot(x, v, color="0.6", lw=1.0, zorder=1)
    for i, ds in enumerate(dss):
        ax.errorbar(x[i:i + 1], v[i:i + 1], xerr=xerr[:, i:i + 1], yerr=yerr[:, i:i + 1],
                    fmt=ds.marker, markerfacecolor=ds.color, markeredgecolor=ds.edge_color,
                    markeredgewidth=0.5, ecolor=ds.color, capsize=ds.capsize,
                    markersize=ds.markersize, zorder=2)
    band = (ref_bands or {}).get(pname)
    if band is not None:
        ax.axhspan(*band, color="grey", alpha=0.30, linewidth=0, zorder=0,
                   label=_REF_BAND_LABELS.get(pname))
        ax.legend(fontsize=14, frameon=False, loc="upper left")
    if ylims and pname in ylims:
        ax.set_ylim(*ylims[pname])
    ax.set_ylabel(_PARAM_LABELS[pname], fontsize=18)
    ax.tick_params(axis="both", which="major", direction="in",
                   labelsize=15, length=6, width=1.6)


def _format_glat_axis(ax, edges, log) -> None:
    """Log (or linear) |b| axis with ticks at the band edges."""
    if log:
        ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(np.unique(edges)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    b_min, b_max = float(edges.min()), float(edges.max())
    ax.set_xlim(*((b_min / 1.15, b_max * 1.1) if log else (b_min - 1.0, b_max + 1.0)))
    ax.set_xlabel(r"$|b|\,[^{\circ}]$", fontsize=18)


def plot_params_vs_glat(
    datasets: Sequence[ParamsDataset],
    bands: Sequence[tuple[float, float]],
    *,
    pnames: tuple[str, str],
    ref_bands: dict[str, tuple[float, float] | None] | None = None,
    ylims: dict[str, tuple[float, float]] | None = None,
    figsize: tuple[float, float] = (6.0, 9.5),
    save: str | Path | None = None,
    show: bool = False,
    dpi: int = 200,
    prior_kind: str | None = None,
    nuisance_mode: str | None = None,
):
    """Two parameters (``pnames``, one panel each) vs ``|b|``, one point per band.

    ``datasets[i]`` is the fit of the |b| band ``bands[i]`` and holds a single
    |l| window (written as the title).  The vertical bar is the
    interval of :func:`window_interval`; for ``Ebreak`` / ``Δγ`` a missing
    side is drawn to the bounds used by :func:`plot_params_vs_glon`.  The
    horizontal bar spans the band.  When every edge is > 0 the |b| axis is
    logarithmic and each point sits at the geometric centre of its band.
    ``ref_bands`` maps a parameter to a grey reference band.
    """
    dss, wfs, x, xerr, edges, log = _glat_points(datasets, bands)

    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.08)

    for ax, pname in zip(axs, pnames):
        _draw_param_vs_glat(ax, pname, dss, wfs, x, xerr, ref_bands=ref_bands, ylims=ylims)
    _format_glat_axis(axs[-1], edges, log)
    axs[0].set_title(rf"${wfs[0].lowl:g}^\circ<|\ell|<{wfs[0].highl:g}^\circ$", fontsize=16)

    stamp_prior_nuisance(axs[0], prior_kind=prior_kind, nuisance_mode=nuisance_mode, color="0.25")

    fig.tight_layout()

    if save is not None:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_params_gammas_vs_glat(
    datasets: Sequence[ParamsDataset],
    bands: Sequence[tuple[float, float]],
    *,
    ref_bands: dict[str, tuple[float, float] | None] | None,
    ylims: dict[str, tuple[float, float]] | None,
    gamma_radius: float,
    gamma_color,
    gamma_offset: float = 0.015,
    figsize: tuple[float, float] = (12.0, 9.5),
    save: str | Path | None = None,
    show: bool = False,
    dpi: int = 200,
    prior_kind: str | None = None,
    nuisance_mode: str | None = None,
):
    """:func:`plot_params_vs_glat` of Ebreak / Δγ (left column) plus ``γ1`` and ``γ2`` vs ``|b|`` (top right).

    The left column is drawn exactly as ``plot_params_vs_glat(pnames=("Ebreak", "dgamma"))``.
    In the top-right panel ``γ1`` is drawn in ``gamma_color`` with circles and
    a solid line through the bands, and ``γ2`` in a darker shade with squares
    and a dashed line (:func:`_gamma_series`), with the interval of
    :func:`window_interval`; a missing side is drawn to the profile-grid edge
    (best ± ``gamma_radius`` = ``cfg.profile_errors.gamma_radius``).  ``γ1``
    / ``γ2`` are shifted by ∓ ``gamma_offset`` (dex on a log |b| axis, else a
    fraction of the |b| span) so both bars stay visible.
    """
    dss, wfs, x, xerr, edges, log = _glat_points(datasets, bands)

    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.08)

    for ax, pname in zip(axs[:, 0], ("Ebreak", "dgamma")):
        _draw_param_vs_glat(ax, pname, dss, wfs, x, xerr, ref_bands=ref_bands, ylims=ylims)
    _format_glat_axis(axs[1, 0], edges, log)
    axs[0, 0].set_title(rf"${wfs[0].lowl:g}^\circ<|\ell|<{wfs[0].highl:g}^\circ$", fontsize=16)

    ax_g = axs[0, 1]
    axs[1, 1].remove()
    span = float(edges.max() - edges.min())
    ref = replace(dss[0], color=gamma_color, marker="o", fill=True)
    for pname, label, ls, color, marker, _, dx in _gamma_series(ref, gamma_offset, shade=True):
        v, lo, hi = np.array([window_interval(wf, ds, pname) for wf, ds in zip(wfs, dss)]).T
        lo = np.where(np.isfinite(lo), lo, v - gamma_radius)
        hi = np.where(np.isfinite(hi), hi, v + gamma_radius)
        ax_g.errorbar(
            x * 10.0 ** dx if log else x + dx * span, v,
            yerr=np.clip(np.vstack([v - lo, hi - v]), 0.0, None),
            fmt=marker, ls=ls, lw=1.5, color=color, markerfacecolor=color,
            markeredgecolor=ref.edge_color, markeredgewidth=0.5, ecolor=color,
            capsize=ref.capsize, markersize=ref.markersize, label=label,
        )
    ax_g.legend(fontsize=14, frameon=False, loc="best")
    ax_g.set_ylabel(r"$\gamma$", fontsize=18)
    ax_g.set_xlabel(r"$|b|\,[^{\circ}]$", fontsize=18)
    ax_g.tick_params(axis="both", which="major", direction="in",
                     labelsize=15, length=6, width=1.6, labelbottom=True)

    stamp_prior_nuisance(axs[0, 0], prior_kind=prior_kind, nuisance_mode=nuisance_mode, color="0.25")

    fig.tight_layout()

    if save is not None:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig
