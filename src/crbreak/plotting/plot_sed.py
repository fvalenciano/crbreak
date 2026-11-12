"""SED grid plot (per-window PL + SBPL).

Single-dataset on purpose: one call → one figure for either PASS8 or IEM.
Supports the full grid (all windows in a multi-panel figure) or a single
window (one Axes), the way the legacy code did it.

Default styling (legacy-faithful):

* data points colored per dataset (``cornflowerblue`` for PASS8,
  ``tab:orange`` for IEM); marker labelled in the legend with the dataset
  name (e.g. ``"PASS8"``);
* PL curve in solid ``tab:red``;
* SBPL curve in ``forestgreen`` with a ``mediumseagreen`` MC envelope;
* CR ``E_break`` shown as a grey vertical band labelled ``"Local CR Break"``;
* per-window best-fit ``E_break`` ±1σ band in the dataset color;
* ROI text (``|l|``, ``|b|``) in the bottom-right corner.

Profiled fits (output of a *bound*
:class:`crbreak.fitting.nuisance.NuisanceModel`) are plotted on top with
a dotted line style: pass the per-window stored names via
``pl_prof_name`` / ``sbpl_prof_name`` and the SED gets the deformed
prediction (:meth:`NuisanceModel.predict_deformed_on_grid`) so the user
can see where the fit pulls the model away from the physical surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter

from ..fitting.envelope import envelope_from_cov
from ..fitting.nuisance import NuisanceModel
from ._run_stamp import stamp_prior_nuisance


# ----------------------------------------------------------------------
# Style container (light-weight; users can pass kwargs directly too).
# ----------------------------------------------------------------------


@dataclass
class SedStyle:
    """Color / line styling for an SED panel."""

    data_color: str = "cornflowerblue"  # PASS8 default; "tab:orange" for IEM
    data_label: str = "PASS8"  # legend label for the data marker
    pl_color: str = "tab:red"
    pl_ls: str = "-"  # solid PL by default
    sbpl_color: str = "forestgreen"
    sbpl_band_color: str = "mediumseagreen"
    sbpl_global_ls: str = ":"
    ebreak_color: str | None = None  # falls back to ``data_color`` if None
    cr_band_color: str = "grey"
    cr_band_alpha: float = 0.40
    corr_sys_alpha: float = 0.14
    # Single-panel annotation helpers (Eb bands + CR band labels).
    # Make the classical band more visible; keep the profiled one lighter.
    eb_uncorr_alpha: float = 0.14  # classical SBPL Eb band ("Uncorr. sys.")
    eb_corr_alpha: float = 0.12  # profiled SBPL Eb band ("Corr. sys. (extreme)")
    band_label_fs: int = 13


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _resolve_window_keys(fitting, window_keys) -> list[str]:
    """Normalize the ``window_keys`` argument to a sorted list of strings."""
    if window_keys is None:
        keys = list(fitting.window_fits.keys())
    elif isinstance(window_keys, str):
        keys = [window_keys]
    else:
        keys = list(window_keys)
    # Sort by ROI longitude so figure layout is stable.
    return sorted(keys, key=lambda k: fitting[k].lowl)


def _make_axes(
    n_panels: int,
    *,
    n_cols: int,
    panel_size: tuple[float, float],
    sharex: bool,
    sharey: bool,
):
    if n_panels == 1:
        fig, ax = plt.subplots(figsize=panel_size)
        return fig, np.array([ax])
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows),
        sharex=sharex,
        sharey=sharey,
    )
    return fig, np.atleast_1d(axes).flatten()


def _draw_one_panel(
    ax,
    wf,
    *,
    key: str,
    g,
    style: SedStyle,
    sbpl_name: str,
    pl_name: str,
    pl_prof_name: str | None,
    sbpl_prof_name: str | None,
    sed_scaling: float,
    sed_band,
    show_envelope: bool,
    show_global: bool,
    show_profiled_curves: bool,
    show_eb_uncorr_band: bool,
    show_eb_corr_band: bool,
    cr_ebr_band: tuple[float, float] | None,
    cr_ebr_fixed: float | None,
    annotate_roi: bool,
    annotate_fit: bool,
    annotate_band_labels: bool,
    e_break_legend_as_vertical_text: bool,
    extra_annotations: dict | None,
    legend: bool,
    n_dense: int,
    dense_pad: float,
    tick_labelsize: int = 20,
) -> None:
    E = np.asarray(wf.E_GeV, dtype=float)
    scale = E**sed_scaling
    ebreak_color = style.ebreak_color or style.data_color

    ax.errorbar(
        E,
        wf.y * scale,
        yerr=wf.sigma * scale,
        fmt="o",
        color=style.data_color,
        ecolor="black",
        elinewidth=1,
        markeredgecolor="black",
        markeredgewidth=0.7,
        capsize=0,
        markersize=8,
        lw=0.8,
        label=style.data_label if legend else None,
    )

    E_dense = np.logspace(
        np.log10(E.min() / dense_pad),
        np.log10(E.max() * dense_pad),
        n_dense,
    )
    scale_dense = E_dense**sed_scaling

    pl = wf[pl_name] if pl_name in wf else None
    sbpl = wf[sbpl_name] if sbpl_name in wf else None
    pl_prof = wf[pl_prof_name] if (pl_prof_name and pl_prof_name in wf) else None
    sbpl_prof = (
        wf[sbpl_prof_name] if (sbpl_prof_name and sbpl_prof_name in wf) else None
    )

    # Style convention:
    #   * profiled curve (the deformed prediction that *actually fits the
    #     data* once nuisance parameters are profiled out)  →  solid.
    #   * classical curve (no deformation; legacy chi²)      →  dashed
    #     when a profiled overlay is also drawn (so they're easy to
    #     distinguish), solid otherwise (legacy default).
    pl_classical_ls = "--" if pl_prof is not None else style.pl_ls
    sbpl_classical_ls = "--" if sbpl_prof is not None else "-"

    if pl is not None:
        ax.plot(
            E_dense,
            pl.predict(E_dense) * scale_dense,
            color=style.pl_color,
            lw=1.4,
            ls=pl_classical_ls,
            label=("PL" if legend and pl_prof is None else None),
        )

    if pl_prof is not None and show_profiled_curves:
        if isinstance(pl_prof.model, NuisanceModel) and pl_prof.model.is_bound:
            w_best = getattr(pl_prof, "w_best", None)
            mu_def = pl_prof.model.predict_deformed_on_grid(
                E_dense, tuple(pl_prof.popt), w=w_best
            )
        else:
            mu_def = pl_prof.predict(E_dense)
        ax.plot(
            E_dense,
            mu_def * scale_dense,
            color=style.pl_color,
            lw=1.6,
            ls="-",
            alpha=1.0,
            label="PL" if legend else None,
        )

    if sbpl is not None:
        ax.plot(
            E_dense,
            sbpl.predict(E_dense) * scale_dense,
            color=style.sbpl_color,
            lw=1.5,
            ls=sbpl_classical_ls,
            label=("SBPL" if legend and sbpl_prof is None else None),
        )

    # Envelope: drawn from the *primary* SBPL fit so the band is always
    # consistent with the curve / point reported elsewhere.  ``primary``
    # is the profiled fit when available (USE_CLASSICAL=False flow) and
    # falls back to the classical fit (USE_CLASSICAL=True or no
    # profiled overlay).
    primary_sbpl = sbpl_prof if sbpl_prof is not None else sbpl
    if primary_sbpl is not None and show_envelope:
        try:
            # MC percentile band on the predicted curve (``plots.sed_band``).
            band = dict(n=sed_band.n_samples, q=tuple(sed_band.q), seed=sed_band.seed)
            if (
                isinstance(primary_sbpl.model, NuisanceModel)
                and primary_sbpl.model.is_bound
            ):
                w_best = getattr(primary_sbpl, "w_best", None)
                central = primary_sbpl.model.predict_deformed_on_grid(
                    E_dense, tuple(primary_sbpl.popt), w=w_best
                )
                lo, hi = envelope_from_cov(
                    lambda E_, *pars: primary_sbpl.model.predict_deformed_on_grid(
                        E_, tuple(pars), w=w_best
                    ),
                    primary_sbpl.popt,
                    primary_sbpl.pcov,
                    E_dense,
                    **band,
                )
            else:
                central = primary_sbpl.predict(E_dense)
                lo, hi = primary_sbpl.envelope(E_dense, **band)
            lo = np.minimum(lo, central)
            hi = np.maximum(hi, central)
            ax.fill_between(
                E_dense,
                lo * scale_dense,
                hi * scale_dense,
                color=style.sbpl_band_color,
                alpha=0.22,
                linewidth=0,
            )
        except Exception:
            pass

    if sbpl_prof is not None and show_profiled_curves:
        if isinstance(sbpl_prof.model, NuisanceModel) and sbpl_prof.model.is_bound:
            w_best = getattr(sbpl_prof, "w_best", None)
            mu_def = sbpl_prof.model.predict_deformed_on_grid(
                E_dense, tuple(sbpl_prof.popt), w=w_best
            )
        else:
            mu_def = sbpl_prof.predict(E_dense)
        ax.plot(
            E_dense,
            mu_def * scale_dense,
            color=style.sbpl_color,
            lw=1.7,
            ls="-",
            alpha=1.0,
            label="SBPL" if legend else None,
        )

    if g is not None and show_global and key in g.window_keys:
        ax.plot(
            E_dense,
            g.predict_window(key, E_dense) * scale_dense,
            color=style.sbpl_color,
            lw=1.4,
            ls=style.sbpl_global_ls,
        )

    # CR reference: grey vertical band.  The central dashed CR line is optional
    # and disabled by default to match the legacy final SED styling.
    if cr_ebr_band is not None:
        lo_b, hi_b = cr_ebr_band
        ax.axvspan(
            lo_b,
            hi_b,
            color=style.cr_band_color,
            alpha=style.cr_band_alpha,
            linewidth=0,
            zorder=0,
        )
        if annotate_band_labels:
            # Anchor label at the *left edge* of the CR band (inside it).
            x_mid = float(lo_b) * 1.01
            ax.text(
                x_mid,
                0.50,
                "Local CR Break",
                transform=ax.get_xaxis_transform(),
                rotation=90,
                ha="left",
                va="center",
                fontsize=style.band_label_fs,
                color="dimgray",
                alpha=0.9,
                zorder=3,
                clip_on=True,
            )
    if cr_ebr_fixed is not None:
        ax.axvline(
            cr_ebr_fixed,
            color=style.cr_band_color,
            ls="--",
            lw=1.0,
            alpha=0.9,
            zorder=1,
        )

    # Best-fit Ebreak bands (SBPL only). For a profiled overlay we show two:
    #   - classical SBPL Eb band ("Uncorr. sys.")
    #   - profiled SBPL Eb band ("Corr. sys. (extreme)")
    # Otherwise we show the classical one only (legacy behaviour).
    def _eb_band_from_fit(fr, name_key: str | None):
        if fr is None or "Ebreak" not in fr.params:
            return float("nan"), float("nan"), float("nan")
        # Profiled SBPL: prefer asymmetric Δχ²=1 errors if present on the WindowFitResult.
        if name_key is not None:
            prof_err = getattr(wf, "profile_errors", {}).get(name_key)
            if prof_err and "Ebreak" in prof_err:
                Eb_best, Eb_lo_abs_, Eb_hi_abs_ = prof_err["Ebreak"]
                return float(Eb_best), float(Eb_lo_abs_), float(Eb_hi_abs_)
        Eb_best, Eb_err = fr.params["Ebreak"]
        if np.isfinite(Eb_err):
            return float(Eb_best), float(Eb_best - Eb_err), float(Eb_best + Eb_err)
        return float(Eb_best), float("nan"), float("nan")

    Eb_c, Eb_c_lo, Eb_c_hi = _eb_band_from_fit(sbpl, None)
    Eb_p, Eb_p_lo, Eb_p_hi = _eb_band_from_fit(sbpl_prof, sbpl_prof_name)

    # Classical (uncorrelated) band.
    if (
        show_eb_uncorr_band
        and np.isfinite(Eb_c)
        and np.isfinite(Eb_c_lo)
        and np.isfinite(Eb_c_hi)
        and Eb_c_hi > Eb_c_lo
    ):
        ax.axvspan(
            Eb_c_lo,
            Eb_c_hi,
            color=ebreak_color,
            alpha=style.eb_uncorr_alpha,
            linewidth=0,
            zorder=2,
        )

    # Profiled (correlated/extreme) band.
    if (
        show_eb_corr_band
        and np.isfinite(Eb_p)
        and np.isfinite(Eb_p_lo)
        and np.isfinite(Eb_p_hi)
        and Eb_p_hi > Eb_p_lo
    ):
        ax.axvspan(
            max(Eb_p_lo, float(E_dense.min())),
            Eb_p_hi,
            color=ebreak_color,
            alpha=style.eb_corr_alpha,
            linewidth=0,
            zorder=2.1,
        )

    # Single-panel figures: label Eb bands like "Local CR Break" (vertical at
    # left edge, blended y) instead of the multi-panel Patch legend.
    if (
        (show_eb_uncorr_band or show_eb_corr_band)
        and e_break_legend_as_vertical_text
        and annotate_band_labels
    ):
        def _eb_sys_label(x_left: float, label: str) -> None:
            ax.text(
                float(x_left) * 1.01,
                0.50,
                label,
                transform=ax.get_xaxis_transform(),
                rotation=90,
                ha="left",
                va="center",
                fontsize=style.band_label_fs,
                color=ebreak_color,
                alpha=0.9,
                zorder=3.2,
                clip_on=True,
            )

        if (
            show_eb_uncorr_band
            and np.isfinite(Eb_c)
            and np.isfinite(Eb_c_lo)
            and np.isfinite(Eb_c_hi)
            and Eb_c_hi > Eb_c_lo
        ):
            _eb_sys_label(Eb_c_lo, "Uncorrelated sys.")
        if (
            show_eb_corr_band
            and np.isfinite(Eb_p)
            and np.isfinite(Eb_p_lo)
            and np.isfinite(Eb_p_hi)
            and Eb_p_hi > Eb_p_lo
        ):
            _eb_sys_label(max(Eb_p_lo, float(E_dense.min())), "Correlated sys.")

    # Central best-fit line for the primary Eb (profiled when present, else classical).
    Eb_primary = Eb_p if np.isfinite(Eb_p) else Eb_c
    if np.isfinite(Eb_primary):
        ax.axvline(Eb_primary, color=ebreak_color, ls="--", lw=1.2, zorder=3)

    # Cosmetic axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(E_dense.min(), E_dense.max())
    y_scaled = wf.y * scale
    y_pos = y_scaled[np.isfinite(y_scaled) & (y_scaled > 0)]
    if y_pos.size:
        ax.set_ylim(0.85 * float(np.min(y_pos)), 1.20 * float(np.max(y_pos)))
    # Keep y tick labels "clean": prefer ticks at integer mantissas only
    # (avoid 1.2×10^n / 2.4×10^n / 3.8×10^n). This is more reliable than
    # LogLocator when matplotlib decides to pick intermediate values on a
    # narrow log-range.
    try:
        ymin, ymax = ax.get_ylim()
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin > 0 and ymax > 0:
            ylo, yhi = (min(ymin, ymax), max(ymin, ymax))
            p_lo = int(np.floor(np.log10(ylo)))
            p_hi = int(np.ceil(np.log10(yhi)))

            ticks: list[float] = []
            for p in range(p_lo - 1, p_hi + 2):
                base = 10.0**p
                m_lo = int(np.ceil(ylo / base - 1e-12))
                m_hi = int(np.floor(yhi / base + 1e-12))
                if m_hi < m_lo:
                    continue
                # Avoid extreme mantissas; keep ticks readable.
                m_lo = max(m_lo, 1)
                m_hi = min(m_hi, 9)
                for m in range(m_lo, m_hi + 1):
                    ticks.append(float(m) * base)

            # If too few ticks (e.g. span crosses mantissa boundaries), add a
            # conservative fallback set.
            if len(ticks) < 2:
                mant = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0], dtype=float)
                for p in range(p_lo - 1, p_hi + 2):
                    ticks.extend((mant * (10.0**p)).tolist())
                ticks = [t for t in ticks if (t >= ylo * 0.999) and (t <= yhi * 1.001)]

            # Downsample if the range is wide enough to create many ticks.
            ticks = sorted(set(ticks))
            if len(ticks) > 10:
                step = int(np.ceil(len(ticks) / 10))
                ticks = ticks[::step]

            if len(ticks) >= 2:
                ax.yaxis.set_major_locator(FixedLocator(ticks))
            else:
                ax.yaxis.set_major_locator(
                    LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0))
                )
        else:
            ax.yaxis.set_major_locator(
                LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0))
            )
    except Exception:
        ax.yaxis.set_major_locator(
            LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0))
        )
    ax.yaxis.set_major_formatter(
        LogFormatterSciNotation(base=10.0, labelOnlyBase=False)
    )
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        labelsize=int(tick_labelsize),
        length=6,
        width=1.5,
    )
    ax.tick_params(axis="both", which="minor", direction="in", length=3.0, width=1.15)

    # ROI text → bottom-right corner.
    if annotate_roi:
        bbox = dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            alpha=0.8,
            edgecolor="none",
        )
        ax.text(
            0.95,
            0.16,
            rf"${wf.lowb:g}^\circ<|b|<{wf.highb:g}^\circ$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=tick_labelsize,
            bbox=bbox,
        )
        ax.text(
            0.95,
            0.05,
            rf"${wf.lowl:g}^\circ<|l|<{wf.highl:g}^\circ$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=tick_labelsize,
            bbox=bbox,
        )

    # Optional fit-info text (Eb, dgamma, dchi2, ...). Off by default; turn on
    # for diagnostics / sanity scripts.
    if annotate_fit and sbpl is not None:
        Eb_w, Eb_w_err = sbpl.params.get("Ebreak", (np.nan, np.nan))
        dg_w, dg_w_err = sbpl.params.get("dgamma", (np.nan, np.nan))
        info = (
            f"$E_b$={Eb_w:.1f}±{Eb_w_err:.1f}\n"
            f"$\\Delta\\gamma$={dg_w:.2f}±{dg_w_err:.2f}"
        )
        if pl is not None:
            d = wf.delta_chi2(pl_name, sbpl_name)
            info += f"\n$\\Delta\\chi^2_{{{pl_name}-{sbpl_name}}}$={d:.1f}"
        if extra_annotations and wf.glon_bin in extra_annotations:
            info += "\n" + str(extra_annotations[wf.glon_bin])
        ax.text(
            0.05,
            0.95,
            info,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.85,
                edgecolor="none",
            ),
        )

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ordered = [x for x in ("SBPL", "PL", style.data_label) if x in by_label]
        ax.legend(
            [by_label[x] for x in ordered],
            ordered,
            frameon=False,
            fontsize=tick_labelsize - 1,
            loc="upper left",
        )


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def plot_sed(
    fitting,
    *,
    g=None,
    window_keys: Sequence[str] | str | None = None,
    style: SedStyle | None = None,
    pl_name: str = "PL",
    sbpl_name: str = "SBPL",
    pl_prof_name: str | None = None,
    sbpl_prof_name: str | None = None,
    sed_scaling: float,
    sed_band,
    cr_ebr_band: tuple[float, float] | None,
    show_envelope: bool = True,
    show_global: bool = False,
    show_profiled_curves: bool = True,
    show_eb_uncorr_band: bool = True,
    show_eb_corr_band: bool = True,
    cr_ebr_fixed: float | None = None,
    annotate_roi: bool = True,
    annotate_fit: bool = False,
    extra_annotations: Mapping[tuple[float, float], str] | None = None,
    n_cols: int = 3,
    panel_size: tuple[float, float] = (6.0, 5.0),
    sharex: bool = True,
    sharey: bool = False,
    save: str | Path | None = None,
    show: bool = False,
    dpi: int = 240,
    n_dense: int = 500,
    dense_pad: float = 1.2,
    prior_kind: str | None = None,
    nuisance_mode: str | None = None,
    ebreak_band_legend: bool = True,
):
    """Plot the SED + per-window fits for a single dataset.

    Parameters
    ----------
    fitting : :class:`crbreak.fitting.main.Fitting`
        Already populated via ``Fitting.fit_per_window([PowerLaw(), SmoothBrokenPowerLaw()])``.
    g : :class:`crbreak.fitting.global_fit.GlobalFitResult`, optional
        If given, the global SBPL prediction is overlaid in each panel
        (same color as the per-window SBPL, dotted; *not* shown in the
        legend by default).
    window_keys : str | sequence of str | None
        ``None``   -> all windows in a multi-panel grid (the default).
        ``str``    -> a single panel for that window.
        ``list``   -> a multi-panel grid restricted to those windows.
    style : :class:`SedStyle`
        Color / line styling.  Defaults to PASS8 colors (data in
        ``cornflowerblue``, label ``"PASS8"``); for IEM use e.g.
        ``SedStyle(data_color='tab:orange', data_label='IEM')``.
    pl_name, sbpl_name : str
        Names under which the classical PL / SBPL :class:`FitResult` were
        stored on each :class:`WindowFitResult` (also used as legend
        labels).
    pl_prof_name, sbpl_prof_name : str, optional
        Optional names of profiled (nuisance) fits stored on the same
        :class:`WindowFitResult`.  When given, the profiled deformed
        prediction (:meth:`NuisanceModel.predict_deformed_on_grid` of
        the bound model) is overlaid as a dotted line in the same color
        as the corresponding classical curve.  If a window does not have
        the requested profiled fit, that overlay is silently skipped.
    sed_scaling
        ``y = E^sed_scaling dN/dE`` on the vertical axis (``data.sed_scaling``).
    sed_band
        ``plots.sed_band`` of the analysis config (``n_samples``, ``q``,
        ``seed`` of the SBPL band).
    cr_ebr_band, cr_ebr_fixed
        If non-``None``, draw the grey CR vertical band and, optionally, a
        thin dashed reference line at the CR ``E_break`` central value.
    annotate_roi : bool
        Show the ``{lowl}<|l|<{highl}, |b|<{highb}`` text box in the
        bottom-right corner of each panel.  ``True`` by default.
    annotate_fit : bool
        Show a top-left text box with ``Eb``, ``Δγ``, ``Δχ²``...  Off by
        default; turn on for diagnostic / sanity scripts.
    extra_annotations
        Mapping ``{wf.glon_bin: text_to_append}`` to extend the fit-info
        box (only effective when ``annotate_fit=True``).
    save : path-like, optional
        Where to save the figure (``.png`` extension recommended).
    show : bool
        Call ``plt.show()`` after saving.
    ebreak_band_legend : bool
        If ``True`` (default), multi-panel figures add a Patch legend on the
        second panel explaining ``Uncorr. sys.`` / ``Corr. sys. (extreme)``
        bands for SBPL \(E_{\rm break}\). Single-panel figures draw the same
        text vertically at each band's left edge (dataset color), like
        ``Local CR Break``. Set ``False`` for control-region plots where
        those bands are not meaningful.

    Returns
    -------
    matplotlib.figure.Figure
    """
    style = style or SedStyle()
    keys = _resolve_window_keys(fitting, window_keys)
    if not keys:
        raise ValueError("No windows selected for plot_sed.")

    n_panels = len(keys)
    if n_panels == 1 and str(style.data_label).upper() == "PASS8":
        style = replace(
            style,
            data_label=r"$\mathit{Fermi}\mathrm{-LAT}$",
        )
    fig, axes = _make_axes(
        n_panels,
        n_cols=min(n_cols, n_panels),
        panel_size=panel_size,
        sharex=sharex,
        sharey=sharey,
    )

    for i, key in enumerate(keys):
        wf = fitting[key]
        tick_ls = 18 if n_panels == 1 else 20
        # Annotate ROI / fit on every panel; only the first panel gets a
        # legend so the figure stays readable.
        _draw_one_panel(
            axes[i],
            wf,
            key=key,
            g=g,
            style=style,
            sbpl_name=sbpl_name,
            pl_name=pl_name,
            pl_prof_name=pl_prof_name,
            sbpl_prof_name=sbpl_prof_name,
            sed_scaling=sed_scaling,
            sed_band=sed_band,
            show_envelope=show_envelope,
            show_global=show_global,
            show_profiled_curves=show_profiled_curves,
            show_eb_uncorr_band=show_eb_uncorr_band,
            show_eb_corr_band=show_eb_corr_band,
            cr_ebr_band=cr_ebr_band,
            cr_ebr_fixed=cr_ebr_fixed,
            annotate_roi=annotate_roi,
            annotate_fit=annotate_fit,
            annotate_band_labels=True,
            e_break_legend_as_vertical_text=(
                n_panels == 1 and ebreak_band_legend
            ),
            extra_annotations=extra_annotations,
            legend=(i == 0),
            n_dense=n_dense,
            dense_pad=dense_pad,
            tick_labelsize=tick_ls,
        )

    # Run configuration stamp (requested: on 3rd panel for grids).
    if prior_kind is not None and nuisance_mode is not None and len(axes) > 0:
        target = axes[2] if len(keys) > 2 and len(axes) > 2 else axes[0]
        stamp_prior_nuisance(
            target,
            prior_kind=prior_kind,
            nuisance_mode=nuisance_mode,
            color="0.25",
        )

    if (
        ebreak_band_legend
        and (show_eb_uncorr_band or show_eb_corr_band)
        and n_panels > 1
        and len(axes) > 1
    ):
        ebreak_color = style.ebreak_color or style.data_color
        eb_handles = []
        if show_eb_uncorr_band:
            eb_handles.append(
                Patch(
                    facecolor=ebreak_color,
                    alpha=style.eb_uncorr_alpha,
                    edgecolor="none",
                    label="Uncorrelated sys.",
                )
            )
        if show_eb_corr_band:
            eb_handles.append(
                Patch(
                    facecolor=ebreak_color,
                    alpha=style.eb_corr_alpha,
                    edgecolor="none",
                    label="Correlated sys.",
                )
            )
        axes[1].legend(
            handles=eb_handles,
            frameon=False,
            fontsize=19,
            loc="upper left",
        )

    # Drop unused axes (when n_panels < n_rows*n_cols)
    for j in range(n_panels, len(axes)):
        try:
            fig.delaxes(axes[j])
        except Exception:
            pass

    s = f"{sed_scaling:g}"
    s_minus_1 = f"{(float(sed_scaling) - 1.0):g}"
    if n_panels > 1:
        fig.supxlabel("Energy [GeV]", fontsize=24, y=0.055)
        fig.supylabel(
            rf"$E^{{{s}}}\,J\;[\mathrm{{GeV^{{{s_minus_1}}}\,cm^{{-2}}\,s^{{-1}}\,sr^{{-1}}}}]$",
            fontsize=24,
            x=0.06,
            y=0.62,
        )
        fig.tight_layout(rect=[0.065, 0.06, 1.0, 0.99])
    else:
        axes[0].set_xlabel("Energy [GeV]", fontsize=18)
        axes[0].set_ylabel(
            rf"$E^{{{s}}}\,J\;[\mathrm{{GeV^{{{s_minus_1}}}\,cm^{{-2}}\,s^{{-1}}\,sr^{{-1}}}}]$",
            fontsize=18,
        )
        fig.tight_layout()

    if save is not None:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig
