"""Per-window nuisance posterior plot (``s(E;w)`` + ``w_j``).

Single-dataset on purpose, like :mod:`crbreak.plotting.plot_sed`: one
call → one figure.  For each window the panel shows

* the deformation curve ``s(E;w) = B(log10 E) @ w_hat`` with 68 % and
  90 % posterior bands obtained by sampling
  ``w ~ N(w_hat, Cov_w)``, where ``Cov_w = inv(H_chi2_w)`` is the
  Hessian of the *inner* problem (``chi2`` as a function of ``w`` only,
  with the spectral parameters held at the profiled best-fit
  ``theta_hat``);
* the per-coefficient ``w_j`` with ±1σ error bars at the basis nodes
  ``x_nuis``;
* both the PL and the SBPL profiled fits overlaid (PL dotted, SBPL
  solid) in the dataset color.

This is the legacy ``plot_s_posterior_vs_logE`` plot from
``legacy/nuisance_fit.py``, ported to the modular ``Fitting`` API and
arranged as a per-window grid (mirroring :func:`plot_sed`).

Axis semantics
--------------
The vertical axis is the dimensionless deformation ``s(E;\mathbf{w})
= B(\log_{10} E)\,\mathbf{w}``.  Shaded curves/bands show the posterior
of ``s(E)``; markers are knot coefficients ``\hat w_j`` (±1σ), which for
this basis satisfy ``w_j = s(E_j)`` at the knot energies.

Common-nuisance mode
--------------------
When the user provides ``pl_prof_global`` and/or ``sbpl_prof_global``
(:class:`~crbreak.fitting.global_fit.GlobalFitResult` from a fit with
``NuisanceModel(share_nuisance=True)``), there is a **single global**
``w_shared`` for the whole dataset (legacy
``plot_s_posterior_vs_logE_shared``).  In that case ``plot_nuisance``
draws a **single panel** with ``s(E; w_shared)`` and ``w_j`` plus the
covariance-derived band, instead of the per-window grid.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from ..fitting.nuisance import NuisanceModel
from ..fitting.nuisance._solver_utils import _fd_hessian
from .plot_sed import _make_axes, _resolve_window_keys
from ._run_stamp import stamp_prior_nuisance

# Match :func:`plot_sed` typography for **single-panel** nuisance figures
# (one longitude window or shared-nuisance mode), per publication request.
_SED_SINGLE_XLABEL_FS = 20
_SED_SINGLE_YLABEL_FS = 24
_SED_SINGLE_TICK_LABELSIZE = 20

# Y-axis (publication): deformation only; ``w_{\mathrm{shared}}`` is not spelled out.
_NUIS_YLABEL = r"$s(E;\,\mathbf{w})$"
# Top-right panel tag (“Common”, …): smaller than ROI text so it stays discreet.
_NUIS_CORNER_TAG_FS = 17


def _nuisance_corner_tag_text(
    fit_nuisance_mode: str | None, *, shared_mode: bool
) -> str | None:
    """Top-right tag for shared-nuisance single panels."""
    if not shared_mode:
        return None
    nm = str(fit_nuisance_mode or "").strip().lower().replace("-", "_")
    if nm in {"control_common", "control_common_wsbpl"}:
        return "All-sky control-common"
    return "Common"


def _draw_nuisance_corner_tag(ax, text: str, *, fontsize: int) -> None:
    ax.text(
        0.97,
        0.97,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=int(fontsize),
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            alpha=0.88,
            edgecolor="none",
        ),
    )


def _apply_sed_single_panel_tickstyle(ax) -> None:
    """Major/minor tick lengths and label sizes aligned with ``plot_sed``."""
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        labelsize=_SED_SINGLE_TICK_LABELSIZE,
        length=6,
        width=1.5,
    )
    ax.tick_params(axis="both", which="minor", direction="in", length=3.0, width=1.15)


def _style_for_single_panel_like_sed(style: NuisanceStyle) -> NuisanceStyle:
    return replace(
        style,
        tick_fs=_SED_SINGLE_TICK_LABELSIZE,
        roi_fs=20,
        legend_fs=19,
    )


# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------


@dataclass
class NuisanceStyle:
    """Color / line styling for a nuisance panel.

    Styling for the PL vs SBPL overlays.

    Default is to *separate models by color* (PL red, SBPL green) rather than
    by dataset color, so the nuisance posterior figure looks consistent across
    PASS8/IEM and matches the SED SBPL palette.
    """

    # Kept for back-compat with older callers; no longer used by default.
    data_color: str = "cornflowerblue"
    data_label: str = "PASS8"
    pl_ls: str = "-"
    sbpl_ls: str = "-"
    pl_color: str = "indianred"
    # Match SED palette: dark green line + lighter green bands.
    sbpl_color: str = "forestgreen"
    sbpl_band_color: str = "mediumseagreen"
    # SBPL bands (PL uses its own alphas below).
    band_alpha_68: float = 0.22
    band_alpha_90: float = 0.12
    # PL band alphas (requested: rgba(178,34,34,α)).
    pl_band_alpha_68: float = 0.25
    pl_band_alpha_90: float = 0.12
    line_alpha: float = 1.0
    cr_band_color: str = "grey"
    # Horizontal offset in log10(E) for the w_j errorbars to avoid overlap.
    w_offset_x: float = 0.10
    # Font sizing (kept local to this figure so it doesn't affect global rcParams).
    label_fs: int = 20
    title_fs: int = 20
    legend_fs: int = 17
    roi_fs: int = 17
    tick_fs: int = 16


# ----------------------------------------------------------------------
# Cov(w) helper
# ----------------------------------------------------------------------


def _robust_inv(H: np.ndarray, ridge: float) -> np.ndarray:
    """Symmetric inverse with a tiny ridge fallback (legacy ``robust_inv``)."""
    H = 0.5 * (np.asarray(H, dtype=float) + np.asarray(H, dtype=float).T)
    n = H.shape[0]
    lam = ridge
    for _ in range(6):
        try:
            return np.linalg.inv(H + lam * np.eye(n))
        except np.linalg.LinAlgError:
            lam *= 10.0
    return np.linalg.pinv(H + lam * np.eye(n))


def _build_chi2_w(
    nm: NuisanceModel,
    y_m: np.ndarray,
    sigma_m: np.ndarray,
    theta: tuple[float, ...],
):
    """Return ``chi2_w(w)``: total chi² as a function of ``w`` only.

    Spectral parameters ``theta`` are held fixed.  Inputs ``y_m``/``sigma_m``
    must already be aligned with ``nm.basis.B``/``nm.Ssys_frac``/``nm.E_data``
    (i.e. the legacy ``isfinite & sigma>0`` mask has been applied).
    """
    E = nm.E_data
    B = nm.basis.B
    S = nm.Ssys_frac
    prior = nm.prior
    if E is None or B is None or S is None or prior is None:
        raise ValueError("NuisanceModel must be bound (call `for_window`).")

    def chi2_w(w: np.ndarray) -> float:
        mu = np.asarray(nm.spectral_model(E, *theta), dtype=float)
        m = np.isfinite(mu)
        if not np.any(m):
            return float("inf")
        yhat = (1.0 + (B[m] @ w) * S[m]) * mu[m]
        r = (y_m[m] - yhat) / sigma_m[m]
        chi2_data = float(np.sum(r * r))
        prior_r = prior.augmented_residuals(np.asarray(w, dtype=float))
        chi2_prior = float(np.sum(prior_r * prior_r)) if prior_r.size else 0.0
        return chi2_data + chi2_prior

    return chi2_w


def _masked_window_arrays(wf) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the legacy ``finite & sigma>0`` mask to the window data.

    Mirrors :meth:`NuisanceModel.for_window` so that the returned ``y``,
    ``sigma`` (and ``E``) are aligned with the bound model's
    ``basis.B`` / ``Ssys_frac`` / ``E_data``.
    """
    E = np.asarray(wf.E_GeV, dtype=float)
    y = np.asarray(wf.y, dtype=float)
    sigma = np.asarray(wf.sigma, dtype=float)
    Ssys = wf.meta.get("Ssys_frac")
    if Ssys is None:
        raise RuntimeError(
            f"Window {wf.glon_bin}x{wf.glat_bin}: meta['Ssys_frac'] is "
            "missing.  Re-run extraction so the systematic Aeff fraction is "
            "propagated, otherwise the inner Cov(w) cannot be reconstructed."
        )
    Ssys = np.asarray(Ssys, dtype=float)
    mask = (
        np.isfinite(E)
        & np.isfinite(y)
        & np.isfinite(sigma)
        & np.isfinite(Ssys)
        & (sigma > 0)
    )
    return E[mask], y[mask], sigma[mask]


def _posterior_w(
    fr,
    y_m: np.ndarray,
    sigma_m: np.ndarray,
    *,
    nsamp: int,
    rng: np.random.Generator,
    eps_hess: float,
    ridge_hess: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample ``w`` from a Gaussian posterior centred on ``last_w``.

    Returns
    -------
    w_hat : (n_nuis,) array
        Inner best-fit nuisance vector at ``theta_hat`` (= ``nm.last_w``).
    w_err : (n_nuis,) array
        ±1σ from ``sqrt(diag(Cov_w))``.
    ws : (nsamp, n_nuis) array
        Multivariate-normal draws used to build the ``s(E)`` posterior.
    """
    nm = fr.model
    if not isinstance(nm, NuisanceModel) or not nm.is_bound:
        raise TypeError(
            f"FitResult '{fr.name}' has no bound NuisanceModel; cannot draw "
            "the inner posterior of w."
        )
    w_hat = np.asarray(nm.last_w, dtype=float)
    chi2_w = _build_chi2_w(nm, y_m, sigma_m, tuple(fr.popt))
    H = _fd_hessian(chi2_w, w_hat, eps=eps_hess)
    cov = _robust_inv(H, ridge_hess)
    cov = 0.5 * (cov + cov.T)
    w_err = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    try:
        ws = rng.multivariate_normal(mean=w_hat, cov=cov, size=nsamp)
    except np.linalg.LinAlgError:
        cov_psd = cov + 1e-8 * np.eye(cov.shape[0])
        ws = rng.multivariate_normal(mean=w_hat, cov=cov_psd, size=nsamp)
    return w_hat, w_err, ws


# ----------------------------------------------------------------------
# Per-panel drawing
# ----------------------------------------------------------------------


def _plot_one_model(
    ax,
    fr,
    *,
    color: str,
    band_color: str,
    ls: str,
    label: str,
    marker: str,
    x_offset: float,
    band_alpha_68: float,
    band_alpha_90: float,
    line_alpha: float,
    logE_dense: np.ndarray,
    y_m: np.ndarray,
    sigma_m: np.ndarray,
    nsamp: int,
    rng: np.random.Generator,
    eps_hess: float,
    ridge_hess: float,
    legend: bool,
) -> None:
    """Draw one model's ``s(E)`` band + ``w_j`` errorbars on ``ax``."""
    nm = fr.model
    w_hat, w_err, ws = _posterior_w(
        fr,
        y_m,
        sigma_m,
        nsamp=nsamp,
        rng=rng,
        eps_hess=eps_hess,
        ridge_hess=ridge_hess,
    )

    B_dense = nm.basis.evaluate(logE_dense)
    s_curves = ws @ B_dense.T
    s_med = np.percentile(s_curves, 50, axis=0)
    s_lo68, s_hi68 = np.percentile(s_curves, [16, 84], axis=0)
    s_lo90, s_hi90 = np.percentile(s_curves, [5, 95], axis=0)

    ax.fill_between(
        logE_dense,
        s_lo90,
        s_hi90,
        color=band_color,
        alpha=band_alpha_90,
        linewidth=0,
    )
    ax.fill_between(
        logE_dense,
        s_lo68,
        s_hi68,
        color=band_color,
        alpha=band_alpha_68,
        linewidth=0,
    )
    ax.plot(
        logE_dense,
        s_med,
        color=color,
        ls=ls,
        lw=1.8,
        alpha=line_alpha,
        label=label if legend else None,
    )

    xg = np.asarray(nm.basis.x_nuis, dtype=float)
    ax.errorbar(
        xg + float(x_offset),
        w_hat,
        yerr=w_err,
        fmt=marker,
        color=color,
        ecolor=color,
        markeredgecolor="black",
        markeredgewidth=0.6,
        markersize=5,
        capsize=3,
        elinewidth=1.0,
        alpha=line_alpha,
    )


def _draw_one_panel(
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
    """Draw the per-window nuisance posterior panel."""
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
        return

    E_m, y_m, sigma_m = _masked_window_arrays(wf)
    if E_m.size == 0:
        return
    logE_dense = np.linspace(
        np.log10(float(E_m.min()) / dense_pad),
        np.log10(float(E_m.max()) * dense_pad),
        n_dense,
    )
    rng = np.random.default_rng(seed)

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

    ax.axhline(0.0, color="black", ls="--", lw=0.8, alpha=0.6, zorder=0)

    # CR reference band/line (in log10 E).
    if cr_ebr_band is not None:
        lo_b, hi_b = cr_ebr_band
        ax.axvspan(
            np.log10(lo_b),
            np.log10(hi_b),
            color=style.cr_band_color,
            alpha=0.20,
            linewidth=0,
            zorder=0,
        )
        x_mid = float(np.log10(np.sqrt(lo_b * hi_b)))
        x_mid = min(x_mid + 0.03, float(np.log10(hi_b)) - 0.02)
        ax.text(
            x_mid,
            0.32,
            "Local CR Break",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="center",
            va="center",
            fontsize=max(10, int(style.roi_fs) - 2),
            color="dimgray",
            alpha=0.9,
            zorder=3,
            clip_on=True,
        )
    if cr_ebr_fixed is not None:
        ax.axvline(
            np.log10(cr_ebr_fixed),
            color=style.cr_band_color,
            ls="--",
            lw=1.0,
            alpha=0.9,
            zorder=1,
        )

    ax.set_xlim(logE_dense.min(), logE_dense.max())
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
        # Match the SED ROI stamp: mathtext |$b$| / |$l$| on two lines.
        bbox = dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.85,
            edgecolor="none",
        )
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


# ----------------------------------------------------------------------
# Shared-nuisance panel: ``w_shared`` + ``s(E; w_shared)`` band
# ----------------------------------------------------------------------


def _global_logE_range(fitting, dense_pad: float) -> tuple[float, float]:
    """Min/max log10(E) across all windows (with a small padding)."""
    los, his = [], []
    for wf in fitting.window_fits.values():
        E = np.asarray(wf.E_GeV, dtype=float)
        E = E[np.isfinite(E) & (E > 0)]
        if E.size:
            los.append(float(E.min()))
            his.append(float(E.max()))
    if not los:
        raise RuntimeError("plot_nuisance: no window has finite energies.")
    return (
        np.log10(min(los) / dense_pad),
        np.log10(max(his) * dense_pad),
    )


def _draw_shared_one_model(
    ax,
    g,
    *,
    fitting,
    color: str,
    band_color: str,
    ls: str,
    label: str,
    marker: str,
    x_offset: float,
    band_alpha_68: float,
    band_alpha_90: float,
    line_alpha: float,
    logE_dense: np.ndarray,
    nsamp: int,
    rng: np.random.Generator,
    legend: bool,
) -> None:
    """Draw ``s(E; w_shared)`` band + ``w_j`` errorbars from a global fit.

    ``g`` must be a :class:`GlobalFitResult` produced by a profiled global
    fit with ``NuisanceModel(share_nuisance=True)``: ``g.w_shared`` and
    ``g.cov_w_shared`` are non-``None`` and the basis nodes are common
    across windows.
    """
    if g.w_shared is None or g.cov_w_shared is None:
        ax.text(
            0.5,
            0.5,
            "no shared w in global fit",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
        )
        return

    # The global result carries the representative bound NM with the shared basis.
    bound = getattr(g.model, "is_bound", False) and g.model
    if not bound:
        # GlobalFitResult.model is always a bound NM in profiled mode.
        raise RuntimeError(
            "plot_nuisance(shared): GlobalFitResult.model is not a bound "
            "NuisanceModel; was the global fit run in profiled+shared mode?"
        )

    w_hat = np.asarray(g.w_shared, dtype=float)
    cov = np.asarray(g.cov_w_shared, dtype=float)
    cov = 0.5 * (cov + cov.T)
    w_err = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    try:
        ws = rng.multivariate_normal(mean=w_hat, cov=cov, size=nsamp)
    except np.linalg.LinAlgError:
        ws = rng.multivariate_normal(
            mean=w_hat, cov=cov + 1e-8 * np.eye(cov.shape[0]), size=nsamp
        )

    # Use the basis from any bound NM (all windows share x_nuis in this mode).
    B_dense = bound.basis.evaluate(logE_dense)
    s_hat = B_dense @ w_hat
    s_curves = ws @ B_dense.T
    s_lo68, s_hi68 = np.percentile(s_curves, [16, 84], axis=0)
    s_lo90, s_hi90 = np.percentile(s_curves, [5, 95], axis=0)

    ax.fill_between(
        logE_dense,
        s_lo90,
        s_hi90,
        color=band_color,
        alpha=band_alpha_90,
        linewidth=0,
    )
    ax.fill_between(
        logE_dense,
        s_lo68,
        s_hi68,
        color=band_color,
        alpha=band_alpha_68,
        linewidth=0,
    )
    ax.plot(
        logE_dense,
        s_hat,
        color=color,
        ls=ls,
        lw=1.8,
        alpha=line_alpha,
        label=label if legend else None,
    )
    xg = np.asarray(bound.basis.x_nuis, dtype=float)
    ax.errorbar(
        xg + float(x_offset),
        w_hat,
        yerr=w_err,
        fmt=marker,
        color=color,
        ecolor=color,
        markeredgecolor="black",
        markeredgewidth=0.6,
        markersize=5,
        capsize=3,
        elinewidth=1.0,
        alpha=line_alpha,
    )


def _draw_shared_panel(
    ax,
    fitting,
    *,
    pl_prof_global,
    sbpl_prof_global,
    style: NuisanceStyle,
    cr_ebr_band: tuple[float, float] | None,
    cr_ebr_fixed: float | None,
    legend: bool,
    nsamp: int,
    seed: int,
    n_dense: int,
    dense_pad: float,
) -> None:
    """Single-panel nuisance posterior in common-nuisance mode."""
    logE_lo, logE_hi = _global_logE_range(fitting, dense_pad)
    logE_dense = np.linspace(logE_lo, logE_hi, n_dense)
    rng = np.random.default_rng(seed)

    if pl_prof_global is not None:
        _draw_shared_one_model(
            ax,
            pl_prof_global,
            fitting=fitting,
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
            nsamp=nsamp,
            rng=rng,
            legend=legend,
        )
    if sbpl_prof_global is not None:
        _draw_shared_one_model(
            ax,
            sbpl_prof_global,
            fitting=fitting,
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
            nsamp=nsamp,
            rng=rng,
            legend=legend,
        )

    ax.axhline(0.0, color="black", ls="--", lw=0.8, alpha=0.6, zorder=0)

    if cr_ebr_band is not None:
        lo_b, hi_b = cr_ebr_band
        ax.axvspan(
            np.log10(lo_b),
            np.log10(hi_b),
            color=style.cr_band_color,
            alpha=0.20,
            linewidth=0,
            zorder=0,
        )
        x_mid = float(np.log10(np.sqrt(lo_b * hi_b)))
        x_mid = min(x_mid + 0.03, float(np.log10(hi_b)) - 0.02)
        ax.text(
            x_mid,
            0.32,
            "Local CR Break",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="center",
            va="center",
            fontsize=max(10, int(style.roi_fs) - 2),
            color="dimgray",
            alpha=0.9,
            zorder=3,
            clip_on=True,
        )
    if cr_ebr_fixed is not None:
        ax.axvline(
            np.log10(cr_ebr_fixed),
            color=style.cr_band_color,
            ls="--",
            lw=1.0,
            alpha=0.9,
            zorder=1,
        )

    ax.set_xlim(logE_dense.min(), logE_dense.max())
    ax.set_ylim(-2.5, 2.5)
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
    if legend:
        ax.legend(frameon=False, fontsize=style.legend_fs, loc="upper left")


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def plot_nuisance(
    fitting,
    *,
    pl_prof_name: str | None = None,
    sbpl_prof_name: str | None = None,
    pl_prof_global=None,
    sbpl_prof_global=None,
    style: NuisanceStyle | None = None,
    window_keys: Sequence[str] | str | None = None,
    cr_ebr_band: tuple[float, float] | None,
    nuisance_band,
    cr_ebr_fixed: float | None = None,
    annotate_roi: bool = True,
    n_cols: int = 3,
    panel_size: tuple[float, float] = (6.0, 5.0),
    sharex: bool = True,
    sharey: bool = False,
    n_dense: int = 400,
    dense_pad: float = 1.05,
    save: str | Path | None = None,
    show: bool = False,
    dpi: int = 240,
    prior_kind: str | None = None,
    nuisance_mode: str | None = None,
    stamp_nuisance_mode: str | None = None,
):
    """Plot the per-window nuisance posterior (``s(E;w)`` + ``w_j``).

    Parameters
    ----------
    fitting : :class:`crbreak.fitting.main.Fitting`
        Already populated via
        ``Fitting.fit_per_window([NuisanceModel(PowerLaw()),
        NuisanceModel(SmoothBrokenPowerLaw())])``.
    pl_prof_name, sbpl_prof_name : str, optional
        Names under which the profiled PL / SBPL :class:`FitResult` were
        stored on each :class:`WindowFitResult`.  Each panel overlays
        whichever of the two is available; windows missing both are
        annotated as ``"no profiled fit"``.  Ignored in common-nuisance
        mode (see ``pl_prof_global`` / ``sbpl_prof_global`` below).
    pl_prof_global, sbpl_prof_global : :class:`GlobalFitResult`, optional
        Profiled global fits with shared nuisances
        (``NuisanceModel(share_nuisance=True)``).  When at least one is
        provided **and** carries a non-``None`` ``w_shared``,
        ``plot_nuisance`` switches to a **single-panel** layout showing
        the global ``s(E; w_shared)`` posterior.  Mutually exclusive
        with the per-window grid (which is rendered when these are
        ``None``).
    style : :class:`NuisanceStyle`, optional
        Color / line styling.  Defaults to PASS8 colors
        (``cornflowerblue``); for IEM use e.g.
        ``NuisanceStyle(data_color='tab:orange', data_label='IEM')``.
    window_keys : str | sequence | None
        Same convention as :func:`plot_sed`.  ``None`` plots all windows.
    cr_ebr_band, cr_ebr_fixed
        Optional grey vertical band/line at the canonical CR
        ``E_break``.  Pass ``None`` to suppress.
    nuisance_band
        ``plots.nuisance_band`` of the analysis config: ``n_samples``
        posterior draws of ``w`` (``seed``; panel ``i`` uses ``seed + i``),
        from the Hessian with step ``hessian_eps`` and ridge ``hessian_ridge``.
    save, show, dpi
        Output handling, identical to :func:`plot_sed`.
    nuisance_mode
        Actual fit nuisance configuration string (e.g. ``"common"``,
        ``"control_common"``).  Used for the optional top-right panel tag on
        shared-nuisance figures.
    stamp_nuisance_mode
        If given, passed to :func:`stamp_prior_nuisance` instead of
        ``nuisance_mode`` (e.g. a short display label like ``"Classical"``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    nsamp_band = nuisance_band.n_samples
    seed = nuisance_band.seed
    eps_hess, ridge_hess = nuisance_band.hessian_eps, nuisance_band.hessian_ridge
    style = style or NuisanceStyle()
    _stamp_nm = (
        stamp_nuisance_mode if stamp_nuisance_mode is not None else nuisance_mode
    )

    # ------------------------------------------------------------------
    # Common-nuisance mode: a single global ``w_shared`` for all windows.
    # When the caller passes ``pl_prof_global`` and/or ``sbpl_prof_global``
    # we draw one panel with the shared posterior and ignore
    # ``pl_prof_name`` / ``sbpl_prof_name`` (those would re-evaluate the
    # *frozen* w_shared per window with no extra information).
    # ------------------------------------------------------------------
    shared_mode = (
        pl_prof_global is not None
        and getattr(pl_prof_global, "w_shared", None) is not None
    ) or (
        sbpl_prof_global is not None
        and getattr(sbpl_prof_global, "w_shared", None) is not None
    )
    if shared_mode:
        # Single panel: match ``plot_sed`` figure size / ticks / axis labels.
        shared_style = _style_for_single_panel_like_sed(style)
        fig, axes = _make_axes(
            1,
            n_cols=1,
            panel_size=panel_size,
            sharex=False,
            sharey=False,
        )
        _draw_shared_panel(
            axes[0],
            fitting,
            pl_prof_global=pl_prof_global,
            sbpl_prof_global=sbpl_prof_global,
            style=shared_style,
            cr_ebr_band=cr_ebr_band,
            cr_ebr_fixed=cr_ebr_fixed,
            legend=True,
            nsamp=nsamp_band,
            seed=seed,
            n_dense=n_dense,
            dense_pad=dense_pad,
        )
        _apply_sed_single_panel_tickstyle(axes[0])
        axes[0].set_xlabel(
            r"$\log_{10}(E/\mathrm{GeV})$", fontsize=_SED_SINGLE_XLABEL_FS
        )
        axes[0].set_ylabel(
            _NUIS_YLABEL,
            fontsize=_SED_SINGLE_YLABEL_FS,
        )
        tag_txt = _nuisance_corner_tag_text(nuisance_mode, shared_mode=True)
        if tag_txt is not None:
            _draw_nuisance_corner_tag(axes[0], tag_txt, fontsize=_NUIS_CORNER_TAG_FS)
        stamp_prior_nuisance(
            axes[0],
            prior_kind=prior_kind,
            nuisance_mode=_stamp_nm,
            color="0.25",
            fontsize=12,
        )
        fig.tight_layout()
        if save is not None:
            save_path = Path(save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    if pl_prof_name is None and sbpl_prof_name is None:
        raise ValueError(
            "plot_nuisance: at least one of `pl_prof_name`, "
            "`sbpl_prof_name`, `pl_prof_global` or `sbpl_prof_global` "
            "must be provided."
        )
    keys = _resolve_window_keys(fitting, window_keys)
    if not keys:
        raise ValueError("No windows selected for plot_nuisance.")

    n_panels = len(keys)
    loop_style = _style_for_single_panel_like_sed(style) if n_panels == 1 else style
    fig, axes = _make_axes(
        n_panels,
        n_cols=min(n_cols, n_panels),
        panel_size=panel_size,
        sharex=sharex,
        sharey=sharey,
    )

    for i, key in enumerate(keys):
        wf = fitting[key]
        _draw_one_panel(
            axes[i],
            wf,
            style=loop_style,
            pl_prof_name=pl_prof_name,
            sbpl_prof_name=sbpl_prof_name,
            cr_ebr_band=cr_ebr_band,
            cr_ebr_fixed=cr_ebr_fixed,
            annotate_roi=annotate_roi,
            legend=(i == 0),
            nsamp=nsamp_band,
            seed=seed + i,  # decorrelate panels for nicer-looking bands
            eps_hess=eps_hess,
            ridge_hess=ridge_hess,
            n_dense=n_dense,
            dense_pad=dense_pad,
        )
        if i == 1:
            axes[i].text(
                0.08,
                0.90,
                loop_style.data_label,
                transform=axes[i].transAxes,
                ha="left",
                va="top",
                fontsize=loop_style.title_fs + 1,
                fontweight="bold",
                color=loop_style.data_color,
            )

    for j in range(n_panels, len(axes)):
        try:
            fig.delaxes(axes[j])
        except Exception:
            pass

    if n_panels > 1:
        fig.supxlabel(r"$\log_{10}(E/\mathrm{GeV})$", fontsize=style.label_fs, y=0.04)
        fig.supylabel(_NUIS_YLABEL, fontsize=style.label_fs, x=0.04)
        # Stamp on the 3rd panel when available (requested for grids).
        target = axes[2] if len(axes) > 2 else axes[0]
        stamp_prior_nuisance(
            target,
            prior_kind=prior_kind,
            nuisance_mode=_stamp_nm,
            color="0.25",
        )
        fig.tight_layout(rect=[0.05, 0.05, 1.0, 1.0])
    else:
        axes[0].set_xlabel(
            r"$\log_{10}(E/\mathrm{GeV})$", fontsize=_SED_SINGLE_XLABEL_FS
        )
        axes[0].set_ylabel(_NUIS_YLABEL, fontsize=_SED_SINGLE_YLABEL_FS)
        _apply_sed_single_panel_tickstyle(axes[0])
        stamp_prior_nuisance(
            axes[0],
            prior_kind=prior_kind,
            nuisance_mode=_stamp_nm,
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
