"""Contour plot of ``Δχ²(Ebreak, dgamma)`` from a parameter scan.

Mirrors the legacy ``_draw_global_overlay_contours`` figure: each dataset's
``Δχ²`` grid is lightly Gaussian-smoothed for cosmetics, drawn as filled
single-color bands of decreasing alpha (1σ, 2σ, 3σ for df=2), with thin
colored contour lines on top labelled by confidence level.

Overlay-capable through the small :class:`ContourDataset` spec, so PASS8
+ IEM (or any third dataset) coexist with their own colors and best-fit
markers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from ._run_stamp import stamp_prior_nuisance


# Labels of the joint 2-D (df=2) Δχ² contour levels (``plots.contour_levels``):
#   2.30 -> 68.3% (1σ), 6.18 -> 95.4% (2σ), 11.83 -> 99.7% (3σ).
_DEFAULT_LEVEL_LABELS = {2.30: "68%", 6.18: "95%", 11.83: "99.7%"}

# Legacy-faithful default palettes:
_PASS8_BANDS = (
    (0.392, 0.584, 0.929, 0.90),
    (0.392, 0.584, 0.929, 0.52),
    (0.392, 0.584, 0.929, 0.24),
)
_IEM_BANDS = (
    (1.0, 0.498, 0.055, 0.68),
    (1.0, 0.498, 0.055, 0.46),
    (1.0, 0.498, 0.055, 0.20),
)


# ----------------------------------------------------------------------
# Dataset spec
# ----------------------------------------------------------------------


@dataclass
class ContourDataset:
    """One Δχ² surface to overlay on the contour figure.

    Parameters
    ----------
    label : str
        Legend label.
    scan : :class:`ChiSquareScan`
        Output of :meth:`Fitting.scan`.  Its
        :attr:`delta_chi2_grid_vs_anchor` (when an anchor was provided) or
        :attr:`delta_chi2_grid_vs_grid` (fallback) is contoured.
    g
        Optional :class:`GlobalFitResult`; if given, a marker is placed at
        the joint best-fit ``(Ebreak, dgamma)`` (with optional ±1σ cross).
    line_color : str
        Color of the thin contour lines.
    band_rgba : sequence of RGBA tuples
        Color of the filled bands (one per Δχ² level, lighter outwards).
    marker, marker_color, marker_size
        Best-fit marker styling.
    use_anchor : bool
        When True (default), :attr:`delta_chi2_grid_vs_anchor` is plotted
        so all datasets share a common χ² zero (the joint free-fit χ²).
        When False, each dataset is centred on its own grid minimum
        (:attr:`delta_chi2_grid_vs_grid`).
    """

    label: str
    scan: object
    g: object = None
    line_color: str = "tab:blue"
    band_rgba: Sequence[tuple[float, float, float, float]] = field(
        default_factory=lambda: tuple(_PASS8_BANDS)
    )
    marker: str = "^"
    marker_color: str = "cornflowerblue"
    marker_size: float = 9.0
    use_anchor: bool = True

    @classmethod
    def pass8(cls, scan, g=None, label: str = "PASS8") -> "ContourDataset":
        return cls(
            label=label, scan=scan, g=g,
            line_color="tab:blue",
            band_rgba=tuple(_PASS8_BANDS),
            marker="^", marker_color="cornflowerblue",
        )

    @classmethod
    def iem(cls, scan, g=None, label: str = "IEM") -> "ContourDataset":
        return cls(
            label=label, scan=scan, g=g,
            line_color="tab:orange",
            band_rgba=tuple(_IEM_BANDS),
            marker="^", marker_color="tab:orange",
        )


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _normalise_datasets(datasets) -> list[ContourDataset]:
    if isinstance(datasets, ContourDataset):
        return [datasets]
    out = list(datasets)
    if not out:
        raise ValueError("plot_contour_plot: at least one dataset required.")
    if not all(isinstance(d, ContourDataset) for d in out):
        raise TypeError(
            "plot_contour_plot expects ContourDataset instances "
            "(or a single one)."
        )
    return out


def _smooth_for_contours(D: np.ndarray, sigma: float) -> np.ndarray:
    """Light Gaussian smoothing with NaN-safe filling, then re-zero.

    Replicates the legacy ``_prep_delta_grid_for_contours``: NaNs are
    replaced by a high value before smoothing (so the contours don't bleed
    into them), and the result is shifted so its minimum is exactly 0
    (which keeps the level numbers meaningful).
    """
    D = np.asarray(D, dtype=float)
    finite = np.isfinite(D)
    if not np.any(finite):
        raise RuntimeError("Δχ² grid has no finite values.")
    if sigma is None or sigma <= 0:
        out = D.copy()
        out[~finite] = float(np.nanmax(out[finite])) + 50.0
        return out - float(np.nanmin(out))
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        # Without scipy, fall back to no smoothing.
        out = D.copy()
        out[~finite] = float(np.nanmax(out[finite])) + 50.0
        return out - float(np.nanmin(out))
    hi = float(np.nanmax(D[finite]))
    D_filled = D.copy()
    D_filled[~finite] = hi + 50.0
    D_smooth = gaussian_filter(D_filled, sigma=float(sigma))
    return D_smooth - float(np.nanmin(D_smooth))


def _asym_err(
    central: float,
    band: tuple[float, float] | None,
) -> list[list[float]] | None:
    """Build the ``[[low_err], [high_err]]`` payload for ``errorbar``.

    ``band`` is the ``(low, high)`` reference range; the returned errors
    are ``central - low`` and ``high - central`` clipped to non-negative.
    Returns ``None`` if ``band`` is ``None``.
    """
    if band is None:
        return None
    lo, hi = band
    err_lo = max(0.0, float(central) - float(lo))
    err_hi = max(0.0, float(hi) - float(central))
    return [[err_lo], [err_hi]]


def _resolve_delta_grid(ds: ContourDataset) -> np.ndarray:
    if ds.use_anchor:
        d = ds.scan.delta_chi2_grid_vs_anchor
        if d is None:
            d = ds.scan.delta_chi2_grid_vs_grid
        return d
    return ds.scan.delta_chi2_grid_vs_grid


def _draw_one_dataset(
    ax,
    ds: ContourDataset,
    *,
    levels: Sequence[float],
    level_labels: dict,
    smooth_sigma: float,
    label_contours: bool,
) -> None:
    Eb = np.asarray(ds.scan.Ebreak_grid, dtype=float)
    Dg = np.asarray(ds.scan.dgamma_grid, dtype=float)
    X, Y = np.meshgrid(Eb, Dg, indexing="ij")
    D = _smooth_for_contours(_resolve_delta_grid(ds), sigma=smooth_sigma)

    band_levels = [0.0, *levels]
    if len(ds.band_rgba) < len(levels):
        raise ValueError(
            f"ContourDataset({ds.label!r}): "
            f"need at least {len(levels)} band colors, got {len(ds.band_rgba)}."
        )
    ax.contourf(
        X, Y, D,
        levels=band_levels,
        colors=tuple(ds.band_rgba[: len(levels)]),
        antialiased=True,
    )
    cs = ax.contour(X, Y, D, levels=list(levels), colors=[ds.line_color], linewidths=1.2)
    if label_contours:
        try:
            ax.clabel(cs, fmt=level_labels, inline=True, fontsize=9)
        except Exception:
            pass

    # Best-fit marker (joint free fit).
    if ds.g is not None:
        try:
            Eb_g, _ = ds.g.shared_params["Ebreak"]
            dg_g, _ = ds.g.shared_params["dgamma"]
            ax.plot(
                Eb_g, dg_g,
                marker=ds.marker, ms=ds.marker_size,
                mec="black", mew=0.8,
                mfc=ds.marker_color, color=ds.marker_color,
                zorder=6, linestyle="none",
            )
        except Exception:
            pass


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def plot_contour_plot(
    datasets: Sequence[ContourDataset] | ContourDataset,
    *,
    smooth_sigma: float,
    levels: Sequence[float],
    level_labels: dict | None = None,
    label_contours: bool = True,
    cr_ref: tuple[float, float] | None,
    cr_ebr_band: tuple[float, float] | None,
    cr_dgamma_band: tuple[float, float] | None,
    cr_color: str = "grey",
    cr_marker: str = "o",
    cr_marker_size: float = 7.0,
    cr_label: str = "Local CR Break",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    highb_text: float | None = None,
    figsize: tuple[float, float] = (8.0, 5.5),
    save: str | Path | None = None,
    show: bool = False,
    dpi: int = 200,
    prior_kind: str | None = None,
    nuisance_mode: str | None = None,
):
    """Overlay ``Δχ²(Ebreak, dgamma)`` contours from one or more scans.

    Parameters
    ----------
    datasets : :class:`ContourDataset` or sequence of them
        Each dataset draws its own filled bands + line contours + best-fit
        marker (if ``ds.g`` is given).
    smooth_sigma : float
        Light Gaussian smoothing for the *display* (statistics unchanged).
        Set to 0 to disable.
    levels : sequence of float
        Δχ² thresholds for the contour lines (default = 1σ, 2σ, 3σ for
        joint df=2).
    level_labels : dict, optional
        ``{level_value: label}`` for ``ax.clabel``.  Defaults to the legacy
        labels (``"68%"``, ``"95%"``, ``"99.7%"``).
    cr_ref
        ``(Ebreak, dgamma)`` of the local-CR reference; drawn as a single
        marker with horizontal/vertical error bars derived from
        ``cr_ebr_band`` and ``cr_dgamma_band``.  Pass ``None`` to suppress.
    cr_ebr_band, cr_dgamma_band
        ``(low, high)`` ranges around ``cr_ref``; rendered as the asymmetric
        x/y error bars on the CR marker (legacy convention).  Pass ``None``
        on either side for a symmetric / no error in that axis.
    cr_color, cr_marker, cr_marker_size, cr_label
        Styling and legend label for the CR marker.
    xlim, ylim
        Axis ranges.  Defaults to the union of the scan grids.
    highb_text : float | None
        If given, drop a small ``|b|<{highb}`` text box.
    save, show, dpi
        Standard matplotlib output controls.

    Returns
    -------
    matplotlib.figure.Figure
    """
    dss = _normalise_datasets(datasets)
    fig, ax = plt.subplots(figsize=figsize)

    # Determine axis ranges from the union of grids if not given.
    if xlim is None:
        xs = np.concatenate([np.asarray(d.scan.Ebreak_grid, float) for d in dss])
        xlim = (float(xs.min()), float(xs.max()))
    if ylim is None:
        ys = np.concatenate([np.asarray(d.scan.dgamma_grid, float) for d in dss])
        ylim = (float(ys.min()), float(ys.max()))

    # Δχ² contours, dataset by dataset.
    for ds in dss:
        _draw_one_dataset(
            ax, ds,
            levels=levels,
            level_labels=level_labels or _DEFAULT_LEVEL_LABELS,
            smooth_sigma=smooth_sigma,
            label_contours=label_contours,
        )

    # CR reference marker with asymmetric error bars derived from the bands.
    if cr_ref is not None:
        Eb_ref, dg_ref = cr_ref
        xerr = _asym_err(Eb_ref, cr_ebr_band)
        yerr = _asym_err(dg_ref, cr_dgamma_band)
        ax.errorbar(
            [Eb_ref], [dg_ref],
            xerr=xerr, yerr=yerr,
            fmt=cr_marker, color=cr_color, ecolor=cr_color,
            mec="black", mew=0.8, ms=cr_marker_size,
            elinewidth=1.2, capsize=3,
            linestyle="none", zorder=5,
        )

    # Build a clean legend with one Patch per dataset + CR marker.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = []
    for ds in dss:
        face = ds.band_rgba[0] if ds.band_rgba else ds.line_color
        handles.append(Patch(facecolor=face, edgecolor="none", label=ds.label))
    if cr_ref is not None:
        handles.append(
            Line2D(
                [0], [0], marker=cr_marker, color=cr_color, mec="black",
                lw=0, ms=6, label=cr_label,
            )
        )
    ax.legend(handles=handles, frameon=False, loc="upper left",
              fontsize=12, ncol=max(1, min(len(handles), 2)))

    if highb_text is not None:
        ax.text(
            0.95, 0.05, f"|b|<{highb_text:g}",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=15,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      alpha=0.8, edgecolor="none"),
        )

    ax.set_xlabel(r"$E_{\rm break}\,[\mathrm{GeV}]$", fontsize=16)
    ax.set_ylabel(r"$\Delta\gamma$", fontsize=16)
    ax.tick_params(axis="both", which="major", direction="in",
                   labelsize=14, length=6, width=1.6)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    stamp_prior_nuisance(
        ax,
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


def plot_contour_heatmap(
    dataset: ContourDataset,
    *,
    use_anchor: bool | None = None,
    cr_ref: tuple[float, float] | None,
    cr_ebr_band: tuple[float, float] | None,
    cr_dgamma_band: tuple[float, float] | None,
    cmap: str = "viridis_r",
    vmax: float = 60.0,
    figsize: tuple[float, float] = (6.5, 5.5),
    save: str | Path | None = None,
    show: bool = False,
    dpi: int = 200,
    prior_kind: str | None = None,
    nuisance_mode: str | None = None,
):
    """Raw per-dataset Δχ² heatmap without smoothing or interpolation."""
    ds = dataset
    if use_anchor is not None:
        ds = ContourDataset(
            label=dataset.label,
            scan=dataset.scan,
            g=dataset.g,
            line_color=dataset.line_color,
            band_rgba=dataset.band_rgba,
            marker=dataset.marker,
            marker_color=dataset.marker_color,
            marker_size=dataset.marker_size,
            use_anchor=bool(use_anchor),
        )
    Eb = np.asarray(ds.scan.Ebreak_grid, dtype=float)
    Dg = np.asarray(ds.scan.dgamma_grid, dtype=float)
    D = np.asarray(_resolve_delta_grid(ds), dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(Eb, Dg, D.T, shading="nearest", cmap=cmap, vmin=0.0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\Delta\chi^2$", fontsize=14)

    if ds.g is not None:
        try:
            Eb_g, _ = ds.g.shared_params["Ebreak"]
            dg_g, _ = ds.g.shared_params["dgamma"]
            ax.plot(
                Eb_g, dg_g,
                marker=ds.marker, ms=ds.marker_size,
                mec="black", mew=0.8,
                mfc=ds.marker_color, color=ds.marker_color,
                linestyle="none", zorder=5,
            )
        except Exception:
            pass

    if cr_ref is not None:
        Eb_ref, dg_ref = cr_ref
        ax.errorbar(
            [Eb_ref], [dg_ref],
            xerr=_asym_err(Eb_ref, cr_ebr_band),
            yerr=_asym_err(dg_ref, cr_dgamma_band),
            fmt="o", color="grey", ecolor="grey",
            mec="black", mew=0.8, ms=7,
            elinewidth=1.2, capsize=3,
            linestyle="none", zorder=5,
        )

    ax.set_title(ds.label)
    ax.set_xlabel(r"$E_{\rm break}\,[\mathrm{GeV}]$", fontsize=16)
    ax.set_ylabel(r"$\Delta\gamma$", fontsize=16)
    ax.tick_params(axis="both", which="major", direction="in",
                   labelsize=14, length=6, width=1.6)
    ax.set_xlim(float(Eb.min()), float(Eb.max()))
    ax.set_ylim(float(Dg.min()), float(Dg.max()))

    stamp_prior_nuisance(
        ax,
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
