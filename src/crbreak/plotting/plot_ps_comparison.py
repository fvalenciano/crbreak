"""Compare PS spectral-model variants (best, PL, LP, PLSEC) against data.

Reproduces the legacy PS-comparison diagnostic figure: raw and
background-subtracted data points on top of the four PS template variants,
log-log axes, 3-column legend below.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Variant key → plot style (matches legacy code).
_PS_STYLE: dict[str, dict] = {
    "best":  dict(label="4FGL-DR4 SpectrumType", color="purple",     ls="-",  lw=1.7),
    "PL":    dict(label="PL",                    color="indigo",     ls="--", lw=1.7),
    "LP":    dict(label="LP",                    color="slateblue",  ls="-.", lw=1.7),
    "PLSEC": dict(label="PLSEC",                 color="darkorchid", ls=":",  lw=2.0),
}

_LEGEND_ORDER = ["best", "PL", "LP", "PLSEC"]


def plot_ps_comparison(
    window_data: dict,
    ps_variants: dict[str, np.ndarray],
    *,
    sed_scaling: float,
    save: str | Path | None = None,
    show: bool = False,
    dpi: int = 240,
    figsize: tuple[float, float] = (6, 5),
) -> plt.Figure:
    """Plot PS template variants alongside raw and subtracted PASS8 data.

    Parameters
    ----------
    window_data:
        Dict returned by :func:`~crbreak.data_extract.data_reduction.reduce_window`.
        Provides ``F_total_scaled`` (raw), ``F_bg_total``, ``sigma``, ``roi``.
    ps_variants:
        Mapping ``{variant_key: SED-scaled array}`` as returned by
        :func:`~crbreak.data_extract.data_reduction.load_ps_scaled`.
        Expected keys (any subset): ``"SV"``, ``"best"``, ``"PL"``, ``"LP"``,
        ``"PLSEC"``.
    sed_scaling:
        Power-law index for the energy weighting (``data.sed_scaling``).
    save:
        Path to save the figure.  ``None`` → not saved.
    show:
        Call ``plt.show()`` after drawing.
    dpi:
        Figure resolution when saving.
    figsize:
        Figure size ``(width, height)`` in inches.
    """
    E = np.asarray(window_data["E_GeV"], dtype=float)
    F_raw = np.asarray(window_data["F_total_scaled"], dtype=float)
    F_bg = np.asarray(window_data["F_bg_total"], dtype=float)
    F_sub = F_raw - F_bg
    sigma = np.asarray(window_data["sigma"], dtype=float)
    roi = window_data.get("roi", {})
    highb = roi.get("highb", 4.0)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Unsubtracted data (white fill, blue edge + errorbars) — behind.
    ax.errorbar(
        E, F_raw, yerr=sigma,
        markeredgecolor="cornflowerblue", markeredgewidth=1.5,
        ecolor="cornflowerblue",
        fmt="o", capsize=0, color="white",
        label="LAT",
        markersize=7, lw=1.5, zorder=5,
    )

    # Background-subtracted data (blue fill) — on top.
    ax.errorbar(
        E, F_sub, yerr=sigma,
        markeredgecolor="black", markeredgewidth=0.5,
        ecolor="cornflowerblue",
        fmt="o", capsize=0, color="cornflowerblue",
        label="LAT subs.", markersize=7, lw=1.5, zorder=6,
    )

    # PS variant lines.
    legend_handles: list = []
    legend_labels: list[str] = []
    plotted: set[str] = set()

    for key in _LEGEND_ORDER:
        if key not in ps_variants:
            continue
        style = _PS_STYLE[key]
        y = np.asarray(ps_variants[key], dtype=float)
        (line,) = ax.plot(
            E, y,
            color=style["color"],
            ls=style["ls"],
            lw=style["lw"],
        )
        legend_handles.append(line)
        legend_labels.append(style["label"])
        plotted.add(key)

    # ROI annotation.
    ax.text(
        0.05, 0.95, rf"$|b|<{highb:.0f}$",
        transform=ax.transAxes, va="top", ha="left", fontsize=18,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="none"),
    )

    ax.tick_params(axis="both", which="major", direction="in", labelsize=17, length=6, width=1.8)
    ax.tick_params(axis="both", which="minor", direction="in", labelsize=14, length=4, width=1.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(3, 800)
    ax.set_ylim(1e-6, 2e-3)

    s = f"{sed_scaling:g}"
    ax.set_xlabel(r"$E_\gamma\ \mathrm{[GeV]}$", fontsize=18)
    ax.set_ylabel(rf"$E^{{{s}}}\,\mathrm{{d}}N/\mathrm{{d}}E$", fontsize=18)

    # Legend, column-major fill: data entries first so "Fermi-LAT" and
    # "Fermi-LAT subs." stack together in column 1.
    data_handles, data_labels = ax.get_legend_handles_labels()
    all_handles = data_handles + legend_handles
    all_labels = data_labels + legend_labels
    ax.legend(
        all_handles, all_labels,
        ncol=3, fontsize=16, frameon=False,
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        columnspacing=1.2, handletextpad=0.4,
    )

    if save is not None:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()

    return fig
