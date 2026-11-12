"""Plot raw PASS8 spectrum overlaid with all background components.

Reproduces the exact style of the legacy FDE_PASS8 diagnostic figure:
log-log axes, blue filled circles for data, one colored line per background
component (Brems H2+HI and IC CMB+ISRF combined, plus a Total line), 3-column
legend below the axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Component key (after grouping, see _GROUPS) → plot style.
_COMP_STYLE: dict[str, dict] = {
    "PS":    dict(label="PS",    color="purple",     ls="-",  lw=1.7),
    "UPS":   dict(label="UPS",   color="orchid",  ls="-", lw=1.5),
    "BUB":   dict(label="FB",    color="tab:red",     ls="-",  lw=1.5),
    "ISO":   dict(label="ISO",   color="orangered",   ls="-",  lw=1.5),
    "IC":    dict(label="IC",    color="goldenrod",   ls="-",  lw=1.5),
    "Brems": dict(label="Brems", color="royalblue",   ls="-",  lw=1.5),
    "Total": dict(label="Sum", color="grey", ls="-", lw=2.0),
}

# Individual bg_components keys that are summed together into one line/legend entry.
_GROUPS: dict[str, list[str]] = {
    "IC": ["IC_cmb", "IC_isrf"],
    "Brems": ["Brems_H2", "Brems_HI"],
}

# Legend order: row-major, 3 columns, matching the reference figure layout.
_LEGEND_ORDER = ["PS", "BUB", "Brems", "UPS", "IC", "ISO", "Total"]


def plot_background_components(
    window_data: dict,
    *,
    sed_scaling: float,
    save: str | Path | None = None,
    show: bool = False,
    dpi: int = 240,
    figsize: tuple[float, float] = (6, 5),
    dragon_curve: tuple[np.ndarray, np.ndarray] | None = None,
    dragon_label: str = r"$\pi^0$",
    dragon_color: str = "forestgreen",
    dragon_ls: str = "-.",
    dragon_show_line: bool = True,
    show_subtracted: bool = False,
) -> plt.Figure:
    """Plot the unsubtracted PASS8 spectrum with individual BG components.

    Parameters
    ----------
    window_data:
        Dict returned by :func:`~crbreak.data_extract.data_reduction.reduce_window`.
        The raw (unsubtracted) flux is taken from ``F_total_scaled``.
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
    dragon_curve:
        Optional ``(E_GeV, F)`` overlay, e.g. a DRAGON pi0 model spectrum
        (see :func:`~crbreak.data_extract.data_loader.load_dragon_pi0_template`).
        ``F`` must already be ``E**sed_scaling``-weighted, matching the
        y-axis convention of this plot. Drawn as its own line and also
        interpolated onto ``E_GeV`` and folded into the "Total foreground"
        sum alongside the other bg components.
    dragon_label:
        Legend label for ``dragon_curve``.
    dragon_color, dragon_ls:
        Line color/style for ``dragon_curve``.
    dragon_show_line:
        If ``False``, ``dragon_curve`` is still folded into the "Total"
        sum but its own line/legend entry is not drawn.
    show_subtracted:
        If ``True``, also plot the background-subtracted data ("LAT subs.",
        filled cornflowerblue circles) alongside the unsubtracted "LAT"
        points, stacked in the same legend column.
    """
    E = np.asarray(window_data["E_GeV"], dtype=float)
    F_raw = np.asarray(window_data["F_total_scaled"], dtype=float)
    F_bg = np.asarray(window_data["F_bg_total"], dtype=float)
    sigma = np.asarray(window_data["sigma"], dtype=float)
    bg_comps: dict[str, np.ndarray] = window_data.get("bg_components", {})
    roi = window_data.get("roi", {})
    highb = roi.get("highb", 4.0)

    # Interpolate the DRAGON curve onto the same energy grid as the other
    # components, so it spans the same x-range instead of its own native grid.
    F_dragon_on_E = None
    if dragon_curve is not None:
        E_dragon_raw, F_dragon_raw = dragon_curve
        F_dragon_on_E = np.interp(
            E, np.asarray(E_dragon_raw, dtype=float), np.asarray(F_dragon_raw, dtype=float)
        )

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Unsubtracted data (white fill, blue edge + errorbars).
    ax.errorbar(
        E, F_raw, yerr=sigma,
        markeredgecolor="cornflowerblue", markeredgewidth=1.5,
        ecolor="cornflowerblue",
        fmt="o", capsize=0, color="white",
        label="LAT", markersize=7, lw=1.5,
        zorder=5,
    )

    # Optional background-subtracted data (filled cornflowerblue circles).
    if show_subtracted:
        F_sub = F_raw - F_bg
        ax.errorbar(
            E, F_sub, yerr=sigma,
            markeredgecolor="black", markeredgewidth=0.5,
            ecolor="cornflowerblue",
            fmt="o", capsize=0, color="cornflowerblue",
            label="LAT subs.", markersize=7, lw=1.5,
            zorder=6,
        )

    # Group individual components (Brems H2+HI, IC CMB+ISRF) into single lines,
    # and add a "Total" line: the sum of every background component.
    grouped_keys: set[str] = set()
    plot_comps: dict[str, np.ndarray] = {}
    for group_key, members in _GROUPS.items():
        present = [m for m in members if m in bg_comps]
        if not present:
            continue
        plot_comps[group_key] = sum(np.asarray(bg_comps[m], dtype=float) for m in present)
        grouped_keys.update(present)
    for key, arr in bg_comps.items():
        if key in grouped_keys:
            continue
        plot_comps[key] = np.asarray(arr, dtype=float)
    if bg_comps:
        F_total = F_bg
        if F_dragon_on_E is not None:
            F_total = F_total + F_dragon_on_E
        plot_comps["Total"] = F_total

    # Background component lines, keyed by component name for legend lookup.
    comp_handles: dict[str, object] = {}
    comp_labels: dict[str, str] = {}
    plotted: set[str] = set()

    for key in _LEGEND_ORDER:
        if key not in plot_comps:
            continue
        style = _COMP_STYLE[key]
        (line,) = ax.plot(
            E, plot_comps[key],
            color=style["color"],
            ls=style["ls"],
            lw=style["lw"],
        )
        comp_handles[key] = line
        comp_labels[key] = style["label"]
        plotted.add(key)

    # Any remaining components not in _LEGEND_ORDER.
    for key, arr in plot_comps.items():
        if key in plotted:
            continue
        style = _COMP_STYLE.get(key, dict(label=key, color="gray", ls=":", lw=1.2))
        (line,) = ax.plot(
            E, arr,
            color=style["color"],
            ls=style["ls"],
            lw=style.get("lw", 1.2),
        )
        comp_handles[key] = line
        comp_labels[key] = style["label"]

    # Optional DRAGON model overlay, on the same energy grid as the other lines.
    dragon_handle = None
    if F_dragon_on_E is not None and dragon_show_line:
        (dragon_handle,) = ax.plot(
            E, F_dragon_on_E,
            color=dragon_color, ls=dragon_ls, lw=2.0, zorder=3,
        )

    # ROI annotation (upper left).
    ax.text(
        0.05, 0.95,
        rf"$|b|<{highb:.0f}$",
        transform=ax.transAxes,
        va="top", ha="left", fontsize=18,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="none"),
    )

    ax.tick_params(axis="both", which="major", direction="in", labelsize=17, length=6, width=1.8)
    ax.tick_params(axis="both", which="minor", direction="in", labelsize=14, length=4, width=1.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(3, 800)
    ax.set_ylim(5e-8, 8e-4)

    s = f"{sed_scaling:g}"
    ax.set_xlabel(r"$E_\gamma\ \mathrm{[GeV]}$", fontsize=18)
    ax.set_ylabel(rf"$E^{{{s}}}\,\mathrm{{d}}N/\mathrm{{d}}E$", fontsize=18)

    # 3-column legend, filled column-major by matplotlib: put the data
    # entry/entries and the Total line first so they stack in column 1.
    auto_handles, auto_labels = ax.get_legend_handles_labels()
    all_handles = list(auto_handles)
    all_labels = list(auto_labels)
    if "Total" in comp_handles:
        all_handles.append(comp_handles.pop("Total"))
        all_labels.append(comp_labels.pop("Total"))
    if dragon_handle is not None:
        all_handles.append(dragon_handle)
        all_labels.append(dragon_label)
    for key in _LEGEND_ORDER:
        if key in comp_handles:
            all_handles.append(comp_handles.pop(key))
            all_labels.append(comp_labels.pop(key))
    for key in list(comp_handles):
        all_handles.append(comp_handles.pop(key))
        all_labels.append(comp_labels.pop(key))
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
