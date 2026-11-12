"""
Quick visual sanity check of the modular extraction.

Builds a longitude grid (single latitude band), extracts the SED-scaled
spectrum for each window using the new pipeline, and plots a 3x3 grid
(one panel per longitude bin) with data points + 1-sigma error bars.

Supports PASS8 and/or IEM overlay.

No fitting, no background subtraction: this only verifies that the data
extraction itself behaves as expected.

Every setting (windows, datasets, files, cuts) comes from the analysis config:

    python -m crbreak.data_extract.diagnostics configs/baseline.yaml [sed|bg_plot|ps_comparison]
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..plotting.plot_background_components import plot_background_components
from ..plotting.plot_ps_comparison import plot_ps_comparison
from .data_loader import load_dragon_pi0_template
from .data_reduction import load_ps_scaled, reduce_window
from .main import Data, SkyMesh


def run(cfg, *, subtraction: bool, outdir: str) -> str:
    """Extract every window of ``cfg.sky`` for each of ``cfg.data.kinds`` and
    plot one SED panel per |l| window."""
    glon_bins = np.asarray(cfg.sky.glon_edges, dtype=float)
    glon_lims = (float(glon_bins[0]), float(glon_bins[-1]))
    glat_lims = (cfg.sky.glat_edges[0], cfg.sky.glat_edges[-1])
    n_glon = int(glon_bins.size - 1)
    mesh = SkyMesh(glon_lims=glon_lims, glat_lims=glat_lims, Nglon=n_glon, Nglat=1, glon_bins=glon_bins)
    kinds_u = cfg.data.kinds

    results_by_kind: dict[str, list[dict]] = {}
    for kind in kinds_u:
        print(
            f"[main_test] Extracting {kind} in {n_glon} longitude windows "
            f"glon_edges={glon_bins.tolist()}, |b| in {glat_lims} deg ..."
        )
        data = Data(mesh)
        data.build(kind, cfg=cfg, subtraction=subtraction)
        windows = sorted(data.data_dict.values(), key=lambda w: w.lowl)
        results_by_kind[kind] = [w.data for w in windows]  # same window order

    n_bins = len(windows)
    n_cols = 3 if n_bins >= 3 else n_bins
    n_rows = math.ceil(n_bins / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharex=True,
        sharey=False,
    )
    axes = np.atleast_1d(axes).flatten()

    color_map = {"PASS8": "cornflowerblue", "IEM": "tab:orange"}
    for i in range(n_bins):
        ax = axes[i]
        xmins: list[float] = []
        xmaxs: list[float] = []
        ymins: list[float] = []
        ymaxs: list[float] = []
        for kind in kinds_u:
            res = results_by_kind[kind][i]
            if res is None:
                continue
            E = np.asarray(res["E_GeV"], dtype=float)
            y = np.asarray(res["y"], dtype=float)
            sigma = np.asarray(res["sigma"], dtype=float)

            # Track per-panel limits (log axes => keep only finite positive y)
            E_ok = np.isfinite(E)
            if E_ok.any():
                xmins.append(float(np.nanmin(E[E_ok])))
                xmaxs.append(float(np.nanmax(E[E_ok])))

            y_ok = np.isfinite(y) & (y > 0)
            if y_ok.any():
                ymins.append(float(np.nanmin(y[y_ok])))
                ymaxs.append(float(np.nanmax(y[y_ok])))

            ax.errorbar(
                E,
                y,
                yerr=sigma,
                fmt="o",
                color=color_map.get(kind, "black"),
                ecolor="black",
                elinewidth=1,
                markeredgecolor="black",
                markeredgewidth=0.7,
                capsize=0,
                markersize=6,
                label=kind,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        if xmins and xmaxs:
            ax.set_xlim(0.9 * min(xmins), 1.1 * max(xmaxs))
        if ymins and ymaxs:
            ax.set_ylim(0.85 * min(ymins), 1.20 * max(ymaxs))
        ax.text(
            0.95,
            0.05,
            f"{mesh.glon_bins[i]:.0f}<|l|<{mesh.glon_bins[i+1]:.0f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=14,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.85,
                edgecolor="none",
            ),
        )
        ax.text(
            0.95,
            0.16,
            f"|b|<{glat_lims[1]:.0f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=14,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.85,
                edgecolor="none",
            ),
        )
        ax.tick_params(axis="both", which="major", direction="in", length=5, width=1.2)
        ax.tick_params(axis="both", which="minor", direction="in", length=3, width=1.0)

        if i == 0:
            ax.legend(frameon=False, fontsize=12, loc="upper left")

    for j in range(n_bins, len(axes)):
        fig.delaxes(axes[j])

    s = f"{cfg.data.sed_scaling:g}"
    fig.supxlabel("Energy [GeV]", fontsize=18, y=0.04)
    fig.supylabel(
        rf"$E^{{{s}}}\,J\;[\mathrm{{GeV^{{{s} - 1}}\,cm^{{-2}}\,s^{{-1}}\,sr^{{-1}}}}]$",
        fontsize=18,
        x=0.04,
    )

    fig.tight_layout(rect=[0.05, 0.05, 1.0, 1.0])

    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kind_tag = "_".join(kinds_u).lower()
    sub_tag = "sub" if subtraction else "raw"
    fname = (
        f"{kind_tag}_{sub_tag}_sanity_{int(glat_lims[0])}b{int(glat_lims[1])}_"
        f"{int(glon_lims[0])}-{int(glon_lims[1])}l_{n_bins}win.png"
    )
    fpath = out_dir / fname
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    print(f"[main_test] Saved: {fpath.resolve()}")
    plt.close(fig)
    return str(fpath)


def run_bg_plot(cfg, *, glon_lims: tuple[float, float], glat_lims: tuple[float, float], outdir: str) -> str:
    """Extract PASS8 over the full ROI and plot all background components.

    Produces a single figure with the unsubtracted data on top of the
    individual BG templates (ISO, PS, UPS, FB, IC CMB, IC ISRF, Brems H2,
    Brems HI), plus the DRAGON pi0 model ``cfg.plots.dragon_model`` (if any).
    """
    print(
        f"[bg_plot] Extracting PASS8 full ROI "
        f"|l|<{glon_lims[1]:.0f}, |b|<{glat_lims[1]:.0f} ..."
    )
    data = reduce_window(
        kind="PASS8",
        lowl=float(glon_lims[0]),
        highl=float(glon_lims[1]),
        lowb=float(glat_lims[0]),
        highb=float(glat_lims[1]),
        glon_sign=cfg.sky.glon_sign,
        data=cfg.data,
        paths=cfg.paths,
        files=cfg.files,
        subtraction=True,  # F_total_scaled (raw) and F_bg_total are always in the dict
    )

    dragon_curve = None
    dragon_label = r"$\pi^0$"
    dragon_model = cfg.plots.dragon_model
    if dragon_model is not None:
        print(f"[bg_plot] Loading DRAGON pi0 model [{dragon_model}] ...")
        dragon_files = cfg.files.dragon_models[dragon_model]
        E_dragon, F_dragon = load_dragon_pi0_template(
            Path(cfg.paths.dragon_dir) / dragon_files["H2"],
            Path(cfg.paths.dragon_dir) / dragon_files["HI"],
            lowb=float(glat_lims[0]),
            highb=float(glat_lims[1]),
            lowl=float(glon_lims[0]),
            highl=float(glon_lims[1]),
            glon_sign=cfg.sky.glon_sign,
        )
        F_dragon_scaled = F_dragon * E_dragon**cfg.data.sed_scaling

        # Normalize to the (subtracted) Fermi-LAT data at the lowest energies,
        # since the DRAGON prediction has no absolute calibration of its own.
        E = np.asarray(data["E_GeV"], dtype=float)
        F_sub = np.asarray(data["F_total_scaled"], dtype=float) - np.asarray(
            data["F_bg_total"], dtype=float
        )
        n_lo = min(cfg.plots.dragon_norm_n_bins, E.size)
        F_dragon_at_data = np.interp(E[:n_lo], E_dragon, F_dragon_scaled)
        norm = np.median(F_sub[:n_lo] / F_dragon_at_data)
        F_dragon_scaled = F_dragon_scaled * norm

        dragon_curve = (E_dragon, F_dragon_scaled)

    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (
        f"bg_components_{int(glat_lims[0])}b{int(glat_lims[1])}_"
        f"{int(glon_lims[0])}-{int(glon_lims[1])}l.png"
    )
    fpath = out_dir / fname

    fig = plot_background_components(
        data, sed_scaling=cfg.data.sed_scaling, save=fpath,
        dragon_curve=dragon_curve, dragon_label=dragon_label,
    )
    plt.close(fig)
    print(f"[bg_plot] Saved: {fpath.resolve()}")
    return str(fpath)


def run_ps_comparison_plot(cfg, *, glon_lims: tuple[float, float], glat_lims: tuple[float, float], outdir: str) -> str:
    """Compare the PS template variants ``cfg.files.point_source_variants``
    against the full-ROI PASS8 data (with BG subtraction), via
    :func:`~crbreak.plotting.plot_ps_comparison`.
    """
    print(
        f"[ps_comparison] Extracting PASS8 full ROI "
        f"|l|<{glon_lims[1]:.0f}, |b|<{glat_lims[1]:.0f} ..."
    )
    data = reduce_window(
        kind="PASS8",
        lowl=float(glon_lims[0]),
        highl=float(glon_lims[1]),
        lowb=float(glat_lims[0]),
        highb=float(glat_lims[1]),
        glon_sign=cfg.sky.glon_sign,
        data=cfg.data,
        paths=cfg.paths,
        files=cfg.files,
        subtraction=True,
    )
    E_GeV = data["E_GeV"]

    print("[ps_comparison] Loading PS template variants ...")
    ps_variants: dict[str, np.ndarray] = {}
    for key, fname in cfg.files.point_source_variants.items():
        try:
            ps_variants[key] = load_ps_scaled(
                Path(cfg.paths.data_dir) / fname,
                E_GeV=E_GeV,
                lowl=float(glon_lims[0]),
                highl=float(glon_lims[1]),
                lowb=float(glat_lims[0]),
                highb=float(glat_lims[1]),
                glon_sign=cfg.sky.glon_sign,
                sed_scaling=cfg.data.sed_scaling,
            )
            print(f"  loaded: {key}")
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {key}: {exc}")

    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname_out = (
        f"ps_comparison_{int(glat_lims[0])}b{int(glat_lims[1])}_"
        f"{int(glon_lims[0])}-{int(glon_lims[1])}l.png"
    )
    fpath = out_dir / fname_out

    fig = plot_ps_comparison(data, ps_variants, sed_scaling=cfg.data.sed_scaling, save=fpath)
    plt.close(fig)
    print(f"[ps_comparison] Saved: {fpath.resolve()}")
    return str(fpath)


if __name__ == "__main__":
    import sys

    from ..config.analysis import load_config

    cfg = load_config(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "sed"
    glon_lims = (cfg.sky.glon_edges[0], cfg.sky.glon_edges[-1])
    glat_lims = (cfg.sky.glat_edges[0], cfg.sky.glat_edges[-1])
    if mode == "bg_plot":
        run_bg_plot(cfg, glon_lims=glon_lims, glat_lims=glat_lims, outdir="plots/bg_components")
    elif mode == "ps_comparison":
        run_ps_comparison_plot(cfg, glon_lims=glon_lims, glat_lims=glat_lims, outdir="plots/ps_comparison")
    else:
        run(cfg, subtraction=True, outdir="plots/extraction_sanity")
