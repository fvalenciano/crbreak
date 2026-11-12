from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._run_stamp import stamp_prior_nuisance

def plot_mc_diagnostics(
    fitting,
    *,
    kind: str,
    mode: str,
    prior_kind: str | None = None,
    nuisance_mode: str | None = None,
    pl_name: str,
    sbpl_name: str,
    g_free=None,
    g_fixed=None,
    outdir: str | Path,
) -> list[Path]:
    """Save MC Δχ² histograms with the observed threshold marked."""

    def draw(ax, mc, title: str) -> None:
        d = np.asarray(mc.deltas, dtype=float)
        ax.hist(d[np.isfinite(d)], bins="auto", color="0.55", alpha=0.75)
        ax.axvline(
            mc.delta_obs,
            color="crimson",
            lw=2,
            label=f"threshold={mc.delta_obs:.3g}",
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"$\Delta\chi^2$")
        ax.set_ylabel("toys")
        ax.legend(fontsize=8)

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    items = [
        (k, wf.mc_results.get((pl_name, sbpl_name)))
        for k, wf in fitting.window_fits.items()
    ]
    items = [(k, mc) for k, mc in items if mc is not None and mc.deltas is not None]
    if items:
        ncols = min(3, len(items))
        nrows = int(np.ceil(len(items) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.0 * ncols, 3.0 * nrows),
            squeeze=False,
        )
        for ax, (k, mc) in zip(axes.ravel(), items):
            draw(ax, mc, str(k))
        # Stamp on 3rd panel when available (requested for grids).
        target = axes.ravel()[2] if axes.size > 2 else axes.ravel()[0]
        stamp_prior_nuisance(
            target,
            prior_kind=prior_kind,
            nuisance_mode=nuisance_mode,
            color="0.25",
        )
        for ax in axes.ravel()[len(items):]:
            ax.axis("off")
        fig.tight_layout()
        path = out / f"{kind.lower()}_{mode}_windows_mc_deltachi2.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(path)

    for tag, gfit in (("global_free", g_free), ("global_fixed", g_fixed)):
        mc = None if gfit is None else gfit.mc_results.get(pl_name)
        if mc is None or mc.deltas is None:
            continue
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        draw(ax, mc, f"{kind.upper()} {mode} {tag}")
        stamp_prior_nuisance(
            ax,
            prior_kind=prior_kind,
            nuisance_mode=nuisance_mode,
            color="0.25",
        )
        fig.tight_layout()
        path = out / f"{kind.lower()}_{mode}_{tag}_mc_deltachi2.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(path)

    return saved
