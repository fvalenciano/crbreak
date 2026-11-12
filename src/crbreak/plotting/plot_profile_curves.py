"""Debug plot: per-window 1-D profile-likelihood ``Δχ²(param)`` curves
for the SBPL profiled fit, one figure per parameter.

For each parameter ``Ebreak`` / ``dgamma``, one figure with one panel per
window arranged in a grid; each panel shows:

* ``Δχ²(param) = chi2_row - chi2_anchor`` versus the 1-D scan grid;
* horizontal reference lines at ``Δχ² = 0`` (anchor) and ``Δχ² = 1`` (1σ);
* vertical mark at the per-window best-fit value;
* vertical marks at the bracketed 1σ crossings (``lo`` and ``hi``).

The data is read from ``WindowFitResult.profile_curves`` (populated by
:meth:`WindowFitResult.compute_profile_errors` during the run), so this
script does **not** re-fit anything.

Throwaway helper -- delete the file and the call site in
``main_test.py`` once the diagnostic is no longer needed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._run_stamp import stamp_prior_nuisance

def _plot_one_param(
    items,
    *,
    pname: str,
    sbpl_prof_name: str,
    global_result=None,
    out_path: Path,
    title_prefix: str,
    delta_ref: float,
    prior_kind: str | None = None,
    nuisance_mode: str | None = None,
) -> None:
    panel_items = list(items)
    if global_result is not None:
        panel_items.append(("GLOBAL", global_result))
    n = len(panel_items)
    if n == 0:
        return
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 3.0 * nrows), squeeze=False
    )

    xlabel = "Ebreak [GeV]" if pname == "Ebreak" else pname

    for idx, (key, obj) in enumerate(panel_items):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        is_global = key == "GLOBAL"
        if is_global:
            curves = getattr(obj, "profile_curves", {}) or {}
            errs = getattr(obj, "profile_errors", {}) or {}
        else:
            curves = obj.profile_curves.get(sbpl_prof_name, {})
            errs = obj.profile_errors.get(sbpl_prof_name, {})
        triplet = curves.get(pname)

        if is_global:
            roi = "joint fit"
        else:
            roi = f"l=({obj.lowl:g},{obj.highl:g})"
        if triplet is None:
            ax.set_title(f"{key}  {roi}\n(no curve)")
            ax.axis("off")
            continue

        grid, chi2_row, chi2_anchor = triplet
        grid = np.asarray(grid, dtype=float)
        chi2_row = np.asarray(chi2_row, dtype=float)
        best, lo, hi = errs.get(pname, (np.nan, np.nan, np.nan))
        if is_global and np.isfinite(best):
            # The global 1-D curve is sampled on a finite grid.  Add the
            # exact best-fit anchor so the diagnostic plot does not imply
            # that the fit prefers a neighbouring grid point.
            if not np.any(np.isclose(grid, best, rtol=0.0, atol=1e-10)):
                grid = np.concatenate([grid, np.array([float(best)])])
                chi2_row = np.concatenate(
                    [chi2_row, np.array([float(chi2_anchor)])]
                )
                order = np.argsort(grid)
                grid = grid[order]
                chi2_row = chi2_row[order]
        d = chi2_row - float(chi2_anchor)

        color = "C3" if is_global else "C0"
        ax.plot(grid, d, "-o", ms=3, lw=1.2, color=color)
        ax.axhline(0.0, color="k", lw=0.6, ls="--")
        ax.axhline(delta_ref, color="r", lw=0.7, ls=":")

        if np.isfinite(best):
            ax.axvline(best, color="g", lw=1.0, ls="-", alpha=0.8, label=f"best={best:.3g}")
        if np.isfinite(lo):
            ax.axvline(lo, color="orange", lw=0.8, ls="--", alpha=0.9, label=f"lo={lo:.3g}")
        if np.isfinite(hi):
            ax.axvline(hi, color="orange", lw=0.8, ls="--", alpha=0.9, label=f"hi={hi:.3g}")

        d_finite = d[np.isfinite(d)]
        if d_finite.size:
            ymin = min(-0.5, float(np.min(d_finite)))
            ymax = max(2.5, float(np.percentile(d_finite, 95)) * 1.1)
            ax.set_ylim(ymin, ymax)
        ax.set_title(f"{key}  {roi}", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Δχ²", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.7)

    # Stamp on 3rd panel when available (requested for grids).
    target = axes.ravel()[2] if axes.size > 2 else axes.ravel()[0]
    stamp_prior_nuisance(
        target,
        prior_kind=prior_kind,
        nuisance_mode=nuisance_mode,
        color="0.25",
    )

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.suptitle(
        f"{title_prefix}Δχ²({pname}) profile-likelihood curves -- {sbpl_prof_name}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"[plot_profile_curves] wrote {out_path}")
    plt.close(fig)


def plot_profile_curves(
    fitting,
    *,
    sbpl_prof_name: str,
    global_result=None,
    out_dir: str | Path,
    out_prefix: str = "profile_curves",
    title_prefix: str = "",
    delta_ref: float = 1.0,
    prior_kind: str | None = None,
    nuisance_mode: str | None = None,
) -> None:
    """Save two PNGs (one per parameter) with the per-window 1-D Δχ² curves.

    Parameters
    ----------
    fitting
        A populated :class:`crbreak.fitting.main.Fitting`; each window's
        ``profile_curves[sbpl_prof_name]`` must be set (it is, when
        :meth:`Fitting.fit_per_window` runs with ``do_profile_errors=True``).
    sbpl_prof_name
        Name of the profiled SBPL fit (e.g. ``"SBPL|prof.box"``).
    global_result
        Optional profiled global SBPL result.  When passed, a ``GLOBAL``
        panel is appended to each figure using ``global_result.profile_curves``.
    out_dir
        Directory where the PNGs are written.
    out_prefix
        Filename prefix; outputs ``<prefix>_Ebreak.png`` and ``<prefix>_dgamma.png``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = list(fitting.window_fits.items())
    for pname in ("Ebreak", "dgamma"):
        _plot_one_param(
            items,
            pname=pname,
            sbpl_prof_name=sbpl_prof_name,
            global_result=global_result,
            out_path=out_dir / f"{out_prefix}_{pname}.png",
            title_prefix=title_prefix,
            delta_ref=delta_ref,
            prior_kind=prior_kind,
            nuisance_mode=nuisance_mode,
        )
