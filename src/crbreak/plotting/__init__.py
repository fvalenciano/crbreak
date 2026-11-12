"""Reusable plotting helpers for the ``crbreak`` analysis.

Four modules, mirroring the legacy ``make_final_plots.py`` figures:

* :mod:`crbreak.plotting.plot_sed`           -- SED grid (PL + SBPL), one
  dataset (PASS8 *or* IEM); supports single-window or multi-window.
* :mod:`crbreak.plotting.plot_params_vs_glon` -- ``E_break`` and ``Δγ`` (or
  ``γ1`` and ``γ2``) vs ``|l|``; overlay-capable (PASS8 + IEM, or |b| bands).
  Also vs ``|b|`` for one |l| window in several |b| bands, and Ebreak / Δγ
  with γ1 and γ2 in one figure.
* :mod:`crbreak.plotting.plot_contour_plot`  -- 2-D ``Δχ²(Ebreak, dgamma)``
  contours from a :class:`ChiSquareScan`; overlay-capable.
* :mod:`crbreak.plotting.plot_nuisance`      -- Per-window nuisance
  posterior ``s(E;w)`` + ``w_j`` (legacy
  ``plot_s_posterior_vs_logE``), one dataset.

The overlay plots use small ``*Dataset`` dataclasses to bundle each curve's
data + style.  This avoids the ``Fitting`` / ``ChiSquareScan`` having to know
about colors or labels, and makes it trivial to extend later (e.g. a third
DRAGON dataset) without touching the plotting code.

Importing this package applies a small set of opinionated ``rcParams`` so
every figure produced by these helpers shares the legacy STIX look-and-feel.
Use :func:`apply_default_rcparams` to re-apply them later (e.g. after a
notebook reset) and :func:`reset_rcparams_to_default` to roll back.
"""

import matplotlib as _mpl

# Snapshot the original rcParams so we can roll back on demand.
_ORIGINAL_RCPARAMS = dict(_mpl.rcParamsDefault)

_DEFAULT_RCPARAMS: dict[str, object] = {
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
}


def apply_default_rcparams() -> None:
    """Apply the package-wide ``rcParams`` (STIX math, STIXGeneral font)."""
    _mpl.rcParams.update(_DEFAULT_RCPARAMS)


def reset_rcparams_to_default() -> None:
    """Restore matplotlib's factory ``rcParams``."""
    _mpl.rcParams.update(_ORIGINAL_RCPARAMS)


# Apply on import so users of ``crbreak.plotting`` get the look automatically.
apply_default_rcparams()


from .plot_contour_plot import ContourDataset, plot_contour_heatmap, plot_contour_plot  # noqa: E402
from .plot_mc_diagnostics import plot_mc_diagnostics  # noqa: E402
from .plot_mc_toy_movie import make_mc_toy_movie  # noqa: E402
from .plot_nuisance import NuisanceStyle, plot_nuisance  # noqa: E402
from .plot_params_vs_glon import (  # noqa: E402
    ParamsDataset,
    plot_gammas_vs_glon,
    plot_params_gammas_vs_glat,
    plot_params_gammas_vs_glon,
    plot_params_vs_glat,
    plot_params_vs_glon,
)
from .params_vs_glon_plus_contour import plot_params_vs_glon_plus_contour  # noqa: E402
from .plot_background_components import plot_background_components  # noqa: E402
from .plot_ps_comparison import plot_ps_comparison  # noqa: E402
from .plot_sed import plot_sed  # noqa: E402
from .plot_sed_plus_nuisances import plot_sed_plus_nuisances  # noqa: E402

__all__ = [
    "plot_sed",
    "plot_params_vs_glon",
    "plot_gammas_vs_glon",
    "plot_params_gammas_vs_glon",
    "plot_params_gammas_vs_glat",
    "plot_params_vs_glat",
    "ParamsDataset",
    "plot_params_vs_glon_plus_contour",
    "plot_contour_plot",
    "plot_contour_heatmap",
    "ContourDataset",
    "plot_mc_diagnostics",
    "make_mc_toy_movie",
    "plot_nuisance",
    "NuisanceStyle",
    "plot_background_components",
    "plot_ps_comparison",
    "plot_sed_plus_nuisances",
    "apply_default_rcparams",
    "reset_rcparams_to_default",
]
