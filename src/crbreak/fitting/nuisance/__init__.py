"""Nuisance-parameter machinery (deformations of the model prediction).

This sub-package describes the *math* of systematic-deformation
nuisance parameters used by the profiled chi-squared:

.. math::
    \\hat y_i(\\theta, w) = (1 + s(E_i; w)\\,S_i)\\,\\mu(E_i; \\theta),
    \\quad s(E; w) = B(\\log_{10} E)\\,w

with ``S(E)`` the per-bin fractional systematic (e.g. ``Aeff`` error,
already extracted upstream as ``Ssys_frac``) and ``w`` a small set of
nuisance coefficients with a prior penalty (Gaussian or box-Gaussian).

Top-level building blocks
-------------------------
* :class:`NuisanceBasis`     -- ``B(log10 E)`` + node positions ``x_nuis``.
* :class:`NuisancePrior`     -- ABC; concrete: :class:`GaussianPrior`,
  :class:`BoxGaussPrior`.  Optional :class:`SmoothnessPenalty` add-on.
* :class:`NuisanceModel`     -- single class with two states:
    - *unbound* (configuration knob the user passes to
      :class:`Fitting`): wraps a :class:`SpectralModel`, a prior factory
      and the basis hyper-parameters.
    - *bound* (output of :meth:`NuisanceModel.for_window`): the same
      object enriched with a basis evaluated on one window's energy
      grid, the corresponding ``Ssys_frac`` array, a fresh prior
      instance and the inner profile machinery (``profile_w``,
      ``predict_deformed_on_grid``...).  Behaves like a
      :class:`SpectralModel` so it slots straight into
      :class:`crbreak.fitting.chi2.FitResult`.

The actual profiled-fit *engines* live next to the classical ones:

* :class:`crbreak.fitting.chi2.ProfiledChi2Fit`              -- per-window.
* :class:`crbreak.fitting.global_chi2.ProfiledGlobalChi2Fit` -- joint.

Convenience entry points
------------------------
* :func:`make_prior` for string-based prior selection
  (``"gaussian"`` / ``"box_gauss"``) matching the legacy
  ``PROFILED_PRIOR_MODE`` knob.
"""

from .basis import NuisanceBasis, smooth_prior_sigma_edges
from .model import NuisanceModel, nuisance_kwargs_from_config
from .priors import (
    BoxGaussPrior,
    GaussianPrior,
    NuisancePrior,
    SmoothnessPenalty,
    make_prior,
)

__all__ = [
    "NuisanceBasis",
    "NuisanceModel",
    "nuisance_kwargs_from_config",
    "NuisancePrior",
    "GaussianPrior",
    "BoxGaussPrior",
    "SmoothnessPenalty",
    "make_prior",
    "smooth_prior_sigma_edges",
]
