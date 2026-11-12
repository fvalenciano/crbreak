"""Monte-Carlo envelope from a Gaussian posterior on the fit parameters.

This is the same recipe used in the legacy ``FDE_PASS8_new_v4.py``:
draw ``n`` samples from :math:`\\mathcal{N}(\\hat\\theta, \\hat C)`,
evaluate the model on each draw, and report the requested percentile band.
``n``, ``q`` and ``seed`` come from the analysis config (``plots.sed_band``).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def envelope_from_cov(
    model_callable: Callable[..., np.ndarray],
    popt: np.ndarray,
    pcov: np.ndarray,
    E_dense: np.ndarray,
    *,
    n: int,
    q: tuple[float, float],
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lo, hi)`` percentile band on the fitted curve.

    Parameters
    ----------
    model_callable
        Any callable ``f(E, *params) -> ndarray``.  Typically a
        :class:`~crbreak.fitting.models.SpectralModel` instance.
    popt, pcov
        Best-fit parameter vector and covariance matrix.
    E_dense
        Energies (GeV) at which to evaluate the envelope.
    n
        Number of Monte-Carlo draws.  Legacy uses 600.
    q
        Percentiles to return.  Default ``(5, 95)``.
    seed
        RNG seed (``None`` to disable reproducibility).

    Notes
    -----
    Draws that contain non-finite values (typically when the random sample
    falls in a region where the model diverges) are dropped before computing
    percentiles.  If *all* draws are non-finite, a NaN array is returned.
    """
    popt = np.asarray(popt, dtype=float)
    pcov = np.asarray(pcov, dtype=float)
    E_dense = np.asarray(E_dense, dtype=float)

    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(popt, pcov, int(n))

    curves = np.empty((samples.shape[0], E_dense.size), dtype=float)
    for i, pars in enumerate(samples):
        curves[i] = model_callable(E_dense, *pars)

    keep = np.all(np.isfinite(curves), axis=1)
    curves = curves[keep]
    if curves.size == 0:
        nan = np.full_like(E_dense, np.nan, dtype=float)
        return nan, nan

    lo, hi = np.percentile(curves, list(q), axis=0)
    return np.asarray(lo, float), np.asarray(hi, float)
