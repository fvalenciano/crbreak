"""Statistical helpers for the chi-squared fitter.

Currently exposes :func:`wilks_pvalue`, the standard Wilks' LRT p-value and
one-sided significance used in the legacy ``FDE_PASS8_new_v4.py`` /
``nuisance_fit.py`` pipelines:

.. math::
    \\Delta\\chi^2_+ = \\max(0,\\,\\Delta\\chi^2)\\,,\\quad
    p_\\mathrm{Wilks} = \\mathcal{P}(\\chi^2_{\\mathrm{dof}} > \\Delta\\chi^2_+)\\,,\\quad
    Z_\\mathrm{Wilks} = \\Phi^{-1}(1 - p)
"""

from __future__ import annotations

from scipy.stats import chi2 as _chi2_dist, norm as _norm

# Floor used to keep ``norm.isf(p)`` finite when ``p == 0``.  Same value as in
# the legacy code (10^-300).
_P_FLOOR: float = 1e-300


def wilks_pvalue(delta_chi2: float, dof: int) -> dict:
    """Return the Wilks LRT p-value and one-sided Z significance.

    Parameters
    ----------
    delta_chi2
        ``chi2(simpler) - chi2(complex)`` for two **nested** models (the
        complex model has more free parameters than the simpler one).
        Negative values are clipped to 0 (legacy convention).
    dof
        Difference in number of free parameters between the two models.

    Returns
    -------
    dict with keys
        - ``delta_chi2``      : the input as float
        - ``delta_chi2_pos``  : ``max(0, delta_chi2)``
        - ``dof``             : the input ``dof`` as int
        - ``p_wilks``         : ``chi2.sf(delta_chi2_pos, dof)``
        - ``Z_wilks``         : ``norm.isf(max(p, 1e-300))`` (one-sided)
    """
    delta = float(delta_chi2)
    delta_pos = max(0.0, delta)
    df = int(dof)
    p = float(_chi2_dist.sf(delta_pos, df))
    z = float(_norm.isf(max(p, _P_FLOOR)))
    return {
        "delta_chi2": delta,
        "delta_chi2_pos": delta_pos,
        "dof": df,
        "p_wilks": p,
        "Z_wilks": z,
    }
