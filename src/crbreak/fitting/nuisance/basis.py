"""Nuisance bases ``s(E; w) = B(log10 E) @ w``.

Two flavours are supported, mirroring the legacy ``make_bspline_basis``:

* :meth:`NuisanceBasis.piecewise_linear` (default, Appendix-C in the
  Fermi-LAT cosmic-ray-break paper).  ``ncoeff`` reference energies are
  picked uniformly in ``log10(E)`` and ``s(E)`` is linear-in-log10 between
  consecutive nodes.  Each ``w_j`` is then literally ``s(E_j)``, which
  makes the priors and the smoothness penalty trivially interpretable.

* :meth:`NuisanceBasis.bspline` (legacy alternative).  Open-uniform knot
  vector with the requested ``degree`` (default cubic).

The ``x_nuis`` attribute returns the nuisance node positions in
``log10(E)`` space and is used by the optional smoothness penalty to
scale per-edge widths with the local node spacing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline


@dataclass
class NuisanceBasis:
    """Container for a nuisance basis evaluated at the data energies.

    Attributes
    ----------
    B : np.ndarray
        ``(N_data, n_nuis)`` design matrix so that ``s(E_data) = B @ w``.
    x_nuis : np.ndarray
        ``(n_nuis,)`` node positions in ``log10(E)`` (used by smoothness).
    n_nuis : int
        Number of nuisance coefficients.
    kind : str
        ``"piecewise_linear"`` or ``"bspline"`` (informational).
    knots : np.ndarray, optional
        B-spline knot vector (only set when ``kind == "bspline"``).  Used
        by :meth:`evaluate` to rebuild the design matrix on an arbitrary
        ``logE`` grid with the same nodes as the original fit.
    degree : int
        B-spline degree (only meaningful for ``kind == "bspline"``).
    """

    B: np.ndarray
    x_nuis: np.ndarray
    n_nuis: int
    kind: str = "piecewise_linear"
    knots: np.ndarray | None = None
    degree: int = 0

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def piecewise_linear(
        cls,
        logE: np.ndarray,
        n_nuis: int,
    ) -> "NuisanceBasis":
        """Linear-in-log10(E) basis with ``n_nuis`` uniformly-spaced nodes.

        ``w_j == s(E_j)`` so each coefficient has a direct physical meaning
        (fractional Aeff distortion at the j-th reference energy).
        """
        if n_nuis < 2:
            raise ValueError("Need n_nuis >= 2 for piecewise-linear basis.")

        logE = np.asarray(logE, dtype=float)
        a, b = float(logE.min()), float(logE.max())
        x_nodes = np.linspace(a, b, n_nuis)
        B = _piecewise_linear_design(logE, x_nodes)
        return cls(B=B, x_nuis=x_nodes, n_nuis=n_nuis, kind="piecewise_linear")

    @classmethod
    def bspline(
        cls,
        logE: np.ndarray,
        n_nuis: int,
        degree: int,
    ) -> "NuisanceBasis":
        """Legacy B-spline basis with open-uniform knot vector."""
        if n_nuis <= degree:
            raise ValueError("Need n_nuis > degree for B-spline basis.")

        logE = np.asarray(logE, dtype=float)
        a, b = float(logE.min()), float(logE.max())
        n_internal = max(0, n_nuis - (degree + 1))
        if n_internal > 0:
            internal = np.linspace(a, b, n_internal + 2)[1:-1]
            knots = np.r_[
                np.repeat(a, degree + 1), internal, np.repeat(b, degree + 1)
            ]
        else:
            knots = np.r_[np.repeat(a, degree + 1), np.repeat(b, degree + 1)]

        B = np.zeros((logE.size, n_nuis), dtype=float)
        for j in range(n_nuis):
            c = np.zeros(n_nuis)
            c[j] = 1.0
            spl = BSpline(knots, c, degree, extrapolate=True)
            B[:, j] = spl(logE)

        x_nuis = _greville_abscissae(knots, degree, n_nuis)
        return cls(
            B=B,
            x_nuis=x_nuis,
            n_nuis=n_nuis,
            kind="bspline",
            knots=knots,
            degree=degree,
        )

    @classmethod
    def from_E(
        cls,
        E_GeV: np.ndarray,
        *,
        n_nuis: int,
        kind: str,
        degree: int,
    ) -> "NuisanceBasis":
        """Convenience: build directly from energies in GeV (``degree``: B-spline only)."""
        logE = np.log10(np.asarray(E_GeV, dtype=float))
        if kind == "piecewise_linear":
            return cls.piecewise_linear(logE, n_nuis)
        if kind == "bspline":
            return cls.bspline(logE, n_nuis, degree=degree)
        raise ValueError(f"Unknown basis kind={kind!r}.")

    # ------------------------------------------------------------------
    # Subsetting (used by the fitter to apply the data-mask)
    # ------------------------------------------------------------------
    def subset(self, mask: np.ndarray) -> "NuisanceBasis":
        """Return a basis with rows of ``B`` selected by ``mask``.

        Nodes (``x_nuis``, ``knots``, ``degree``) are preserved so the
        returned basis can be re-evaluated on any ``logE`` grid via
        :meth:`evaluate` and recover the same nuisance functional form.
        """
        mask = np.asarray(mask, dtype=bool)
        return NuisanceBasis(
            B=self.B[mask],
            x_nuis=self.x_nuis.copy(),
            n_nuis=self.n_nuis,
            kind=self.kind,
            knots=None if self.knots is None else self.knots.copy(),
            degree=self.degree,
        )

    # ------------------------------------------------------------------
    # Re-evaluation on an arbitrary log10(E) grid
    # ------------------------------------------------------------------
    def evaluate(self, logE: np.ndarray) -> np.ndarray:
        """Design matrix on ``logE`` reusing the *original* nodes.

        Lets callers (e.g. plotting code) draw the deformed prediction
        on a dense energy grid without rebuilding the basis from a
        different node set.

        * ``piecewise_linear``: linear interpolation on ``self.x_nuis``;
          values outside the original range are clamped to the endpoint
          nodes (mirroring the fit-time behaviour).
        * ``bspline``: re-evaluates the same B-spline functions defined
          by ``self.knots`` / ``self.degree``; ``extrapolate=True``
          extends them gracefully outside the original support.
        """
        logE_new = np.asarray(logE, dtype=float)
        if self.kind == "piecewise_linear":
            return _piecewise_linear_design(logE_new, self.x_nuis)
        if self.kind == "bspline":
            if self.knots is None:
                raise ValueError(
                    "B-spline basis is missing its knot vector; rebuild "
                    "with NuisanceBasis.bspline(...) first."
                )
            B = np.zeros((logE_new.size, self.n_nuis), dtype=float)
            for j in range(self.n_nuis):
                c = np.zeros(self.n_nuis)
                c[j] = 1.0
                spl = BSpline(self.knots, c, self.degree, extrapolate=True)
                B[:, j] = spl(logE_new)
            return B
        raise ValueError(f"Unknown basis kind={self.kind!r}.")


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _piecewise_linear_design(logE: np.ndarray, x_nodes: np.ndarray) -> np.ndarray:
    """Design matrix for linear interpolation on fixed ``log10(E)`` nodes."""
    logE = np.asarray(logE, dtype=float)
    x_nodes = np.asarray(x_nodes, dtype=float)
    n_nuis = x_nodes.size

    B = np.zeros((logE.size, n_nuis), dtype=float)

    idx = np.searchsorted(x_nodes, logE, side="right") - 1
    idx = np.clip(idx, 0, n_nuis - 2)

    x0 = x_nodes[idx]
    x1 = x_nodes[idx + 1]
    t = (logE - x0) / (x1 - x0)

    rows = np.arange(logE.size)
    B[rows, idx] = 1.0 - t
    B[rows, idx + 1] = t

    # Clamp outside the range to the endpoint nodes.
    left = logE <= x_nodes[0]
    right = logE >= x_nodes[-1]
    if np.any(left):
        B[left, :] = 0.0
        B[left, 0] = 1.0
    if np.any(right):
        B[right, :] = 0.0
        B[right, -1] = 1.0
    return B


def _greville_abscissae(
    knots: np.ndarray, degree: int, n_nuis: int
) -> np.ndarray:
    """Greville abscissae of a B-spline (used as nuisance node positions)."""
    t = np.asarray(knots, dtype=float)
    out = np.empty(n_nuis, dtype=float)
    for j in range(n_nuis):
        out[j] = float(np.mean(t[j + 1 : j + 1 + degree]))
    return out


def smooth_prior_sigma_edges(
    x_nuis: np.ndarray,
    *,
    delta_logE_ref: float = 0.5,
    max_delta_w_ref: float = 2.0,
) -> np.ndarray:
    """Per-edge sigma for ``(w_{j+1}-w_j)``, scaled with local node spacing.

    Same prescription as the legacy ``_smooth_prior_sigma_edges``: an edge
    spanning ``delta_logE_ref`` (default 0.5 dex) gets sigma
    ``max_delta_w_ref`` (default 2.0); narrower/wider edges scale linearly.
    """
    x_nuis = np.asarray(x_nuis, dtype=float)
    dx = np.diff(x_nuis)
    dx_safe = np.where(dx > 0.0, dx, delta_logE_ref)
    return max_delta_w_ref * (dx_safe / float(delta_logE_ref))
