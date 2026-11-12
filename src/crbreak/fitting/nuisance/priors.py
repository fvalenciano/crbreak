"""Nuisance priors and smoothness penalty.

A *prior* assigns a chi-squared penalty to a nuisance vector ``w``.

* :class:`GaussianPrior`  -- ``Σ (w_j / σ_w)²``  (quadratic).
* :class:`BoxGaussPrior`  -- flat box of half-width ``a``, Gaussian tails of
  width ``τ = box_tail_sigma_frac · a`` (non-quadratic).

A :class:`SmoothnessPenalty` is an *add-on* that penalises differences
between adjacent nodes; it is always quadratic and can be combined with
either prior via :meth:`NuisancePrior.with_smoothness`.

Every prior exposes:

* ``chi2(w)`` -- value of the penalty at ``w``.
* ``augmented_residuals(w)`` -- pseudo-residual vector ``r`` with
  ``||r||² = chi2(w)`` (so it slots straight into ``scipy.optimize.least_squares``).
* ``is_quadratic`` -- if ``True``, ``linear_components()`` returns
  ``(A, b)`` with ``chi2(w) = ||A w - b||²``, enabling a single
  ``np.linalg.lstsq`` inner solve.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .basis import smooth_prior_sigma_edges


# ----------------------------------------------------------------------
# Smoothness penalty (composable add-on)
# ----------------------------------------------------------------------


class SmoothnessPenalty:
    """Adjacent-difference penalty ``λ Σ ((w_{j+1}-w_j)/σ_edge_j)²``."""

    is_quadratic = True

    def __init__(
        self,
        lam: float,
        x_nuis: np.ndarray,
        *,
        delta_logE_ref: float = 0.5,
        max_delta_w_ref: float = 2.0,
    ):
        self.lam = float(lam)
        self.x_nuis = np.asarray(x_nuis, dtype=float)
        self.n_nuis = self.x_nuis.size
        self.sigma_edges = smooth_prior_sigma_edges(
            self.x_nuis,
            delta_logE_ref=delta_logE_ref,
            max_delta_w_ref=max_delta_w_ref,
        )
        # Cache forward-difference matrix D so that D @ w == np.diff(w).
        n = self.n_nuis
        D = np.zeros((n - 1, n), dtype=float)
        rows = np.arange(n - 1)
        D[rows, rows] = -1.0
        D[rows, rows + 1] = 1.0
        self.D = D

    @property
    def n_pseudo(self) -> int:
        return self.n_nuis - 1

    def chi2(self, w: np.ndarray) -> float:
        if self.lam <= 0.0 or self.n_nuis < 2:
            return 0.0
        dw = np.diff(np.asarray(w, dtype=float))
        return float(self.lam * np.sum((dw / self.sigma_edges) ** 2))

    def augmented_residuals(self, w: np.ndarray) -> np.ndarray:
        if self.lam <= 0.0 or self.n_nuis < 2:
            return np.zeros(0, dtype=float)
        dw = np.diff(np.asarray(w, dtype=float))
        return np.sqrt(self.lam) * dw / self.sigma_edges

    def linear_components(self) -> tuple[np.ndarray, np.ndarray]:
        """``(A, b)`` such that ``chi2(w) = ||A w - b||²``."""
        if self.lam <= 0.0 or self.n_nuis < 2:
            return np.zeros((0, self.n_nuis)), np.zeros(0)
        scale = np.sqrt(self.lam) / self.sigma_edges
        A = scale[:, None] * self.D
        b = np.zeros(self.n_nuis - 1, dtype=float)
        return A, b


# ----------------------------------------------------------------------
# Priors
# ----------------------------------------------------------------------


class NuisancePrior(ABC):
    """Abstract per-node prior on the nuisance vector."""

    n_nuis: int
    is_quadratic: bool = False
    short_name: str = "prior"

    def __init__(
        self, n_nuis: int, *, smoothness: SmoothnessPenalty | None = None
    ):
        self.n_nuis = int(n_nuis)
        self.smoothness = smoothness

    # --- API -----------------------------------------------------------
    @abstractmethod
    def _node_chi2(self, w: np.ndarray) -> float: ...

    @abstractmethod
    def _node_residuals(self, w: np.ndarray) -> np.ndarray:
        """Pseudo-residuals ``r`` such that ``||r||² == _node_chi2(w)``."""

    def chi2(self, w: np.ndarray) -> float:
        c = float(self._node_chi2(w))
        if self.smoothness is not None:
            c += self.smoothness.chi2(w)
        return c

    def augmented_residuals(self, w: np.ndarray) -> np.ndarray:
        parts = [self._node_residuals(w)]
        if self.smoothness is not None:
            parts.append(self.smoothness.augmented_residuals(w))
        return np.concatenate(parts) if len(parts) > 1 else parts[0]

    @property
    def n_pseudo(self) -> int:
        n = self.n_nuis
        if self.smoothness is not None:
            n += self.smoothness.n_pseudo
        return n

    # Quadratic-only API.  Default raises (non-quadratic priors must override
    # ``is_quadratic = False`` and just inherit the default).
    def linear_components(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            f"{type(self).__name__} is not quadratic; "
            "no closed-form linear_components()."
        )

    # Convenience: composability with a smoothness penalty.
    def with_smoothness(self, smoothness: SmoothnessPenalty | None) -> "NuisancePrior":
        self.smoothness = smoothness
        return self

    # --- Bounds (for non-quadratic inner L-BFGS-B) ----------------------
    def w_bounds(self) -> list[tuple[float, float]] | None:
        return None  # unbounded by default

    def __repr__(self) -> str:
        sm = " +smooth" if self.smoothness is not None else ""
        return f"{type(self).__name__}(n_nuis={self.n_nuis}{sm})"


class GaussianPrior(NuisancePrior):
    """``χ²(w) = Σ (w_j / σ_w)²``."""

    is_quadratic = True
    short_name = "gauss"

    def __init__(
        self,
        n_nuis: int,
        sigma: float = 1.0,
        *,
        smoothness: SmoothnessPenalty | None = None,
    ):
        super().__init__(n_nuis, smoothness=smoothness)
        if sigma <= 0.0:
            raise ValueError("GaussianPrior requires sigma > 0.")
        self.sigma = float(sigma)

    def _node_chi2(self, w: np.ndarray) -> float:
        w = np.asarray(w, dtype=float)
        return float(np.sum(w * w) / (self.sigma ** 2))

    def _node_residuals(self, w: np.ndarray) -> np.ndarray:
        return np.asarray(w, dtype=float) / self.sigma

    def linear_components(self) -> tuple[np.ndarray, np.ndarray]:
        node_A = np.eye(self.n_nuis, dtype=float) / self.sigma
        node_b = np.zeros(self.n_nuis, dtype=float)
        if self.smoothness is None:
            return node_A, node_b
        sm_A, sm_b = self.smoothness.linear_components()
        return np.vstack([node_A, sm_A]), np.concatenate([node_b, sm_b])


class BoxGaussPrior(NuisancePrior):
    """Flat box of half-width ``a`` with Gaussian tails of width ``τ``.

    .. math::

        \\chi^2_\\text{node}(w_j) =
        \\begin{cases}
        0 & |w_j| \\le a \\\\
        \\bigl(\\tfrac{|w_j| - a}{\\tau}\\bigr)^2 & |w_j| > a
        \\end{cases}

    with ``τ = box_tail_sigma_frac · a``.

    Not quadratic in ``w`` (kink at ``|w|=a``), so the inner solve falls
    back to a small L-BFGS-B run warm-started from the Gaussian-equivalent
    closed-form solution.
    """

    is_quadratic = False
    short_name = "box"

    def __init__(
        self,
        n_nuis: int,
        width: float,
        tail_sigma_frac: float = 0.1,
        *,
        smoothness: SmoothnessPenalty | None = None,
    ):
        super().__init__(n_nuis, smoothness=smoothness)
        if width <= 0.0:
            raise ValueError("BoxGaussPrior requires width > 0.")
        if tail_sigma_frac <= 0.0:
            raise ValueError("BoxGaussPrior requires tail_sigma_frac > 0.")
        self.width = float(width)
        self.tail_sigma_frac = float(tail_sigma_frac)
        self.tau = float(width * tail_sigma_frac)

    def _excess(self, w: np.ndarray) -> np.ndarray:
        return np.maximum(np.abs(np.asarray(w, dtype=float)) - self.width, 0.0)

    def _node_chi2(self, w: np.ndarray) -> float:
        return float(np.sum((self._excess(w) / self.tau) ** 2))

    def _node_residuals(self, w: np.ndarray) -> np.ndarray:
        return self._excess(w) / self.tau

    def gaussian_equivalent(self) -> "GaussianPrior":
        """Pure-quadratic prior used as a warm-start for the inner solver.

        Uses the box width as the effective sigma so the optimum lives near
        ``|w| ~ a`` for ``|excess| ≪ τ``.  The smoothness add-on is reused
        verbatim.
        """
        return GaussianPrior(
            self.n_nuis,
            sigma=self.width,
            smoothness=self.smoothness,
        )

    def __repr__(self) -> str:
        sm = " +smooth" if self.smoothness is not None else ""
        return (
            f"BoxGaussPrior(n_nuis={self.n_nuis}, width={self.width:g}, "
            f"tail_sigma_frac={self.tail_sigma_frac:g}{sm})"
        )


# ----------------------------------------------------------------------
# Convenience factories (used by NuisanceModel / Fitting.fit_per_window)
# ----------------------------------------------------------------------


def make_prior(
    kind: str,
    n_nuis: int,
    *,
    sigma: float,
    width: float | None,
    tail_sigma_frac: float,
    smoothness: SmoothnessPenalty | None = None,
) -> NuisancePrior:
    """String-based factory matching the legacy ``PROFILED_PRIOR_MODE`` knobs.

    Examples
    --------
    >>> make_prior("gaussian", 6, sigma=1.0)
    GaussianPrior(n_nuis=6)
    >>> make_prior("box_gauss", 6, width=1.0, tail_sigma_frac=0.1)
    BoxGaussPrior(n_nuis=6, width=1, tail_sigma_frac=0.1)
    """
    k = str(kind).strip().lower()
    if k in ("gaussian", "gauss"):
        return GaussianPrior(n_nuis, sigma=sigma, smoothness=smoothness)
    if k in ("box_gauss", "box+gauss", "boxgauss", "box"):
        eff_width = float(width) if width is not None else float(sigma)
        return BoxGaussPrior(
            n_nuis,
            width=eff_width,
            tail_sigma_frac=tail_sigma_frac,
            smoothness=smoothness,
        )
    raise ValueError(
        f"Unknown prior kind={kind!r}. Use 'gaussian' or 'box_gauss'."
    )
