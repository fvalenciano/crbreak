"""Per-window chi-squared fitters and their result container.

Two engines live here, both filling the same :class:`FitResult`:

* :class:`Chi2Fit` -- *classical* fit: ``scipy.optimize.curve_fit`` with
  ``absolute_sigma=True``, errors entering only via the ``sigma`` array
  (statistical and systematic combined in quadrature *upstream*).
* :class:`ProfiledChi2Fit` -- *profiled* fit on the chi-squared
  augmented with systematic-deformation nuisance parameters:
  outer ``L-BFGS-B`` minimisation of the scalar ``chi2_prof(theta)``,
  inner profile over ``w`` handled by a *bound*
  :class:`~crbreak.fitting.nuisance.NuisanceModel` (output of
  :meth:`NuisanceModel.for_window`).

Both fitters are model-agnostic at the per-window level; their global
counterparts live in :mod:`crbreak.fitting.global_chi2`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import curve_fit, minimize

from .envelope import envelope_from_cov
from .models import SpectralModel
from .nuisance import NuisanceModel
from .nuisance._solver_utils import _fd_hessian, _pcov_from_hessian, _zip_bounds


@dataclass
class FitResult:
    """Outcome of a single-model, single-window chi-squared fit.

    All array attributes refer to the *masked* sample actually used by the
    fitter (after dropping NaNs / non-positive sigma).
    """

    model: SpectralModel
    popt: np.ndarray
    pcov: np.ndarray
    chi2: float
    ndof: int
    E: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    converged: bool = True
    solver_message: str = ""
    w_best: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    @property
    def perr(self) -> np.ndarray:
        return np.sqrt(np.diag(self.pcov))

    @property
    def chi2_red(self) -> float:
        return float(self.chi2 / self.ndof) if self.ndof > 0 else float("nan")

    @property
    def params(self) -> dict[str, tuple[float, float]]:
        """``{param_name: (best_value, 1-sigma error)}``."""
        return {
            n: (float(v), float(e))
            for n, v, e in zip(self.model.param_names, self.popt, self.perr)
        }

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    def predict(self, E: np.ndarray) -> np.ndarray:
        return self.model(np.asarray(E, dtype=float), *self.popt)

    def envelope(
        self,
        E_dense: np.ndarray,
        *,
        n: int,
        q: tuple[float, float],
        seed: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Monte-Carlo percentile band of the fitted curve (see :func:`envelope_from_cov`)."""
        return envelope_from_cov(
            self.model, self.popt, self.pcov, E_dense, n=n, q=q, seed=seed
        )

    def residuals(self) -> np.ndarray:
        """Pearson residuals on the fit sample: ``(y - model)/sigma``."""
        return (self.y - self.predict(self.E)) / self.sigma

    def __repr__(self) -> str:
        bits = ", ".join(
            f"{k}={v:.3g}±{e:.2g}" for k, (v, e) in self.params.items()
        )
        return (
            f"FitResult({self.model.name}: chi2={self.chi2:.2f}/{self.ndof}, "
            f"{bits})"
        )


class Chi2Fit:
    """Plain chi-squared fitter wrapping :func:`scipy.optimize.curve_fit`.

    Parameters
    ----------
    model : :class:`SpectralModel`
        Model to fit.  Initial guess and bounds are taken from the model's
        ``default_p0`` / ``default_bounds`` unless overridden in
        :meth:`fit`.
    solver
        ``solver`` section of the analysis config (``chi2_maxfev``: maximum
        number of function evaluations).
    """

    def __init__(self, model: SpectralModel, *, solver):
        self.model = model
        self.maxfev = int(solver.chi2_maxfev)

    def fit(
        self,
        E: np.ndarray,
        y: np.ndarray,
        sigma: np.ndarray,
        *,
        p0: Sequence[float] | None = None,
        bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    ) -> FitResult:
        """Fit ``self.model(E, *params) ≈ y`` with errors ``sigma``."""
        E = np.asarray(E, dtype=float)
        y = np.asarray(y, dtype=float)
        sigma = np.asarray(sigma, dtype=float)

        mask = (
            np.isfinite(E) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
        )
        E_, y_, sig_ = E[mask], y[mask], sigma[mask]
        if E_.size <= self.model.n_params:
            raise RuntimeError(
                f"{self.model.name}: only {E_.size} usable bins for "
                f"{self.model.n_params} parameters."
            )

        p0_ = list(p0) if p0 is not None else list(self.model.default_p0(E_, y_))
        bounds_ = bounds if bounds is not None else self.model.default_bounds()

        popt, pcov = curve_fit(
            self.model,
            E_,
            y_,
            p0=p0_,
            sigma=sig_,
            bounds=bounds_,
            absolute_sigma=True,
            maxfev=self.maxfev,
        )
        chi2 = float(np.sum(((y_ - self.model(E_, *popt)) / sig_) ** 2))
        ndof = int(E_.size - len(popt))
        return FitResult(
            model=self.model,
            popt=np.asarray(popt, dtype=float),
            pcov=np.asarray(pcov, dtype=float),
            chi2=chi2,
            ndof=ndof,
            E=E_,
            y=y_,
            sigma=sig_,
            converged=True,
            solver_message="",
            w_best=None,
        )


# ----------------------------------------------------------------------
# Profiled per-window fit
# ----------------------------------------------------------------------


class ProfiledChi2Fit:
    """Profiled chi-squared fitter for a single window.

    Mirrors the legacy :func:`fit_profiled` driver: outer minimisation
    over ``theta`` is done with :func:`scipy.optimize.minimize` using
    L-BFGS-B on the *scalar* profiled chi-squared, with the inner
    profile over ``w`` handled by a *bound*
    :class:`~crbreak.fitting.nuisance.NuisanceModel` (basis, ``Ssys``
    and prior already attached to one window).  At convergence,
    ``pcov`` is built from a numerical Hessian of the same scalar
    ``chi2_prof(theta)`` (legacy ``2 · inv(H)``).

    Parameters
    ----------
    nm_bound
        Bound :class:`NuisanceModel` (output of
        :meth:`NuisanceModel.for_window`).  Its basis rows must match
        the unmasked input ``E_GeV`` row-for-row.
    solver
        ``solver`` section of the analysis config: ``profiled_max_iter``
        (L-BFGS-B ``maxiter``) and ``hessian_eps`` / ``hessian_ridge``
        (covariance from the numerical Hessian).
    """

    def __init__(self, nm_bound: NuisanceModel, *, solver):
        if not isinstance(nm_bound, NuisanceModel) or not nm_bound.is_bound:
            raise TypeError(
                "ProfiledChi2Fit requires a *bound* NuisanceModel "
                "(call .for_window(wf) first)."
            )
        self.predictor = nm_bound
        self.max_iter = int(solver.profiled_max_iter)
        self.hessian_eps = float(solver.hessian_eps)
        self.hessian_ridge = float(solver.hessian_ridge)

    @property
    def model(self) -> NuisanceModel:
        return self.predictor

    def fit(
        self,
        E: np.ndarray,
        y: np.ndarray,
        sigma: np.ndarray,
        *,
        p0: Sequence[float] | None = None,
        bounds: tuple[Sequence[float], Sequence[float]] | None = None,
        theta_classical: Sequence[float] | None = None,
        legacy_safety_log: list | None = None,
        warn_on_fail: bool = True,
        compute_covariance: bool = True,
    ) -> FitResult:
        """Minimise ``chi2_prof(theta)`` and return a :class:`FitResult`.

        Parameters
        ----------
        theta_classical
            If provided, mirrors the legacy ``FDE_PASS8`` safety net:
            after the L-BFGS-B outer converges, ``chi2_prof`` is also
            evaluated at ``theta_classical`` (e.g. the ``curve_fit``
            best-fit) and **the smaller** of the two is kept.  This
            guards against outer-optimizer regressions on non-smooth
            priors (BoxGauss).  ``pcov`` is then computed at the
            chosen ``theta`` via finite-difference Jacobian.
        legacy_safety_log
            Optional list to which a small dict is appended describing
            which solution won (``"profiled_outer"`` vs
            ``"classical_seed"``).  Useful for diagnostics.
        """
        E = np.asarray(E, dtype=float)
        y = np.asarray(y, dtype=float)
        sigma = np.asarray(sigma, dtype=float)

        mask = np.isfinite(E) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
        E_ = E[mask]
        y_ = y[mask]
        sig_ = sigma[mask]
        n_outer = self.predictor.n_params
        if E_.size <= n_outer:
            raise RuntimeError(
                f"{self.predictor.name}: only {E_.size} usable bins for "
                f"{n_outer} outer parameters."
            )

        if self.predictor.basis.B.shape[0] != E.size:
            raise ValueError(
                f"Predictor basis rows ({self.predictor.basis.B.shape[0]}) "
                f"do not match input E size ({E.size}). Build the basis "
                "from the same energy grid."
            )
        pred = self.predictor.subset(mask)
        # Reset the warm-start cache for this fit (the legacy
        # ``w_cache`` is local to ``fit_profiled``).
        pred.last_w = np.zeros(pred.n_nuis, dtype=float)

        if p0 is None:
            p0 = list(pred.default_p0(E_, y_))
        else:
            p0 = list(p0)
        if bounds is None:
            lo, hi = pred.default_bounds()
        else:
            lo, hi = bounds
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)

        eps = np.where(np.isfinite(hi), 1e-12 * np.maximum(np.abs(hi), 1.0), 0.0)
        theta0 = np.minimum(np.maximum(np.asarray(p0, dtype=float), lo + eps), hi - eps)

        def obj(theta: np.ndarray) -> float:
            return pred.chi2(E_, y_, sig_, *theta)

        def obj_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
            return pred.chi2_external_grad(E_, y_, sig_, *theta)

        sol = minimize(
            obj_grad,
            x0=theta0,
            jac=True,
            method="L-BFGS-B",
            bounds=_zip_bounds(lo, hi),
            options={"maxiter": self.max_iter},
        )
        if (not sol.success) and warn_on_fail:
            print(f"[ProfiledChi2Fit] solver returned: {sol.message}")

        theta_outer = np.asarray(sol.x, dtype=float)
        chi2_outer = float(sol.fun)

        # Legacy safety net: evaluate ``chi2_prof`` at the classical seed
        # and keep whichever theta yields the smaller profiled chi-squared.
        chosen_source = "profiled_outer"
        chi2_classical: float | None = None
        if theta_classical is not None:
            th_cls = np.asarray(theta_classical, dtype=float)
            th_cls = np.minimum(np.maximum(th_cls, lo + eps), hi - eps)
            chi2_classical = pred.chi2(E_, y_, sig_, *th_cls)
            if chi2_classical < chi2_outer - 1e-12:
                theta_hat = th_cls
                chi2_final = chi2_classical
                chosen_source = "classical_seed"
            else:
                theta_hat = theta_outer
                chi2_final = pred.chi2(E_, y_, sig_, *theta_hat)
        else:
            theta_hat = theta_outer
            chi2_final = pred.chi2(E_, y_, sig_, *theta_hat)

        if legacy_safety_log is not None:
            legacy_safety_log.append(
                {
                    "model": pred.name,
                    "chi2_outer": chi2_outer,
                    "chi2_classical": chi2_classical,
                    "chosen": chosen_source,
                    "theta_outer": np.asarray(theta_outer, float),
                    "theta_classical": (
                        np.asarray(theta_classical, float)
                        if theta_classical is not None
                        else None
                    ),
                }
            )

        if compute_covariance:
            # Legacy-faithful covariance: numerical Hessian of the *scalar*
            # ``chi2_prof(theta)`` at the chosen ``theta_hat`` (each FD
            # evaluation re-runs the inner ``profile_w`` solve), then
            # ``Cov = 2 * inv(H)``.  Matches ``cov_theta_profiled_PL`` in
            # ``legacy/nuisance_fit.py`` and gives tight, legacy-style
            # envelopes -- including when the safety-net falls back to
            # ``theta_classical`` (where a Gauss-Newton ``inv(J^T J)`` would
            # over-estimate the curvature and inflate the band).
            def _chi2_prof(theta: np.ndarray) -> float:
                return float(pred.chi2(E_, y_, sig_, *theta))

            H = _fd_hessian(_chi2_prof, theta_hat, eps=self.hessian_eps)
            cov = _pcov_from_hessian(H, ridge=self.hessian_ridge)

            # FD perturbations leave ``last_w`` at the last perturbed point;
            # re-pin it to ``w_hat(theta_hat)``.
            pred.chi2(E_, y_, sig_, *theta_hat)
        else:
            cov = np.full((n_outer, n_outer), np.nan, dtype=float)

        ndof = int(E_.size - n_outer)

        # Re-bind the warm-started w on the original predictor so later
        # callers (envelopes, MC, plots) can inspect it via ``predictor.last_w``.
        self.predictor.last_w = pred.last_w.copy()
        w_best = pred.last_w.copy()

        return FitResult(
            # Use the masked predictor so its basis rows match the returned E/y/sigma.
            model=pred,
            popt=theta_hat,
            pcov=np.asarray(cov, dtype=float),
            chi2=float(chi2_final),
            ndof=ndof,
            E=E_,
            y=y_,
            sigma=sig_,
            converged=bool(sol.success),
            solver_message=str(sol.message),
            w_best=w_best,
        )
