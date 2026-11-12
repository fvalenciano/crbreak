from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares, minimize

from .adapters import prepare_fit_entry
from .models import SpectralModel
from .nuisance import NuisanceModel
from .nuisance._solver_utils import _fd_hessian, _pcov_from_hessian, _zip_bounds


# ----------------------------------------------------------------------
# Classical joint fit
# ----------------------------------------------------------------------


class GlobalChi2Fit:
    """Joint chi² fit with per-window + shared parameters.

    Parameters
    ----------
    model : :class:`SpectralModel`
        Model fitted to every window.
    shared : sequence of str
        Names of parameters to *share* across all windows (must be a subset
        of ``model.param_names``).  All other parameters become per-window.
    solver
        ``solver`` section of the analysis config (``global_max_nfev`` and
        ``global_x_scale`` of :func:`scipy.optimize.least_squares`).
    """

    def __init__(self, model: SpectralModel, *, shared: Sequence[str], solver):
        unknown = [n for n in shared if n not in model.param_names]
        if unknown:
            raise ValueError(
                f"Shared parameter names not in {model.name}.param_names: {unknown}"
            )
        self.model = model
        self.solver = solver
        self.shared = tuple(shared)
        self.free = tuple(p for p in model.param_names if p not in self.shared)
        self._model_idx = {p: i for i, p in enumerate(model.param_names)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _assemble_window_params(
        self, free_vec: np.ndarray, shared_vec: np.ndarray
    ) -> np.ndarray:
        """Reorder ``[free..., shared...]`` into the model's native order."""
        out = np.empty(self.model.n_params, dtype=float)
        for j, name in enumerate(self.free):
            out[self._model_idx[name]] = free_vec[j]
        for j, name in enumerate(self.shared):
            out[self._model_idx[name]] = shared_vec[j]
        return out

    def _build_bounds(self, n_windows: int) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.model.default_bounds()
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        free_lo = np.array([lo[self._model_idx[n]] for n in self.free])
        free_hi = np.array([hi[self._model_idx[n]] for n in self.free])
        shared_lo = np.array([lo[self._model_idx[n]] for n in self.shared])
        shared_hi = np.array([hi[self._model_idx[n]] for n in self.shared])
        full_lo = np.concatenate([np.tile(free_lo, n_windows), shared_lo])
        full_hi = np.concatenate([np.tile(free_hi, n_windows), shared_hi])
        return full_lo, full_hi

    def _initial_guess(self, fit_datas: list[dict]) -> np.ndarray:
        # Shared block: take from the first window's default_p0 (any window
        # works, the optimiser will refine).
        p0_first = self.model.default_p0(fit_datas[0]["E_GeV"], fit_datas[0]["y"])
        shared_init = np.array([p0_first[self._model_idx[n]] for n in self.shared])
        free_inits = []
        for fd in fit_datas:
            p0 = self.model.default_p0(fd["E_GeV"], fd["y"])
            free_inits.append([p0[self._model_idx[n]] for n in self.free])
        return np.concatenate([np.array(free_inits, float).ravel(), shared_init])

    @staticmethod
    def _coerce_fit_entry(entry) -> dict:
        """Normalize a windows-mapping entry into a simple dict of fit arrays."""
        return prepare_fit_entry(entry)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        windows: Mapping[str, object],
        *,
        verbose: int = 0,
        store_fit_data: bool = False,
        p0_full: np.ndarray | None = None,
        warn_on_fail: bool = True,
    ):
        """Run the joint fit on a ``{key: window-or-extract-dict}`` mapping."""
        from .global_fit import GlobalFitResult

        keys = list(windows.keys())
        raw_data = [self._coerce_fit_entry(windows[k]) for k in keys]

        # Mask each window's data
        clean: list[dict] = []
        for fd in raw_data:
            m = (
                np.isfinite(fd["E_GeV"])
                & np.isfinite(fd["y"])
                & np.isfinite(fd["sigma"])
                & (fd["sigma"] > 0)
            )
            clean.append(
                {
                    "E_GeV": fd["E_GeV"][m],
                    "y": fd["y"][m],
                    "sigma": fd["sigma"][m],
                    "meta": fd.get("meta", {}),
                }
            )

        n_windows = len(keys)
        n_free = len(self.free)
        n_shared = len(self.shared)
        n_points_total = int(sum(c["E_GeV"].size for c in clean))
        n_params_total = n_free * n_windows + n_shared
        if n_points_total <= n_params_total:
            raise RuntimeError(
                f"Global fit underdetermined: {n_points_total} points vs "
                f"{n_params_total} parameters."
            )

        # Residual function for least_squares (already normalised by sigma).
        # Note: ``theta[-0:]`` would return the full array, so we guard it.
        empty_shared = np.empty(0, dtype=float)

        def residuals(theta: np.ndarray) -> np.ndarray:
            shared_vec = theta[-n_shared:] if n_shared > 0 else empty_shared
            chunks = []
            for i, fd in enumerate(clean):
                free_vec = theta[i * n_free : (i + 1) * n_free]
                pars = self._assemble_window_params(free_vec, shared_vec)
                model_y = self.model(fd["E_GeV"], *pars)
                chunks.append((fd["y"] - model_y) / fd["sigma"])
            return np.concatenate(chunks)

        if p0_full is not None:
            theta0 = np.asarray(p0_full, dtype=float).ravel()
            if theta0.size != n_params_total:
                raise ValueError(
                    f"p0_full has size {theta0.size} but expected "
                    f"{n_params_total} for N={n_windows} windows."
                )
        else:
            theta0 = self._initial_guess(clean)
        lo, hi = self._build_bounds(n_windows)
        # Keep theta0 strictly inside bounds (least_squares requirement).
        eps = np.where(np.isfinite(hi), 1e-12 * np.maximum(np.abs(hi), 1.0), 0.0)
        theta0 = np.minimum(np.maximum(theta0, lo + eps), hi - eps)

        sol = least_squares(
            residuals,
            theta0,
            bounds=(lo, hi),
            max_nfev=self.solver.global_max_nfev,
            x_scale=self.solver.global_x_scale,
            verbose=verbose,
        )
        if (not sol.success) and warn_on_fail:
            print(f"[GlobalChi2Fit] solver returned: {sol.message}")

        J = sol.jac
        try:
            cov = np.linalg.inv(J.T @ J)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(J.T @ J)

        chi2 = float(np.sum(sol.fun**2))
        ndof = int(n_points_total - n_params_total)

        return GlobalFitResult(
            model=self.model,
            window_keys=tuple(keys),
            free_param_names=tuple(self.free),
            shared_param_names=tuple(self.shared),
            popt_full=np.asarray(sol.x, dtype=float),
            pcov_full=np.asarray(cov, dtype=float),
            chi2=chi2,
            ndof=ndof,
            converged=bool(sol.success),
            solver_message=str(sol.message),
            fit_data=({k: c for k, c in zip(keys, clean)} if store_fit_data else None),
        )


# ----------------------------------------------------------------------
# Profiled joint fit
# ----------------------------------------------------------------------


class ProfiledGlobalChi2Fit:
    """Joint profiled chi-squared fit across many windows.

    Mirrors the legacy :func:`global_profiled_lrt_pl_vs_bkpl` driver:
    outer minimisation is done with :func:`scipy.optimize.minimize`
    using L-BFGS-B on the *scalar* total profiled chi-squared (sum over
    windows), with each window's inner profile over ``w`` handled by
    its own *bound* :class:`NuisanceModel`.

    Two modes, selected by the bound NuisanceModel(s):

    * **independent** (``predictors[k].share_nuisance == False``): each
      window's ``w_j`` is profiled inside the objective; the free
      vector is just ``[θ_per_window, θ_shared]``.
    * **shared / common-nuisance** (``share_nuisance == True``): a
      single ``w_shared`` is appended to the free vector, the prior
      χ² is evaluated **once** on it, and the data residuals use
      ``B_j @ w_shared`` for every window.  This is the legacy
      ``COMMON_NUISANCE=True`` flavour.  After the fit ``w_shared`` is
      propagated to every bound NM (so any subsequent per-window
      profiled call automatically reuses it as a *frozen* deformation).

    Parameters
    ----------
    model
        Underlying :class:`SpectralModel` (same model fitted to every
        window).
    shared
        Names of parameters shared across all windows (subset of
        ``model.param_names``).  All other parameters become per-window.
    solver
        ``solver`` section of the analysis config: ``profiled_max_iter``
        (L-BFGS-B ``maxiter``) and ``hessian_eps`` / ``hessian_ridge``.
    """

    def __init__(self, model: SpectralModel, *, shared: Sequence[str], solver):
        unknown = [n for n in shared if n not in model.param_names]
        if unknown:
            raise ValueError(
                f"Shared parameter names not in {model.name}.param_names: {unknown}"
            )
        self.model = model
        self.shared = tuple(shared)
        self.free = tuple(p for p in model.param_names if p not in self.shared)
        self._model_idx = {p: i for i, p in enumerate(model.param_names)}
        self.max_iter = int(solver.profiled_max_iter)
        self.hessian_eps = float(solver.hessian_eps)
        self.hessian_ridge = float(solver.hessian_ridge)

    # ------------------------------------------------------------------
    # Internals (mirror GlobalChi2Fit's helpers)
    # ------------------------------------------------------------------
    def _assemble_window_params(
        self, free_vec: np.ndarray, shared_vec: np.ndarray
    ) -> np.ndarray:
        out = np.empty(self.model.n_params, dtype=float)
        for j, name in enumerate(self.free):
            out[self._model_idx[name]] = free_vec[j]
        for j, name in enumerate(self.shared):
            out[self._model_idx[name]] = shared_vec[j]
        return out

    def _build_bounds(self, n_windows: int) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.model.default_bounds()
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        free_lo = np.array([lo[self._model_idx[n]] for n in self.free])
        free_hi = np.array([hi[self._model_idx[n]] for n in self.free])
        shared_lo = np.array([lo[self._model_idx[n]] for n in self.shared])
        shared_hi = np.array([hi[self._model_idx[n]] for n in self.shared])
        full_lo = np.concatenate([np.tile(free_lo, n_windows), shared_lo])
        full_hi = np.concatenate([np.tile(free_hi, n_windows), shared_hi])
        return full_lo, full_hi

    def _initial_guess(
        self, predictors: Sequence[NuisanceModel], data: Sequence[dict]
    ) -> np.ndarray:
        p0_first = predictors[0].default_p0(data[0]["E"], data[0]["y"])
        shared_init = np.array([p0_first[self._model_idx[n]] for n in self.shared])
        free_inits = []
        for pred, d in zip(predictors, data):
            p0 = pred.default_p0(d["E"], d["y"])
            free_inits.append([p0[self._model_idx[n]] for n in self.free])
        return np.concatenate([np.array(free_inits, dtype=float).ravel(), shared_init])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        predictors: Mapping[str, NuisanceModel],
        data: Mapping[str, dict],
        *,
        p0_full: np.ndarray | None = None,
        theta_classical_full: np.ndarray | None = None,
        legacy_safety_log: list | None = None,
        w_shared0: np.ndarray | None = None,
        warn_on_fail: bool = True,
        compute_covariance: bool = True,
    ):
        """Run the joint profiled fit.

        Parameters
        ----------
        predictors
            ``{window_key: bound NuisanceModel}`` (output of
            :meth:`NuisanceModel.for_window`).  Each must be built with
            the basis/``Ssys_frac`` of *its* window.
        data
            ``{window_key: {"E": ..., "y": ..., "sigma": ...}}`` arrays
            already corresponding to the predictor's basis rows (no
            additional masking is applied here -- the caller should pass
            clean fit-ready arrays, e.g. ``WindowFitResult.E_GeV/y/sigma``).
        p0_full
            Optional warm-start for the full ``[free..., shared]`` vector.
        theta_classical_full
            Optional classical (non-profiled) global best-fit (same
            layout as ``p0_full``).  Triggers the legacy safety net:
            after the L-BFGS-B outer converges, the total profiled
            chi-squared is also evaluated at this vector and the
            smaller of the two is kept.
        legacy_safety_log
            Optional list to which a diagnostic dict is appended
            (``"profiled_outer"`` vs ``"classical_seed"``).
        w_shared0
            *Common-nuisance only.*  Initial guess for the shared
            nuisance vector ``w_shared``.  When ``None`` (default) the
            block is initialised at zero, which sits in the basin of
            the prior and frequently gets stuck there for non-trivial
            deformations.  Pass e.g. the per-window mean of the
            independent ``w_j`` (run a non-shared profiled fit first)
            to escape that basin.  Ignored when ``p0_full`` is provided
            (``p0_full`` already specifies the full vector) or when the
            predictors are not in shared mode.
        """
        from .global_fit import GlobalFitResult

        keys = list(predictors.keys())
        if list(data.keys()) != keys:
            raise ValueError("predictors and data must share the same key order.")
        preds = [predictors[k] for k in keys]
        clean = []
        for k, p in zip(keys, preds):
            d = data[k]
            E = np.asarray(d["E"], dtype=float)
            y = np.asarray(d["y"], dtype=float)
            sigma = np.asarray(d["sigma"], dtype=float)
            if p.basis.B.shape[0] != E.size:
                raise ValueError(
                    f"Window {k!r}: predictor basis rows "
                    f"({p.basis.B.shape[0]}) do not match E size ({E.size})."
                )
            clean.append({"E": E, "y": y, "sigma": sigma})

        n_windows = len(keys)
        n_free = len(self.free)
        n_shared = len(self.shared)
        # Common-nuisance: detected from the bound NMs.  When True we
        # also need the shared nuisance prior (a single one across
        # windows; the prior is *the same instance* on every bound NM
        # by construction of ``NuisanceModel.for_global``).
        share = bool(preds[0].share_nuisance)
        if share and not all(p.share_nuisance for p in preds):
            raise ValueError(
                "ProfiledGlobalChi2Fit.fit: mixed share_nuisance flags "
                "across predictors; build them all with for_global() or "
                "all with for_window()."
            )
        n_w = preds[0].n_nuis if share else 0
        n_points_total = int(sum(c["E"].size for c in clean))
        n_spec = n_free * n_windows + n_shared
        n_params_total = n_spec + n_w
        if n_points_total <= n_params_total:
            raise RuntimeError(
                f"Global profiled fit underdetermined: {n_points_total} "
                f"points vs {n_params_total} outer parameters."
            )

        empty_shared = np.empty(0, dtype=float)

        # Reset per-window warm-start caches (legacy w_cache lives
        # inside the optimizer call, not across calls).  In shared mode
        # also drop any stale ``w_shared`` so the inner short-circuit
        # in ``profile_w`` does not fire during the joint optimisation.
        for p in preds:
            p.last_w = np.zeros(p.n_nuis, dtype=float)
            if share:
                p.w_shared = None

        def _per_window_pars(theta: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
            # In shared mode the last n_w entries hold w_shared, just
            # before that come n_shared spectral-shared params.
            tail_shared = theta[-(n_shared + n_w) : theta.size - n_w] \
                if n_shared > 0 else empty_shared
            shared_vec = tail_shared
            pars_list = []
            for i in range(n_windows):
                free_vec = theta[i * n_free : (i + 1) * n_free]
                pars_list.append(self._assemble_window_params(free_vec, shared_vec))
            return pars_list, shared_vec

        if share:
            def obj(theta: np.ndarray) -> float:
                pars_list, _ = _per_window_pars(theta)
                w_vec = theta[-n_w:]
                total = 0.0
                for pred, d, pars in zip(preds, clean, pars_list):
                    mu = np.asarray(
                        pred.spectral_model(d["E"], *pars), dtype=float
                    )
                    m = (
                        np.isfinite(mu)
                        & np.isfinite(d["y"])
                        & np.isfinite(d["sigma"])
                        & (d["sigma"] > 0)
                    )
                    if not np.any(m):
                        return float("inf")
                    yhat = (
                        1.0
                        + (pred.basis.B[m] @ w_vec) * pred.Ssys_frac[m]
                    ) * mu[m]
                    r = (d["y"][m] - yhat) / d["sigma"][m]
                    total += float(np.sum(r * r))
                # Prior evaluated **once** on the shared vector.
                prior_r = preds[0].prior.augmented_residuals(w_vec)
                if prior_r.size:
                    total += float(np.sum(prior_r * prior_r))
                return float(total)
        else:
            def obj(theta: np.ndarray) -> float:
                pars_list, _ = _per_window_pars(theta)
                total = 0.0
                for pred, d, pars in zip(preds, clean, pars_list):
                    total += pred.chi2(d["E"], d["y"], d["sigma"], *pars)
                return float(total)

            def obj_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
                pars_list, _ = _per_window_pars(theta)
                total = 0.0
                grad = np.zeros_like(theta, dtype=float)
                shared0 = n_free * n_windows
                for i, (pred, d, pars) in enumerate(zip(preds, clean, pars_list)):
                    chi2_i, grad_i = pred.chi2_external_grad(
                        d["E"], d["y"], d["sigma"], *pars
                    )
                    total += chi2_i
                    for j, name in enumerate(self.free):
                        grad[i * n_free + j] = grad_i[self._model_idx[name]]
                    for j, name in enumerate(self.shared):
                        grad[shared0 + j] += grad_i[self._model_idx[name]]
                return float(total), grad

        if p0_full is not None:
            theta0 = np.asarray(p0_full, dtype=float).ravel()
            if theta0.size != n_params_total:
                raise ValueError(
                    f"p0_full has size {theta0.size} but expected "
                    f"{n_params_total} for N={n_windows} windows "
                    f"(n_w={n_w})."
                )
        else:
            theta0 = self._initial_guess(preds, clean)
            if share:
                if w_shared0 is None:
                    w0_block = np.zeros(n_w, dtype=float)
                else:
                    w0_block = np.asarray(w_shared0, dtype=float).ravel()
                    if w0_block.size != n_w:
                        raise ValueError(
                            f"w_shared0 has size {w0_block.size} but expected "
                            f"{n_w} (n_nuis of the predictors)."
                        )
                theta0 = np.concatenate([theta0, w0_block])
        lo, hi = self._build_bounds(n_windows)
        if share:
            # Bounds for ``w_shared``: use the prior's own bounds when
            # available (e.g. BoxGaussPrior caps |w|), otherwise leave
            # them open.
            prior = preds[0].prior
            wb = prior.w_bounds()
            if wb is None:
                w_lo = np.full(n_w, -np.inf, dtype=float)
                w_hi = np.full(n_w, +np.inf, dtype=float)
            else:
                w_lo = np.asarray([b[0] for b in wb], dtype=float)
                w_hi = np.asarray([b[1] for b in wb], dtype=float)
            lo = np.concatenate([lo, w_lo])
            hi = np.concatenate([hi, w_hi])
        eps = np.where(np.isfinite(hi), 1e-12 * np.maximum(np.abs(hi), 1.0), 0.0)
        theta0 = np.minimum(np.maximum(theta0, lo + eps), hi - eps)

        sol = minimize(
            obj if share else obj_grad,
            x0=theta0,
            jac=None if share else True,
            method="L-BFGS-B",
            bounds=_zip_bounds(lo, hi),
            options={"maxiter": self.max_iter},
        )
        if (not sol.success) and warn_on_fail:
            print(f"[ProfiledGlobalChi2Fit] solver returned: {sol.message}")

        theta_outer = np.asarray(sol.x, dtype=float)
        chi2_outer = float(sol.fun)

        # Legacy safety net at the global level.  Also run a second
        # profiled minimisation from the classical global solution; if it
        # finds a lower chi2, that solution becomes the one used everywhere
        # downstream (Wilks, MC, scans, plots).
        chosen_source = "profiled_outer"
        chi2_classical: float | None = None
        chi2_classical_min: float | None = None
        theta_hat = theta_outer
        chi2_final = obj(theta_hat)
        if theta_classical_full is not None:
            th_cls = np.asarray(theta_classical_full, dtype=float).ravel()
            if share and th_cls.size == n_spec:
                w0_cls = (
                    np.zeros(n_w, dtype=float)
                    if w_shared0 is None
                    else np.asarray(w_shared0, dtype=float).ravel()
                )
                th_cls = np.concatenate([th_cls, w0_cls])
            if th_cls.size != n_params_total:
                raise ValueError(
                    f"theta_classical_full has size {th_cls.size} but "
                    f"expected {n_params_total} for N={n_windows} windows "
                    f"(n_w={n_w})."
                )
            th_cls = np.minimum(np.maximum(th_cls, lo + eps), hi - eps)
            chi2_classical = obj(th_cls)
            sol_cls = minimize(
                obj if share else obj_grad,
                x0=th_cls,
                jac=None if share else True,
                method="L-BFGS-B",
                bounds=_zip_bounds(lo, hi),
                options={"maxiter": self.max_iter},
            )
            if (not sol_cls.success) and warn_on_fail:
                print(
                    "[ProfiledGlobalChi2Fit safety-net] classical-start "
                    f"solver returned: {sol_cls.message}"
                )
            theta_classical_min = np.asarray(sol_cls.x, dtype=float)
            chi2_classical_min = float(sol_cls.fun)
            if chi2_classical_min < chi2_final - 1e-12:
                theta_hat = theta_classical_min
                chi2_final = chi2_classical_min
                chosen_source = "classical_start_min"
            if chi2_classical < chi2_final - 1e-12:
                theta_hat = th_cls
                chi2_final = chi2_classical
                chosen_source = "classical_seed"

        if legacy_safety_log is not None:
            legacy_safety_log.append(
                {
                    "model": self.model.name,
                    "chi2_outer": chi2_outer,
                    "chi2_classical": chi2_classical,
                    "chi2_classical_min": chi2_classical_min,
                    "chosen": chosen_source,
                }
            )

        if compute_covariance:
            # Legacy-faithful covariance: numerical Hessian of the *scalar*
            # global ``chi2_prof(theta_full)`` at ``theta_hat``, then
            # ``Cov = 2 * inv(H)``.  In shared mode we slice the spectral
            # block (top-left, n_spec x n_spec) for ``pcov_full`` and the
            # nuisance block (bottom-right, n_w x n_w) for ``cov_w_shared``.
            H = _fd_hessian(obj, theta_hat, eps=self.hessian_eps)
            cov = _pcov_from_hessian(H, ridge=self.hessian_ridge)
        else:
            cov = np.full((n_params_total, n_params_total), np.nan, dtype=float)
        if share:
            popt_spec = theta_hat[:n_spec].copy()
            cov_spec = cov[:n_spec, :n_spec].copy()
            w_hat = theta_hat[-n_w:].copy()
            cov_w_shared = cov[-n_w:, -n_w:].copy()
            # Propagate ``w_shared`` to every bound NM so any subsequent
            # per-window profiled call (Wilks, MC, scan) automatically
            # uses the *frozen* common deformation.
            for p in preds:
                p.w_shared = w_hat.copy()
                p.last_w = w_hat.copy()
        else:
            popt_spec = theta_hat
            cov_spec = cov
            w_hat = None
            cov_w_shared = None
            # Re-pin per-window ``last_w`` to ``w_hat(theta_hat)``.
            obj(theta_hat)
        ndof = int(n_points_total - n_params_total)

        # Pick a representative predictor for the result container.  All
        # per-window predictors share the same param_names / __call__
        # delegation to the underlying SpectralModel, so any of them is OK
        # for plotting/envelopes; MC/scan just need ``.model.name`` and
        # ``.model(E, *theta)``, which are identical across them.
        return GlobalFitResult(
            model=preds[0],
            window_keys=tuple(keys),
            free_param_names=tuple(self.free),
            shared_param_names=tuple(self.shared),
            popt_full=np.asarray(popt_spec, dtype=float),
            pcov_full=np.asarray(cov_spec, dtype=float),
            chi2=float(chi2_final),
            ndof=ndof,
            converged=bool(sol.success),
            solver_message=str(sol.message),
            fit_data={
                k: {
                    "E_GeV": c["E"],
                    "y": c["y"],
                    "sigma": c["sigma"],
                    "meta": {},
                }
                for k, c in zip(keys, clean)
            },
            w_shared=w_hat,
            cov_w_shared=cov_w_shared,
        )
