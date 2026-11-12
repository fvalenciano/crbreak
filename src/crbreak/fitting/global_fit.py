"""Joint chi-squared fit result container across many sky windows.

This module hosts only the **result + dispatcher**: :class:`GlobalFitResult`.
The underlying solvers live in :mod:`crbreak.fitting.global_chi2`
(:class:`~crbreak.fitting.global_chi2.GlobalChi2Fit` for classical and
:class:`~crbreak.fitting.global_chi2.ProfiledGlobalChi2Fit` for profiled).

Some model parameters are *shared* across all windows (typically the
shape parameters, e.g. ``Ebreak`` and ``dgamma`` of an SBPL), while the
remaining parameters are fitted *independently* per window.  The full
parameter vector packed for the optimisers is ::

    theta = [ K_1, gamma1_1,  K_2, gamma1_2,  ...,  K_N, gamma1_N,
              shared_1, shared_2, ... ]

Public entry point
------------------
:meth:`GlobalFitResult.fit` is the **constructor classmethod** that runs
the joint fit and returns the populated container.  It dispatches by the
type of ``model``:

* :class:`~crbreak.fitting.models.SpectralModel` → classical chi² via
  :class:`~crbreak.fitting.global_chi2.GlobalChi2Fit`.
* :class:`~crbreak.fitting.nuisance.NuisanceModel` → profiled chi² via
  :class:`~crbreak.fitting.global_chi2.ProfiledGlobalChi2Fit` (with the
  legacy safety net wired through ``classical_seed``).

The orchestrator :meth:`crbreak.fitting.main.Fitting.fit_global` then
becomes a thin wrapper that supplies the windows and runs cross-fit
analysis (Wilks, joint Monte-Carlo).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .envelope import envelope_from_cov
from .models import SpectralModel
from .stats import wilks_pvalue


# ----------------------------------------------------------------------
# Result container
# ---------------------------------- ------------------------------------


@dataclass
class GlobalFitResult:
    """Outcome of a joint multi-window fit."""

    model: SpectralModel
    window_keys: tuple[str, ...]
    free_param_names: tuple[str, ...]  # per-window parameter names
    shared_param_names: tuple[str, ...]  # globally shared parameters
    popt_full: np.ndarray
    pcov_full: np.ndarray
    chi2: float
    ndof: int
    converged: bool = True
    solver_message: str = ""
    # Optional: per-window fit arrays (unscaled) used by the solver.
    # When present, each entry is a dict with keys: E_GeV, y, sigma, meta.
    fit_data: dict[str, dict] | None = None
    # Post-fit analysis caches.  Both keyed by ``name_simpler`` (the
    # null hypothesis); the alternative is always *self*.
    # ``wilks_results`` stores the dict produced by :func:`wilks_pvalue`;
    # ``mc_results`` stores :class:`MCPValueResult` instances.
    wilks_results: dict[str, dict] = field(default_factory=dict)
    mc_results: dict[str, object] = field(default_factory=dict)
    # Profiled-only: legacy safety-net diagnostic emitted by
    # :class:`~crbreak.fitting.global_chi2.ProfiledGlobalChi2Fit` (None
    # for classical fits or when the safety net was disabled / had no
    # classical seed).
    safety_log: dict | None = None
    # Common-nuisance only: filled by
    # :class:`~crbreak.fitting.global_chi2.ProfiledGlobalChi2Fit` when
    # the input :class:`NuisanceModel` has ``share_nuisance=True``.
    # ``w_shared`` is the joint best-fit nuisance vector and
    # ``cov_w_shared`` its covariance (sub-block of the full Hessian).
    w_shared: np.ndarray | None = None
    cov_w_shared: np.ndarray | None = None
    # Asymmetric ±1σ profile-likelihood errors on the shared parameters.
    # ``{param_name: (best, lo, hi)}`` with absolute values; ``lo``/``hi``
    # is ``nan`` when the scan grid does not span the ``Δχ² = 1``
    # crossing.  Currently populated only for SBPL profiled fits via
    # :meth:`compute_profile_errors`; classical fits leave it ``None``.
    profile_errors: dict[str, tuple[float, float, float]] | None = None
    # Optional debug cache for the 1-D global profile-likelihood curves.
    # ``{param_name: (grid, chi2_row, chi2_anchor)}``, populated together
    # with :attr:`profile_errors`.
    profile_curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------
    # Constructor entry point
    # ------------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        model,
        windows: Mapping[str, object],
        *,
        shared: Sequence[str],
        solver,
        classical_seed: "GlobalFitResult | None" = None,
        p0_full: np.ndarray | None = None,
        w_shared0: np.ndarray | None = None,
    ) -> "GlobalFitResult":
        """Run the joint fit and return the populated :class:`GlobalFitResult`.

        Dispatch is by the type of ``model``:

        * :class:`SpectralModel`  → classical chi² via
          :class:`~crbreak.fitting.global_chi2.GlobalChi2Fit`.
        * :class:`NuisanceModel`  → profiled chi² via
          :class:`~crbreak.fitting.global_chi2.ProfiledGlobalChi2Fit`.
          ``classical_seed`` (a previous :class:`GlobalFitResult` from a
          classical fit) activates the legacy safety net at the global
          level: the smaller of ``chi2_prof(theta_outer)`` and
          ``chi2_prof(theta_classical)`` is kept; the diagnostic dict is
          stashed on ``g.safety_log``.

        Parameters
        ----------
        model
            :class:`SpectralModel` or
            :class:`~crbreak.fitting.nuisance.NuisanceModel`.
        windows
            Either ``Mapping[str, WindowFitResult]`` (typical: pass
            ``Fitting.window_fits`` after ``fit_per_window``) or
            ``Mapping[str, Window]`` (raw extraction; classical only).
            Profiled fits *require* :class:`WindowFitResult` instances
            because :meth:`NuisanceModel.for_window` reads the
            window's ``meta["Ssys_frac"]``.
        shared
            Names of parameters shared across windows.
        solver
            ``solver`` section of the analysis config (optimiser budgets
            and Hessian settings of the underlying engine).
        classical_seed
            *Profiled only.*  Previous classical :class:`GlobalFitResult`
            whose ``popt_full`` seeds the safety-net comparison.

        Raises
        ------
        ValueError
            If ``classical_seed`` is given for a classical fit (the seed
            only makes sense for the profiled flavour).
        """
        # Lazy imports to break the global_fit ↔ {nuisance, global_chi2} cycles.
        from .global_chi2 import GlobalChi2Fit
        from .nuisance import NuisanceModel

        if isinstance(model, NuisanceModel):
            return cls._fit_profiled(
                model, windows,
                shared=tuple(shared),
                solver=solver,
                classical_seed=classical_seed,
                p0_full=p0_full,
                w_shared0=w_shared0,
            )
        if w_shared0 is not None:
            raise ValueError(
                "w_shared0 is only meaningful for a profiled "
                "(NuisanceModel) global fit."
            )
        if classical_seed is not None:
            raise ValueError(
                "classical_seed is only meaningful for a profiled "
                "(NuisanceModel) global fit."
            )
        gfit = GlobalChi2Fit(model, shared=tuple(shared), solver=solver)
        return gfit.fit(windows, p0_full=p0_full)

    @classmethod
    def _fit_profiled(
        cls,
        nm,
        windows: Mapping[str, object],
        *,
        shared: tuple[str, ...],
        solver,
        classical_seed: "GlobalFitResult | None" = None,
        p0_full: np.ndarray | None = None,
        w_shared0: np.ndarray | None = None,
    ) -> "GlobalFitResult":
        """Profiled joint fit (helper for :meth:`fit`).

        Builds bound :class:`NuisanceModel` instances for every window:

        * if ``nm.share_nuisance`` is ``False`` (default), each window
          gets its own independent basis via :meth:`NuisanceModel.for_window`;
        * if ``nm.share_nuisance`` is ``True`` (legacy ``COMMON_NUISANCE``),
          all windows share the same ``x_nuis`` nodes via
          :meth:`NuisanceModel.for_global` so the joint fit can solve for
          a single ``w_shared``.
        """
        from .global_chi2 import ProfiledGlobalChi2Fit
        from .nuisance import NuisanceModel as _NuisanceModel

        keys = list(windows.keys())
        predictors: dict[str, _NuisanceModel] = {}
        data: dict[str, dict[str, np.ndarray]] = {}

        if nm.share_nuisance:
            wfs = [windows[k] for k in keys]
            bounds, arrays = nm.for_global(wfs)
            for key, pred, arr in zip(keys, bounds, arrays):
                predictors[key] = pred
                data[key] = arr
        else:
            for key in keys:
                pred, arr = nm.for_window(windows[key])
                predictors[key] = pred
                data[key] = arr

        gfit = ProfiledGlobalChi2Fit(nm.spectral_model, shared=shared, solver=solver)

        theta_classical_full = None
        if nm.legacy_safety_net and classical_seed is not None:
            theta_classical_full = np.asarray(
                classical_seed.popt_full, dtype=float
            )

        safety_log: list = []
        result = gfit.fit(
            predictors, data,
            p0_full=p0_full,
            theta_classical_full=theta_classical_full,
            legacy_safety_log=safety_log,
            w_shared0=w_shared0,
        )
        if safety_log:
            result.safety_log = safety_log[0]
        # Common-nuisance: propagate ``w_shared`` to the *unbound* NM
        # the user passed in.  Any subsequent :meth:`for_window` will
        # therefore inherit it (via :meth:`_with_bound_state`) so
        # downstream per-window profiled fits, scans and MC use the
        # legacy "frozen common deformation" semantics automatically.
        if nm.share_nuisance and result.w_shared is not None:
            nm.w_shared = np.asarray(result.w_shared, dtype=float).copy()
        return result

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    @property
    def n_windows(self) -> int:
        return len(self.window_keys)

    @property
    def n_free(self) -> int:
        return len(self.free_param_names)

    @property
    def n_shared(self) -> int:
        return len(self.shared_param_names)

    def _full_param_indices_for_window(self, key: str) -> np.ndarray:
        i = self.window_keys.index(key)
        free_idx = np.arange(i * self.n_free, (i + 1) * self.n_free)
        shared_offset = self.n_free * self.n_windows
        shared_idx = np.arange(shared_offset, shared_offset + self.n_shared)
        return np.concatenate([free_idx, shared_idx])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def perr_full(self) -> np.ndarray:
        return np.sqrt(np.diag(self.pcov_full))

    @property
    def chi2_red(self) -> float:
        return float(self.chi2 / self.ndof) if self.ndof > 0 else float("nan")

    @property
    def shared_params(self) -> dict[str, tuple[float, float]]:
        """``{name: (value, 1-sigma error)}`` for the shared parameters."""
        offset = self.n_free * self.n_windows
        vals = self.popt_full[offset:]
        errs = self.perr_full[offset:]
        return {
            n: (float(v), float(e))
            for n, v, e in zip(self.shared_param_names, vals, errs)
        }

    def window_params(self, key: str) -> dict[str, tuple[float, float]]:
        """All parameters (per-window + shared) reorganised in model order."""
        # Use the same packing as :meth:`_assemble_window_params` would.
        free_part = self._free_part(key)
        shared_part = list(self.shared_params.items())
        free_part_named = [
            (n, (float(v), float(e)))
            for n, v, e in zip(self.free_param_names, free_part[0], free_part[1])
        ]
        return dict(free_part_named + shared_part)

    def _free_part(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        i = self.window_keys.index(key)
        sl = slice(i * self.n_free, (i + 1) * self.n_free)
        return self.popt_full[sl], self.perr_full[sl]

    def per_window(self) -> dict[str, dict[str, tuple[float, float]]]:
        return {k: self.window_params(k) for k in self.window_keys}

    @property
    def n_params_total(self) -> int:
        """Total number of free parameters in the joint fit."""
        return self.n_free * self.n_windows + self.n_shared

    # ------------------------------------------------------------------
    # Profile-likelihood errors on the shared parameters
    # ------------------------------------------------------------------
    def compute_profile_errors(
        self,
        fitting,
        *,
        params: tuple[str, ...],
        nuisance_config,
        grids: dict,
        delta: float,
        n_workers: int | None = None,
    ) -> dict[str, tuple[float, float, float]]:
        """Asymmetric ±1σ profile-likelihood errors on the joint shared
        parameters (``Ebreak``, ``dgamma`` for SBPL).

        Reuses :func:`crbreak.fitting.scan_params.profile_errors_1d_sbpl`,
        which itself reuses :func:`scan_chi2_sbpl` along each axis with
        the **other** one frozen at its joint best-fit value.  Every
        per-window ``[K, gamma1]`` is reprofiled at every grid point and,
        in shared-nuisance mode, ``w_shared`` is held at its joint
        best-fit -- exactly as in the 2-D contour scan -- so the
        resulting ``Δχ²`` profile is consistent with ``self.chi2``.

        Parameters
        ----------
        nuisance_config
            The :class:`NuisanceModel` driving the scan.
        grids
            ``{"Ebreak": array, "dgamma": array}`` of absolute parameter
            values (other keys are ignored: only the shared break
            parameters are profiled at the global level).

        Result is also cached in :attr:`profile_errors` and returned.
        """
        from .scan_params import profile_errors_1d_sbpl

        if "Ebreak" not in self.shared_param_names or "dgamma" not in self.shared_param_names:
            raise ValueError(
                "GlobalFitResult.compute_profile_errors: only SBPL "
                "(Ebreak, dgamma) profiled fits are supported."
            )

        # Best-fit shared values.
        sp = self.shared_params  # {"Ebreak": (val, hess_err), "dgamma": (val, hess_err)}
        Eb_best = float(sp["Ebreak"][0])
        dg_best = float(sp["dgamma"][0])

        init_pw = (
            self.popt_full[: self.n_free * self.n_windows]
            .reshape(self.n_windows, self.n_free)
            .copy()
        )
        out = profile_errors_1d_sbpl(
            fitting, nuisance_config.spectral_model,
            Eb_best=Eb_best, dg_best=dg_best,
            Eb_grid=grids["Ebreak"], dg_grid=grids["dgamma"],
            chi2_anchor=float(self.chi2),
            nuisance_config=nuisance_config,
            init_per_window=init_pw,
            n_workers=n_workers,
            delta=delta,
        )
        # Strip the ChiSquareScan diagnostic from the cached entry; keep
        # the (best, lo, hi) tuples that are the actual public result.
        clean = {k: (v[0], v[1], v[2]) for k, v in out.items() if k in params}
        curves = {}
        if "Ebreak" in out:
            scan = out["Ebreak"][3]
            curves["Ebreak"] = (
                scan.Ebreak_grid.copy(),
                scan.chi2_grid[:, 0].copy(),
                float(self.chi2),
            )
        if "dgamma" in out:
            scan = out["dgamma"][3]
            curves["dgamma"] = (
                scan.dgamma_grid.copy(),
                scan.chi2_grid[0, :].copy(),
                float(self.chi2),
            )
        self.profile_errors = clean
        self.profile_curves = curves
        return clean

    # ------------------------------------------------------------------
    # Wilks' theorem
    # ------------------------------------------------------------------
    def wilks_vs(
        self,
        chi2_simpler: float,
        n_params_simpler: int,
        *,
        dof: int | None = None,
        name_simpler: str | None = None,
    ) -> dict:
        """Wilks LRT against a less-complex (nested) reference fit.

        For the typical PL-vs-SBPL global comparison, ``chi2_simpler`` is the
        sum of per-window PL chi² and ``n_params_simpler`` the sum of per-
        window PL free-parameter counts; ``dof`` then reduces to ``n_shared``
        (=2 for SBPL: ``Ebreak`` and ``dgamma``).

        If ``name_simpler`` is given the result is cached in
        :attr:`wilks_results` keyed by that name.

        Returns the dict produced by :func:`crbreak.fitting.stats.wilks_pvalue`.
        """
        if dof is None:
            dof = int(self.n_params_total - int(n_params_simpler))
        out = wilks_pvalue(float(chi2_simpler) - float(self.chi2), dof)
        if name_simpler is not None:
            self.wilks_results[str(name_simpler)] = out
        return out

    # ------------------------------------------------------------------
    # Monte-Carlo p-value (joint, this fit as the alternative)
    # ------------------------------------------------------------------
    def mc_pvalue(
        self,
        fitting,
        *,
        name_pl: str,
        N: int,
        seed: int,
        sample_theta: bool,
        warm_start: bool,
        store_deltas: bool,
        n_workers: int | None = None,
        chunksize: int = 1,
        batch_size: int | None = None,
        print_every: int = 0,
        null_global: "GlobalFitResult | None" = None,
        control_global: "GlobalFitResult | None" = None,
    ):
        """Joint parametric-bootstrap MC p-value for ``H0=name_pl`` vs *self*.

        Runs ``N`` toys under the null and refits the null plus the joint
        alternative (this :class:`GlobalFitResult`) on each toy.  In the
        default path the observed test statistic is
        ``Δχ²_obs = Σ_i χ²(PL_i) - χ²(self)``.  When ``null_global`` is
        supplied (common-nuisance global MC), the threshold is instead
        ``Δχ²_obs = χ²(null_global) - χ²(self)``.  ``control_global``
        supplies an externally calibrated shared nuisance vector/covariance
        for control-common toys while keeping the default threshold.  The
        exact threshold used is always stored on the returned
        ``MCPValueResult.delta_obs``.

        Parameters
        ----------
        fitting
            The :class:`crbreak.fitting.main.Fitting` whose ``window_fits``
            already contain the ``name_pl`` per-window fits referenced by
            ``self.window_keys``.

        Returns the :class:`crbreak.fitting.montecarlo.MCPValueResult`.
        """
        from . import montecarlo as _mc

        out = _mc.mc_pvalue_global(
            fitting,
            self,
            name_pl=name_pl,
            N=N,
            seed=seed,
            sample_theta=sample_theta,
            warm_start=warm_start,
            n_workers=n_workers,
            chunksize=chunksize,
            batch_size=batch_size,
            print_every=print_every,
            store_deltas=store_deltas,
            null_global=null_global,
            control_global=control_global,
        )
        self.mc_results[str(name_pl)] = out
        return out

    # ------------------------------------------------------------------
    # Predictions and envelopes
    # ------------------------------------------------------------------
    def _model_args_for_window(self, key: str) -> np.ndarray:
        """Pack ``[free_i, shared]`` into the model's native ``param_names`` order."""
        free_vec, _ = self._free_part(key)
        shared_vec = self.popt_full[self.n_free * self.n_windows :]
        out = np.empty(self.model.n_params, dtype=float)
        idx = {p: i for i, p in enumerate(self.model.param_names)}
        for j, name in enumerate(self.free_param_names):
            out[idx[name]] = free_vec[j]
        for j, name in enumerate(self.shared_param_names):
            out[idx[name]] = shared_vec[j]
        return out

    def _model_pcov_for_window(self, key: str) -> np.ndarray:
        """Sub-covariance for the parameters that drive this window's curve.

        Returned in the same order as ``_model_args_for_window`` -- i.e. the
        model's native ``param_names`` order -- so it can be fed straight to
        :func:`envelope_from_cov`.
        """
        idx_full = self._full_param_indices_for_window(key)
        # Reorder to model.param_names order
        names_concat = list(self.free_param_names) + list(self.shared_param_names)
        order = [names_concat.index(p) for p in self.model.param_names]
        idx_native = idx_full[order]
        return self.pcov_full[np.ix_(idx_native, idx_native)]

    def predict_window(self, key: str, E: np.ndarray) -> np.ndarray:
        params = self._model_args_for_window(key)
        return self.model(np.asarray(E, dtype=float), *params)

    def envelope_window(
        self,
        key: str,
        E_dense: np.ndarray,
        *,
        n: int,
        q: tuple[float, float],
        seed: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        params = self._model_args_for_window(key)
        cov = self._model_pcov_for_window(key)
        return envelope_from_cov(self.model, params, cov, E_dense, n=n, q=q, seed=seed)

    def __repr__(self) -> str:
        bits = ", ".join(
            f"{n}={v:.3g}±{e:.2g}" for n, (v, e) in self.shared_params.items()
        )
        return (
            f"GlobalFitResult({self.model.name}, N={self.n_windows}, "
            f"chi2={self.chi2:.1f}/{self.ndof}, shared: {bits})"
        )
