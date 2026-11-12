"""Per-window fit result container.

:class:`WindowFitResult` mirrors :class:`crbreak.data_extract.main.Window`:

* ``Window``        holds the ROI definition + extracted data.
* ``WindowFitResult`` holds the ROI definition + the data fitted + the
  fits performed on it.

A window knows **how to fit itself** to a model.  The single public entry
point :meth:`WindowFitResult.fit` accepts either a
:class:`~crbreak.fitting.models.SpectralModel` (classical chi²) or a
:class:`~crbreak.fitting.nuisance.NuisanceModel` (profiled chi² with
systematic deformations) and dispatches internally.  The fit is stored in
:attr:`fits` keyed by the model's name; per-window post-fit analysis
(Wilks, Monte-Carlo) lives in :meth:`wilks` and :meth:`mc_pvalue`, with
their outputs cached in :attr:`wilks_results` / :attr:`mc_results`.

The orchestrator :meth:`crbreak.fitting.main.Fitting.fit_per_window` then
becomes a thin loop over windows × models that delegates to
:meth:`fit`; cross-window analysis (joint fits, Wilks vs sum-of-PL,
joint MC) lives at the :class:`Fitting` level since it inherently
crosses windows.

Iterating over :class:`crbreak.fitting.main.Fitting` yields one
``WindowFitResult`` per sky region, the same way iterating over ``Data``
yields one ``Window`` per region.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from .adapters import prepare_fit_data_from_window, fit_meta_from_window
from .chi2 import Chi2Fit, FitResult, ProfiledChi2Fit
from .models import SpectralModel
from .nuisance import NuisanceModel
from .stats import wilks_pvalue


# Public alias mirroring the one in :mod:`crbreak.fitting.main`.
ModelLike = Union[SpectralModel, NuisanceModel]


class WindowFitResult:
    """All fits performed on a single sky window."""

    def __init__(
        self,
        glon_bin: Tuple[float, float],
        glat_bin: Tuple[float, float],
        *,
        E_GeV: np.ndarray,
        y: np.ndarray,
        sigma: np.ndarray,
        meta: dict | None = None,
        reduction: dict | None = None,
    ):
        self.glon_bin = glon_bin
        self.glat_bin = glat_bin

        self.lowl = float(glon_bin[0])
        self.highl = float(glon_bin[1])
        self.lowb = float(glat_bin[0])
        self.highb = float(glat_bin[1])

        self.E_GeV = np.asarray(E_GeV, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.meta = meta or {}
        # Optional: keep a pointer to the original reduction dict for debugging/plots.
        self.reduction = reduction
        self.fits: dict[str, FitResult] = {}
        # Post-fit analysis caches.  ``wilks_results`` and ``mc_results``
        # are keyed by ``(name_simpler, name_complex)``; ``safety_logs``
        # is keyed by ``fit_name`` and stores the per-window legacy
        # safety-net diagnostic emitted by :class:`ProfiledChi2Fit`.
        self.wilks_results: dict[tuple[str, str], dict] = {}
        self.mc_results: dict[tuple[str, str], object] = {}
        self.safety_logs: dict[str, dict] = {}
        # Asymmetric ±1σ profile-likelihood errors per profiled fit.
        # Keyed by ``fit_name``, value is ``{param_name: (best, lo, hi)}``
        # with absolute parameter values (``nan`` when the grid does not
        # span the ``Δχ² = 1`` crossing).  Currently populated only for
        # SBPL profiled fits via :meth:`compute_profile_errors`.
        self.profile_errors: dict[str, dict[str, tuple[float, float, float]]] = {}
        # Optional debug cache: per-fit-name, per-param ``(grid, chi2_row,
        # chi2_anchor)`` so the 1-D Δχ² profiles can be plotted afterwards.
        self.profile_curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray, float]]] = {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_window(cls, window) -> "WindowFitResult":
        """Build a :class:`WindowFitResult` from a populated ``Window``."""
        E, y, sigma = prepare_fit_data_from_window(window)
        return cls(
            glon_bin=window.glon_bin,
            glat_bin=window.glat_bin,
            E_GeV=E,
            y=y,
            sigma=sigma,
            meta=fit_meta_from_window(window),
            reduction=window.data,
        )

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, model: ModelLike, *, solver) -> FitResult:
        """Fit ``model`` to this window and store the result.

        Dispatch is by type:

        * :class:`SpectralModel`  → classical chi² via :class:`Chi2Fit`.
        * :class:`NuisanceModel`  → profiled chi² via
          :class:`ProfiledChi2Fit` (systematic deformations as nuisance
          parameters).  The legacy safety net is honoured automatically:
          when a *classical* fit for the same underlying spectral model
          is already in :attr:`fits` and ``model.legacy_safety_net`` is
          ``True``, its ``popt`` is used to compare ``chi2_prof`` at
          both the profiled and classical solutions, keeping the better
          one.  The diagnostic dict (``chi2_outer``, ``chi2_classical``,
          ``chosen``) is cached in :attr:`safety_logs[fit_name]`.

        ``solver`` is the ``solver`` section of the analysis config
        (optimiser budgets and Hessian settings).

        Returns the :class:`FitResult` and also stores it in
        :attr:`fits` keyed by the model's ``name``.
        """
        if isinstance(model, NuisanceModel):
            return self._fit_profiled(model, solver)
        return self._fit_classical(model, solver)

    def _fit_classical(self, model: SpectralModel, solver) -> FitResult:
        res = Chi2Fit(model, solver=solver).fit(self.E_GeV, self.y, self.sigma)
        self.fits[model.name] = res
        return res

    def _fit_profiled(self, nm: NuisanceModel, solver) -> FitResult:
        if nm.share_nuisance and nm.w_shared is None:
            raise RuntimeError(
                f"Per-window profiled fit on {self.glon_bin}x{self.glat_bin}: "
                "`NuisanceModel(share_nuisance=True)` requires the global "
                "joint fit to run first so `w_shared` can be propagated.\n"
                "Workflow: fitting.fit_global(nm)  →  fitting.fit_per_window([nm])."
            )
        pred, arr = nm.for_window(self)
        fitter = ProfiledChi2Fit(pred, solver=solver)
        # Legacy safety net: pull the classical fit's popt for this
        # underlying model (if available) and pass it down so the
        # profiled fitter can compare ``chi2_prof(theta_classical)``
        # against the outer solution and keep the better one.
        theta_classical = None
        if nm.legacy_safety_net:
            classical_fit = self.fits.get(nm.spectral_model.name)
            if classical_fit is not None:
                theta_classical = np.asarray(classical_fit.popt, dtype=float)
        safety_log: list = []
        res = fitter.fit(
            arr["E"], arr["y"], arr["sigma"],
            theta_classical=theta_classical,
            legacy_safety_log=safety_log,
        )
        self.fits[pred.name] = res
        if safety_log:
            self.safety_logs[pred.name] = safety_log[0]
        return res

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------
    def __getitem__(self, name: str) -> FitResult:
        return self.fits[name]

    def __contains__(self, name: str) -> bool:
        return name in self.fits

    def __iter__(self):
        return iter(self.fits.items())

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.fits.keys())

    def delta_chi2(self, name_a: str, name_b: str) -> float:
        """``chi2(name_a) - chi2(name_b)`` on this window."""
        return float(self.fits[name_a].chi2 - self.fits[name_b].chi2)

    def compute_profile_errors(
        self,
        name: str,
        *,
        params: tuple[str, ...],
        grids: dict,
        delta: float,
        solver,
        n_workers: int | None = None,
    ) -> dict[str, tuple[float, float, float]]:
        """Asymmetric ±1σ profile-likelihood errors on ``params`` for the
        profiled fit ``name``.

        Reuses the bound :class:`NuisanceModel` that produced
        ``self.fits[name]`` (so the basis / prior / ``Ssys_frac`` and --
        in shared mode -- the global ``w_shared`` are exactly the ones of
        the best-fit) and reprofiles ``w`` and the remaining spectral
        parameters at every grid cell.  See
        :func:`crbreak.fitting.scan_params.profile_errors_1d_window`.

        ``n_workers`` is forwarded to the underlying scan, where it
        parallelises **cells** (not windows) via a process pool.

        Result is cached in :attr:`profile_errors[name]` and returned.
        """
        from .scan_params import profile_errors_1d_window

        out = profile_errors_1d_window(
            self, name, params=params, grids=grids, delta=delta, solver=solver,
            n_workers=n_workers, return_curves=True,
        )
        # ``out`` now contains both the bracketed errors and the raw 1-D
        # Δχ² curves under the ``"_curves"`` key (debug cache for plots).
        curves = out.pop("_curves", {}) if isinstance(out, dict) else {}
        self.profile_errors[name] = out
        self.profile_curves[name] = curves
        return out

    def wilks(
        self,
        name_simpler: str,
        name_complex: str,
        *,
        dof: int | None = None,
    ) -> dict:
        """Wilks' LRT for ``H0=name_simpler`` vs ``H1=name_complex``.

        The result is also cached in :attr:`wilks_results` keyed by
        ``(name_simpler, name_complex)``, so callers can either use the
        return value or iterate over the dict afterwards.
        """
        a = self.fits[name_simpler]
        b = self.fits[name_complex]
        if dof is None:
            dof = int(b.model.n_params - a.model.n_params)
        out = wilks_pvalue(a.chi2 - b.chi2, dof)
        self.wilks_results[(name_simpler, name_complex)] = out
        return out

    def mc_pvalue(
        self,
        *,
        name_pl: str,
        name_alt: str,
        N: int,
        seed: int,
        sample_theta: bool,
        warm_start: bool,
        store_deltas: bool,
        solver,
        n_workers: int | None = None,
        chunksize: int = 4,
        batch_size: int | None = None,
        print_every: int = 0,
    ):
        """Parametric-bootstrap MC p-value for ``H0=name_pl`` vs ``name_alt``.

        Generates ``N`` toy datasets under the null (``name_pl``) and refits
        both models on each toy to build the empirical Δχ² distribution, then
        compares against the observed Δχ² in this window.  Both ``name_pl``
        and ``name_alt`` must already have been fitted (e.g. via
        :meth:`crbreak.fitting.main.Fitting.fit_per_window`).

        Returns the :class:`crbreak.fitting.montecarlo.MCPValueResult`.
        """
        from . import montecarlo as _mc

        out = _mc.mc_pvalue_window(
            self,
            name_pl=name_pl,
            name_alt=name_alt,
            N=N,
            seed=seed,
            sample_theta=sample_theta,
            warm_start=warm_start,
            solver=solver,
            n_workers=n_workers,
            chunksize=chunksize,
            batch_size=batch_size,
            print_every=print_every,
            store_deltas=store_deltas,
        )
        self.mc_results[(name_pl, name_alt)] = out
        return out

    def __repr__(self) -> str:
        roi = (
            f"|l|=({self.lowl:g},{self.highl:g}), "
            f"|b|=({self.lowb:g},{self.highb:g})"
        )
        if not self.fits:
            return f"WindowFitResult({roi}, no fits)"
        bits = ", ".join(
            f"{n}: chi2={r.chi2:.1f}/{r.ndof}" for n, r in self.fits.items()
        )
        return f"WindowFitResult({roi}, {bits})"
