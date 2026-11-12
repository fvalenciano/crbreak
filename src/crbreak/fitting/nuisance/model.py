"""Profiled (nuisance) spectral model.

A :class:`NuisanceModel` is the configuration knob a user passes to
:meth:`crbreak.fitting.main.Fitting.fit_per_window` (and friends) when they
want the deformed-model flavour instead of the plain
:class:`crbreak.fitting.chi2.Chi2Fit`.

The class lives in **two states**:

* *unbound* (what the user constructs) -- holds only the configuration:
  spectral model, prior factory, basis kind, ``n_nuis`` etc.  Does not
  know about any window.
* *bound* (returned by :meth:`for_window`) -- a fresh copy whose
  ``basis`` / ``Ssys_frac`` / ``prior`` / ``E_data`` / ``last_w`` slots
  are populated for one specific window.  This is the object the
  profiled fitters (``ProfiledChi2Fit`` / ``ProfiledGlobalChi2Fit``)
  actually consume; it also implements the :class:`SpectralModel`
  surface (``__call__``, ``default_p0``, ``default_bounds``) so it slots
  straight into :class:`crbreak.fitting.chi2.FitResult`.

Mathematically, the deformed prediction at the data energies is

.. math::
    \\hat y_i(\\theta, w) = (1 + (Bw)_i\\,S_i)\\,\\mu(E_i; \\theta).

For fixed ``\\theta`` (and therefore fixed ``\\mu``), ``\\hat y`` is
linear in ``w``, so :meth:`profile_w` reduces to a single
``np.linalg.lstsq`` whenever the prior is quadratic; non-quadratic
priors (e.g. :class:`BoxGaussPrior`) fall back to L-BFGS-B with a
gaussian-equivalent warm start.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Sequence, TYPE_CHECKING

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize

from ..models import SpectralModel
from .basis import NuisanceBasis, _piecewise_linear_design
from .priors import BoxGaussPrior, NuisancePrior, make_prior

if TYPE_CHECKING:
    from ..window_fit import WindowFitResult


def _const_prior_factory(prior: NuisancePrior, n_nuis: int) -> NuisancePrior:
    """Picklable ``prior_factory`` replacement for *bound* NuisanceModels.

    After :meth:`NuisanceModel.for_window`, ``prior`` is already cached
    on the bound copy and the user's original ``prior_factory`` (often
    a ``lambda``) is no longer needed for any internal operation.  We
    swap it for ``functools.partial(_const_prior_factory, prior)`` so
    the bound model survives ``pickle`` round-trips -- which is what
    :func:`crbreak.fitting.scan_params.scan_chi2_sbpl` needs to ship
    bound NuisanceModels into worker processes (``n_workers > 1``).
    """
    del n_nuis
    return prior


def nuisance_kwargs_from_config(nu) -> dict:
    """Every :class:`NuisanceModel` argument except ``spectral_model`` and
    ``share_nuisance``, from the ``nuisance`` section of the analysis config."""
    return dict(
        prior_factory=partial(
            make_prior,
            nu.prior,
            sigma=nu.prior_sigma,
            width=nu.prior_width,
            tail_sigma_frac=nu.prior_tail,
        ),
        n_nuis=nu.n_nodes,
        basis_kind=nu.basis,
        bspline_degree=nu.bspline_degree,
        legacy_safety_net=nu.safety_net,
        enforce_s_bounds=nu.enforce_s_bounds,
        max_dw=nu.max_dw,
    )


@dataclass
class NuisanceModel:
    """Window-agnostic configuration *and* per-window bound instance.

    Parameters
    ----------
    spectral_model
        Underlying :class:`SpectralModel` (e.g.
        ``PowerLaw.from_config(cfg.model)``).
    prior_factory
        ``f(n_nuis) -> NuisancePrior`` (e.g.
        ``lambda n: BoxGaussPrior(n, width=1.0)``).  A *fresh* prior is
        instantiated per window, which keeps caches isolated and makes
        parallelisation trivial.
    n_nuis
        Number of nuisance coefficients ``w_j``.
    basis_kind, bspline_degree
        Forwarded to :meth:`NuisanceBasis.from_E`.
    legacy_safety_net
        If ``True``, the profiled fitter also evaluates
        ``chi2_prof`` at the corresponding *classical* best-fit (when one
        is available in the window's ``fits`` dict) and keeps whichever
        ``theta`` yields the smaller value.  Replicates the legacy
        ``FDE_PASS8`` behaviour.
    share_nuisance
        Common-nuisance mode (legacy ``COMMON_NUISANCE=True``).  When
        ``True`` the deformation ``s(E;w)`` is assumed to have the **same
        shape across all ROIs** — i.e. a single ``w_shared`` is fit
        globally for all windows instead of an independent ``w_j`` per
        window.  This flag is honoured by
        :class:`~crbreak.fitting.global_chi2.ProfiledGlobalChi2Fit`
        (vector-extended joint fit), by
        :class:`~crbreak.fitting.scan_params.scan_chi2_sbpl` (uses
        ``w_shared`` frozen to the global solution per cell) and by
        :class:`~crbreak.fitting.chi2.ProfiledChi2Fit` (the per-window
        χ² uses ``w_shared`` *frozen* — the inner solve is skipped).
        ``share_nuisance=True`` and ``share_nuisance=False`` are
        **mutually exclusive analyses**: pick one and run the whole
        pipeline with it.
    w_shared
        Bound-only.  Set by
        :class:`~crbreak.fitting.global_chi2.ProfiledGlobalChi2Fit`
        after the global joint fit.  When non-``None`` the inner
        :meth:`profile_w` short-circuits and returns this vector
        verbatim — that is what makes the per-window ``ProfiledChi2Fit``
        use the *common* ``w_shared`` automatically and produce
        legacy-faithful per-window deltachi²/Wilks/MC.

    Bound state
    -----------
    The remaining attributes (``basis``, ``Ssys_frac``, ``prior``,
    ``E_data``, ``last_w``) are populated by :meth:`for_window` and are
    ``None`` on the unbound configuration object.  Calling any of the
    bound-only methods (``profile_w``, ``predict_deformed``,
    ``augmented_residuals``, ``chi2``, ``subset``) on an unbound model
    raises :class:`RuntimeError`.
    """

    spectral_model: SpectralModel
    prior_factory: Callable[[int], NuisancePrior]
    n_nuis: int
    basis_kind: str
    bspline_degree: int
    legacy_safety_net: bool
    share_nuisance: bool
    # When True, the inner w-solve enforces s(E;w) = (B@w)_i ∈ (-1, 1) for
    # all data bins.  For piecewise_linear this is exact (w_j = s(E_j));
    # for bspline the convex-hull property makes it a sufficient bound.
    enforce_s_bounds: bool
    # Hard limit on the increment |w_{j+1} - w_j| between adjacent nuisance
    # nodes.  Only active for piecewise_linear basis (where w_j = s(E_j)).
    # When max_dw < 2.0, _refine_nonquadratic reparametrises as v[0]=w[0],
    # v[j]=w[j]-w[j-1] and adds L-BFGS-B box bounds v[j] ∈ (-max_dw, max_dw),
    # giving a TRUE hard constraint at no gradient-computation cost.
    # 2.0 = inactive (full range allowed; w ∈ [-1,1] means max diff ≤ 2).
    max_dw: float

    basis: NuisanceBasis | None = field(default=None, repr=False)
    Ssys_frac: np.ndarray | None = field(default=None, repr=False)
    prior: NuisancePrior | None = field(default=None, repr=False)
    E_data: np.ndarray | None = field(default=None, repr=False)
    last_w: np.ndarray | None = field(default=None, repr=False)
    w_shared: np.ndarray | None = field(default=None, repr=False)
    w_shared_counts_prior: bool = field(default=False, repr=False)
    # Optional externally calibrated nuisance, keyed by ``tuple(glon_bin)``.
    # Used for control-region workflows: fit w in high-latitude windows,
    # then freeze the corresponding vector in the low-latitude signal window.
    w_shared_by_glon: dict[tuple[float, float], np.ndarray] | None = field(
        default=None, repr=False
    )

    _cached_name: str | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # SpectralModel-compatible surface (always available)
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Composite name, e.g. ``"SBPL|prof.box"``."""
        if self._cached_name is None:
            probe = self.prior or self.prior_factory(self.n_nuis)
            self._cached_name = (
                f"{self.spectral_model.name}|prof.{probe.short_name}"
            )
        return self._cached_name

    @property
    def param_names(self) -> tuple[str, ...]:
        return self.spectral_model.param_names

    @property
    def n_params(self) -> int:
        return self.spectral_model.n_params

    @property
    def E0(self) -> float:
        return self.spectral_model.E0

    def __call__(self, E: np.ndarray, *theta: float) -> np.ndarray:
        """Physical model prediction (no deformation).

        Used by :class:`FitResult.predict`, envelopes, plots.  The deformed
        prediction is :meth:`predict_deformed` (data grid) or
        :meth:`predict_deformed_on_grid` (arbitrary grid).
        """
        return self.spectral_model(E, *theta)

    def default_p0(self, E: np.ndarray, y: np.ndarray) -> tuple[float, ...]:
        return self.spectral_model.default_p0(E, y)

    def default_bounds(self):
        return self.spectral_model.default_bounds()

    # ------------------------------------------------------------------
    # Bind a configuration to a window
    # ------------------------------------------------------------------
    @property
    def is_bound(self) -> bool:
        """``True`` once :meth:`for_window` has been called."""
        return self.basis is not None

    def for_window(
        self, wf: "WindowFitResult"
    ) -> tuple["NuisanceModel", dict[str, np.ndarray]]:
        """Return a *bound* copy of this :class:`NuisanceModel` for one window.

        Applies the standard ``finite & sigma>0`` mask once and slices the
        nuisance arrays to match; the resulting bound model's basis rows
        align with the returned ``E/y/sigma``.

        ``wf.meta["Ssys_frac"]`` must be present (extraction populates it
        via :func:`crbreak.fitting.adapters.fit_meta_from_window`); a
        clear error is raised otherwise.

        Returns
        -------
        nm_bound : NuisanceModel
            Same configuration, with ``basis``/``Ssys_frac``/``prior``/
            ``E_data``/``last_w`` populated for ``wf``.
        arrays : dict
            ``{"E", "y", "sigma"}`` of the masked window data ready to
            feed :class:`~crbreak.fitting.chi2.ProfiledChi2Fit` or
            :class:`~crbreak.fitting.global_chi2.ProfiledGlobalChi2Fit`.
        """
        Ssys = wf.meta.get("Ssys_frac")
        if Ssys is None:
            raise RuntimeError(
                f"Window {wf.glon_bin}x{wf.glat_bin}: meta['Ssys_frac'] "
                "is missing.  Re-run extraction so the systematic Aeff "
                "fraction is propagated, or use a classical SpectralModel."
            )
        E = np.asarray(wf.E_GeV, dtype=float)
        y = np.asarray(wf.y, dtype=float)
        sigma = np.asarray(wf.sigma, dtype=float)
        Ssys = np.asarray(Ssys, dtype=float)
        mask = (
            np.isfinite(E)
            & np.isfinite(y)
            & np.isfinite(sigma)
            & np.isfinite(Ssys)
            & (sigma > 0)
        )
        E_, y_, sig_, S_ = E[mask], y[mask], sigma[mask], Ssys[mask]

        basis = NuisanceBasis.from_E(
            E_,
            n_nuis=self.n_nuis,
            kind=self.basis_kind,
            degree=self.bspline_degree,
        )
        prior = self.prior_factory(self.n_nuis)
        w_saved = self.w_shared
        w_counts_prior = False
        if self.w_shared_by_glon:
            key = tuple(float(x) for x in wf.glon_bin)
            if key in self.w_shared_by_glon:
                w_saved = np.asarray(self.w_shared_by_glon[key], dtype=float)
                w_counts_prior = True
        nm_bound = self._with_bound_state(
            basis=basis,
            Ssys_frac=S_,
            prior=prior,
            E_data=E_,
        )
        nm_bound.w_shared = None if w_saved is None else np.asarray(w_saved, dtype=float).copy()
        nm_bound.w_shared_counts_prior = bool(w_counts_prior)
        return nm_bound, {"E": E_, "y": y_, "sigma": sig_}

    def for_global(
        self,
        wfs: Sequence["WindowFitResult"],
    ) -> tuple[list["NuisanceModel"], list[dict[str, np.ndarray]]]:
        """Bind one :class:`NuisanceModel` per window with a **shared** basis.

        Required when :attr:`share_nuisance` is ``True``.  The nuisance
        nodes ``x_nuis`` are placed uniformly in ``log10(E)`` over the
        **global** energy range (min/max across all ``wfs``); each
        per-window basis ``B_j`` is then evaluated on that window's
        log10(E) grid using **the same nodes**, so the coefficient
        vector ``w_shared`` has a single, well-defined meaning across
        windows.

        All bound NMs share the **same prior instance** (with the same
        ``x_nuis``), so the prior chi² evaluated on ``w_shared`` is
        identical to what each window-bound model would compute.

        Returns
        -------
        bounds : list[NuisanceModel]
            One bound NM per window.  All share ``x_nuis``, ``prior``,
            and ``share_nuisance=True``; ``w_shared`` is ``None`` until
            the global fit fills it.
        arrays : list[dict]
            ``[{'E','y','sigma'}, ...]`` aligned with ``bounds``, ready
            for :class:`~crbreak.fitting.global_chi2.ProfiledGlobalChi2Fit`.
        """
        if not self.share_nuisance:
            raise ValueError(
                "NuisanceModel.for_global is only meaningful when "
                "`share_nuisance=True`.  For independent nuisances per "
                "window use `for_window` (called automatically by the "
                "fitters)."
            )

        # 1. Mask each window and stash the kept (E, y, sigma, S, logE).
        per_win: list[dict[str, np.ndarray]] = []
        for wf in wfs:
            Ssys = wf.meta.get("Ssys_frac")
            if Ssys is None:
                raise RuntimeError(
                    f"Window {wf.glon_bin}x{wf.glat_bin}: meta['Ssys_frac'] "
                    "is missing.  Re-run extraction so the systematic Aeff "
                    "fraction is propagated."
                )
            E = np.asarray(wf.E_GeV, dtype=float)
            y = np.asarray(wf.y, dtype=float)
            sigma = np.asarray(wf.sigma, dtype=float)
            S = np.asarray(Ssys, dtype=float)
            mask = (
                np.isfinite(E)
                & np.isfinite(y)
                & np.isfinite(sigma)
                & np.isfinite(S)
                & (sigma > 0)
            )
            E_, y_, sig_, S_ = E[mask], y[mask], sigma[mask], S[mask]
            if E_.size == 0:
                raise RuntimeError(
                    f"Window {wf.glon_bin}x{wf.glat_bin}: no finite bin "
                    "after the standard mask; cannot build a shared basis."
                )
            per_win.append(
                {"E": E_, "y": y_, "sigma": sig_, "S": S_, "logE": np.log10(E_)}
            )

        # 2. Build the shared nuisance nodes on the GLOBAL log10(E) range.
        all_logE = np.concatenate([d["logE"] for d in per_win])
        logE_lo = float(np.nanmin(all_logE))
        logE_hi = float(np.nanmax(all_logE))
        if not (logE_hi > logE_lo):
            raise RuntimeError(
                "for_global: degenerate global log10(E) range "
                f"[{logE_lo}, {logE_hi}]."
            )
        kind = self.basis_kind
        if kind == "piecewise_linear":
            x_nuis_shared = np.linspace(logE_lo, logE_hi, self.n_nuis)
            knots_shared = None
            degree_shared = 0
        elif kind == "bspline":
            degree_shared = int(self.bspline_degree)
            if self.n_nuis <= degree_shared:
                raise ValueError("Need n_nuis > bspline_degree for bspline.")
            n_internal = max(0, self.n_nuis - (degree_shared + 1))
            if n_internal > 0:
                internal = np.linspace(logE_lo, logE_hi, n_internal + 2)[1:-1]
                knots_shared = np.r_[
                    np.repeat(logE_lo, degree_shared + 1),
                    internal,
                    np.repeat(logE_hi, degree_shared + 1),
                ]
            else:
                knots_shared = np.r_[
                    np.repeat(logE_lo, degree_shared + 1),
                    np.repeat(logE_hi, degree_shared + 1),
                ]
            # Greville abscissae of the shared knot vector.
            x_nuis_shared = np.empty(self.n_nuis, dtype=float)
            for j in range(self.n_nuis):
                x_nuis_shared[j] = float(
                    np.mean(knots_shared[j + 1 : j + 1 + degree_shared])
                )
        else:
            raise ValueError(f"Unknown basis_kind={kind!r}.")

        # 3. Per-window basis on the SHARED nodes, plus the shared prior.
        prior_shared = self.prior_factory(self.n_nuis)
        bounds: list[NuisanceModel] = []
        arrays: list[dict[str, np.ndarray]] = []
        for d in per_win:
            if kind == "piecewise_linear":
                B_j = _piecewise_linear_design(d["logE"], x_nuis_shared)
            else:
                B_j = np.zeros((d["logE"].size, self.n_nuis), dtype=float)
                for j in range(self.n_nuis):
                    c = np.zeros(self.n_nuis)
                    c[j] = 1.0
                    spl = BSpline(knots_shared, c, degree_shared, extrapolate=True)
                    B_j[:, j] = spl(d["logE"])
            basis_j = NuisanceBasis(
                B=B_j,
                x_nuis=x_nuis_shared.copy(),
                n_nuis=self.n_nuis,
                kind=kind,
                knots=None if knots_shared is None else knots_shared.copy(),
                degree=degree_shared,
            )
            nm_bound = self._with_bound_state(
                basis=basis_j,
                Ssys_frac=d["S"],
                prior=prior_shared,
                E_data=d["E"],
            )
            bounds.append(nm_bound)
            arrays.append({"E": d["E"], "y": d["y"], "sigma": d["sigma"]})
        return bounds, arrays

    def _with_bound_state(
        self,
        *,
        basis: NuisanceBasis,
        Ssys_frac: np.ndarray,
        prior: NuisancePrior,
        E_data: np.ndarray | None,
    ) -> "NuisanceModel":
        """Build a bound copy with the given basis/Ssys/prior/E_data."""
        if prior.n_nuis != basis.n_nuis:
            raise ValueError(
                f"prior.n_nuis ({prior.n_nuis}) != basis.n_nuis "
                f"({basis.n_nuis})."
            )
        Ssys_frac = np.asarray(Ssys_frac, dtype=float)
        if Ssys_frac.shape[0] != basis.B.shape[0]:
            raise ValueError(
                f"Ssys_frac size ({Ssys_frac.shape[0]}) != basis rows "
                f"({basis.B.shape[0]})."
            )
        bound = NuisanceModel(
            spectral_model=self.spectral_model,
            prior_factory=partial(_const_prior_factory, prior),
            n_nuis=self.n_nuis,
            basis_kind=self.basis_kind,
            bspline_degree=self.bspline_degree,
            legacy_safety_net=self.legacy_safety_net,
            share_nuisance=self.share_nuisance,
            enforce_s_bounds=self.enforce_s_bounds,
            max_dw=self.max_dw,
            basis=basis,
            Ssys_frac=Ssys_frac,
            prior=prior,
            E_data=(None if E_data is None else np.asarray(E_data, dtype=float)),
            last_w=np.zeros(basis.n_nuis, dtype=float),
            w_shared=None if self.w_shared is None else np.asarray(
                self.w_shared, dtype=float
            ).copy(),
            w_shared_counts_prior=bool(self.w_shared_counts_prior),
            w_shared_by_glon=None if self.w_shared_by_glon is None else {
                k: np.asarray(v, dtype=float).copy()
                for k, v in self.w_shared_by_glon.items()
            },
        )
        return bound

    def _require_bound(self, op: str) -> None:
        if not self.is_bound:
            raise RuntimeError(
                f"{op}: NuisanceModel is unbound.  Call .for_window(wf) "
                "first to build a bound instance for a specific window."
            )

    def with_spectral_model(self, new_spectral: SpectralModel) -> "NuisanceModel":
        """Return a bound copy with a different :class:`SpectralModel`.

        Reuses ``basis`` / ``Ssys_frac`` / ``prior`` / ``E_data`` exactly
        (since none of them depend on the spectral shape) and just swaps
        in ``new_spectral`` plus a fresh zero ``last_w``.

        This is the cheap way to drive a 2-D parameter scan: the basis
        and the prior are built **once per window**, and each grid cell
        only changes the spectral model (typically a
        :class:`FrozenSpectralModel` with ``Ebreak``/``dgamma`` fixed at
        the cell values).
        """
        self._require_bound("NuisanceModel.with_spectral_model")
        bound = NuisanceModel(
            spectral_model=new_spectral,
            prior_factory=self.prior_factory,
            n_nuis=self.n_nuis,
            basis_kind=self.basis_kind,
            bspline_degree=self.bspline_degree,
            legacy_safety_net=self.legacy_safety_net,
            share_nuisance=self.share_nuisance,
            enforce_s_bounds=self.enforce_s_bounds,
            max_dw=self.max_dw,
            basis=self.basis,
            Ssys_frac=self.Ssys_frac,
            prior=self.prior,
            E_data=self.E_data,
            last_w=np.zeros(self.basis.n_nuis, dtype=float),
            w_shared=None if self.w_shared is None else np.asarray(
                self.w_shared, dtype=float
            ).copy(),
            w_shared_counts_prior=bool(self.w_shared_counts_prior),
            w_shared_by_glon=None if self.w_shared_by_glon is None else {
                k: np.asarray(v, dtype=float).copy()
                for k, v in self.w_shared_by_glon.items()
            },
        )
        return bound

    def __repr__(self) -> str:
        share = "shared" if self.share_nuisance else "indep"
        if not self.is_bound:
            return (
                f"NuisanceModel({self.spectral_model.name}, "
                f"prior_factory={self.prior_factory!r}, "
                f"n_nuis={self.n_nuis}, basis_kind={self.basis_kind!r}, "
                f"{share}, unbound)"
            )
        wflag = "frozen" if (self.share_nuisance and self.w_shared is not None) else share
        return (
            f"NuisanceModel({self.spectral_model.name}, "
            f"basis={self.basis.kind}, n_nuis={self.n_nuis}, "
            f"prior={self.prior!r}, {wflag}, bound)"
        )

    # ------------------------------------------------------------------
    # Bound-only: subsetting (used by ProfiledChi2Fit)
    # ------------------------------------------------------------------
    def subset(self, mask: np.ndarray) -> "NuisanceModel":
        """Return a bound copy with basis/Ssys_frac restricted to ``mask``."""
        self._require_bound("NuisanceModel.subset")
        mask = np.asarray(mask, dtype=bool)
        sub = self._with_bound_state(
            basis=self.basis.subset(mask),
            Ssys_frac=self.Ssys_frac[mask],
            prior=self.prior,
            E_data=None if self.E_data is None else self.E_data[mask],
        )
        sub.last_w = self.last_w.copy() if self.last_w is not None else None
        return sub

    # ------------------------------------------------------------------
    # Bound-only: inner profile
    # ------------------------------------------------------------------
    def profile_w(
        self,
        y: np.ndarray,
        sigma: np.ndarray,
        mu: np.ndarray,
        *,
        w0: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve ``argmin_w chi2_data(theta, w) + chi2_prior(w)`` at fixed ``theta``.

        ``mu`` is the *physical* model prediction at the data energies
        (i.e. ``self.spectral_model(E_data, *theta)``) and is taken as
        input to avoid recomputing it inside the inner solve.

        A finite-bin mask ``m = isfinite(mu) & isfinite(y) & isfinite(sigma)
        & (sigma > 0)`` is applied **inside** the solve (legacy-style)
        so that pathological bins (e.g. ``mu = nan/inf`` from extreme
        spectral parameters) cannot poison ``numpy.linalg.lstsq`` /
        LAPACK with NaNs.  If no bin is finite the inner solve is
        skipped and ``w = 0`` is returned; the caller is expected to
        treat that ``theta`` as ``chi2 = +inf``.

        Returns
        -------
        w_hat : np.ndarray
            Best-fit nuisance vector.  Also cached on :attr:`last_w`.

        Common-nuisance short-circuit
        -----------------------------
        When ``share_nuisance=True`` and ``w_shared`` is set, this method
        returns ``w_shared`` verbatim **without** running the inner
        solve.  That is the single switch that turns the per-window
        profiled chi² into a deformed chi² with frozen ``w`` (legacy
        ``WINDOW_REANALYZE_WITH_SHARED_SHAPE``): every downstream caller
        — :class:`~crbreak.fitting.chi2.ProfiledChi2Fit`, :meth:`chi2`,
        :meth:`augmented_residuals` — automatically inherits the
        common-nuisance behaviour.
        """
        self._require_bound("NuisanceModel.profile_w")
        if self.w_shared is not None:
            self.last_w = np.asarray(self.w_shared, dtype=float)
            return self.last_w

        y = np.asarray(y, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        mu = np.asarray(mu, dtype=float)

        m = np.isfinite(mu) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
        if not np.any(m):
            w_hat = np.zeros(self.basis.n_nuis, dtype=float)
            self.last_w = w_hat
            return w_hat

        y_m, sig_m, mu_m = y[m], sigma[m], mu[m]
        S_m, B_m = self.Ssys_frac[m], self.basis.B[m]

        if self.prior.is_quadratic:
            w_hat = self._lstsq_solve(y_m, sig_m, mu_m, S_m, B_m, self.prior)
            if self.enforce_s_bounds and np.any(np.abs(w_hat) >= 1.0):
                # lstsq has no bound support; clip and refine with bounded L-BFGS-B.
                w0_clipped = np.clip(w_hat, -1.0 + 1e-6, 1.0 - 1e-6)
                w_hat = self._refine_nonquadratic(
                    y_m, sig_m, mu_m, S_m, B_m, w0_clipped
                )
        else:
            if w0 is None:
                w0_ = self._lstsq_solve(
                    y_m, sig_m, mu_m, S_m, B_m,
                    prior=self._gaussian_seed_prior(),
                )
                if self.enforce_s_bounds:
                    w0_ = np.clip(w0_, -1.0 + 1e-6, 1.0 - 1e-6)
            else:
                w0_ = np.asarray(w0, dtype=float)
            w_hat = self._refine_nonquadratic(y_m, sig_m, mu_m, S_m, B_m, w0_)

        self.last_w = np.asarray(w_hat, dtype=float)
        return self.last_w

    # ------------------------------------------------------------------
    # Bound-only: predictions
    # ------------------------------------------------------------------
    def predict_deformed(
        self,
        E: np.ndarray,
        theta: tuple,
        w: np.ndarray | None = None,
    ) -> np.ndarray:
        """``(1 + (B @ w) * S) * mu(E, theta)`` at the data grid.

        Only meaningful at the *data* energies (``B`` and ``S`` are bound
        to those bins).  Use :meth:`predict_deformed_on_grid` to draw a
        smooth deformed line on a dense energy axis.
        """
        self._require_bound("NuisanceModel.predict_deformed")
        if w is None:
            w = self.last_w
        mu = self.spectral_model(E, *theta)
        return (1.0 + (self.basis.B @ w) * self.Ssys_frac) * mu

    def predict_deformed_on_grid(
        self,
        E_eval: np.ndarray,
        theta: tuple,
        w: np.ndarray | None = None,
    ) -> np.ndarray:
        """Deformed prediction on an arbitrary energy grid.

        Re-evaluates the basis at ``log10(E_eval)`` reusing the original
        nodes (so the nuisance functional form is the same as during the
        fit) and linearly interpolates ``Ssys_frac`` from the data grid
        (clamped at the endpoints).  This is what plotting code should
        call to draw a smooth deformed line on a dense energy axis,
        instead of a piecewise segment connecting only the data points.

        Returns
        -------
        y_def : np.ndarray
            ``(1 + B(logE_eval) @ w * S(E_eval)) * mu(E_eval, theta)``.
        """
        self._require_bound("NuisanceModel.predict_deformed_on_grid")
        if self.E_data is None:
            raise ValueError(
                "Bound NuisanceModel was built without E_data; cannot "
                "interpolate Ssys_frac on a new grid.  Use "
                "predict_deformed instead."
            )
        if w is None:
            w = self.last_w
        E_eval = np.asarray(E_eval, dtype=float)
        logE_eval = np.log10(E_eval)
        B_eval = self.basis.evaluate(logE_eval)
        S_eval = np.interp(
            logE_eval,
            np.log10(self.E_data),
            self.Ssys_frac,
        )
        mu = self.spectral_model(E_eval, *theta)
        return (1.0 + (B_eval @ w) * S_eval) * mu

    # ------------------------------------------------------------------
    # Bound-only: augmented residuals + scalar chi2
    # ------------------------------------------------------------------
    def augmented_residuals(
        self,
        E: np.ndarray,
        y: np.ndarray,
        sigma: np.ndarray,
        *theta: float,
        w0: np.ndarray | None = None,
    ) -> np.ndarray:
        """Stacked residual vector ``[(y - yhat)/sigma; prior_pseudo(w_hat)]``.

        Calling :func:`numpy.sum` of its square gives ``chi2_data + chi2_prior``
        at the profiled minimum of ``w``.  Used by
        :class:`~crbreak.fitting.chi2.ProfiledChi2Fit`.

        The data-block has the **same length** as ``y`` even when some
        bins are masked out (``mu`` non-finite or ``sigma <= 0``); those
        bins simply contribute ``0`` to the residual vector, mirroring
        the masked profiling done in :meth:`profile_w`.
        """
        self._require_bound("NuisanceModel.augmented_residuals")
        E = np.asarray(E, dtype=float)
        y = np.asarray(y, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        mu = np.asarray(self.spectral_model(E, *theta), dtype=float)

        m = np.isfinite(mu) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
        w_hat = self.profile_w(
            y, sigma, mu,
            w0=self.last_w if w0 is None else w0,
        )

        data_r = np.zeros_like(y)
        if np.any(m):
            yhat_m = (1.0 + (self.basis.B[m] @ w_hat) * self.Ssys_frac[m]) * mu[m]
            data_r[m] = (y[m] - yhat_m) / sigma[m]

        # Global common-nuisance with frozen w_shared: drop the prior block
        # because the prior was already counted once in the global joint fit.
        # Control-mode w_shared_by_glon values are different per longitude and
        # must still contribute their own prior penalty.
        if self.w_shared is not None and not self.w_shared_counts_prior:
            return data_r

        prior_r = self.prior.augmented_residuals(w_hat)
        if prior_r.size == 0:
            return data_r
        return np.concatenate([data_r, prior_r])

    def chi2(
        self,
        E: np.ndarray,
        y: np.ndarray,
        sigma: np.ndarray,
        *theta: float,
    ) -> float:
        """Profiled chi² ``min_w [chi2_data(theta, w) + chi2_prior(w)]``.

        Bins where ``mu`` is non-finite or ``sigma <= 0`` are masked out
        (legacy ``profiled_chi2`` semantics): if **no** bin survives,
        the cell is reported as ``+inf`` so the outer optimiser steps
        away from that ``theta``.  Otherwise the inner ``w`` is profiled
        on the masked sub-system and the chi² is summed only on those
        bins (plus the prior).

        Global common-nuisance (``share_nuisance=True`` and ``w_shared`` set):
        the prior term is dropped from the per-window chi² (it was
        already counted *once* in the global joint fit) so the
        residual-only "conditional" statistic is what drives Wilks /
        MC / scan, exactly mirroring the legacy ``data_fixedw``
        bookkeeping in ``WINDOW_REANALYZE_WITH_SHARED_SHAPE``.

        Control-mode fixed nuisances (``w_shared_by_glon``) are different
        per longitude and are therefore still penalised once per window.
        """
        self._require_bound("NuisanceModel.chi2")
        E = np.asarray(E, dtype=float)
        y = np.asarray(y, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        mu = np.asarray(self.spectral_model(E, *theta), dtype=float)

        m = np.isfinite(mu) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
        if not np.any(m):
            return float("inf")

        w_hat = self.profile_w(y, sigma, mu, w0=self.last_w)
        yhat_m = (1.0 + (self.basis.B[m] @ w_hat) * self.Ssys_frac[m]) * mu[m]
        data_r = (y[m] - yhat_m) / sigma[m]
        chi2_data = float(np.sum(data_r * data_r))
        if self.w_shared is not None and not self.w_shared_counts_prior:
            return chi2_data
        prior_r = self.prior.augmented_residuals(w_hat)
        chi2_prior = float(np.sum(prior_r * prior_r)) if prior_r.size else 0.0
        return chi2_data + chi2_prior

    def chi2_external_grad(
        self,
        E: np.ndarray,
        y: np.ndarray,
        sigma: np.ndarray,
        *theta: float,
    ) -> tuple[float, np.ndarray]:
        """Profiled chi2 plus envelope-theorem gradient wrt spectral theta."""
        self._require_bound("NuisanceModel.chi2_external_grad")
        E = np.asarray(E, dtype=float)
        y = np.asarray(y, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        theta = np.asarray(theta, dtype=float)
        mu = np.asarray(self.spectral_model(E, *theta), dtype=float)
        m = np.isfinite(mu) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
        if not np.any(m):
            return float("inf"), np.zeros(theta.size, dtype=float)

        w_hat = self.profile_w(y, sigma, mu, w0=self.last_w)
        A = 1.0 + (self.basis.B[m] @ w_hat) * self.Ssys_frac[m]
        yhat = A * mu[m]
        r = (y[m] - yhat) / sigma[m]
        chi2_data = float(np.sum(r * r))
        if self.w_shared is not None and not self.w_shared_counts_prior:
            chi2_prior = 0.0
        else:
            prior_r = self.prior.augmented_residuals(w_hat)
            chi2_prior = float(np.sum(prior_r * prior_r)) if prior_r.size else 0.0
        jac_mu = self._spectral_jacobian_fd(E[m], theta)
        weight = ((y[m] - yhat) / (sigma[m] * sigma[m]) * A)[:, None]
        grad = -2.0 * np.sum(weight * jac_mu, axis=0)
        return chi2_data + chi2_prior, grad

    def _spectral_jacobian_fd(self, E: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Small FD Jacobian of the physical spectral model only."""
        lo, hi = self.default_bounds()
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        theta = np.asarray(theta, dtype=float)
        jac = np.empty((np.asarray(E).size, theta.size), dtype=float)
        floors = {"K": 1e-3, "Ebreak": 1.0}
        for j, name in enumerate(self.param_names):
            h = 1e-6 * max(abs(theta[j]), floors.get(name, 1.0))
            tp = theta.copy()
            tm = theta.copy()
            if theta[j] - h > lo[j] and theta[j] + h < hi[j]:
                tp[j] += h
                tm[j] -= h
                jac[:, j] = (
                    self.spectral_model(E, *tp) - self.spectral_model(E, *tm)
                ) / (2.0 * h)
            elif theta[j] + h < hi[j]:
                tp[j] += h
                jac[:, j] = (
                    self.spectral_model(E, *tp) - self.spectral_model(E, *theta)
                ) / h
            else:
                tm[j] -= h
                jac[:, j] = (
                    self.spectral_model(E, *theta) - self.spectral_model(E, *tm)
                ) / h
        return jac

    # ------------------------------------------------------------------
    # Bound-only: internals (linear / non-linear inner solves)
    # ------------------------------------------------------------------
    @staticmethod
    def _lstsq_solve(
        y: np.ndarray,
        sigma: np.ndarray,
        mu: np.ndarray,
        S: np.ndarray,
        B: np.ndarray,
        prior: NuisancePrior,
    ) -> np.ndarray:
        """One-shot linear inner solve assuming a quadratic prior.

        Solves ``argmin_w || M w - r0 ||^2 + || A w - b ||^2`` by stacking
        the data and prior blocks and calling :func:`numpy.linalg.lstsq`.
        """
        scale = mu * S
        M = (B * scale[:, None]) / sigma[:, None]
        r0 = (y - mu) / sigma

        A_p, b_p = prior.linear_components()
        A = np.vstack([M, A_p]) if A_p.size else M
        b = np.concatenate([r0, b_p]) if A_p.size else r0
        w_hat, *_ = np.linalg.lstsq(A, b, rcond=None)
        return np.asarray(w_hat, dtype=float)

    def _gaussian_seed_prior(self) -> NuisancePrior:
        if isinstance(self.prior, BoxGaussPrior):
            return self.prior.gaussian_equivalent()
        from .priors import GaussianPrior  # local import to avoid cycle
        return GaussianPrior(
            self.basis.n_nuis, sigma=1.0, smoothness=self.prior.smoothness
        )

    def _refine_nonquadratic(
        self,
        y: np.ndarray,
        sigma: np.ndarray,
        mu: np.ndarray,
        S: np.ndarray,
        B: np.ndarray,
        w0: np.ndarray,
    ) -> np.ndarray:
        """L-BFGS-B refinement for non-quadratic priors (e.g. box_gauss).

        When ``max_dw < 2.0`` and ``basis_kind == 'piecewise_linear'``, the
        optimisation is reparametrised as
            v[0] = w[0],  v[j] = w[j] - w[j-1]  (j ≥ 1)
        so that w = cumsum(v).  L-BFGS-B box bounds on the increments
            v[j] ∈ (-max_dw, max_dw)
        then enforce the adjacent-difference restriction as a TRUE hard
        constraint with no extra gradient-call cost (analytical gradient
        via the chain rule: g_v = reverse-cumsum(g_w)).
        """
        prior = self.prior
        inv_sig2 = 1.0 / (sigma * sigma)
        n = int(w0.size)

        def _obj(w: np.ndarray) -> float:
            yhat = (1.0 + (B @ w) * S) * mu
            r = y - yhat
            return float(np.sum((r * r) * inv_sig2)) + prior.chi2(w)

        # ---- reparametrised path (increment space, true hard constraint) ----
        use_reparam = (
            isinstance(prior, BoxGaussPrior)
            and self.max_dw < 2.0
            and self.basis_kind == "piecewise_linear"
        )
        if use_reparam:
            max_dw = float(self.max_dw)
            first_lo = -1.0 if self.enforce_s_bounds else -2.0
            first_hi = 1.0 if self.enforce_s_bounds else 2.0
            v_bounds = [(first_lo, first_hi)] + [(-max_dw, max_dw)] * (n - 1)
            lo = np.array([b[0] for b in v_bounds])
            hi = np.array([b[1] for b in v_bounds])

            def _obj_grad_v(v: np.ndarray) -> tuple[float, np.ndarray]:
                w = np.cumsum(v)
                A = B * (mu * S)[:, None]
                r = y - mu - A @ w
                data = float(np.sum((r * r) * inv_sig2))
                g_w = -2.0 * (A.T @ (r * inv_sig2))
                excess = np.maximum(np.abs(w) - prior.width, 0.0)
                g_w += 2.0 * excess * np.sign(w) / (prior.tau * prior.tau)
                # Chain rule: ∂f/∂v[j] = Σ_{k≥j} ∂f/∂w[k]
                g_v = np.cumsum(g_w[::-1])[::-1]
                return data + prior.chi2(w), g_v

            candidates: list[tuple[float, np.ndarray]] = []
            for w_try in (np.zeros(n, dtype=float), np.asarray(w0, dtype=float)):
                try:
                    candidates.append((_obj(w_try), w_try.copy()))
                except (FloatingPointError, ValueError):
                    continue
            try:
                v0 = np.concatenate([[w0[0]], np.diff(np.asarray(w0, dtype=float))])
                v0 = np.clip(v0, lo + 1e-9, hi - 1e-9)
                res = minimize(
                    _obj_grad_v, x0=v0, method="L-BFGS-B",
                    jac=True, bounds=v_bounds,
                )
                if np.all(np.isfinite(res.x)):
                    w_res = np.cumsum(res.x)
                    candidates.append((_obj(w_res), w_res))
            except (FloatingPointError, ValueError, RuntimeError):
                pass
            if not candidates:
                return np.zeros(n, dtype=float)
            best = min(candidates, key=lambda t: t[0])
            return np.asarray(best[1], dtype=float)

        # ---- original paths (analytical gradient or slow numerical) --------
        def _obj_grad_box(w: np.ndarray) -> tuple[float, np.ndarray]:
            A = B * (mu * S)[:, None]
            r = y - mu - A @ w
            data = float(np.sum((r * r) * inv_sig2))
            grad = -2.0 * (A.T @ (r * inv_sig2))
            excess = np.maximum(np.abs(w) - prior.width, 0.0)
            grad += 2.0 * excess * np.sign(w) / (prior.tau * prior.tau)
            return data + prior.chi2(w), grad

        lbfgsb_bounds = [(-1.0, 1.0)] * n if self.enforce_s_bounds else None
        candidates2: list[tuple[float, np.ndarray]] = []
        for w_try in (np.zeros_like(w0), np.asarray(w0, dtype=float)):
            if self.enforce_s_bounds:
                w_try = np.clip(w_try, -1.0 + 1e-6, 1.0 - 1e-6)
            try:
                candidates2.append((_obj(w_try), w_try))
            except (FloatingPointError, ValueError):
                continue
        try:
            if isinstance(prior, BoxGaussPrior) and prior.smoothness is None:
                res = minimize(
                    _obj_grad_box, x0=w0, method="L-BFGS-B",
                    jac=True, bounds=lbfgsb_bounds,
                )
            else:
                res = minimize(
                    _obj, x0=w0, method="L-BFGS-B", bounds=lbfgsb_bounds,
                )
            if np.all(np.isfinite(res.x)):
                candidates2.append((float(res.fun), np.asarray(res.x, float)))
        except (FloatingPointError, ValueError, RuntimeError):
            pass
        if not candidates2:
            return np.zeros_like(w0)
        best2 = min(candidates2, key=lambda t: t[0])
        return np.asarray(best2[1], dtype=float)


__all__ = ["NuisanceModel"]
