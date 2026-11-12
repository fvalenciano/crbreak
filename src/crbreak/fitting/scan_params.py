"""2-D parameter-space scan in ``(Ebreak, dgamma)`` for an SBPL fit.

For each grid cell ``(Ebreak_i, dgamma_j)`` we refit SBPL with those two
parameters frozen via :class:`crbreak.fitting.models.FrozenSpectralModel`
(``K`` and ``gamma1`` stay free per window) and store the resulting
per-window chi² sum.  ``Δχ²`` relative to the **joint free-fit** chi²
is what feeds the standard 2-D contour plots.

Two scoring paths share the same scaffolding:

* **Classical** (default): each cell sums ``Chi2Fit`` chi² across N
  independent per-window classical fits.
* **Profiled**: pass an *unbound* :class:`NuisanceModel` via
  ``nuisance_config=...``.  Each cell does **one** joint
  :func:`scipy.optimize.minimize` over
  ``[K_0, gamma1_0, ..., K_{N-1}, gamma1_{N-1}]`` with ``Ebreak`` /
  ``dgamma`` frozen; the inner ``w`` is profiled per window inside
  ``bound.chi2(...)`` (and cached on each bound NM as ``last_w`` --
  legacy ``w_cache``).
* **Profiled, common nuisance** (``nm.share_nuisance=True``,
  ``nm.w_shared`` already set by a prior global fit): the inner
  ``w`` short-circuits to the *frozen* ``w_shared`` in every window
  (no inner solve), so each cell only minimises ``[K, gamma1]`` per
  window with the global deformation fixed.  All windows share the
  **same** basis (built once via :meth:`NuisanceModel.for_global`,
  identical to the basis used by the global joint fit) so the
  per-cell chi² stays on the same scale as ``g_prof.chi2``.  The
  prior on ``w_shared`` is added back **once** per cell as a
  constant (it does not depend on ``[K, gamma1]``, so the minimum
  location is unchanged) -- this aligns the scan level with
  ``chi2_anchor = g_prof.chi2``, which already counted the prior
  once globally.

Statistical interpretation (large-N / Wilks, df=2 for joint contours)::

    Δχ² = 2.30   ->  68.3 %  (1 σ)
    Δχ² = 6.18   ->  95.4 %  (2 σ)
    Δχ² = 11.83  ->  99.73 % (3 σ)

(See :func:`scipy.stats.chi2.ppf(p, df=2)`.)

Implementation notes
--------------------
- Parallelism is **by rows**: one worker = one ``dgamma`` value swept
  across all ``Ebreak`` values, with a *snake* warm-start centred on
  the middle of ``Ebreak_grid`` (assumed to be the joint best-fit
  cell when callers build the grid that way).
- Per-window state (``E, y, sigma``, the unfrozen SBPL and -- in
  profiled mode -- the bound :class:`NuisanceModel`s) is published once
  via ``Pool(initializer=_pool_init, initargs=(state,))``.
- The initial parameter vector is, in priority order:
  ``init_per_window`` (typically the per-window block of a global
  profiled fit) → the per-window PL ``popt`` (``warm_start_from``).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import chi2 as _chi2_dist

from .chi2 import Chi2Fit
from .models import FrozenSpectralModel, SpectralModel
from .nuisance import NuisanceModel


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------


@dataclass
class ChiSquareScan:
    """Outcome of a 2-D ``(Ebreak, dgamma)`` chi² scan."""

    Ebreak_grid: np.ndarray  # 1-D, length nE
    dgamma_grid: np.ndarray  # 1-D, length nDg
    chi2_grid: np.ndarray  # 2-D, shape (nE, nDg)
    fail_grid: np.ndarray  # 2-D bool, True where the cell fit failed
    chi2_anchor: float | None = None  # ``g.chi2`` of the joint free fit
    elapsed_s: float = 0.0
    label: str = ""
    # Diagnostic histogram of failure modes across all cells, e.g.
    # ``{"non_finite": 39}`` or ``{"<class 'ValueError'>: ...": 7}``.
    # Empty when no cell failed.
    fail_reasons: dict = None  # populated by scan_chi2_sbpl

    @property
    def shape(self) -> tuple[int, int]:
        return self.chi2_grid.shape

    @property
    def n_cells(self) -> int:
        return int(self.chi2_grid.size)

    @property
    def n_fail(self) -> int:
        return int(self.fail_grid.sum())

    @property
    def chi2_min(self) -> float:
        finite = self.chi2_grid[np.isfinite(self.chi2_grid)]
        return float(finite.min()) if finite.size else float("nan")

    @property
    def best_idx(self) -> tuple[int, int]:
        masked = np.where(np.isfinite(self.chi2_grid), self.chi2_grid, np.inf)
        idx = np.unravel_index(int(np.argmin(masked)), self.chi2_grid.shape)
        return int(idx[0]), int(idx[1])

    @property
    def best_cell(self) -> tuple[float, float]:
        i, j = self.best_idx
        return float(self.Ebreak_grid[i]), float(self.dgamma_grid[j])

    @property
    def delta_chi2_grid_vs_anchor(self) -> np.ndarray | None:
        if self.chi2_anchor is None:
            return None
        return self.chi2_grid - float(self.chi2_anchor)

    @property
    def delta_chi2_grid_vs_grid(self) -> np.ndarray:
        return self.chi2_grid - self.chi2_min

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.Ebreak_grid, self.dgamma_grid, indexing="ij")

    @staticmethod
    def confidence_levels(
        probs: Sequence[float] = (0.683, 0.90, 0.954, 0.99, 0.9973),
        df: int = 2,
    ) -> dict[float, float]:
        return {float(p): float(_chi2_dist.ppf(p, df=df)) for p in probs}

    def __repr__(self) -> str:
        Eb_min, dg_min = self.best_cell
        anchor = "None" if self.chi2_anchor is None else f"{self.chi2_anchor:.2f}"
        reasons = ""
        if self.fail_reasons:
            top = sorted(
                self.fail_reasons.items(), key=lambda kv: kv[1], reverse=True
            )[:3]
            reasons = ", reasons={" + ", ".join(
                f"{k}: {v}" for k, v in top
            ) + "}"
        return (
            f"ChiSquareScan(nE={len(self.Ebreak_grid)}, "
            f"ndg={len(self.dgamma_grid)}, "
            f"chi2_min={self.chi2_min:.2f} @ Eb={Eb_min:.2f}, dg={dg_min:.3f}, "
            f"anchor={anchor}, fails={self.n_fail}/{self.n_cells}, "
            f"elapsed={self.elapsed_s:.1f}s{reasons})"
        )


# ----------------------------------------------------------------------
# Worker (one row of the grid).  Module-level so multiprocessing can
# pickle it.  State is published once via Pool(initializer=...).
# ----------------------------------------------------------------------

_WORKER_STATE: dict = {}


def _pool_init(state: dict) -> None:
    _WORKER_STATE.clear()
    _WORKER_STATE.update(state)


def _make_cell_evaluator(s: dict, dg: float):
    """Return ``_eval_cell(j, p_warm) -> (chi2, p_used, ok)`` bound to
    one ``dgamma`` value.

    The closure dispatches on ``s["mode"]`` only **once per row**, not
    once per cell, and the resulting per-cell call is identical between
    classical and profiled paths (same signature, same return tuple),
    which is what lets a single snake loop drive both.
    """
    Es, ys, sigmas = s["Es"], s["ys"], s["sigmas"]
    nwin = len(Es)
    n_free = int(s["n_free"])
    model = s["model_sbpl"]

    if s["mode"] == "classical":

        def _eval_cell(j: int, p_warm: np.ndarray):
            frozen = FrozenSpectralModel(
                model, Ebreak=float(s["Ebreak_grid"][j]), dgamma=dg
            )
            popt = np.empty_like(p_warm)
            chi2_total = 0.0
            for i in range(nwin):
                try:
                    r = Chi2Fit(frozen, solver=s["solver"]).fit(Es[i], ys[i], sigmas[i], p0=p_warm[i])
                except Exception as exc:
                    return float("nan"), p_warm, False, f"{type(exc).__name__}: {exc!s:.80s}"
                popt[i] = np.asarray(r.popt, dtype=float)
                chi2_total += float(r.chi2)
            if not np.isfinite(chi2_total):
                return float("nan"), p_warm, False, "non_finite_chi2"
            return chi2_total, popt, True, None

        return _eval_cell

    bound_nms = s["bound_nms"]
    joint_bounds = s["joint_bounds"]
    max_iter = int(s["solver"].profiled_max_iter)
    # Common-nuisance constant: ``g_prof.chi2`` from the joint fit
    # already includes the prior chi² evaluated *once* on
    # ``w_shared``, but ``NuisanceModel.chi2`` deliberately returns
    # the data-only conditional statistic in shared mode (so it can
    # be summed N times without double-counting the prior).  We add
    # the constant prior term once per cell to keep the scan on the
    # same level as ``chi2_anchor=g_prof.chi2``; it does not depend
    # on (K, gamma1), so the minimum location is unchanged.
    prior_const = float(s["prior_const"])

    def _eval_cell(j: int, p_warm: np.ndarray):
        frozen = FrozenSpectralModel(
            model, Ebreak=float(s["Ebreak_grid"][j]), dgamma=dg
        )
        cell_bounds = [bound_nms[i].with_spectral_model(frozen) for i in range(nwin)]

        def _chi2_total(v: np.ndarray) -> float:
            tot = 0.0
            for i in range(nwin):
                pars = v[i * n_free : (i + 1) * n_free]
                tot += float(cell_bounds[i].chi2(Es[i], ys[i], sigmas[i], *pars))
                if not np.isfinite(tot):
                    return float("inf")
            return tot

        try:
            res = minimize(
                _chi2_total,
                x0=np.asarray(p_warm, dtype=float).reshape(-1),
                method="L-BFGS-B",
                bounds=joint_bounds,
                options={"maxiter": max_iter},
            )
            chi2_opt = float(_chi2_total(res.x)) + prior_const
        except Exception as exc:
            return float("nan"), p_warm, False, f"{type(exc).__name__}: {exc!s:.80s}"
        if not np.isfinite(chi2_opt):
            return float("nan"), p_warm, False, "non_finite_chi2"
        return (
            chi2_opt,
            np.asarray(res.x, dtype=float).reshape(nwin, n_free),
            True,
            None,
        )

    return _eval_cell


def _scan_row(args: tuple) -> tuple[int, np.ndarray, np.ndarray, dict]:
    """Sweep one ``dgamma`` row across ``Ebreak_grid`` with snake
    warm-start anchored at the middle of the grid.

    Returns
    -------
    idg, chi2_row, fail_row, reasons
        ``reasons`` is a ``{tag: count}`` histogram of failure modes
        seen on this row (e.g. ``{"non_finite_chi2": 1}``).
    """
    idg, dg = args
    s = _WORKER_STATE
    nE = len(s["Ebreak_grid"])
    chi2_row = np.full(nE, np.nan, dtype=float)
    fail_row = np.zeros(nE, dtype=bool)
    reasons: dict[str, int] = {}

    eval_cell = _make_cell_evaluator(s, dg)
    seed_idx = nE // 2
    p_init = np.asarray(s["init_per_window"], dtype=float)

    def _bump(tag: str | None) -> None:
        if tag is None:
            return
        reasons[tag] = reasons.get(tag, 0) + 1

    c, p_used, ok, why = eval_cell(seed_idx, p_init)
    chi2_row[seed_idx] = c
    fail_row[seed_idx] = not ok
    if not ok:
        _bump(why)
    p_anchor = p_used if ok else p_init

    for sweep in (range(seed_idx + 1, nE), range(seed_idx - 1, -1, -1)):
        p_warm = p_anchor
        for j in sweep:
            c, p_used, ok, why = eval_cell(j, p_warm)
            chi2_row[j] = c
            fail_row[j] = not ok
            if not ok:
                _bump(why)
            if ok:
                p_warm = p_used

    return idg, chi2_row, fail_row, reasons


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def _resolve_n_workers(n_workers: int | None) -> int:
    if n_workers is None:
        return max(1, (os.cpu_count() or 1) - 1)
    return max(1, int(n_workers))


def scan_chi2_sbpl(
    fitting,
    model_sbpl: SpectralModel,
    *,
    Ebreak_grid: Sequence[float],
    dgamma_grid: Sequence[float],
    chi2_anchor: float | None = None,
    warm_start_from: str | None = "PL",
    nuisance_config: NuisanceModel | None = None,
    init_per_window: Sequence[Sequence[float]] | np.ndarray | None = None,
    n_workers: int | None = None,
    chunksize: int = 1,
    print_every: int = 0,
    label: str = "scan",
) -> ChiSquareScan:
    """Grid-scan SBPL chi² over ``(Ebreak, dgamma)``.

    Parallelises **by rows** (one worker per ``dgamma``) with a snake
    warm-start centred on the middle of ``Ebreak_grid``.

    Parameters
    ----------
    fitting
        :class:`Fitting` instance with ``window_fits`` already populated.
    model_sbpl
        Unfrozen SBPL.  In profiled mode, must match
        ``nuisance_config.spectral_model``.
    Ebreak_grid, dgamma_grid
        1-D sequences of values to scan.  The grid centre is used as
        the snake-warm-start anchor, so build it centred on the joint
        best-fit ``(Ebreak, dgamma)`` for cleanest contours.
    chi2_anchor
        Reference chi² for ``delta_chi2_grid_vs_anchor`` (typically the
        joint free-fit ``g.chi2`` -- profiled with a profiled scan,
        classical with a classical scan).
    warm_start_from
        Per-window fit name whose ``popt`` is the **fallback** initial
        guess if ``init_per_window`` is not provided.  ``"PL"`` by
        default; pass ``None`` to disable.
    nuisance_config
        If given, switches to profiled chi² (single joint ``minimize``
        per cell) and uses the per-window bound :class:`NuisanceModel`s
        for the inner ``w`` profile.
    init_per_window
        Optional ``(nwin, n_free)`` array of initial ``[K, gamma1]``
        per window -- typically the per-window free block of a global
        profiled fit (so the snake starts at the joint best-fit).
    n_workers, chunksize, print_every, label
        Standard parallelisation knobs.  ``print_every`` here counts
        **rows**.
    """
    Ebreak_grid = np.asarray(Ebreak_grid, dtype=float).ravel()
    dgamma_grid = np.asarray(dgamma_grid, dtype=float).ravel()
    nE, ndg = len(Ebreak_grid), len(dgamma_grid)
    if nE < 1 or ndg < 1:
        raise ValueError(f"Empty grid: nE={nE}, ndg={ndg}")
    if not getattr(fitting, "window_fits", None):
        raise RuntimeError(
            "fitting.window_fits is empty; call Fitting.fit_per_window(...) first."
        )
    if nuisance_config is not None and not isinstance(nuisance_config, NuisanceModel):
        raise TypeError(
            f"nuisance_config must be a NuisanceModel, got {type(nuisance_config)!r}."
        )
    if (
        nuisance_config is not None
        and nuisance_config.share_nuisance
        and nuisance_config.w_shared is None
    ):
        raise RuntimeError(
            "scan_chi2_sbpl: NuisanceModel(share_nuisance=True) requires "
            "fit_global(nm) to run first so w_shared can be propagated.\n"
            "Workflow: fit_global(nm) then scan(nm, ...)."
        )

    # SBPL with shared (Ebreak, dgamma) -> free per window: [K, gamma1]
    # (+ s when model.sbpl_s_fixed is null).
    n_free = max(0, len(model_sbpl.param_names) - 2)
    if n_free <= 0:
        raise ValueError(
            f"model_sbpl has no free parameters left after freezing "
            f"(Ebreak, dgamma): param_names={model_sbpl.param_names!r}"
        )

    # Per-window data (+ bound NMs in profiled mode) and the per-window
    # warm-start matrix (init_per_window > PL popt > flat 1.0).
    keys = tuple(fitting.window_fits.keys())
    nwin = len(keys)
    Es: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    bound_nms: list[NuisanceModel | None] = []
    init = np.ones((nwin, n_free), dtype=float)
    if init_per_window is not None:
        init = np.asarray(init_per_window, dtype=float).reshape(nwin, n_free).copy()

    # In shared-nuisance mode we MUST use the same basis (shared
    # ``x_nuis``) that the global fit used, otherwise ``w_shared``
    # gets re-interpreted on each window's local basis and the
    # per-cell chi² is no longer comparable to ``g_prof.chi2``
    # (the contour drifts away from the joint best-fit).  Build all
    # bound NMs once via ``for_global`` so every window shares the
    # same nodes / prior / ``w_shared``.
    if nuisance_config is not None and nuisance_config.share_nuisance:
        wfs = [fitting.window_fits[k] for k in keys]
        bounds_global, arrays_global = nuisance_config.for_global(wfs)
    else:
        bounds_global = None
        arrays_global = None

    for i, k in enumerate(keys):
        wf = fitting.window_fits[k]
        if nuisance_config is None:
            Es.append(np.asarray(wf.E_GeV, dtype=float))
            ys.append(np.asarray(wf.y, dtype=float))
            sigmas.append(np.asarray(wf.sigma, dtype=float))
            bound_nms.append(None)
        elif bounds_global is not None:
            nm_bound = bounds_global[i]
            arr = arrays_global[i]
            Es.append(arr["E"])
            ys.append(arr["y"])
            sigmas.append(arr["sigma"])
            bound_nms.append(nm_bound)
        else:
            nm_bound, arr = nuisance_config.for_window(wf)
            Es.append(arr["E"])
            ys.append(arr["y"])
            sigmas.append(arr["sigma"])
            bound_nms.append(nm_bound)
        if init_per_window is None and warm_start_from and warm_start_from in wf.fits:
            v = np.asarray(wf.fits[warm_start_from].popt, dtype=float).ravel()
            init[i, : min(v.size, n_free)] = v[:n_free]

    # Joint bounds for the profiled minimize (replicated per window), by
    # name: the free parameters of the cell model, whatever their position.
    lo, hi = FrozenSpectralModel(
        model_sbpl, Ebreak=float(Ebreak_grid[0]), dgamma=float(dgamma_grid[0])
    ).default_bounds()
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    joint_bounds = [(float(lo[k]), float(hi[k])) for _ in range(nwin) for k in range(n_free)]
    # A PL warm start fills only [K, gamma1]: keep the rest (e.g. s) inside its bounds.
    init = np.clip(init, lo, hi)

    # In shared mode the per-window chi² is data-only; we add the
    # frozen prior(w_shared) once per cell so the level matches
    # ``g_prof.chi2`` (which counted it once globally).  In every
    # other mode this constant is zero.
    prior_const = 0.0
    if (
        nuisance_config is not None
        and nuisance_config.share_nuisance
        and nuisance_config.w_shared is not None
        and bounds_global is not None
    ):
        prior_r = bounds_global[0].prior.augmented_residuals(
            np.asarray(nuisance_config.w_shared, dtype=float)
        )
        prior_const = float(np.sum(prior_r * prior_r)) if prior_r.size else 0.0

    state = {
        "Es": tuple(Es),
        "ys": tuple(ys),
        "sigmas": tuple(sigmas),
        "model_sbpl": model_sbpl,
        "mode": ("profiled" if nuisance_config is not None else "classical"),
        "bound_nms": tuple(bound_nms),
        "solver": fitting.solver,
        "init_per_window": init,
        "joint_bounds": joint_bounds,
        "n_free": n_free,
        "Ebreak_grid": Ebreak_grid,
        "prior_const": prior_const,
    }

    chi2_grid = np.full((nE, ndg), np.nan, dtype=float)
    fail_grid = np.zeros((nE, ndg), dtype=bool)
    fail_reasons: dict[str, int] = {}
    nw = _resolve_n_workers(n_workers)
    rows = [(j, float(dgamma_grid[j])) for j in range(ndg)]
    t0 = time.perf_counter()
    n_done = 0

    def _consume(item: tuple) -> None:
        nonlocal n_done
        idg, chi2_row, fail_row, reasons = item
        chi2_grid[:, idg] = chi2_row
        fail_grid[:, idg] = fail_row
        for k, v in reasons.items():
            fail_reasons[k] = fail_reasons.get(k, 0) + int(v)
        n_done += 1
        if print_every and (n_done % print_every == 0 or n_done == ndg):
            el = time.perf_counter() - t0
            rate = f"{n_done / el:.2f} rows/s" if el > 0 else "n/a"
            print(
                f"[scan {label}] rows {n_done}/{ndg} "
                f"fails={int(fail_grid.sum())} elapsed={el:.1f}s ({rate})",
                flush=True,
            )

    if nw <= 1:
        _pool_init(state)
        for r in rows:
            _consume(_scan_row(r))
    else:
        ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
        with ctx.Pool(processes=nw, initializer=_pool_init, initargs=(state,)) as pool:
            for item in pool.imap_unordered(_scan_row, rows, chunksize=chunksize):
                _consume(item)

    return ChiSquareScan(
        Ebreak_grid=Ebreak_grid,
        dgamma_grid=dgamma_grid,
        chi2_grid=chi2_grid,
        fail_grid=fail_grid,
        chi2_anchor=(None if chi2_anchor is None else float(chi2_anchor)),
        elapsed_s=time.perf_counter() - t0,
        label=label,
        fail_reasons=fail_reasons,
    )


# ----------------------------------------------------------------------
# 1-D profile-likelihood errors via two degenerate-axis scans
# ----------------------------------------------------------------------


def _bracket_delta(
    grid: np.ndarray,
    chi2_row: np.ndarray,
    *,
    x_best: float,
    chi2_min: float,
    delta: float,
) -> tuple[float, float]:
    """Linearly-interpolated ``Δχ² = delta`` crossings on each side of
    ``x_best``.  Returns ``(x_lo, x_hi)`` as **absolute** grid values
    (not offsets).  Either side is ``nan`` when the grid does not span
    the crossing on that side.
    """
    g = np.asarray(grid, dtype=float)
    d = np.asarray(chi2_row, dtype=float) - float(chi2_min)
    finite = np.isfinite(d)
    if not np.any(finite):
        return float("nan"), float("nan")

    # If x_best is not already a grid point, insert it with d=0 so the
    # crossing search always starts from the anchor (Δχ²=0) rather than
    # from whatever grid point happens to be closest.  Without this, a
    # coarse grid can place the first left/right point above delta=1,
    # causing _cross to miss the bracket entirely and return nan.
    x_best_f = float(x_best)
    if not np.any(np.isclose(g, x_best_f, rtol=0.0, atol=1e-10)):
        g = np.concatenate([g, [x_best_f]])
        d = np.concatenate([d, [0.0]])
        order = np.argsort(g)
        g = g[order]
        d = d[order]

    def _cross(g_seq: np.ndarray, d_seq: np.ndarray) -> float:
        for k in range(1, g_seq.size):
            d0, d1 = d_seq[k - 1], d_seq[k]
            if not (np.isfinite(d0) and np.isfinite(d1)):
                continue
            if (d0 - delta) * (d1 - delta) <= 0.0 and d1 != d0:
                t = (delta - d0) / (d1 - d0)
                return float(g_seq[k - 1] + t * (g_seq[k] - g_seq[k - 1]))
        return float("nan")

    left = g <= x_best_f
    right = g >= x_best_f
    x_lo = _cross(g[left][::-1], d[left][::-1])
    x_hi = _cross(g[right], d[right])
    return x_lo, x_hi


def _zero_compatible_lower_bound(
    pname: str,
    x_best: float,
    x_lo: float,
    grid: np.ndarray,
    chi2_row: np.ndarray,
) -> float:
    """Treat an open lower ``Ebreak`` profile as compatible with zero.

    In weak-break windows the 1-D profile in ``Ebreak`` can stay below
    ``Delta chi2 = 1`` all the way down to the lowest scanned energy.
    Then the lower 1-sigma crossing is formally open.  For reporting and
    plotting, encode this as an absolute lower limit of zero, i.e. a
    lower error equal to the best-fit value.
    """
    if pname != "Ebreak" or np.isfinite(x_lo) or not np.isfinite(x_best):
        return x_lo
    g = np.asarray(grid, dtype=float)
    row = np.asarray(chi2_row, dtype=float)
    finite_below = g[np.isfinite(g) & np.isfinite(row) & (g < float(x_best))]
    if finite_below.size == 0:
        return x_lo
    return 0.0


def profile_errors_1d_sbpl(
    fitting,
    model_sbpl: SpectralModel,
    *,
    Eb_best: float,
    dg_best: float,
    Eb_grid: Sequence[float],
    dg_grid: Sequence[float],
    chi2_anchor: float,
    delta: float,
    nuisance_config: NuisanceModel | None = None,
    init_per_window: Sequence[Sequence[float]] | np.ndarray | None = None,
    n_workers: int | None = None,
) -> dict:
    """Profile-likelihood ``Δχ² = delta`` errors on ``(Ebreak, dgamma)``.

    Reuses :func:`scan_chi2_sbpl` along each axis with the **other** one
    frozen at its joint best-fit value.  Each axis run produces a 1-D
    chi² profile that already has every per-window ``[K, gamma1]``
    re-minimised at every grid point (and, in shared-nuisance mode,
    keeps ``w_shared`` frozen at the global best-fit, exactly as in the
    2-D contour scan).  We then linearly interpolate the
    ``Δχ² = delta`` crossing on each side of the best-fit value, which
    yields **asymmetric** ±1σ errors by construction.

    Returns
    -------
    dict
        ``{"Ebreak": (best, lo, hi, scan), "dgamma": (best, lo, hi, scan)}``
        where ``lo``/``hi`` are absolute grid values and ``scan`` is the
        underlying :class:`ChiSquareScan` (kept for diagnostics).  ``lo``
        or ``hi`` is ``nan`` when the grid does not span the crossing,
        except that an open lower ``Ebreak`` side is reported as ``lo=0``
        to encode compatibility with no finite break scale.
    """
    sE = scan_chi2_sbpl(
        fitting, model_sbpl,
        Ebreak_grid=Eb_grid,
        dgamma_grid=np.array([dg_best], dtype=float),
        chi2_anchor=chi2_anchor,
        nuisance_config=nuisance_config,
        init_per_window=init_per_window,
        n_workers=n_workers,
        label="prof1d_Eb",
    )
    sD = scan_chi2_sbpl(
        fitting, model_sbpl,
        Ebreak_grid=np.array([Eb_best], dtype=float),
        dgamma_grid=dg_grid,
        chi2_anchor=chi2_anchor,
        nuisance_config=nuisance_config,
        init_per_window=init_per_window,
        n_workers=n_workers,
        label="prof1d_dg",
    )
    Eb_lo, Eb_hi = _bracket_delta(
        sE.Ebreak_grid, sE.chi2_grid[:, 0],
        x_best=Eb_best, chi2_min=float(chi2_anchor), delta=delta,
    )
    Eb_lo = _zero_compatible_lower_bound(
        "Ebreak", float(Eb_best), Eb_lo, sE.Ebreak_grid, sE.chi2_grid[:, 0],
    )
    dg_lo, dg_hi = _bracket_delta(
        sD.dgamma_grid, sD.chi2_grid[0, :],
        x_best=dg_best, chi2_min=float(chi2_anchor), delta=delta,
    )
    return {
        "Ebreak": (float(Eb_best), Eb_lo, Eb_hi, sE),
        "dgamma": (float(dg_best), dg_lo, dg_hi, sD),
    }


# --- Pool-init / worker for profile_errors_1d_window (per-cell parallel) -----
_PEW_STATE: dict | None = None


def _pew_init(state: dict) -> None:
    """Pool initializer: publish ``(bound, inner, E, y, sigma, solver,
    p0_by_param)`` in the worker's module-global ``_PEW_STATE`` so each
    cell job only needs to send ``(pname, val)``."""
    global _PEW_STATE
    _PEW_STATE = state


def _pew_cell(args: tuple[str, float]) -> float:
    """Worker: re-fit the profiled SBPL with ``pname`` frozen at ``val``,
    using the per-window best-fit ``[K, gamma1, ...]`` (with the frozen
    parameter dropped) as the warm-start ``p0``.  Returns ``chi2``
    (``nan`` on failure)."""
    from .chi2 import ProfiledChi2Fit

    pname, val = args
    s = _PEW_STATE
    assert s is not None
    bound = s["bound"]
    inner = s["inner"]
    frozen = FrozenSpectralModel(inner, **{pname: float(val)})
    cell_nm = bound.with_spectral_model(frozen)
    p0_cell = s["p0_by_param"].get(pname)
    try:
        r = ProfiledChi2Fit(cell_nm, solver=s["solver"]).fit(
            s["E"], s["y"], s["sigma"], p0=p0_cell,
        )
        c = float(r.chi2)
        return c if np.isfinite(c) else float("nan")
    except Exception:
        return float("nan")


def profile_errors_1d_window(
    wf,
    name: str,
    *,
    params: Sequence[str],
    grids: dict,
    delta: float,
    solver,
    n_workers: int | None = None,
    return_curves: bool = False,
) -> dict:
    """Per-window profile-likelihood ``Δχ² = delta`` errors.

    Reuses the bound :class:`NuisanceModel` of ``wf.fits[name]`` (so the
    basis / prior / ``Ssys_frac`` are exactly those of the window's own
    profiled SBPL fit), wraps it in a :class:`FrozenSpectralModel` to
    freeze one parameter at a grid value, and runs a fresh
    :class:`ProfiledChi2Fit` at every cell **warm-started from the
    per-window best-fit** ``[K, gamma1, ...]`` (with the frozen parameter
    dropped).  ``w`` is reprofiled cell by cell, exactly as the profiled
    best-fit does -- no Hessian assumption.

    ``params`` can name any spectral parameter (e.g. ``gamma1``) and, for
    an SBPL, the derived high-energy slope ``gamma2 = gamma1 - dgamma``.

    Robustness
    ----------
    The Δχ² anchor is taken as ``min(wf.fits[name].chi2,
    nanmin(chi2_row))`` so that, in windows where the per-window safety
    net selected ``θ_classical`` (i.e. ``fit.chi2`` overestimates the true
    profiled minimum), the scan ``Δχ²`` is still ``≥ 0`` and brackets the
    ``Δχ² = delta`` crossing instead of polluting it with negative
    offsets.

    Parallelism
    -----------
    When ``n_workers`` is ``None`` or ``<= 1`` the cells run serially.
    Otherwise a ``multiprocessing.Pool`` distributes **cells** (not
    windows): the bound NM, ``E, y, sigma`` and the warm-start ``p0`` are
    published once via ``Pool(initializer=...)`` and each worker only
    ships back the cell chi².

    Returns
    -------
    dict
        ``{param_name: (best, lo, hi)}`` with absolute grid values
        (``nan`` when the grid does not span the ``Δχ² = delta`` crossing,
        except that an open lower ``Ebreak`` side is reported as ``lo=0``).
        When ``return_curves=True`` the dict also contains a key
        ``"_curves"`` mapping each param name to ``(grid, chi2_row,
        chi2_anchor)``.
    """
    from .chi2 import ProfiledChi2Fit  # noqa: F401  (kept for the serial path import-cost balance)

    fit = wf.fits[name]
    bound = fit.model
    if not isinstance(bound, NuisanceModel) or not bound.is_bound:
        raise TypeError(
            f"profile_errors_1d_window: {name!r} is not a profiled fit "
            "(model is not a bound NuisanceModel)."
        )
    inner = bound.spectral_model
    chi2_anchor_fit = float(fit.chi2)
    E = np.asarray(wf.E_GeV, dtype=float)
    y = np.asarray(wf.y, dtype=float)
    sigma = np.asarray(wf.sigma, dtype=float)
    popt_full = np.asarray(fit.popt, dtype=float)
    popt_by_name = dict(zip(inner.param_names, popt_full))

    nw = max(1, int(n_workers)) if n_workers else 1

    # Build cell job list across all requested params.  Per-param warm-start
    # vector: drop the index of the parameter that the cell will freeze.
    jobs: list[tuple[str, float]] = []
    grids_per: dict[str, np.ndarray] = {}
    x_best_per: dict[str, float] = {}
    p0_by_param: dict[str, np.ndarray] = {}
    for pname in params:
        if pname == "gamma2" and {"gamma1", "dgamma"} <= set(inner.param_names):
            # Derived slope: FrozenSpectralModel(gamma2=...) drops gamma1.
            x_best_per[pname] = float(popt_by_name["gamma1"] - popt_by_name["dgamma"])
            dropped = "gamma1"
        elif pname in inner.param_names:
            x_best_per[pname] = float(popt_by_name[pname])
            dropped = pname
        else:
            continue
        if pname not in grids:
            raise ValueError(f"profile_errors_1d_window: missing grid for {pname!r}.")
        g = np.asarray(grids[pname], dtype=float).ravel()
        grids_per[pname] = g
        free_mask = [n != dropped for n in inner.param_names]
        p0_by_param[pname] = popt_full[free_mask]
        for v in g:
            jobs.append((pname, float(v)))

    if not jobs:
        return {} if not return_curves else {"_curves": {}}

    # Run all cells (serial or pool over cells, never over windows).
    state = {
        "bound": bound,
        "inner": inner,
        "E": E,
        "y": y,
        "sigma": sigma,
        "solver": solver,
        "p0_by_param": p0_by_param,
    }
    if nw > 1 and len(jobs) > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=nw, initializer=_pew_init, initargs=(state,)) as pool:
            chi2_flat = pool.map(_pew_cell, jobs)
    else:
        _pew_init(state)
        chi2_flat = [_pew_cell(j) for j in jobs]
        # Clear the module-global so a subsequent serial pass starts clean.
        globals()["_PEW_STATE"] = None

    # Re-shape into per-param chi² rows, re-anchor and bracket the crossings.
    out: dict[str, tuple[float, float, float]] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    cursor = 0
    for pname in params:
        if pname not in grids_per:
            continue
        g = grids_per[pname]
        row = np.asarray(chi2_flat[cursor : cursor + g.size], dtype=float)
        cursor += g.size
        # Re-anchor to ``min(fit.chi2, nanmin(scan))`` so safety-net biased
        # ``fit.chi2`` cannot turn the Δχ² profile negative.
        finite = row[np.isfinite(row)]
        chi2_anchor = chi2_anchor_fit
        if finite.size:
            chi2_anchor = min(chi2_anchor_fit, float(np.min(finite)))
        x_lo, x_hi = _bracket_delta(
            g, row, x_best=x_best_per[pname], chi2_min=chi2_anchor, delta=delta,
        )
        x_lo = _zero_compatible_lower_bound(pname, x_best_per[pname], x_lo, g, row)
        out[pname] = (x_best_per[pname], x_lo, x_hi)
        if return_curves:
            curves[pname] = (g.copy(), row.copy(), float(chi2_anchor))

    if return_curves:
        out["_curves"] = curves  # type: ignore[assignment]
    return out
