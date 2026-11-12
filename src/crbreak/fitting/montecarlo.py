"""Parametric-bootstrap Monte-Carlo p-values for PL vs an alternative model.

Two scopes only:

* :func:`mc_pvalue_window` -- single window: ``H0=PL`` refit vs an alternative
  model refit on one window's pseudo-data.
* :func:`mc_pvalue_global` -- joint over ``N`` windows: ``H0=PL`` refit per
  window vs a joint refit of the alternative.  Classical alternatives use
  :class:`~crbreak.fitting.global_chi2.GlobalChi2Fit`; profiled nuisance
  alternatives use :class:`~crbreak.fitting.global_chi2.ProfiledGlobalChi2Fit`.
  The alternative is always provided as a
  :class:`~crbreak.fitting.global_fit.GlobalFitResult`, which covers both the
  *free-shape* case (e.g. SBPL with ``Ebreak``/``dgamma`` shared across windows)
  and the *fixed-shape* case (SBPL with those parameters pinned via a
  :class:`~crbreak.fitting.models.FrozenSpectralModel` and ``shared=()``); the
  joint solver decouples mathematically in the latter, but the API stays
  uniform.

Both flavours share the same scaffolding: pseudo-data sampler, Pool driver
with reproducible :class:`numpy.random.SeedSequence` seeds, and the legacy
PDG-floored summary ``p = max(n_exceed/N, 1/(N+1))``.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import norm as _norm

from .chi2 import Chi2Fit, ProfiledChi2Fit
from .global_chi2 import GlobalChi2Fit, ProfiledGlobalChi2Fit
from .global_fit import GlobalFitResult
from .models import FrozenSpectralModel, SpectralModel
from .nuisance import BoxGaussPrior, NuisanceModel


# ----------------------------------------------------------------------
# Result container

@dataclass
class MCPValueResult:
    delta_obs: float
    p_value: float
    Z: float
    n_exceed: int
    n_kept: int
    n_fail: int
    N: int
    seed: int | None
    sample_theta: bool
    n_nonconverged: int = 0
    n_error: int = 0
    deltas: np.ndarray | None = None
    elapsed_s: float = 0.0
    label: str = ""
    p_is_upper_limit: bool = False

    def to_dict(self) -> dict:
        d = {
            "delta_obs": float(self.delta_obs),
            "p_value": float(self.p_value),
            "Z": float(self.Z),
            "n_exceed": int(self.n_exceed),
            "n_kept": int(self.n_kept),
            "n_fail": int(self.n_fail),
            "n_nonconverged": int(self.n_nonconverged),
            "n_error": int(self.n_error),
            "N": int(self.N),
            "seed": self.seed,
            "sample_theta": bool(self.sample_theta),
            "elapsed_s": float(self.elapsed_s),
            "label": self.label,
            "p_is_upper_limit": bool(self.p_is_upper_limit),
        }
        if self.deltas is not None:
            d["deltas"] = np.asarray(self.deltas, dtype=float)
        return d

    def __repr__(self) -> str:
        p_op = "<" if self.p_is_upper_limit else ""
        z_op = ">" if self.p_is_upper_limit else ""
        return (
            f"MCPValueResult({self.label}: delta_obs={self.delta_obs:.3f}, "
            f"p={p_op}{self.p_value:.3g}, Z={z_op}{self.Z:.2f}, "
            f"exceed={self.n_exceed}, kept={self.n_kept}/{self.N}, "
            f"fail={self.n_fail}, "
            f"nonconv={self.n_nonconverged}, error={self.n_error}, "
            f"elapsed={self.elapsed_s:.1f}s)"
        )


# ----------------------------------------------------------------------
# Pseudo-data generation

def sample_pseudo_y_pl(
    model_pl: SpectralModel,
    E: np.ndarray,
    sigma: np.ndarray,
    popt_pl: np.ndarray,
    pcov_pl: np.ndarray | None,
    *,
    rng: np.random.Generator,
    sample_theta: bool = True,
    w_ref: np.ndarray | None = None,
) -> np.ndarray:
    """Draw one pseudo-dataset under H0 = PL.

    With ``sample_theta=True`` (legacy default) ``theta`` is drawn from
    ``MVN(popt_pl, pcov_pl)``; otherwise it stays at ``popt_pl``.  Gaussian
    noise of width ``sigma`` is then added on top of the null prediction.  For
    a bound :class:`NuisanceModel`, the null prediction includes the fitted
    deformation ``w_ref``.
    """
    if sample_theta and pcov_pl is not None:
        try:
            theta = rng.multivariate_normal(popt_pl, pcov_pl, check_valid="ignore")
        except np.linalg.LinAlgError:
            theta = np.asarray(popt_pl, dtype=float)
    else:
        theta = np.asarray(popt_pl, dtype=float)
    E = np.asarray(E, dtype=float)
    if isinstance(model_pl, NuisanceModel) and model_pl.is_bound:
        w_use = model_pl.last_w if w_ref is None else np.asarray(w_ref, dtype=float)
        mu = model_pl.predict_deformed(E, tuple(theta), w=w_use)
    else:
        mu = model_pl(E, *theta)
    # in a future we could add here the * (1 + s * Ssys_frac) depending on s, different nuisance config
    return mu + rng.normal(0.0, np.asarray(sigma, dtype=float))


# ----------------------------------------------------------------------
# Workers --- module-level so multiprocessing can pickle them.
# State is published via Pool(initializer=_pool_init, initargs=(state,)).

_WORKER_STATE: dict = {}


def _pool_init(state: dict) -> None:
    _WORKER_STATE.clear()
    _WORKER_STATE.update(state)


def _draw_mvn_or_mean(
    rng: np.random.Generator,
    mean: np.ndarray,
    cov: np.ndarray | None,
) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    if cov is None:
        return mean.copy()
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (mean.size, mean.size) or not np.all(np.isfinite(cov)):
        return mean.copy()
    try:
        draw = rng.multivariate_normal(mean, cov, check_valid="ignore")
    except (ValueError, np.linalg.LinAlgError):
        return mean.copy()
    return draw if np.all(np.isfinite(draw)) else mean.copy()


def _draw_mvn_box_reject_or_mean(
    rng: np.random.Generator,
    mean: np.ndarray,
    cov: np.ndarray | None,
    *,
    box_width: float | None,
    max_redraw: int = 10_000,
) -> np.ndarray:
    """Draw from MVN, rejecting box-gauss common-nuisance draws outside |w|<=a."""
    mean = np.asarray(mean, dtype=float)
    if box_width is None:
        return _draw_mvn_or_mean(rng, mean, cov)
    a = float(box_width)
    if not np.isfinite(a) or a <= 0.0:
        return _draw_mvn_or_mean(rng, mean, cov)
    cov = None if cov is None else np.asarray(cov, dtype=float)
    if cov is None or cov.shape != (mean.size, mean.size) or not np.all(np.isfinite(cov)):
        return mean.copy()
    for _ in range(max(1, int(max_redraw))):
        try:
            draw = rng.multivariate_normal(mean, cov, check_valid="ignore")
        except (ValueError, np.linalg.LinAlgError):
            return mean.copy()
        draw = np.asarray(draw, dtype=float)
        if np.all(np.isfinite(draw)) and np.all(np.abs(draw) <= a):
            return draw
    raise RuntimeError("Failed to draw box-bounded shared nuisance within max_redraw attempts.")


def _fit_ok(result) -> bool:
    return bool(getattr(result, "converged", True))


def _toy_window(seed: int) -> tuple[float, str]:
    rng = np.random.default_rng(seed)
    s = _WORKER_STATE
    try:
        y_pseudo = sample_pseudo_y_pl(
            s["model_pl"],
            s["E"],
            s["sigma"],
            s["popt_pl"],
            s["pcov_pl"],
            rng=rng,
            sample_theta=s["sample_theta"],
            w_ref=s["w_pl_gen"],
        )
        if isinstance(s["model_pl"], NuisanceModel):
            fit_pl = ProfiledChi2Fit(s["model_pl"], solver=s["solver"]).fit(
                s["E"],
                y_pseudo,
                s["sigma"],
                p0=s["popt_pl"],
                warn_on_fail=False,
                compute_covariance=False,
            )
        else:
            fit_pl = Chi2Fit(s["model_pl"], solver=s["solver"]).fit(
                s["E"], y_pseudo, s["sigma"], p0=s["popt_pl"]
            )
        if not _fit_ok(fit_pl):
            return (float("nan"), "nonconverged")
        alt_p0 = s["popt_alt_warm"] if s["warm_start"] else None
        if isinstance(s["model_alt"], NuisanceModel):
            fit_alt = ProfiledChi2Fit(s["model_alt"], solver=s["solver"]).fit(
                s["E"],
                y_pseudo,
                s["sigma"],
                p0=alt_p0,
                warn_on_fail=False,
                compute_covariance=False,
            )
        else:
            fit_alt = Chi2Fit(s["model_alt"], solver=s["solver"]).fit(
                s["E"], y_pseudo, s["sigma"], p0=alt_p0
            )
        if not _fit_ok(fit_alt):
            return (float("nan"), "nonconverged")
        delta = float(fit_pl.chi2 - fit_alt.chi2)
        if not np.isfinite(delta):
            return (float("nan"), "error")
        return (delta, "ok")
    except Exception:
        return (float("nan"), "error")


def _toy_window_batch(task: tuple[int, int]) -> tuple[np.ndarray, int, int, int, int]:
    """Run a batch of window toys in one worker process.

    Returns ``(ok_deltas, n_attempted, n_fail, n_nonconverged, n_error)``.
    The parent process decides whether to retain ``ok_deltas`` or only use
    them to update the exceedance counter.
    """
    seed_root, n_draw = task
    ss = np.random.SeedSequence(int(seed_root))
    ok: list[float] = []
    n_fail = 0
    n_nonconverged = 0
    n_error = 0
    for child in ss.spawn(int(n_draw)):
        seed = int(child.generate_state(1)[0])
        d, status = _toy_window(seed)
        if status == "ok":
            ok.append(float(d))
        else:
            n_fail += 1
            if status == "nonconverged":
                n_nonconverged += 1
            else:
                n_error += 1
    return (
        np.asarray(ok, dtype=float),
        int(n_draw),
        int(n_fail),
        int(n_nonconverged),
        int(n_error),
    )


def _toy_global(seed: int) -> tuple[float, str]:
    """One global toy: per-window pseudo-PL → refit PL per window + joint alt.

    Required state keys
    -------------------
    ``keys`` : ``tuple[str, ...]``
        Window IDs in the layout order used by ``popt_alt_warm``.
    ``Es``, ``sigmas``, ``popt_pls``, ``pcov_pls`` : tuples of ``np.ndarray``
        Per-window arrays (same length as ``keys``).
    ``model_pls`` : tuple[:class:`SpectralModel`]
    ``model_alt`` : :class:`SpectralModel`
        Free SBPL or :class:`FrozenSpectralModel`-wrapped SBPL.
    ``shared_alt`` : ``tuple[str, ...]``
        Names of parameters shared across windows in the joint alt fit.
        ``("Ebreak", "dgamma")`` for the free case, ``()`` for fixed-shape
        (in which case :class:`GlobalChi2Fit` decouples the windows but
        still solves them in a single ``least_squares`` call).
    ``popt_alt_warm`` : ``np.ndarray | None``
        Warm-start vector for the joint alternative fit (classical:
        ``[free_1, free_2, ..., free_N, shared]``; common profiled:
        same spectral block plus the final ``w_shared`` block).
    ``alt_predictors`` : tuple[:class:`NuisanceModel`, ...]
        Bound per-window nuisance predictors, present only for profiled
        alternatives.
    ``sample_theta``, ``warm_start`` : ``bool``
    """
    rng = np.random.default_rng(seed)
    s = _WORKER_STATE
    try:
        keys = s["keys"]
        Es = s["Es"]
        sigmas = s["sigmas"]
        popt_pls = s["popt_pls"]
        pcov_pls = s["pcov_pls"]
        model_pls = s["model_pls"]
        model_alt = s["model_alt"]
        shared_alt = s["shared_alt"]
        sample_theta = s["sample_theta"]
        warm = s["warm_start"]

        w_global_gen = None
        if s.get("w_global_gen_mean") is not None:
            w_global_gen = _draw_mvn_box_reject_or_mean(
                rng,
                s["w_global_gen_mean"],
                s["w_global_gen_cov"],
                box_width=s.get("w_global_gen_box_width"),
                max_redraw=s.get("w_global_gen_box_max_redraw", 10_000),
            )

        y_pseudos = []
        for i in range(len(keys)):
            w_gen = w_global_gen if w_global_gen is not None else s["w_pl_gens"][i]
            y_pseudos.append(
                sample_pseudo_y_pl(
                    model_pls[i],
                    Es[i],
                    sigmas[i],
                    popt_pls[i],
                    pcov_pls[i],
                    rng=rng,
                    sample_theta=sample_theta,
                    w_ref=w_gen,
                )
            )

        # Per-window PL refits (warm-started at observed popt_pl).  If
        # the stored PL is profiled, keep the toy refit in the same
        # profiled likelihood family instead of silently falling back to
        # a classical Chi2Fit on the physical PL curve.
        chi2_pl_total = 0.0
        for i in range(len(keys)):
            model_pl = model_pls[i]
            if isinstance(model_pl, NuisanceModel):
                if w_global_gen is not None:
                    model_pl.w_shared = np.asarray(w_global_gen, dtype=float).copy()
                    model_pl.last_w = np.asarray(w_global_gen, dtype=float).copy()
                r = ProfiledChi2Fit(model_pl, solver=s["solver"]).fit(
                    Es[i],
                    y_pseudos[i],
                    sigmas[i],
                    p0=popt_pls[i],
                    warn_on_fail=False,
                    compute_covariance=False,
                )
            else:
                r = Chi2Fit(model_pl, solver=s["solver"]).fit(
                    Es[i],
                    y_pseudos[i],
                    sigmas[i],
                    p0=popt_pls[i],
                )
            if not _fit_ok(r):
                return (float("nan"), "nonconverged")
            chi2_pl_total += float(r.chi2)

        # Joint alternative refit.  Dispatch matches GlobalFitResult.fit:
        # classical alternatives use GlobalChi2Fit; profiled nuisance
        # alternatives use ProfiledGlobalChi2Fit on the pre-bound
        # per-window nuisance models.
        if isinstance(model_alt, NuisanceModel):
            predictors = dict(zip(keys, s["alt_predictors"]))
            data = {
                keys[i]: {
                    "E": Es[i],
                    "y": y_pseudos[i],
                    "sigma": sigmas[i],
                }
                for i in range(len(keys))
            }
            gfit = ProfiledGlobalChi2Fit(
                model_alt.spectral_model, shared=shared_alt, solver=s["solver"]
            ).fit(
                predictors,
                data,
                p0_full=(s["popt_alt_warm"] if warm else None),
                warn_on_fail=False,
                compute_covariance=False,
            )
        else:
            windows_in = {
                keys[i]: {
                    "E_GeV": Es[i],
                    "y": y_pseudos[i],
                    "sigma": sigmas[i],
                    "meta": {},
                }
                for i in range(len(keys))
            }
            gfit = GlobalChi2Fit(model_alt, shared=shared_alt, solver=s["solver"]).fit(
                windows_in,
                p0_full=(s["popt_alt_warm"] if warm else None),
                warn_on_fail=False,
            )
        if not _fit_ok(gfit):
            return (float("nan"), "nonconverged")
        delta = chi2_pl_total - float(gfit.chi2)
        if not np.isfinite(delta):
            return (float("nan"), "error")
        return (delta, "ok")
    except Exception:
        return (float("nan"), "error")


def _toy_global_common(seed: int) -> tuple[float, str]:
    """One common-nuisance global toy: draw one ``w_shared`` and refit globally."""
    rng = np.random.default_rng(seed)
    s = _WORKER_STATE
    try:
        keys = s["keys"]
        Es = s["Es"]
        sigmas = s["sigmas"]
        theta = (
            _draw_mvn_or_mean(rng, s["popt_null"], s["pcov_null"])
            if s["sample_theta"]
            else np.asarray(s["popt_null"], dtype=float)
        )
        w_toy = _draw_mvn_box_reject_or_mean(
            rng,
            s["w_null"],
            s["cov_w_null"],
            box_width=s.get("w_box_width"),
            max_redraw=s.get("w_box_max_redraw", 10_000),
        )
        n_free = int(s["n_null_free"])
        null_predictors = s["null_predictors"]
        y_pseudos = []
        for i, pred in enumerate(null_predictors):
            pars = theta[i * n_free : (i + 1) * n_free]
            mu = pred.predict_deformed(Es[i], tuple(pars), w=w_toy)
            y_pseudos.append(mu + rng.normal(0.0, sigmas[i]))

        data = {
            keys[i]: {"E": Es[i], "y": y_pseudos[i], "sigma": sigmas[i]}
            for i in range(len(keys))
        }
        null_p0 = (
            np.concatenate([s["popt_null"], w_toy])
            if s["warm_start"]
            else None
        )
        fit_null = ProfiledGlobalChi2Fit(
            s["null_model"].spectral_model, shared=s["null_shared"], solver=s["solver"]
        ).fit(
            dict(zip(keys, null_predictors)),
            data,
            p0_full=null_p0,
            warn_on_fail=False,
            compute_covariance=False,
        )
        if not _fit_ok(fit_null):
            return (float("nan"), "nonconverged")

        alt_p0 = s["popt_alt_warm"] if s["warm_start"] else None
        fit_alt = ProfiledGlobalChi2Fit(
            s["model_alt"].spectral_model, shared=s["shared_alt"], solver=s["solver"]
        ).fit(
            dict(zip(keys, s["alt_predictors"])),
            data,
            p0_full=alt_p0,
            warn_on_fail=False,
            compute_covariance=False,
        )
        if not _fit_ok(fit_alt):
            return (float("nan"), "nonconverged")
        delta = float(fit_null.chi2 - fit_alt.chi2)
        if not np.isfinite(delta):
            return (float("nan"), "error")
        return (delta, "ok")
    except Exception:
        return (float("nan"), "error")


def _toy_global_batch(task: tuple[int, int]) -> tuple[np.ndarray, int, int, int, int]:
    """Run a batch of global toys in one worker process."""
    seed_root, n_draw = task
    ss = np.random.SeedSequence(int(seed_root))
    ok: list[float] = []
    n_fail = 0
    n_nonconverged = 0
    n_error = 0
    for child in ss.spawn(int(n_draw)):
        seed = int(child.generate_state(1)[0])
        d, status = _toy_global(seed)
        if status == "ok":
            ok.append(float(d))
        else:
            n_fail += 1
            if status == "nonconverged":
                n_nonconverged += 1
            else:
                n_error += 1
    return (
        np.asarray(ok, dtype=float),
        int(n_draw),
        int(n_fail),
        int(n_nonconverged),
        int(n_error),
    )


def _toy_global_common_batch(task: tuple[int, int]) -> tuple[np.ndarray, int, int, int, int]:
    """Run a batch of common-nuisance global toys in one worker process."""
    seed_root, n_draw = task
    ss = np.random.SeedSequence(int(seed_root))
    ok: list[float] = []
    n_fail = 0
    n_nonconverged = 0
    n_error = 0
    for child in ss.spawn(int(n_draw)):
        seed = int(child.generate_state(1)[0])
        d, status = _toy_global_common(seed)
        if status == "ok":
            ok.append(float(d))
        else:
            n_fail += 1
            if status == "nonconverged":
                n_nonconverged += 1
            else:
                n_error += 1
    return (
        np.asarray(ok, dtype=float),
        int(n_draw),
        int(n_fail),
        int(n_nonconverged),
        int(n_error),
    )


# ----------------------------------------------------------------------
# Pool driver -- shared by both flavours


def _resolve_n_workers(n_workers: int | None) -> int:
    if n_workers is None:
        return max(1, (os.cpu_count() or 1) - 1)
    return max(1, int(n_workers))


def _run_pool(
    state: dict,
    toy_fn: Callable[[int], tuple[float, str]],
    *,
    N: int,
    seed: int,
    n_workers: int,
    chunksize: int = 1,
    print_every: int = 0,
    label: str = "",
) -> tuple[np.ndarray, int, int, int, int]:
    """Run toys until ``N`` finite, converged deltas are accepted.

    Returns ``(deltas, n_kept, n_fail, n_nonconverged, n_error)``.
    ``deltas`` collects only finite values from toys whose fits converged.
    Per-toy seeds are derived from a master :class:`SeedSequence`; failed
    attempts get fresh replacement seeds until the target is reached.
    """
    ss = np.random.SeedSequence(seed)

    deltas: list[float] = []
    n_fail = 0
    n_nonconverged = 0
    n_error = 0
    n_attempted = 0
    t0 = time.perf_counter()

    def _new_seeds(n: int) -> list[int]:
        return [int(child.generate_state(1)[0]) for child in ss.spawn(n)]

    def _record(d: float, status: str) -> None:
        nonlocal n_fail, n_nonconverged, n_error
        if status == "ok":
            deltas.append(d)
        else:
            n_fail += 1
            if status == "nonconverged":
                n_nonconverged += 1
            else:
                n_error += 1

    if n_workers <= 1:
        _pool_init(state)
        while len(deltas) < N:
            sd = _new_seeds(1)[0]
            d, status = toy_fn(sd)
            n_attempted += 1
            _record(d, status)
            if print_every and (n_attempted % print_every == 0 or len(deltas) == N):
                _print_progress(
                    label, n_attempted, len(deltas), N, n_fail, n_nonconverged, t0
                )
    else:
        ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
        with ctx.Pool(
            processes=n_workers,
            initializer=_pool_init,
            initargs=(state,),
        ) as pool:
            while len(deltas) < N:
                seeds = _new_seeds(N - len(deltas))
                for d, status in pool.imap_unordered(
                    toy_fn, seeds, chunksize=chunksize
                ):
                    n_attempted += 1
                    _record(d, status)
                    if print_every and (
                        n_attempted % print_every == 0 or len(deltas) == N
                    ):
                        _print_progress(
                            label,
                            n_attempted,
                            len(deltas),
                            N,
                            n_fail,
                            n_nonconverged,
                            t0,
                        )

    return (
        np.asarray(deltas, dtype=float),
        len(deltas),
        n_fail,
        n_nonconverged,
        n_error,
    )


def _run_pool_batched(
    state: dict,
    batch_fn: Callable[[tuple[int, int]], tuple[np.ndarray, int, int, int, int]],
    *,
    N: int,
    seed: int,
    n_workers: int,
    delta_obs: float,
    store_deltas: bool,
    batch_size: int,
    chunksize: int = 1,
    print_every: int = 0,
    label: str = "",
) -> tuple[np.ndarray, int, int, int, int, int]:
    """Run toys in worker-local batches until ``N`` accepted deltas exist.

    This keeps the statistical contract of :func:`_run_pool`: ``N`` means
    accepted, converged toys.  Each wave launches at most the number of draws
    still needed; failed toys are replaced by later waves.

    Returns ``(deltas, n_kept, n_fail, n_nonconverged, n_error, n_exceed)``.
    ``deltas`` is populated only when ``store_deltas=True``; otherwise the
    parent keeps only the exceedance count.
    """
    ss = np.random.SeedSequence(seed)
    batch_size = max(1, int(batch_size))

    deltas: list[np.ndarray] = []
    n_kept = 0
    n_fail = 0
    n_nonconverged = 0
    n_error = 0
    n_exceed = 0
    n_attempted = 0
    t0 = time.perf_counter()
    next_print = int(print_every) if print_every else 0

    def _new_tasks(n_draw_total: int) -> list[tuple[int, int]]:
        sizes: list[int] = []
        remaining = int(n_draw_total)
        while remaining > 0:
            n = min(batch_size, remaining)
            sizes.append(n)
            remaining -= n
        roots = [int(child.generate_state(1)[0]) for child in ss.spawn(len(sizes))]
        return list(zip(roots, sizes))

    def _record_batch(
        ok_deltas: np.ndarray,
        attempted_i: int,
        fail_i: int,
        nonconv_i: int,
        error_i: int,
    ) -> None:
        nonlocal n_kept, n_fail, n_nonconverged, n_error, n_exceed, n_attempted
        n_attempted += int(attempted_i)
        n_fail += int(fail_i)
        n_nonconverged += int(nonconv_i)
        n_error += int(error_i)

        ok_deltas = np.asarray(ok_deltas, dtype=float)
        if ok_deltas.size:
            # A wave never contains more draws than remaining accepted toys, so
            # this slice is normally the full array.  It also guards future
            # changes that might launch larger waves.
            take = min(ok_deltas.size, int(N) - n_kept)
            if take > 0:
                use = ok_deltas[:take]
                n_kept += int(take)
                n_exceed += int(np.sum(use >= float(delta_obs)))
                if store_deltas:
                    deltas.append(use.copy())

    def _maybe_print() -> None:
        nonlocal next_print
        if not print_every:
            return
        if n_attempted >= next_print or n_kept >= N:
            _print_progress(
                label, n_attempted, n_kept, N, n_fail, n_nonconverged, t0
            )
            while next_print and n_attempted >= next_print:
                next_print += int(print_every)

    if n_workers <= 1:
        _pool_init(state)
        while n_kept < N:
            for task in _new_tasks(N - n_kept):
                out = batch_fn(task)
                _record_batch(*out)
                _maybe_print()
                if n_kept >= N:
                    break
    else:
        ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
        with ctx.Pool(
            processes=n_workers,
            initializer=_pool_init,
            initargs=(state,),
        ) as pool:
            while n_kept < N:
                tasks = _new_tasks(N - n_kept)
                for out in pool.imap_unordered(
                    batch_fn, tasks, chunksize=chunksize
                ):
                    _record_batch(*out)
                    _maybe_print()
                    if n_kept >= N:
                        break

    if store_deltas:
        deltas_arr = np.concatenate(deltas) if deltas else np.empty(0, dtype=float)
    else:
        deltas_arr = np.empty(0, dtype=float)
    return (
        deltas_arr,
        int(n_kept),
        int(n_fail),
        int(n_nonconverged),
        int(n_error),
        int(n_exceed),
    )


def _print_progress(
    label: str,
    n_attempted: int,
    n_kept: int,
    N: int,
    n_fail: int,
    n_nonconverged: int,
    t0: float,
) -> None:
    el = time.perf_counter() - t0
    rate = n_attempted / el if el > 0 else float("inf")
    print(
        f"[MC {label}] kept={n_kept}/{N} attempted={n_attempted} "
        f"fails={n_fail} nonconv={n_nonconverged} "
        f"elapsed={el:.1f}s ({rate:.1f} toys/s)",
        flush=True,
    )


def _summarise(
    delta_obs: float,
    deltas: np.ndarray,
    n_kept: int,
    n_fail: int,
    n_nonconverged: int,
    n_error: int,
    N: int,
    seed: int,
    sample_theta: bool,
    store_deltas: bool,
    elapsed: float,
    label: str,
    n_exceed_override: int | None = None,
) -> MCPValueResult:
    if n_kept <= 0:
        p = 1.0
        n_exceed = 0
        p_is_upper_limit = False
    else:
        if n_exceed_override is None:
            n_exceed = int(np.sum(deltas >= float(delta_obs)))
        else:
            n_exceed = int(n_exceed_override)
        p_is_upper_limit = n_exceed == 0
        p = max(n_exceed / float(n_kept), 1.0 / (float(n_kept) + 1.0))
    z = float(_norm.isf(p))
    return MCPValueResult(
        delta_obs=float(delta_obs),
        p_value=float(p),
        Z=z,
        n_exceed=int(n_exceed),
        n_kept=int(n_kept),
        n_fail=int(n_fail),
        N=int(N),
        seed=int(seed),
        sample_theta=bool(sample_theta),
        n_nonconverged=int(n_nonconverged),
        n_error=int(n_error),
        deltas=(deltas.copy() if store_deltas else None),
        elapsed_s=float(elapsed),
        label=label,
        p_is_upper_limit=bool(p_is_upper_limit),
    )

 
# ----------------------------------------------------------------------
# Public API: per-window MC

def mc_pvalue_window(
    wfit,
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
) -> MCPValueResult:
    """Per-window Monte-Carlo p-value of PL vs an alternative model.

    Parameters
    ----------
    wfit
        :class:`WindowFitResult` containing fits for both ``name_pl`` and
        ``name_alt`` (typically ``"PL"`` and ``"SBPL"``).
    N
        Number of accepted toys.  Toys with non-converged fits are discarded
        and regenerated; see ``n_nonconverged`` / ``n_fail`` in the result.
    """
    fit_pl = wfit.fits[name_pl]
    fit_alt = wfit.fits[name_alt]
    delta_obs = float(fit_pl.chi2 - fit_alt.chi2)

    state = {
        "model_pl": fit_pl.model,
        "model_alt": fit_alt.model,
        "E": np.asarray(fit_pl.E, dtype=float),
        "sigma": np.asarray(fit_pl.sigma, dtype=float),
        "popt_pl": np.asarray(fit_pl.popt, dtype=float),
        "pcov_pl": np.asarray(fit_pl.pcov, dtype=float),
        "popt_alt_warm": np.asarray(fit_alt.popt, dtype=float),
        "w_pl_gen": (
            np.asarray(fit_pl.model.last_w, dtype=float).copy()
            if isinstance(fit_pl.model, NuisanceModel)
            and fit_pl.model.last_w is not None
            else None
        ),
        "sample_theta": bool(sample_theta),
        "warm_start": bool(warm_start),
        "solver": solver,
    }

    nw = _resolve_n_workers(n_workers)
    label = (
        f"window {getattr(wfit, 'lowl', '?')}-{getattr(wfit, 'highl', '?')} "
        f"({name_pl} vs {name_alt})"
    )
    t0 = time.perf_counter()
    n_exceed_override = None
    if batch_size is not None and int(batch_size) > 1:
        (
            deltas,
            n_kept,
            n_fail,
            n_nonconverged,
            n_error,
            n_exceed_override,
        ) = _run_pool_batched(
            state,
            _toy_window_batch,
            N=N,
            seed=seed,
            n_workers=nw,
            delta_obs=delta_obs,
            store_deltas=store_deltas,
            batch_size=int(batch_size),
            chunksize=chunksize,
            print_every=print_every,
            label=label,
        )
    else:
        deltas, n_kept, n_fail, n_nonconverged, n_error = _run_pool(
            state,
            _toy_window,
            N=N,
            seed=seed,
            n_workers=nw,
            chunksize=chunksize,
            print_every=print_every,
            label=label,
        )
    elapsed = time.perf_counter() - t0
    return _summarise(
        delta_obs,
        deltas,
        n_kept,
        n_fail,
        n_nonconverged,
        n_error,
        N,
        seed,
        sample_theta,
        store_deltas,
        elapsed,
        label,
        n_exceed_override=n_exceed_override,
    )


# ----------------------------------------------------------------------
# Public API: global MC -- works for free *and* fixed-shape alternatives.
# ----------------------------------------------------------------------


def mc_pvalue_global(
    fitting,
    alternative: GlobalFitResult,
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
    null_global: GlobalFitResult | None = None,
    control_global: GlobalFitResult | None = None,
) -> MCPValueResult:
    """Joint MC p-value: per-window PL vs joint alternative.

    ``alternative`` is always a :class:`GlobalFitResult`.  Both flavours are
    supported transparently:

    * **Free-shape**: ``alternative`` carries the global SBPL fit with
      ``Ebreak``/``dgamma`` (or whichever names were declared shared) joint
      across windows — produced by ``Fitting.fit_global(SBPL(), shared=...)``.
    * **Fixed-shape**: ``alternative`` carries a global fit whose ``model`` is a
      :class:`FrozenSpectralModel` and ``shared=()`` — produced by
      ``Fitting.fit_global(FrozenSpectralModel(SBPL(), Ebreak=..., dgamma=...),
      shared=())``.  Mathematically the joint solver decouples into N
      independent fits, but the API stays uniform.

    ``delta_obs = sum_i chi2(PL_i) - chi2(alt)`` in the generic path.
    In the common-nuisance path with ``null_global`` it is
    ``delta_obs = chi2(null_global) - chi2(alt)``.
    ``control_global`` optionally supplies an externally calibrated shared
    nuisance vector/covariance used to draw a frozen common deformation in the
    generic path (for control-common workflows).
    ``N`` is the number of accepted toys; toys with non-converged fits are
    discarded and regenerated, with counts stored in the returned result.
    """
    if not isinstance(alternative, GlobalFitResult):
        raise TypeError(
            "`alternative` must be a GlobalFitResult "
            f"(got {type(alternative).__name__})."
        )

    if (
        isinstance(alternative.model, NuisanceModel)
        and alternative.model.share_nuisance
        and null_global is not None
    ):
        if (
            not isinstance(null_global, GlobalFitResult)
            or not isinstance(null_global.model, NuisanceModel)
            or not null_global.model.share_nuisance
            or tuple(null_global.window_keys) != tuple(alternative.window_keys)
        ):
            raise ValueError("`null_global` must be a matching common-nuisance GlobalFitResult.")
        if null_global.shared_param_names:
            raise ValueError("Common-nuisance global MC expects the PL null to have no shared spectral parameters.")
        if null_global.w_shared is None:
            raise ValueError("`null_global.w_shared` is missing; refit the global PL common-nuisance null first.")

        keys = tuple(alternative.window_keys)
        wfs = [fitting.window_fits[k] for k in keys]
        null_predictors, null_arrays = null_global.model.for_global(wfs)
        alt_predictors, _ = alternative.model.for_global(wfs)
        Es = tuple(np.asarray(a["E"], dtype=float) for a in null_arrays)
        sigmas = tuple(np.asarray(a["sigma"], dtype=float) for a in null_arrays)
        popt_alt_warm = None
        if alternative.w_shared is not None:
            popt_alt_warm = np.concatenate(
                [
                    np.asarray(alternative.popt_full, dtype=float),
                    np.asarray(alternative.w_shared, dtype=float).ravel(),
                ]
            )

        state = {
            "keys": keys,
            "Es": Es,
            "sigmas": sigmas,
            "popt_null": np.asarray(null_global.popt_full, dtype=float),
            "pcov_null": np.asarray(null_global.pcov_full, dtype=float),
            "w_null": np.asarray(null_global.w_shared, dtype=float),
            "cov_w_null": (
                None
                if null_global.cov_w_shared is None
                else np.asarray(null_global.cov_w_shared, dtype=float)
            ),
            "w_box_width": (
                float(null_predictors[0].prior.width)
                if isinstance(null_predictors[0].prior, BoxGaussPrior)
                else None
            ),
            "w_box_max_redraw": 10_000,
            "n_null_free": len(null_global.free_param_names),
            "null_model": null_global.model,
            "null_shared": tuple(null_global.shared_param_names),
            "null_predictors": tuple(null_predictors),
            "model_alt": alternative.model,
            "shared_alt": tuple(alternative.shared_param_names),
            "popt_alt_warm": popt_alt_warm,
            "alt_predictors": tuple(alt_predictors),
            "sample_theta": bool(sample_theta),
            "warm_start": bool(warm_start),
            "solver": fitting.solver,
        }

        nw = _resolve_n_workers(n_workers)
        label = f"global common legacy (N_win={len(keys)})"
        delta_obs = float(null_global.chi2 - alternative.chi2)
        t0 = time.perf_counter()
        n_exceed_override = None
        if batch_size is not None and int(batch_size) > 1:
            (
                deltas,
                n_kept,
                n_fail,
                n_nonconverged,
                n_error,
                n_exceed_override,
            ) = _run_pool_batched(
                state,
                _toy_global_common_batch,
                N=N,
                seed=seed,
                n_workers=nw,
                delta_obs=delta_obs,
                store_deltas=store_deltas,
                batch_size=int(batch_size),
                chunksize=chunksize,
                print_every=print_every,
                label=label,
            )
        else:
            deltas, n_kept, n_fail, n_nonconverged, n_error = _run_pool(
                state,
                _toy_global_common,
                N=N,
                seed=seed,
                n_workers=nw,
                chunksize=chunksize,
                print_every=print_every,
                label=label,
            )
        return _summarise(
            delta_obs,
            deltas,
            n_kept,
            n_fail,
            n_nonconverged,
            n_error,
            N,
            seed,
            sample_theta,
            store_deltas,
            time.perf_counter() - t0,
            label,
            n_exceed_override=n_exceed_override,
        )

    keys = tuple(alternative.window_keys)
    if isinstance(alternative.model, NuisanceModel) and alternative.model.share_nuisance:
        fpl0 = fitting.window_fits[keys[0]].fits[name_pl]
        if isinstance(fpl0.model, NuisanceModel) and fpl0.model.share_nuisance:
            raise ValueError(
                "Common-nuisance global MC requires `null_global`: pass the "
                "global PL fit with the same shared nuisance."
            )
    Es: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    popt_pls: list[np.ndarray] = []
    pcov_pls: list[np.ndarray] = []
    chi2_pl_total = 0.0
    model_pls: list[SpectralModel] = []
    w_pl_gens: list[np.ndarray | None] = []

    for key in keys:
        wf = fitting.window_fits[key]
        fpl = wf.fits[name_pl]
        model_pls.append(fpl.model)
        w_pl_gens.append(
            np.asarray(fpl.model.last_w, dtype=float).copy()
            if isinstance(fpl.model, NuisanceModel)
            and fpl.model.last_w is not None
            else None
        )
        Es.append(np.asarray(fpl.E, dtype=float))
        sigmas.append(np.asarray(fpl.sigma, dtype=float))
        popt_pls.append(np.asarray(fpl.popt, dtype=float))
        pcov_pls.append(np.asarray(fpl.pcov, dtype=float))
        chi2_pl_total += float(fpl.chi2)

    delta_obs = float(chi2_pl_total - alternative.chi2)

    alt_predictors = None
    popt_alt_warm = np.asarray(alternative.popt_full, dtype=float)
    if isinstance(alternative.model, NuisanceModel):
        wfs = [fitting.window_fits[k] for k in keys]
        if alternative.model.share_nuisance:
            alt_predictors, _ = alternative.model.for_global(wfs)
            if alternative.w_shared is not None:
                popt_alt_warm = np.concatenate(
                    [
                        popt_alt_warm,
                        np.asarray(alternative.w_shared, dtype=float).ravel(),
                    ]
                )
            else:
                popt_alt_warm = None
        else:
            alt_predictors = [
                alternative.model.for_window(fitting.window_fits[k])[0]
                for k in keys
            ]

    w_global_gen_mean = None
    w_global_gen_cov = None
    w_global_gen_box_width = None
    if control_global is not None:
        if (
            not isinstance(control_global, GlobalFitResult)
            or not isinstance(control_global.model, NuisanceModel)
            or control_global.w_shared is None
        ):
            raise ValueError(
                "`control_global` must be a GlobalFitResult with a fitted "
                "shared nuisance vector."
            )
        w_global_gen_mean = np.asarray(control_global.w_shared, dtype=float)
        w_global_gen_cov = (
            None
            if control_global.cov_w_shared is None
            else np.asarray(control_global.cov_w_shared, dtype=float)
        )
        prior = getattr(control_global.model, "prior", None)
        if isinstance(prior, BoxGaussPrior):
            w_global_gen_box_width = float(prior.width)

    state = {
        "keys": keys,
        "Es": tuple(Es),
        "sigmas": tuple(sigmas),
        "popt_pls": tuple(popt_pls),
        "pcov_pls": tuple(pcov_pls),
        "model_pls": tuple(model_pls),
        "w_pl_gens": tuple(w_pl_gens),
        "model_alt": alternative.model,
        "shared_alt": tuple(alternative.shared_param_names),
        "popt_alt_warm": popt_alt_warm,
        "alt_predictors": tuple(alt_predictors) if alt_predictors is not None else (),
        "w_global_gen_mean": w_global_gen_mean,
        "w_global_gen_cov": w_global_gen_cov,
        "w_global_gen_box_width": w_global_gen_box_width,
        "w_global_gen_box_max_redraw": 10_000,
        "sample_theta": bool(sample_theta),
        "warm_start": bool(warm_start),
        "solver": fitting.solver,
    }

    nw = _resolve_n_workers(n_workers)
    label_model = (
        alternative.model.spectral_model
        if isinstance(alternative.model, NuisanceModel)
        else alternative.model
    )
    if isinstance(label_model, FrozenSpectralModel):
        fixed_repr = ",".join(
            f"{k}={v:g}" for k, v in sorted(label_model.fixed.items())
        )
        label = f"global fixed [{fixed_repr}] (N_win={len(keys)})"
    else:
        label = f"global free (N_win={len(keys)})"

    t0 = time.perf_counter()
    n_exceed_override = None
    if batch_size is not None and int(batch_size) > 1:
        (
            deltas,
            n_kept,
            n_fail,
            n_nonconverged,
            n_error,
            n_exceed_override,
        ) = _run_pool_batched(
            state,
            _toy_global_batch,
            N=N,
            seed=seed,
            n_workers=nw,
            delta_obs=delta_obs,
            store_deltas=store_deltas,
            batch_size=int(batch_size),
            chunksize=chunksize,
            print_every=print_every,
            label=label,
        )
    else:
        deltas, n_kept, n_fail, n_nonconverged, n_error = _run_pool(
            state,
            _toy_global,
            N=N,
            seed=seed,
            n_workers=nw,
            chunksize=chunksize,
            print_every=print_every,
            label=label,
        )
    elapsed = time.perf_counter() - t0
    return _summarise(
        delta_obs,
        deltas,
        n_kept,
        n_fail,
        n_nonconverged,
        n_error,
        N,
        seed,
        sample_theta,
        store_deltas,
        elapsed,
        label,
        n_exceed_override=n_exceed_override,
    )
