from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Union

import numpy as np

from ..data_extract.main import Data, Window
from .chi2 import FitResult
from .global_fit import GlobalFitResult
from .models import SpectralModel
from . import scan_params as _scan
from .nuisance import BoxGaussPrior, NuisanceModel
from .window_fit import WindowFitResult


# Public alias: every fit entry-point accepts either flavour.
ModelLike = Union[SpectralModel, NuisanceModel]

def _is_sbpl_profiled(model) -> bool:
    """Profiled SBPL, also with a fixed Ebreak (``model.sbpl_ebreak_fixed``)."""
    if not isinstance(model, NuisanceModel):
        return False
    pn = getattr(model.spectral_model, "param_names", ())
    return ("gamma1" in pn) and ("dgamma" in pn)


def _gamma_grids_for_window(
    wf: WindowFitResult, name: str, *, radius: float, n_points: int
) -> dict:
    """``gamma1`` / ``gamma2`` profile grids: this window's best fit +- ``radius``."""
    popt = dict(zip(wf.fits[name].model.param_names, np.asarray(wf.fits[name].popt, dtype=float)))
    g1 = float(popt["gamma1"])
    g2 = g1 - float(popt["dgamma"])
    return {
        "gamma1": np.linspace(g1 - radius, g1 + radius, int(n_points)),
        "gamma2": np.linspace(g2 - radius, g2 + radius, int(n_points)),
    }


class Fitting:
    def __init__(self, data: Data | Mapping[str, Window], *, solver):
        """``solver`` is the ``solver`` section of the analysis config; every
        fit, scan, profile and MC toy launched from here uses it."""
        self._data_obj: Data | None = data
        self.windows: dict[str, Window] = dict(data.data_dict)
        self.solver = solver

        if not self.windows:
            raise ValueError(
                "Fitting received an empty windows mapping. Did you call "
                "Data.build(...) first?"
            )

        self.window_fits: dict[str, WindowFitResult] = {}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_window_fits(self) -> None:
        if self.window_fits:
            return
        for key, window in self.windows.items():
            self.window_fits[key] = WindowFitResult.from_window(window)

    @staticmethod
    def _print_window_fit(
        wf: WindowFitResult, key: str, name: str, res: FitResult
    ) -> None:
        kind = name.split("|", 1)[0]
        roi = f"l=({wf.lowl:g},{wf.highl:g}) b=({wf.lowb:g},{wf.highb:g})"
        print(f"\n----- [{key} {roi}] {kind}")
        print(f"  model: {name}")
        print(f"  chi2/ndof: {res.chi2:.3g}/{res.ndof}")
        for p, (v, e) in res.params.items():
            print(f"  {p}: {v:.6g} ± {e:.3g}")

    @staticmethod
    def _print_safety_net(prefix: str, info: dict) -> None:
        if info.get("chi2_classical") is None:
            return
        delta = info["chi2_outer"] - info["chi2_classical"]
        chosen_label = {
            "classical_seed": "theta_classical",
            "classical_start_min": "theta_classical_min",
        }.get(info.get("chosen"), "theta_outer")
        extra = (
            f"\n{prefix}   chi2_prof@theta_classical_min: "
            f"{info['chi2_classical_min']:.3f}"
            if info.get("chi2_classical_min") is not None
            else ""
        )
        print(
            f"{prefix} safety-net:\n"
            f"{prefix}   chosen: {chosen_label}\n"
            f"{prefix}   chi2_prof@theta_outer: {info['chi2_outer']:.3f}\n"
            f"{prefix}   chi2_prof@theta_classical: {info['chi2_classical']:.3f}\n"
            f"{prefix}   diff outer-classical: {delta:+.3f}{extra}"
        )

    @staticmethod
    def _print_box_prior_diagnostic(
        prefix: str,
        label: str,
        model: SpectralModel,
        w: np.ndarray | None,
    ) -> None:
        if not isinstance(model, NuisanceModel) or w is None:
            return
        prior = getattr(model, "prior", None)
        if not isinstance(prior, BoxGaussPrior):
            return
        w = np.asarray(w, dtype=float)
        if w.size == 0 or not np.all(np.isfinite(w)):
            return
        excess = np.maximum(np.abs(w) - prior.width, 0.0)
        idx = np.flatnonzero(excess > 1e-10)
        if idx.size == 0:
            return
        node_chi2 = (excess / prior.tau) ** 2
        total_node = float(np.sum(node_chi2))
        total = float(prior.chi2(w))
        print(f"{prefix} nuisance box-gauss prior: {label}")
        print(
            f"{prefix}   width={prior.width:g}, tau={prior.tau:g}, "
            f"outside_box={idx.size}/{w.size}"
        )
        for j in idx:
            print(
                f"{prefix}   w[{j}]={w[j]:+.4g}  "
                f"excess={excess[j]:.4g}  chi2_prior_j={node_chi2[j]:.4g}"
            )
        print(f"{prefix}   chi2_prior_box_nodes={total_node:.4g}")
        if abs(total - total_node) > 1e-8:
            print(f"{prefix}   chi2_prior_total={total:.4g}")

    # ------------------------------------------------------------------
    # Per-window fits
    # ------------------------------------------------------------------
    def fit_per_window(
        self,
        models: Sequence[ModelLike],
        *,
        wilks_pairs: Sequence[tuple[str, str]] | None = None,
        do_montecarlo: bool = False,
        mc_pairs: Sequence[tuple[str, str]] | None = None,
        mc_kwargs: dict | None = None,
        do_profile_errors: bool = True,
        profile_errors_kwargs: dict | None = None,
        progress: bool = False,
    ) -> dict[str, WindowFitResult]:
        """Fit every model in every window.

        ``profile_errors_kwargs`` (needed when a profiled SBPL is fitted with
        ``do_profile_errors``): ``params``, ``delta``, ``grids`` (absolute
        ``Ebreak`` / ``dgamma`` grids), ``gamma_radius``, ``gamma_n_points``
        and ``n_workers``.  ``mc_kwargs`` go to :meth:`WindowFitResult.mc_pvalue`.
        """
        models = list(models)
        self._ensure_window_fits()

        for key, window in self.windows.items():
            wf = self.window_fits.get(key)
            if wf is None:
                wf = WindowFitResult.from_window(window)
                self.window_fits[key] = wf

            for model in models:
                res = wf.fit(model, solver=self.solver)
                if progress:
                    self._print_window_fit(wf, key, res.model.name, res)
                    info = wf.safety_logs.get(res.model.name)
                    if info is not None:
                        self._print_safety_net("           ", info)
                    self._print_box_prior_diagnostic(
                        "           ",
                        res.model.name,
                        res.model,
                        getattr(res.model, "last_w", None),
                    )

        sbpl_prof = [m for m in models if _is_sbpl_profiled(m)]
        if do_profile_errors and sbpl_prof:
            pek = profile_errors_kwargs
            # Windows in serial; *cells* inside each window are parallelised
            # by ``profile_errors_1d_window`` via ``n_workers``.
            if progress:
                print("\n----- 1D error scanning")
                print(f"  Scanning {', '.join(pek['params'])} per window for profile errors.")
            for model in sbpl_prof:
                name = model.name
                for key, wf in self.window_fits.items():
                    if name not in wf.fits:
                        continue
                    # Absolute Ebreak/dgamma grids; the slopes get a grid
                    # centred on this window's best fit.
                    grids = {
                        **pek["grids"],
                        **_gamma_grids_for_window(
                            wf, name, radius=pek["gamma_radius"], n_points=pek["gamma_n_points"]
                        ),
                    }
                    wf.compute_profile_errors(
                        name,
                        params=pek["params"],
                        grids=grids,
                        delta=pek["delta"],
                        solver=self.solver,
                        n_workers=pek["n_workers"],
                    )

        if wilks_pairs:
            for wf in self.window_fits.values():
                for s, c in wilks_pairs:
                    wf.wilks(s, c)

        if do_montecarlo:
            pairs = (
                list(mc_pairs)
                if mc_pairs
                else (list(wilks_pairs) if wilks_pairs else [])
            )
            if not pairs:
                raise ValueError(
                    "fit_per_window: do_montecarlo=True requires either "
                    "mc_pairs=[(name_pl, name_alt), ...] or wilks_pairs."
                )
            for wf in self.window_fits.values():
                for s, c in pairs:
                    wf.mc_pvalue(name_pl=s, name_alt=c, solver=self.solver, **mc_kwargs)

        return self.window_fits

    # ------------------------------------------------------------------
    # Global fit (free shape, shared parameters)
    # ------------------------------------------------------------------
    def fit_global(
        self,
        model: ModelLike,
        *,
        shared: Iterable[str],
        classical_seed: GlobalFitResult | None = None,
        w_shared0: np.ndarray | None = None,
        wilks_against: Sequence[str] | None = None,
        do_montecarlo: bool = False,
        mc_against: Sequence[str] | None = None,
        mc_kwargs: dict | None = None,
        polish_break_scan_kwargs: dict | None = None,
        do_profile_errors: bool = True,
        profile_errors_kwargs: dict | None = None,
        progress: bool = False,
    ) -> GlobalFitResult:
        if isinstance(model, NuisanceModel):
            self._ensure_window_fits()
        shared_tuple = tuple(shared)
        windows_in = self.window_fits if self.window_fits else self.windows
        if (
            progress
            and isinstance(model, NuisanceModel)
            and getattr(model, "share_nuisance", False)
            and w_shared0 is not None
        ):
            w0 = np.asarray(w_shared0, dtype=float)
            print(
                f"[profiled global] warm-start w_shared0: "
                f"n_nuis={w0.size}, |w0|_2={np.linalg.norm(w0):.3g}, "
                f"|w0|_inf={np.max(np.abs(w0)) if w0.size else 0.0:.3g}"
            )
        g = GlobalFitResult.fit(
            model,
            windows_in,
            shared=shared_tuple,
            solver=self.solver,
            classical_seed=classical_seed,
            w_shared0=w_shared0,
        )
        if progress and g.safety_log is not None:
            self._print_safety_net("[profiled global]", g.safety_log)

        if (
            polish_break_scan_kwargs is not None
            and _is_sbpl_profiled(model)
            and ("Ebreak" in g.shared_param_names and "dgamma" in g.shared_param_names)
        ):
            g = self._polish_global_break_with_scan(
                g,
                model,
                windows_in=windows_in,
                shared=shared_tuple,
                classical_seed=classical_seed,
                w_shared0=w_shared0,
                kwargs=polish_break_scan_kwargs,
                progress=progress,
            )

        if (
            do_profile_errors
            and _is_sbpl_profiled(model)
            and ("Ebreak" in g.shared_param_names and "dgamma" in g.shared_param_names)
        ):
            pek = profile_errors_kwargs
            g.compute_profile_errors(
                self,
                nuisance_config=model,
                params=pek["params"],
                grids=pek["grids"],
                delta=pek["delta"],
                n_workers=pek["n_workers"],
            )

        if progress:
            self._print_box_prior_diagnostic(
                "[profiled global]",
                "w_shared",
                g.model,
                getattr(g, "w_shared", None),
            )

        if wilks_against:
            for name_simpler in wilks_against:
                chi2_simpler = self.total_chi2(name_simpler)
                n_params_simpler = self.total_n_params(name_simpler)
                g.wilks_vs(
                    chi2_simpler,
                    n_params_simpler,
                    name_simpler=name_simpler,
                )

        if do_montecarlo:
            against = (
                list(mc_against)
                if mc_against
                else (list(wilks_against) if wilks_against else [])
            )
            if not against:
                raise ValueError(
                    "fit_global: do_montecarlo=True requires either "
                    "mc_against=[name_pl, ...] or wilks_against."
                )
            for name_pl in against:
                g.mc_pvalue(self, name_pl=name_pl, **mc_kwargs)

        return g

    def _polish_global_break_with_scan(
        self,
        g: GlobalFitResult,
        model: NuisanceModel,
        *,
        windows_in,
        shared: tuple[str, ...],
        classical_seed: GlobalFitResult | None,
        w_shared0: np.ndarray | None,
        kwargs: dict,
        progress: bool,
    ) -> GlobalFitResult:
        """Local 2-D scan around a profiled global SBPL solution.

        The scan keeps the existing common ``w_shared`` fixed, matching
        ``scan_chi2_sbpl`` semantics, then a continuous global fit is restarted
        from the best grid cell if the scan finds a lower chi2 neighborhood.

        ``kwargs``: ``radius_Ebreak``, ``radius_dgamma``, ``n_Ebreak``,
        ``n_dgamma``, ``min_improvement``, ``n_workers``, ``print_every``.
        """
        sp = g.shared_params
        Eb0 = float(sp["Ebreak"][0])
        dg0 = float(sp["dgamma"][0])
        if not (np.isfinite(Eb0) and np.isfinite(dg0) and np.isfinite(g.chi2)):
            return g

        spectral = model.spectral_model
        lo_all, hi_all = spectral.default_bounds()
        bounds = {
            name: (float(lo_all[i]), float(hi_all[i]))
            for i, name in enumerate(spectral.param_names)
        }
        Eb_lo, Eb_hi = bounds["Ebreak"]
        dg_lo, dg_hi = bounds["dgamma"]

        rE = float(kwargs["radius_Ebreak"])
        rdg = float(kwargs["radius_dgamma"])
        nE = int(kwargs["n_Ebreak"])
        ndg = int(kwargs["n_dgamma"])
        if nE < 2 or ndg < 2 or rE <= 0.0 or rdg <= 0.0:
            return g

        E_grid = np.linspace(max(Eb_lo, Eb0 - rE), min(Eb_hi, Eb0 + rE), nE)
        dg_grid = np.linspace(max(dg_lo, dg0 - rdg), min(dg_hi, dg0 + rdg), ndg)
        if E_grid.size < 2 or dg_grid.size < 2:
            return g

        if progress:
            print(
                "[profiled global polish] local 2D scan around "
                f"Ebreak={Eb0:.6g}, dgamma={dg0:.6g} "
                f"({E_grid.size}x{dg_grid.size})"
            )

        scan = self.scan(
            model,
            Ebreak_grid=E_grid,
            dgamma_grid=dg_grid,
            chi2_anchor=float(g.chi2),
            warm_start_from="PL",
            init_from_global=g,
            n_workers=kwargs["n_workers"],
            print_every=int(kwargs["print_every"]),
            label="global_break_polish",
        )
        if not np.any(np.isfinite(scan.chi2_grid)):
            if progress:
                print("[profiled global polish] scan produced no finite cells")
            return g

        flat = int(np.nanargmin(scan.chi2_grid))
        iE, idg = np.unravel_index(flat, scan.chi2_grid.shape)
        chi2_scan = float(scan.chi2_grid[iE, idg])
        Eb_scan = float(scan.Ebreak_grid[iE])
        dg_scan = float(scan.dgamma_grid[idg])
        min_improvement = float(kwargs["min_improvement"])
        if progress:
            print(
                "[profiled global polish] scan min: "
                f"Ebreak={Eb_scan:.6g}, dgamma={dg_scan:.6g}, "
                f"chi2={chi2_scan:.6g} (current {g.chi2:.6g})"
            )
        if chi2_scan >= float(g.chi2) - min_improvement:
            self._propagate_result_w_shared(model, g)
            return g

        p0 = np.asarray(g.popt_full, dtype=float).copy()
        offset = g.n_free * g.n_windows
        shared_index = {name: offset + i for i, name in enumerate(g.shared_param_names)}
        p0[shared_index["Ebreak"]] = Eb_scan
        p0[shared_index["dgamma"]] = dg_scan
        if model.share_nuisance and g.w_shared is not None:
            p0 = np.concatenate([p0, np.asarray(g.w_shared, dtype=float).ravel()])

        w_before = None if g.w_shared is None else np.asarray(g.w_shared, dtype=float)
        g_retry = GlobalFitResult.fit(
            model,
            windows_in,
            shared=shared,
            solver=self.solver,
            classical_seed=classical_seed,
            p0_full=p0,
            w_shared0=w_shared0,
        )
        if progress and g_retry.safety_log is not None:
            self._print_safety_net("[profiled global polish refit]", g_retry.safety_log)

        if float(g_retry.chi2) < float(g.chi2) - min_improvement:
            if progress:
                print(
                    "[profiled global polish] accepted refit: "
                    f"chi2 {g.chi2:.6g} -> {g_retry.chi2:.6g}"
                )
                if w_before is not None and g_retry.w_shared is not None:
                    dw = np.asarray(g_retry.w_shared, dtype=float) - w_before
                    print(
                        "[profiled global polish] "
                        f"|delta w_shared|_2={np.linalg.norm(dw):.3g}, "
                        f"|delta w_shared|_inf={np.max(np.abs(dw)) if dw.size else 0.0:.3g}"
                    )
            self._propagate_result_w_shared(model, g_retry)
            return g_retry

        if progress:
            print(
                "[profiled global polish] rejected refit: "
                f"chi2 {g_retry.chi2:.6g} >= {g.chi2:.6g}"
            )
        self._propagate_result_w_shared(model, g)
        return g

    @staticmethod
    def _propagate_result_w_shared(model: NuisanceModel, g: GlobalFitResult) -> None:
        if model.share_nuisance and g.w_shared is not None:
            model.w_shared = np.asarray(g.w_shared, dtype=float).copy()

    # ------------------------------------------------------------------
    # Convenience iteration / access
    # ------------------------------------------------------------------
    def __getitem__(self, key: str) -> WindowFitResult:
        return self.window_fits[key]

    def __iter__(self):
        return iter(self.window_fits.items())

    def __len__(self) -> int:
        return len(self.window_fits)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self.window_fits.keys())

    def delta_chi2(self, name_a: str, name_b: str) -> dict[str, float]:
        return {k: w.delta_chi2(name_a, name_b) for k, w in self.window_fits.items()}

    def total_chi2(self, name: str) -> float:
        return float(sum(w.fits[name].chi2 for w in self.window_fits.values()))

    def total_ndof(self, name: str) -> int:
        return int(sum(w.fits[name].ndof for w in self.window_fits.values()))

    def total_n_params(self, name: str) -> int:
        return int(sum(w.fits[name].model.n_params for w in self.window_fits.values()))

    # ------------------------------------------------------------------
    # 2-D parameter-space scan (Ebreak, dgamma)
    # ------------------------------------------------------------------
    def scan(
        self,
        model_sbpl: ModelLike,
        *,
        Ebreak_grid,
        dgamma_grid,
        chi2_anchor: float | None = None,
        warm_start_from: str | None = "PL",
        init_from_global: GlobalFitResult | None = None,
        n_workers: int | None = None,
        chunksize: int = 1,
        print_every: int = 0,
        label: str | None = None,
    ) -> "_scan.ChiSquareScan":
        if isinstance(model_sbpl, NuisanceModel):
            spectral = model_sbpl.spectral_model
            nuisance = model_sbpl
            default_label = "scan_profiled"
        else:
            spectral = model_sbpl
            nuisance = None
            default_label = "scan"

        # GlobalFitResult.popt_full layout is
        # [free_w0, free_w1, ..., free_w(N-1), shared] with n_free
        # entries per window -- pull just the per-window block.
        init_per_window = None
        if init_from_global is not None:
            nf = init_from_global.n_free
            nW = init_from_global.n_windows
            init_per_window = (
                init_from_global.popt_full[: nf * nW].reshape(nW, nf).copy()
            )

        return _scan.scan_chi2_sbpl(
            self,
            spectral,
            Ebreak_grid=Ebreak_grid,
            dgamma_grid=dgamma_grid,
            chi2_anchor=chi2_anchor,
            warm_start_from=warm_start_from,
            nuisance_config=nuisance,
            init_per_window=init_per_window,
            n_workers=n_workers,
            chunksize=chunksize,
            print_every=print_every,
            label=(label if label is not None else default_label),
        )

    def __repr__(self) -> str:
        return (
            f"Fitting(N_windows={len(self.windows)}, "
            f"fit_done={len(self.window_fits)}/{len(self.windows)})"
        )
