from __future__ import annotations

import gzip
import json
import pickle
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config.analysis import AnalysisConfig, config_to_dict
from ..data_extract.main import Data, SkyMesh
from ..plotting import (
    ContourDataset,
    NuisanceStyle,
    ParamsDataset,
    plot_contour_heatmap,
    plot_contour_plot,
    plot_mc_diagnostics,
    plot_nuisance,
    plot_params_gammas_vs_glon,
    plot_params_vs_glon_plus_contour,
    plot_params_vs_glon,
    plot_sed,
)
from ..plotting.plot_sed import SedStyle
from .control_nuisance import ControlNuisanceBuilder
from .global_fit import GlobalFitResult
from .main import Fitting
from .models import FrozenSpectralModel, PowerLaw, SmoothBrokenPowerLaw
from .nuisance import NuisanceModel, nuisance_kwargs_from_config
from .scan_params import ChiSquareScan


# ----------------------------------------------------------------------
# Per-dataset bundle (all the artifacts produced by ``run_one`` for one kind).


@dataclass
class _DatasetRun:
    kind: str
    fitting: Fitting
    g: GlobalFitResult
    scan: ChiSquareScan | None
    # Profiled (nuisance) fits (optional).
    g_prof: GlobalFitResult | None = None
    pl_prof_name: str | None = None
    sbpl_prof_name: str | None = None
    # Common-nuisance mode only: profiled PL global with shared ``w``.
    g_pl_prof: GlobalFitResult | None = None
    share_nuisance: bool = False
    control_fitting: Fitting | None = None
    control_pl_prof_name: str | None = None
    # Optional PASS8 "raw" (subtraction=False) per-window fit.
    raw_fitting: Fitting | None = None
    raw_sbpl_name: str | None = None


# ----------------------------------------------------------------------
# Per-dataset full pipeline (extract + fit + global + scan).


def _dataset_banner(kind: str) -> None:
    line = "=" * 88
    print(f"\n{line}\n{str(kind).upper():^88}\n{line}")


def _sep(title: str) -> None:
    print(f"\n{'-' * 72}\n{title}")


def _subsep(title: str) -> None:
    print(f"\n----- {title}")


_MC_FINAL_PRINT_EVERY = 10**12


def _mesh_summary(mesh: SkyMesh) -> str:
    l0, l1 = float(mesh.glon_bins[0]), float(mesh.glon_bins[-1])
    b0, b1 = float(mesh.glat_bins[0]), float(mesh.glat_bins[-1])
    lon = "window" if mesh.Nglon == 1 else "windows"
    lat = "band" if mesh.Nglat == 1 else "bands"
    return (
        f"SkyMesh {l0:g}<|l|<{l1:g}, {b0:g}<|b|<{b1:g} "
        f"with {mesh.Nglon} longitude {lon} and {mesh.Nglat} latitude {lat}"
    )


def _fmt_pm(x: float, e: float, *, fmt: str) -> str:
    if np.isfinite(x) and np.isfinite(e):
        return f"{x:{fmt}} ± {e:{fmt}}"
    if np.isfinite(x):
        return f"{x:{fmt}} ± nan"
    return "nan ± nan"


def _fmt_fixed(x: float, *, fmt: str, unit: str = "") -> str:
    u = f" {unit}" if unit else ""
    return f"{x:{fmt}} fixed{u}" if np.isfinite(x) else f"nan fixed{u}"


def _fmt_asym(best: float, lo: float, hi: float, *, fmt: str, unit: str = "") -> str:
    """``best (-Δlo +Δhi) [unit]`` for asymmetric errors.

    If the grid does not span the ``Δχ²`` crossing on one side, print
    ``open`` for that side instead of a misleading ``nan``.
    """
    minus = None if not np.isfinite(lo) else best - lo
    plus = None if not np.isfinite(hi) else hi - best
    minus_txt = "open" if minus is None else f"{minus:{fmt}}"
    plus_txt = "open" if plus is None else f"{plus:{fmt}}"
    u = f" {unit}" if unit else ""
    return f"{best:{fmt}} (-{minus_txt} +{plus_txt}){u}"


def _fmt_fit_param(
    fit,
    pname: str,
    *,
    fmt: str,
    unit: str = "",
    profile_errors: dict | None = None,
) -> str:
    if profile_errors and pname in profile_errors:
        best, lo, hi = profile_errors[pname]
        return _fmt_asym(best, lo, hi, fmt=fmt, unit=unit)
    if pname in fit.params:
        best, err = fit.params[pname]
        u = f" {unit}" if unit else ""
        return f"{_fmt_pm(best, err, fmt=fmt)}{u}"
    fixed = getattr(fit.model, "fixed", None)
    if isinstance(fixed, dict) and pname in fixed:
        return _fmt_fixed(float(fixed[pname]), fmt=fmt, unit=unit)
    return "n/a"


def _fmt_global_param(
    g: GlobalFitResult,
    pname: str,
    *,
    fmt: str,
    unit: str = "",
    profile_errors: dict | None = None,
) -> str:
    if profile_errors and pname in profile_errors:
        best, lo, hi = profile_errors[pname]
        return _fmt_asym(best, lo, hi, fmt=fmt, unit=unit)
    if pname in g.shared_params:
        best, err = g.shared_params[pname]
        u = f" {unit}" if unit else ""
        return f"{_fmt_pm(best, err, fmt=fmt)}{u}"
    fixed = getattr(g.model, "fixed", None)
    if isinstance(fixed, dict) and pname in fixed:
        return _fmt_fixed(float(fixed[pname]), fmt=fmt, unit=unit)
    inner = getattr(g.model, "spectral_model", None)
    fixed = getattr(inner, "fixed", None)
    if isinstance(fixed, dict) and pname in fixed:
        return _fmt_fixed(float(fixed[pname]), fmt=fmt, unit=unit)
    return "n/a"


def _fmt_mc(mc) -> str:
    if mc is None:
        return "nan/nan (n/a)"
    p_op = "<" if bool(getattr(mc, "p_is_upper_limit", False)) else ""
    z_op = ">" if bool(getattr(mc, "p_is_upper_limit", False)) else ""
    return (
        f"{p_op}{float(mc.p_value):.3g}/{z_op}{float(mc.Z):.2f} "
        f"({int(mc.n_exceed)}/{int(mc.n_kept)})"
    )


def _print_global_mc_delta_check(label: str, d_chi2_table: float, mc) -> None:
    """Warn when the displayed global Δχ² is not the MC threshold."""
    if mc is None:
        return
    mc_delta = float(getattr(mc, "delta_obs", np.nan))
    if not (np.isfinite(d_chi2_table) and np.isfinite(mc_delta)):
        return
    diff = float(mc_delta - d_chi2_table)
    scale = max(1.0, abs(mc_delta), abs(d_chi2_table))
    if abs(diff) <= 1e-6 * scale:
        return
    print(
        f"  [delta-check {label}] table Δχ²={d_chi2_table:.6g}; "
        f"MC delta_obs={mc_delta:.6g}; diff={diff:+.6g}"
    )


def _print_window_fit_table(
    fitting: Fitting,
    *,
    kind: str,
    title: str,
    pl_name: str,
    sbpl_name: str,
    do_mc: bool,
    use_profile_errors: bool = False,
) -> None:
    keys = sorted(fitting.window_fits.keys(), key=lambda k: fitting[k].lowl)
    print(f"[fit_test] {kind}: {title}")
    hdr = (
        "  window_key           |  |l| range    |  Ebreak [GeV]"
        "             |  dgamma                  | chi2/ndof      |"
        "  Δχ²     | Wilks p/Z       | MC p/Z (exc/N)"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for k in keys:
        wf = fitting[k]
        fit = wf.fits.get(sbpl_name)
        if fit is None:
            continue
        prof_err = (
            getattr(wf, "profile_errors", {}).get(sbpl_name)
            if use_profile_errors
            else None
        )
        eb_txt = _fmt_fit_param(
            fit, "Ebreak", fmt=".2f", unit="GeV", profile_errors=prof_err
        )
        dg_txt = _fmt_fit_param(fit, "dgamma", fmt=".3f", profile_errors=prof_err)
        d_chi2 = (
            float(wf.delta_chi2(pl_name, sbpl_name))
            if (pl_name in wf and sbpl_name in wf)
            else float("nan")
        )
        w = wf.wilks_results.get((pl_name, sbpl_name), {})
        p_w = w.get("p_wilks", float("nan"))
        z_w = w.get("Z_wilks", float("nan"))
        mc = wf.mc_results.get((pl_name, sbpl_name)) if do_mc else None
        mc_txt = _fmt_mc(mc)
        glon_txt = f"{wf.lowl:4.0f}-{wf.highl:4.0f}"
        print(
            f"  {k:<19} | {glon_txt:>11} | {eb_txt:<25} | "
            f"{dg_txt:<23} | {fit.chi2:7.1f}/{fit.ndof:<4d} | "
            f"{d_chi2:7.2f} | {p_w:7.3g}/{z_w:5.2f} | {mc_txt}"
        )


def _print_global_fit_table(
    *,
    kind: str,
    title: str,
    fitting: Fitting,
    rows: Sequence[tuple[str, GlobalFitResult | None, str]],
    do_mc: bool,
    use_profile_errors: bool = False,
) -> None:
    print(f"[fit_test] {kind}: {title}")
    hdr = (
        "  fit                 |  Ebreak [GeV]             |  dgamma"
        "                  | chi2/ndof      |  Δχ² vs PL | Wilks p/Z       | MC p/Z (exc/N)"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, g, pl_name in rows:
        if g is None:
            continue
        prof_err = getattr(g, "profile_errors", None) if use_profile_errors else None
        eb_txt = _fmt_global_param(
            g, "Ebreak", fmt=".2f", unit="GeV", profile_errors=prof_err
        )
        dg_txt = _fmt_global_param(g, "dgamma", fmt=".3f", profile_errors=prof_err)
        chi2_pl = (
            fitting.total_chi2(pl_name)
            if pl_name in next(iter(fitting.window_fits.values())).fits
            else float("nan")
        )
        d_chi2 = float(chi2_pl - g.chi2) if np.isfinite(chi2_pl) else float("nan")
        w = g.wilks_results.get(pl_name, {})
        p_w = w.get("p_wilks", float("nan"))
        z_w = w.get("Z_wilks", float("nan"))
        mc = g.mc_results.get(pl_name) if do_mc else None
        mc_txt = _fmt_mc(mc)
        print(
            f"  {label:<19} | {eb_txt:<25} | {dg_txt:<23} | "
            f"{g.chi2:7.1f}/{g.ndof:<4d} | {d_chi2:9.2f} | "
            f"{p_w:7.3g}/{z_w:5.2f} | {mc_txt}"
        )
        _print_global_mc_delta_check(label, d_chi2, mc)


def _print_per_window_summary(
    fitting: Fitting,
    *,
    kind: str,
    pl_name: str = "PL",
    sbpl_name: str = "SBPL",
    pl_prof_name: str | None = None,
    sbpl_prof_name: str | None = None,
    do_mc: bool = False,
) -> None:
    # When profiled names are given AND both profiled fits exist, the
    # whole table (Δχ², Wilks, MC) is built from the profiled pair --
    # otherwise we fall back to the classical (PL, SBPL) entries.
    use_prof = bool(pl_prof_name) and bool(sbpl_prof_name)
    a_name = pl_prof_name if use_prof else pl_name
    b_name = sbpl_prof_name if use_prof else sbpl_name

    keys = sorted(fitting.window_fits.keys(), key=lambda k: fitting[k].lowl)
    flavour = "profiled" if use_prof else "classical"
    print(
        f"[fit_test] {kind}: Per-window summary ({len(keys)} windows, "
        f"{flavour} {a_name} -> {b_name})"
    )
    hdr = (
        f"  window_key           |  |l| range    |  Δχ²({a_name}→{b_name})  |  "
        "Wilks p / Z   |  MC p / Z (exc/N)"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for k in keys:
        wf = fitting[k]
        d_chi2 = (
            float(wf.delta_chi2(a_name, b_name))
            if (a_name in wf and b_name in wf)
            else float("nan")
        )
        w = wf.wilks_results.get((a_name, b_name), {})
        p_w = w.get("p_wilks", float("nan"))
        z_w = w.get("Z_wilks", float("nan"))
        mc = wf.mc_results.get((a_name, b_name)) if do_mc else None
        mc_txt = _fmt_mc(mc)

        glon_txt = f"{wf.lowl:4.0f}-{wf.highl:4.0f}"
        print(
            f"  {k:<19} | {glon_txt:>11} | {d_chi2:>12.2f}   | "
            f"{p_w:>7.3g} / {z_w:>5.2f} | {mc_txt}"
        )


def _print_profile_errors_summary(
    fitting: Fitting,
    *,
    kind: str,
    sbpl_prof_name: str,
    g_prof=None,
) -> None:
    """Per-window asymmetric ±1σ Δχ²=1 profile-likelihood errors on
    ``(Ebreak, dgamma)`` for the profiled SBPL fit, plus the joint row.

    Reads ``wf.profile_errors[sbpl_prof_name]`` (populated by
    :meth:`Fitting.fit_per_window`) and ``g_prof.profile_errors``
    (populated by :meth:`Fitting.fit_global`); does **not** recompute.
    """
    keys = sorted(fitting.window_fits.keys(), key=lambda k: fitting[k].lowl)
    print(
        f"\n[fit_test] {kind}: Per-window profile-likelihood ±1σ "
        f"(Δχ²=1, {sbpl_prof_name})"
    )
    hdr = (
        f"  window_key           |  |l| range    |   Ebreak [GeV]"
        f"               |   dgamma"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    def _row(label: str, glon_txt: str, errs: dict | None) -> None:
        if not errs:
            eb_txt = "             n/a"
            dg_txt = "          n/a"
        else:
            eb = errs.get("Ebreak")
            dg = errs.get("dgamma")
            eb_txt = (
                _fmt_asym(eb[0], eb[1], eb[2], fmt=".2f")
                if eb is not None
                else "             n/a"
            )
            dg_txt = (
                _fmt_asym(dg[0], dg[1], dg[2], fmt=".3f")
                if dg is not None
                else "          n/a"
            )
        print(f"  {label:<19} | {glon_txt:>11} |  {eb_txt:<24}  |  {dg_txt}")

    for k in keys:
        wf = fitting[k]
        glon_txt = f"{wf.lowl:4.0f}-{wf.highl:4.0f}"
        errs = getattr(wf, "profile_errors", {}).get(sbpl_prof_name)
        _row(k, glon_txt, errs)

    if g_prof is not None and getattr(g_prof, "profile_errors", None):
        l_lo = min(fitting[k].lowl for k in keys)
        l_hi = max(fitting[k].highl for k in keys)
        glob_txt = f"{l_lo:4.0f}-{l_hi:4.0f}"
        _row("GLOBAL (joint SBPL)", glob_txt, g_prof.profile_errors)


def _run_one_dataset(
    *,
    kind: str,
    mesh: SkyMesh,
    cfg: AnalysisConfig,
    n_workers: int,
    outdir: str,
) -> _DatasetRun:
    """Run extraction + fits + scan for a single ``kind``.

    Returns the bundle holding the populated :class:`Fitting`, the joint
    :class:`GlobalFitResult`, and the optional :class:`ChiSquareScan`
    (per-window Wilks / MC results live on each ``WindowFitResult``).
    """
    nu, mc, pe, sc = cfg.nuisance, cfg.mc, cfg.profile_errors, cfg.scan
    cr = cfg.cr_reference
    do_profiled = not nu.classical_only
    mode = nu.mode
    # Global SBPL fits share the free break parameters (a fixed Ebreak,
    # model.sbpl_ebreak_fixed, is not one); a free s stays per window.
    # The CR fixed-shape fits freeze whichever of (Ebreak, dgamma) is free.
    sbpl_names = SmoothBrokenPowerLaw.from_config(cfg.model).param_names
    sbpl_shared = tuple(p for p in ("Ebreak", "dgamma") if p in sbpl_names)
    cr_shape = {p: v for p, v in (("Ebreak", cr.ebreak_gev), ("dgamma", cr.dgamma)) if p in sbpl_names}

    _dataset_banner(kind)
    _sep(f"[{kind}] Extraction")
    print(f"[fit_test] Extracting {kind} ...")
    data = Data(mesh)
    data.build(kind, cfg=cfg, subtraction=True)
    print(f"[fit_test] Extraction successful for {_mesh_summary(mesh)}.")
    store_classical_deltas = mc.store_deltachi2 and not do_profiled
    store_profiled_deltas = mc.store_deltachi2 and do_profiled
    # Settings shared by every MC call (per-window and global, classical and profiled).
    mc_toys = dict(
        seed=mc.seed,
        sample_theta=mc.sample_theta,
        warm_start=mc.warm_start,
        n_workers=n_workers,
        batch_size=mc.batch_size,
    )

    _sep(f"[{kind}] CLASSICAL fits")
    print("[fit_test] Classical per-window fits: PL + SBPL ...")
    if not mc.do_windows:
        print("[fit_test] MC per-window: not running (mc.do_windows=false).")
    if not mc.do_global:
        print("[fit_test] MC global free/fixed: not running (mc.do_global=false).")
    fitting = Fitting(data, solver=cfg.solver)
    fitting.fit_per_window(
        [PowerLaw.from_config(cfg.model), SmoothBrokenPowerLaw.from_config(cfg.model)],
        wilks_pairs=[("PL", "SBPL")],
        do_montecarlo=mc.do_windows,
        mc_kwargs=dict(N=mc.n_toys_window, print_every=0, store_deltas=store_classical_deltas, **mc_toys),
    )

    print(f"[fit_test] Classical global SBPL (shared {','.join(sbpl_shared)}) ...")
    g = fitting.fit_global(
        SmoothBrokenPowerLaw.from_config(cfg.model),
        shared=sbpl_shared,
        wilks_against=["PL"],
        do_montecarlo=mc.do_global,
        mc_against=["PL"] if mc.do_global else None,
        mc_kwargs=dict(N=mc.n_toys_global_free, print_every=_MC_FINAL_PRINT_EVERY, store_deltas=store_classical_deltas, **mc_toys),
    )

    print(
        f"[fit_test] Fixed-shape SBPL at Ebreak={cr.ebreak_gev} GeV, "
        f"dgamma={cr.dgamma}"
    )
    g_fix = fitting.fit_global(
        FrozenSpectralModel(SmoothBrokenPowerLaw.from_config(cfg.model), **cr_shape),
        shared=(),
        wilks_against=["PL"],
        do_montecarlo=mc.do_global,
        mc_against=["PL"] if mc.do_global else None,
        mc_kwargs=dict(N=mc.n_toys_global_fixed, print_every=_MC_FINAL_PRINT_EVERY, store_deltas=store_classical_deltas, **mc_toys),
    )

    _print_window_fit_table(
        fitting,
        kind=kind,
        title="Classical per-window SBPL",
        pl_name="PL",
        sbpl_name="SBPL",
        do_mc=mc.do_windows,
    )
    _print_global_fit_table(
        kind=kind,
        title="Classical global SBPL",
        fitting=fitting,
        rows=[
            ("free", g, "PL"),
            ("fixed", g_fix, "PL"),
        ],
        do_mc=mc.do_global,
    )

    # Profiled (nuisance) fits ------------------------------------------
    # NB: we run profiled fits *before* the scan so that, if the user
    # asks for both, the scan can be driven by the same NuisanceModel
    # (i.e. profiled chi² in every cell, anchored at ``g_prof.chi2``).
    # Without this, a scan paired with a profiled global anchor would
    # mix classical and profiled chi² and the contour would be
    # statistically meaningless.
    #
    # Two flavours, mutually exclusive (see ``NuisanceModel.share_nuisance``):
    #   * independent (default): one ``w_j`` per window; order is
    #     fit_per_window → fit_global → scan (legacy behaviour).
    #   * common / shared (``share_nuisance=True``): a single
    #     ``w_shared`` for the whole dataset; we MUST run
    #     fit_global(nm_pl) and fit_global(nm_sbpl) FIRST so that
    #     ``w_shared`` is determined and propagated, and only then
    #     fit_per_window/scan (which will use the frozen ``w_shared``
    #     via the short-circuit in ``NuisanceModel.profile_w``).
    g_prof: GlobalFitResult | None = None
    g_pl_prof: GlobalFitResult | None = None
    g_fix_prof: GlobalFitResult | None = None
    pl_prof_name: str | None = None
    sbpl_prof_name: str | None = None
    nm_sbpl: NuisanceModel | None = None
    control_fitting: Fitting | None = None
    control_pl_prof_name: str | None = None
    share_for_fit = mode == "common"
    mc_kwargs_win = dict(N=mc.n_toys_window, print_every=0, store_deltas=store_profiled_deltas, **mc_toys)
    mc_kwargs_glob_free = dict(N=mc.n_toys_global_free, print_every=_MC_FINAL_PRINT_EVERY, store_deltas=store_profiled_deltas, **mc_toys)
    mc_kwargs_glob_fixed = dict(N=mc.n_toys_global_fixed, print_every=_MC_FINAL_PRINT_EVERY, store_deltas=store_profiled_deltas, **mc_toys)
    # 1-D profile-likelihood error scans.  Ebreak/dgamma use the absolute
    # config grids (windows and global fit); gamma1/gamma2 get per-window
    # grids of +-gamma_radius around each window's best fit
    # (``Fitting.fit_per_window``).  The global fit only profiles the
    # shared Ebreak/dgamma.
    pe_kwargs = {
        "n_workers": n_workers,
        "params": pe.params,
        "delta": pe.delta_chi2,
        "grids": {
            "Ebreak": np.linspace(*pe.ebreak_grid),
            "dgamma": np.linspace(*pe.dgamma_grid),
        },
        "gamma_radius": pe.gamma_radius,
        "gamma_n_points": pe.gamma_n_points,
    }
    bp = cfg.break_polish
    break_polish_kwargs = {
        "radius_Ebreak": bp.radius_ebreak,
        "radius_dgamma": bp.radius_dgamma,
        "n_Ebreak": bp.n_ebreak,
        "n_dgamma": bp.n_dgamma,
        "n_workers": n_workers,
        "print_every": 0,
        "min_improvement": bp.min_improvement,
    }

    # Plot stamp: when running in classical mode, show it explicitly.
    stamp_nuisance_mode = "Classical" if not do_profiled else mode
    stamp_prior = nu.prior if cfg.plots.stamp_prior_nuisance else None

    if do_profiled:
        signed_sbpl = mode in {"control_wsbpl", "control_common_wsbpl"}
        flavour = {
            "common": "shared w (common nuisance)",
            "control": "fixed w from high-latitude PL control",
            "control_common": "fixed all-sky PL w + low-latitude common SBPL w",
            "control_wsbpl": "fixed w from separate high-latitude PL/SBPL controls",
            "control_common_wsbpl": (
                "fixed all-sky w from separate low+high PL/SBPL controls"
            ),
        }.get(mode, "independent w")
        _sep(f"[{kind}] PROFILED nuisance fits")
        print(f"[fit_test] Profiled mode: {flavour}")
        nm_kwargs = nuisance_kwargs_from_config(nu)
        # Resolve the per-window predictor names (PL|prof.X / SBPL|prof.X).
        short = nm_kwargs["prior_factory"](nu.n_nodes).short_name
        pl_prof_name = f"PL|prof.{short}"
        sbpl_prof_name = f"SBPL|prof.{short}"
        print(
            f"[fit_test] Profiled fits: prior={nu.prior} (short={short}), "
            f"n_nuis={nu.n_nodes}, basis={nu.basis}, "
            f"mode={mode}"
        )
        nm_pl = NuisanceModel(
            spectral_model=PowerLaw.from_config(cfg.model), share_nuisance=share_for_fit, **nm_kwargs
        )
        nm_sbpl = NuisanceModel(
            spectral_model=SmoothBrokenPowerLaw.from_config(cfg.model, signed_dgamma=signed_sbpl),
            share_nuisance=share_for_fit,
            **nm_kwargs,
        )
        if mode in {
            "control",
            "control_wsbpl",
            "control_common",
            "control_common_wsbpl",
        }:
            control_pl_prof_name = pl_prof_name
            print("[fit_test] Profiled control: high-latitude nuisance ...")
            control_builder = ControlNuisanceBuilder(
                kind=kind,
                signal_fitting=fitting,
                mesh=mesh,
                cfg=cfg,
            )
            control_method = (
                control_builder.fit_allsky
                if mode in {"control_common", "control_common_wsbpl"}
                else control_builder.fit_by_glon
            )
            w_pl_by_glon = control_method(lambda: PowerLaw.from_config(cfg.model), pl_prof_name)
            nm_pl.w_shared_by_glon = w_pl_by_glon
            if mode == "control_common":
                g_pl_prof = control_builder.allsky_globals.get(pl_prof_name)
            if mode == "control_common":
                nm_sbpl = replace(nm_sbpl, share_nuisance=True)
                nm_sbpl_seed = replace(nm_sbpl, share_nuisance=False)
                fitting.fit_per_window(
                    [nm_sbpl_seed],
                    do_profile_errors=False,
                    progress=False,
                )
                w_init_sbpl = np.mean(
                    np.stack(
                        [
                            np.asarray(wf.fits[sbpl_prof_name].model.last_w, float)
                            for wf in fitting.window_fits.values()
                        ],
                        axis=0,
                    ),
                    axis=0,
                )
                _subsep(f"[{kind}] Global Profiled free-shape")
                print(
                    "[fit_test] Profiled global SBPL "
                    "(low-latitude common w, shared Ebreak,dgamma) ..."
                )
                g_prof = fitting.fit_global(
                    nm_sbpl,
                    shared=sbpl_shared,
                    classical_seed=g,
                    w_shared0=w_init_sbpl,
                    polish_break_scan_kwargs=break_polish_kwargs,
                    profile_errors_kwargs=pe_kwargs,
                    progress=True,
                )
            elif mode in {"control_wsbpl", "control_common_wsbpl"}:
                w_sbpl_by_glon = control_method(
                    lambda: SmoothBrokenPowerLaw.from_config(cfg.model, signed_dgamma=True),
                    sbpl_prof_name,
                    refit_control=True,
                )
                nm_sbpl.w_shared_by_glon = w_sbpl_by_glon
            control_fitting = control_builder.control_fitting
            fitting.fit_per_window(
                [nm_pl, nm_sbpl],
                wilks_pairs=[(pl_prof_name, sbpl_prof_name)],
                do_montecarlo=mc.do_windows,
                mc_kwargs=mc_kwargs_win,
                profile_errors_kwargs=pe_kwargs,
                progress=True,
            )
            if mode == "control_common":
                g_prof.wilks_vs(
                    fitting.total_chi2(pl_prof_name),
                    fitting.total_n_params(pl_prof_name),
                    name_simpler=pl_prof_name,
                )
                if mc.do_global:
                    g_prof.mc_pvalue(
                        fitting,
                        name_pl=pl_prof_name,
                        control_global=g_pl_prof,
                        **mc_kwargs_glob_free,
                    )
            else:
                _subsep(f"[{kind}] Global Profiled free-shape")
                print(f"[fit_test] Profiled global SBPL (control-fixed w) ...")
                g_prof = fitting.fit_global(
                    nm_sbpl,
                    shared=sbpl_shared,
                    classical_seed=g,
                    wilks_against=[pl_prof_name],
                    do_montecarlo=mc.do_global,
                    mc_against=[pl_prof_name] if mc.do_global else None,
                    mc_kwargs=mc_kwargs_glob_free,
                    polish_break_scan_kwargs=break_polish_kwargs,
                    profile_errors_kwargs=pe_kwargs,
                    progress=True,
                )
        elif share_for_fit:
            # 0) Warm-start seed for ``w_shared``: the joint outer
            #    has a strong basin near ``w = 0`` (the prior origin),
            #    so cold-starting from zero often gets stuck there.
            #    Run a *throwaway* per-window profiled fit in
            #    independent mode and average ``last_w`` across
            #    windows -- separately for PL and SBPL since their
            #    deformations need not coincide.  These average
            #    vectors seed the joint optimisers below.
            print(
                f"[fit_test] Warm-start seed: per-window profiled "
                f"(independent w) for PL and SBPL ..."
            )
            nm_pl_seed = replace(nm_pl, share_nuisance=False)
            nm_sbpl_seed = replace(nm_sbpl, share_nuisance=False)
            fitting.fit_per_window(
                [nm_pl_seed, nm_sbpl_seed],
                do_profile_errors=False,
                progress=False,
            )

            def _avg_last_w(name: str) -> np.ndarray:
                ws = []
                for k, wf in fitting.window_fits.items():
                    if name in wf.fits:
                        ws.append(np.asarray(wf.fits[name].model.last_w, float))
                if not ws:
                    return np.zeros(nu.n_nodes, dtype=float)
                return np.mean(np.stack(ws, axis=0), axis=0)

            w_init_pl = _avg_last_w(pl_prof_name)
            w_init_sbpl = _avg_last_w(sbpl_prof_name)
            print(
                f"[fit_test] seed |w0_PL|_2={np.linalg.norm(w_init_pl):.3g}, "
                f"|w0_SBPL|_2={np.linalg.norm(w_init_sbpl):.3g}"
            )

            # 1) Global profiled PL with a single ``w_shared_PL`` across
            #    all windows (no spectral params shared).
            print(f"[fit_test] Profiled global PL (shared w) ...")
            g_pl_prof = fitting.fit_global(
                nm_pl,
                shared=(),
                w_shared0=w_init_pl,
                progress=True,
            )
            # 2) Global profiled SBPL with shared (Ebreak, dgamma) AND a
            #    single ``w_shared_SBPL``.  ``classical_seed=g`` keeps
            #    the legacy safety net.  Wilks/MC against PL profiled is
            #    deferred to step 4 because PL profiled per-window does
            #    not exist yet.
            _subsep(f"[{kind}] Global Profiled free-shape")
            print(
                f"[fit_test] Profiled global SBPL (shared w, "
                f"shared Ebreak,dgamma) ..."
            )
            g_prof = fitting.fit_global(
                nm_sbpl,
                shared=sbpl_shared,
                classical_seed=g,
                w_shared0=w_init_sbpl,
                polish_break_scan_kwargs=break_polish_kwargs,
                profile_errors_kwargs=pe_kwargs,
                progress=True,
            )
            # 3) Per-window analysis NOW: ``w_shared_*`` is already
            #    propagated into ``nm_pl`` / ``nm_sbpl`` by ``fit_global``,
            #    so the inner short-circuit in ``profile_w`` keeps
            #    ``w_shared_*`` frozen for every window.  Wilks + MC are
            #    computed on the *profiled* (PL|prof, SBPL|prof) pair.
            print(f"[fit_test] Profiled per-window fits (w_shared frozen) ...")
            fitting.fit_per_window(
                [nm_pl, nm_sbpl],
                wilks_pairs=[(pl_prof_name, sbpl_prof_name)],
                do_montecarlo=mc.do_windows,
                mc_kwargs=mc_kwargs_win,
                profile_errors_kwargs=pe_kwargs,
                progress=True,
            )
            # 4) Now total_chi2(pl_prof_name) is well defined → close the
            #    global Wilks + MC against the profiled PL.
            g_prof.wilks_vs(
                fitting.total_chi2(pl_prof_name),
                fitting.total_n_params(pl_prof_name),
                name_simpler=pl_prof_name,
            )
            if mc.do_global:
                g_prof.mc_pvalue(
                    fitting,
                    name_pl=pl_prof_name,
                    null_global=g_pl_prof,
                    **mc_kwargs_glob_free,
                )
        else:
            # Independent w per window: legacy order works because PL
            # profiled per-window is finished before fit_global needs it.
            fitting.fit_per_window(
                [nm_pl, nm_sbpl],
                wilks_pairs=[(pl_prof_name, sbpl_prof_name)],
                do_montecarlo=mc.do_windows,
                mc_kwargs=mc_kwargs_win,
                profile_errors_kwargs=pe_kwargs,
                progress=True,
            )
            _subsep(f"[{kind}] Global Profiled free-shape")
            print(f"[fit_test] Profiled global SBPL (shared {','.join(sbpl_shared)}) ...")
            g_prof = fitting.fit_global(
                nm_sbpl,
                shared=sbpl_shared,
                classical_seed=g,
                wilks_against=[pl_prof_name],
                do_montecarlo=mc.do_global,
                mc_against=[pl_prof_name] if mc.do_global else None,
                mc_kwargs=mc_kwargs_glob_free,
                polish_break_scan_kwargs=break_polish_kwargs,
                profile_errors_kwargs=pe_kwargs,
                progress=True,
            )

        # Profiled fixed-shape SBPL vs profiled PL (analogue of the
        # classical fixed-shape block, but driven by a NuisanceModel so
        # chi² is profiled in every window).
        _subsep(f"[{kind}] Global Profiled fixed-shape")
        print(
            f"[fit_test] Fixed-shape SBPL at Ebreak={cr.ebreak_gev} GeV, "
            f"dgamma={cr.dgamma} (profiled)"
        )
        sbpl_common_for_fit = share_for_fit or mode == "control_common"
        nm_sbpl_fixed = NuisanceModel(
            spectral_model=FrozenSpectralModel(SmoothBrokenPowerLaw.from_config(cfg.model), **cr_shape),
            share_nuisance=sbpl_common_for_fit,
            **nm_kwargs,
        )
        g_fix_prof = fitting.fit_global(
            nm_sbpl_fixed,
            shared=(),
            classical_seed=g_fix,
            w_shared0=(
                w_init_sbpl
                if (sbpl_common_for_fit and "w_init_sbpl" in locals())
                else None
            ),
            wilks_against=[pl_prof_name],
            do_montecarlo=mc.do_global,
            mc_against=[pl_prof_name] if mc.do_global else None,
            mc_kwargs=(
                {**mc_kwargs_glob_fixed, "null_global": g_pl_prof}
                if share_for_fit
                else (
                    {**mc_kwargs_glob_fixed, "control_global": g_pl_prof}
                    if mode == "control_common"
                    else mc_kwargs_glob_fixed
                )
            ),
            progress=True,
        )

        _print_window_fit_table(
            fitting,
            kind=kind,
            title="Profiled per-window SBPL",
            pl_name=pl_prof_name,
            sbpl_name=sbpl_prof_name,
            use_profile_errors=True,
            do_mc=mc.do_windows,
        )
        _print_global_fit_table(
            kind=kind,
            title="Profiled global SBPL",
            fitting=fitting,
            rows=[
                ("free", g_prof, pl_prof_name),
                ("fixed", g_fix_prof, pl_prof_name),
            ],
            do_mc=mc.do_global,
            use_profile_errors=True,
        )
        # ----------------------------------------------------------------
        # Throwaway diagnostic: per-window 1-D Δχ² profile-likelihood plots
        # for Ebreak / dgamma.  Useful to debug ``nan`` brackets; remove
        # once no longer needed.  See ``plotting/plot_profile_curves.py``.
        # ----------------------------------------------------------------
        try:
            from ..plotting.plot_profile_curves import plot_profile_curves

            plot_profile_curves(
                fitting,
                sbpl_prof_name=sbpl_prof_name,
                global_result=g_prof,
                out_dir=outdir,
                out_prefix=f"{kind.lower()}_profile_curves",
                title_prefix=f"[{kind}] ",
                prior_kind=stamp_prior,
                nuisance_mode=stamp_nuisance_mode,
            )
        except Exception as exc:  # pragma: no cover -- diagnostic only
            print(f"[fit_test] plot_profile_curves failed: {exc}")

    if mc.store_deltachi2:
        if do_profiled and pl_prof_name and sbpl_prof_name:
            for p in plot_mc_diagnostics(
                fitting,
                kind=kind,
                mode="profiled",
                prior_kind=stamp_prior,
                nuisance_mode=stamp_nuisance_mode,
                pl_name=pl_prof_name,
                sbpl_name=sbpl_prof_name,
                g_free=g_prof,
                g_fixed=g_fix_prof,
                outdir=Path(outdir) / "mc_diagnostics",
            ):
                print(f"[fit_test] Saved MC diagnostics: {p.resolve()}")
        else:
            for p in plot_mc_diagnostics(
                fitting,
                kind=kind,
                mode="classical",
                prior_kind=stamp_prior,
                nuisance_mode=stamp_nuisance_mode,
                pl_name="PL",
                sbpl_name="SBPL",
                g_free=g,
                g_fixed=g_fix,
                outdir=Path(outdir) / "mc_diagnostics",
            ):
                print(f"[fit_test] Saved MC diagnostics: {p.resolve()}")

    scan = None
    if sc.enabled:
        Eb_grid = np.linspace(*sc.ebreak_range, sc.n_ebreak)
        dg_grid = np.linspace(*sc.dgamma_range, sc.n_dgamma)
        Eb_lo, Eb_hi = float(Eb_grid[0]), float(Eb_grid[-1])
        dg_lo, dg_hi = float(dg_grid[0]), float(dg_grid[-1])

        if g_prof is not None and nm_sbpl is not None:
            scan_model: NuisanceModel | SmoothBrokenPowerLaw = nm_sbpl
            scan_anchor = g_prof.chi2
            scan_init_global = g_prof
            scan_flavour = "profiled"
        else:
            scan_model = SmoothBrokenPowerLaw.from_config(cfg.model)
            scan_anchor = g.chi2
            scan_init_global = g
            scan_flavour = "classical"

        _sep(f"[{kind}] SCAN ({scan_flavour})")
        print(
            f"[fit_test] Scan SBPL χ² over {sc.n_ebreak}×{sc.n_dgamma} grid "
            f"(Eb∈[{Eb_lo:.2f},{Eb_hi:.2f}], dγ∈[{dg_lo:.3f},{dg_hi:.3f}]) "
            f"[{scan_flavour}] ..."
        )
        scan = fitting.scan(
            scan_model,
            Ebreak_grid=Eb_grid,
            dgamma_grid=dg_grid,
            chi2_anchor=scan_anchor,
            warm_start_from="PL",
            init_from_global=scan_init_global,
            n_workers=n_workers,
            print_every=max(1, sc.n_dgamma // 5),
        )
        print(f"  -> {scan}")

    # ------------------------------------------------------------------
    # PASS8-raw (subtraction=False): only per-window SBPL parameters.
    # No global/scan/MC/Wilks extra work. In profiled flows we reuse the
    # nuisance calibration already computed in the main run:
    #   - common mode: reuse nm_sbpl.w_shared (from the global profiled fit)
    #   - control modes: reuse nm_sbpl.w_shared_by_glon (from the control builder)
    # so the raw points are consistent with the chosen nuisance semantics.
    # ------------------------------------------------------------------
    raw_fitting: Fitting | None = None
    raw_sbpl_name: str | None = None
    if cfg.data.pass8_raw_comparison and kind == "PASS8":
        try:
            _sep("[PASS8] PASS8 raw comparison")
            print("[fit_test] Extracting PASS8 raw (subtraction=False) ...")
            raw_data = Data(mesh)
            raw_data.build(kind, cfg=cfg, subtraction=False)
            print(f"[fit_test] Raw extraction successful for {_mesh_summary(mesh)}.")
            raw_fitting = Fitting(raw_data, solver=cfg.solver)

            # Classical per-window PL + SBPL (seed the profiled safety net).
            print("[fit_test] PASS8-raw classical per-window fits ...")
            raw_fitting.fit_per_window(
                [PowerLaw.from_config(cfg.model), SmoothBrokenPowerLaw.from_config(cfg.model)],
                progress=False,
            )

            if do_profiled and sbpl_prof_name is not None and nm_sbpl is not None:
                nm_sbpl_raw = replace(nm_sbpl)
                if getattr(nm_sbpl, "w_shared", None) is not None:
                    nm_sbpl_raw.w_shared = np.asarray(
                        nm_sbpl.w_shared, dtype=float
                    ).copy()
                if getattr(nm_sbpl, "w_shared_by_glon", None) is not None:
                    nm_sbpl_raw.w_shared_by_glon = {
                        k: np.asarray(v, dtype=float).copy()
                        for k, v in nm_sbpl.w_shared_by_glon.items()
                    }
                raw_fitting.fit_per_window(
                    [nm_sbpl_raw],
                    do_profile_errors=True,
                    profile_errors_kwargs=pe_kwargs,
                    progress=False,
                )
                raw_sbpl_name = sbpl_prof_name
            else:
                raw_sbpl_name = "SBPL"

        except Exception as exc:  # pragma: no cover -- plotting aid only
            print(f"[fit_test] PASS8-raw extraction/fits failed: {exc}")
            raw_fitting = None
            raw_sbpl_name = None

    return _DatasetRun(
        kind=kind,
        fitting=fitting,
        g=g,
        scan=scan,
        g_prof=g_prof,
        pl_prof_name=pl_prof_name,
        sbpl_prof_name=sbpl_prof_name,
        g_pl_prof=g_pl_prof,
        share_nuisance=share_for_fit or mode == "control_common",
        control_fitting=control_fitting,
        control_pl_prof_name=control_pl_prof_name,
        raw_fitting=raw_fitting,
        raw_sbpl_name=raw_sbpl_name,
    )


# ----------------------------------------------------------------------
# Style helpers
# ----------------------------------------------------------------------


_KIND_DEFAULTS = {
    "PASS8": dict(color="cornflowerblue", marker="o"),
    "IEM": dict(color="tab:orange", marker="^"),
    "DRAGON": dict(color="tab:green", marker="D"),
}


def _kind_color(kind: str) -> str:
    return _KIND_DEFAULTS.get(kind.upper(), {"color": "tab:gray"})["color"]


def _kind_marker(kind: str) -> str:
    return _KIND_DEFAULTS.get(kind.upper(), {"marker": "s"})["marker"]


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def run(cfg: AnalysisConfig, *, outdir: str, n_workers: int) -> dict[str, str]:
    """Run the pipeline for every dataset in ``cfg.data.kinds`` and save plots.

    Returns
    -------
    dict[str, str]
        Mapping ``{plot_name: absolute_path}`` of all figures produced.
    """
    nu = cfg.nuisance
    glon_edges = cfg.sky.glon_edges
    glat_lims = (cfg.sky.glat_edges[0], cfg.sky.glat_edges[-1])
    control_glat_lims = cfg.sky.control_glat_lims
    scan_smooth_sigma = cfg.plots.scan_smooth_sigma
    # Plot stamp: when running in classical mode, show it explicitly.
    stamp_nuisance_mode = "Classical" if nu.classical_only else nu.mode
    stamp_prior = nu.prior if cfg.plots.stamp_prior_nuisance else None
    # CR reference drawn on the figures.
    cr = cfg.cr_reference
    cr_bands = dict(cr_ebr_band=cr.ebreak_band, cr_dgamma_band=cr.dgamma_band)
    cr_point = (cr.ebreak_gev, cr.dgamma)
    sed_kwargs = dict(sed_scaling=cfg.data.sed_scaling, sed_band=cfg.plots.sed_band, cr_ebr_band=cr.ebreak_band)
    nuisance_kwargs = dict(cr_ebr_band=cr.ebreak_band, nuisance_band=cfg.plots.nuisance_band)

    glon_bins = np.asarray(glon_edges, dtype=float)
    n_glon = int(glon_bins.size - 1)
    mesh = SkyMesh(
        glon_lims=(float(glon_bins[0]), float(glon_bins[-1])),
        glat_lims=glat_lims,
        Nglon=n_glon,
        Nglat=1,
        glon_bins=glon_bins,
    )

    runs = [
        _run_one_dataset(kind=kind, mesh=mesh, cfg=cfg, n_workers=n_workers, outdir=outdir)
        for kind in cfg.data.kinds
    ]

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(saved: dict[str, str]) -> None:
        if not cfg.run.save_checkpoint:
            return
        ckpt_path = out / "checkpoint.pkl.gz"
        meta_path = out / "checkpoint_meta.json"
        state = dict(
            created_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            outdir=str(out.resolve()),
            config=config_to_dict(cfg),
            # Flat keys read by the replot scripts (also present in older
            # checkpoints, which have no ``config``).
            glon_edges=glon_edges,
            glat_lims=glat_lims,
            kinds=cfg.data.kinds,
            prior_kind=nu.prior,
            nuisance_mode=nu.mode,
            basis_kind_prof=nu.basis,
            stamp_nuisance_mode=stamp_nuisance_mode,
            scan_smooth_sigma=scan_smooth_sigma,
            control_glat_lims=control_glat_lims,
            saved=dict(saved),
            runs=runs,
        )
        with gzip.open(ckpt_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in state.items() if k != "runs"},
                f,
                indent=2,
                sort_keys=True,
            )
        print(f"[fit_test] Saved checkpoint: {ckpt_path.resolve()}")

    # Fits first, so a failing figure does not lose them.
    _save_checkpoint({})

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    glon_tag = f"{int(glon_bins[0])}-{int(glon_bins[-1])}l_{n_glon}win"
    glat_tag = f"{int(glat_lims[0])}b{int(glat_lims[1])}"
    saved: dict[str, str] = {}

    # 1) Per-dataset SED (always one figure per dataset).
    #    When profiled fits are available, the SED shows the deformed
    #    prediction (solid) and the underlying physical prediction
    #    (dashed) for both PL and SBPL; the global SBPL line uses the
    #    profiled global fit if available so all curves are consistent.
    for r in runs:
        kind_u = r.kind.upper()
        style = SedStyle(
            data_color=_kind_color(kind_u),
            data_label=kind_u,
            ebreak_color=_kind_color(kind_u),
        )
        sed_path = out / f"{r.kind.lower()}_fit_sanity_{glat_tag}_{glon_tag}.png"
        plot_sed(
            r.fitting,
            g=(r.g_prof or r.g),
            style=style,
            pl_prof_name=r.pl_prof_name,
            sbpl_prof_name=r.sbpl_prof_name,
            **sed_kwargs,
            prior_kind=stamp_prior,
            nuisance_mode=stamp_nuisance_mode,
            save=sed_path,
        )
        print(f"[fit_test] Saved SED ({kind_u}): {sed_path.resolve()}")
        saved[f"sed_{kind_u}"] = str(sed_path.resolve())

        if r.control_fitting is not None and r.control_pl_prof_name is not None:
            c0, c1 = control_glat_lims
            ctrl_path = out / (
                f"{r.kind.lower()}_control_highlat_pl_"
                f"{int(c0)}b{int(c1)}_{glon_tag}.png"
            )
            plot_sed(
                r.control_fitting,
                g=None,
                style=style,
                pl_prof_name=r.control_pl_prof_name,
                sbpl_prof_name=r.sbpl_prof_name,
                **sed_kwargs,
                show_global=False,
                show_envelope=False,
                ebreak_band_legend=False,
                save=ctrl_path,
            )
            print(f"[fit_test] Saved control SED ({kind_u}): {ctrl_path.resolve()}")
            saved[f"sed_control_{kind_u}"] = str(ctrl_path.resolve())

        # Nuisance posterior plot.  Two flavours, mutually exclusive:
        #   * shared mode  -> single panel with ``w_shared`` and
        #                     ``s(E; w_shared)`` band from ``g_*_prof``.
        #   * independent  -> per-window grid using the bound NMs stored
        #                     on each WindowFitResult.
        if r.pl_prof_name or r.sbpl_prof_name:
            nuis_style = NuisanceStyle(
                data_color=_kind_color(kind_u),
                data_label=kind_u,
            )
            nuis_path = out / f"{r.kind.lower()}_nuisance_{glat_tag}_{glon_tag}.png"
            if r.share_nuisance and (r.g_pl_prof is not None or r.g_prof is not None):
                plot_nuisance(
                    r.fitting,
                    pl_prof_global=r.g_pl_prof,
                    sbpl_prof_global=r.g_prof,
                    style=nuis_style,
                    **nuisance_kwargs,
                    prior_kind=stamp_prior,
                    nuisance_mode=nu.mode,
                    stamp_nuisance_mode=stamp_nuisance_mode,
                    save=nuis_path,
                )
            else:
                plot_nuisance(
                    r.fitting,
                    pl_prof_name=r.pl_prof_name,
                    sbpl_prof_name=r.sbpl_prof_name,
                    style=nuis_style,
                    **nuisance_kwargs,
                    prior_kind=stamp_prior,
                    nuisance_mode=nu.mode,
                    stamp_nuisance_mode=stamp_nuisance_mode,
                    save=nuis_path,
                )
            print(f"[fit_test] Saved nuisance ({kind_u}): {nuis_path.resolve()}")
            saved[f"nuisance_{kind_u}"] = str(nuis_path.resolve())

    # 2) Overlay-capable params_vs_glon (1 dataset → single curve, N → overlay).
    params_specs = [
        ParamsDataset(
            label=r.kind.upper(),
            fitting=r.fitting,
            sbpl_prof_name=r.sbpl_prof_name,
            color=_kind_color(r.kind),
            marker=_kind_marker(r.kind),
            glon_offset_deg=(0.0 if i == 0 else (1.0 if i == 1 else -1.0)),
        )
        for i, r in enumerate(runs)
    ]
    # Optional PASS8-raw overlay: only for the combined params+contour plot.
    raw_specs = []
    for r in runs:
        if r.raw_fitting is None or r.raw_sbpl_name is None:
            continue
        if str(r.kind).upper() != "PASS8":
            continue
        raw_specs.append(
            ParamsDataset(
                label="PASS8-raw",
                fitting=r.raw_fitting,
                sbpl_name="SBPL",
                sbpl_prof_name=(
                    r.raw_sbpl_name if "prof." in str(r.raw_sbpl_name) else None
                ),
                color="cornflowerblue",
                marker="o",
                glon_offset_deg=-1.0,
                edge_color="cornflowerblue",
                fill=False,
                error_ls=":",
            )
        )
    if len(runs) == 1:
        stem = f"{runs[0].kind.lower()}_fit_sanity_{glat_tag}_{glon_tag}"
    else:
        stem = (
            f"overlay_{'_vs_'.join(r.kind.lower() for r in runs)}_{glat_tag}_{glon_tag}"
        )
    params_path = out / f"{stem}_params_vs_glon.png"
    plot_params_vs_glon(
        params_specs,
        **cr_bands,
        prior_kind=stamp_prior,
        nuisance_mode=stamp_nuisance_mode,
        save=params_path,
        highb_text=glat_lims[1],
        dgamma_ylim=(
            (-0.40, 0.50)
            if nu.mode in {"control_wsbpl", "control_common_wsbpl"}
            else (0.0, 0.50)
        ),
    )
    print(f"[fit_test] Saved params: {params_path.resolve()}")
    saved["params"] = str(params_path.resolve())

    # 2b) Same, plus γ1 / γ2 vs |l| (top right).
    params_gammas_path = out / f"{stem}_params_gammas_vs_glon.png"
    plot_params_gammas_vs_glon(
        params_specs,
        **cr_bands,
        gamma_radius=cfg.profile_errors.gamma_radius,
        s_ref=cfg.model.sbpl.p0.get("s"),   # free s: its start value, the fixed s of the other runs
        prior_kind=stamp_prior,
        nuisance_mode=stamp_nuisance_mode,
        save=params_gammas_path,
        highb_text=glat_lims[1],
        dgamma_ylim=(
            (-0.40, 0.50)
            if nu.mode in {"control_wsbpl", "control_common_wsbpl"}
            else (0.0, 0.50)
        ),
    )
    print(f"[fit_test] Saved params + gammas: {params_gammas_path.resolve()}")
    saved["params_gammas"] = str(params_gammas_path.resolve())

    # 3) Overlay-capable contour plot.
    if any(r.scan is not None for r in runs):
        contour_specs = []
        for r in runs:
            if r.scan is None:
                continue
            g_for_scan = r.g_prof if r.g_prof is not None else r.g
            if r.kind.upper() == "PASS8":
                contour_specs.append(
                    ContourDataset.pass8(r.scan, g=g_for_scan, label="PASS8")
                )
            elif r.kind.upper() == "IEM":
                contour_specs.append(
                    ContourDataset.iem(r.scan, g=g_for_scan, label="IEM")
                )
            else:
                contour_specs.append(
                    ContourDataset(label=r.kind.upper(), scan=r.scan, g=g_for_scan)
                )
        contour_path = out / f"{stem}_scan.png"
        plot_contour_plot(
            contour_specs,
            smooth_sigma=scan_smooth_sigma,
            levels=cfg.plots.contour_levels,
            cr_ref=cr_point,
            **cr_bands,
            prior_kind=stamp_prior,
            nuisance_mode=stamp_nuisance_mode,
            save=contour_path,
            highb_text=glat_lims[1],
        )
        print(f"[fit_test] Saved scan:   {contour_path.resolve()}")
        saved["scan"] = str(contour_path.resolve())

        # 4) Combined params-vs-glon + contour figure (single page).
        combo_path = out / f"{stem}_params_plus_contour.png"
        plot_params_vs_glon_plus_contour(
            params_datasets=(params_specs + raw_specs),
            contour_datasets=contour_specs,
            smooth_sigma=scan_smooth_sigma,
            levels=cfg.plots.contour_levels,
            cr_ref=cr_point,
            **cr_bands,
            prior_kind=stamp_prior,
            nuisance_mode=stamp_nuisance_mode,
            save=combo_path,
            highb_text=glat_lims[1],
            dgamma_ylim=(
                (-0.40, 0.49)
                if nu.mode
                in {"control_wsbpl", "control_common_wsbpl"}
                else (0.0, 0.49)
            ),
        )
        print(f"[fit_test] Saved combo:  {combo_path.resolve()}")
        saved["params_plus_contour"] = str(combo_path.resolve())

        for ds in contour_specs:
            heatmap_path = (
                out / f"{ds.label.lower()}_scan_heatmap_{glat_tag}_{glon_tag}.png"
            )
            plot_contour_heatmap(
                ds,
                cr_ref=cr_point,
                **cr_bands,
                prior_kind=stamp_prior,
                nuisance_mode=stamp_nuisance_mode,
                save=heatmap_path,
            )
            print(
                f"[fit_test] Saved scan heatmap ({ds.label}): {heatmap_path.resolve()}"
            )
            saved[f"scan_heatmap_{ds.label.upper()}"] = str(heatmap_path.resolve())

    # Again, now with the figure paths.
    _save_checkpoint(saved)

    return saved
