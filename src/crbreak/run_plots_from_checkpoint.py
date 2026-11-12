from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path

import numpy as np

from crbreak.checkpoint import load_pickle
from crbreak.config.analysis import config_from_checkpoint
from crbreak.plotting import (
    plot_contour_plot,
    plot_nuisance,
    plot_params_gammas_vs_glon,
    plot_params_vs_glon,
    plot_sed,
)
from crbreak.plotting.plot_sed import SedStyle
from crbreak.plotting import ContourDataset, NuisanceStyle, ParamsDataset, plot_contour_heatmap, plot_params_vs_glon_plus_contour


_KIND_DEFAULTS = {
    "PASS8": dict(color="cornflowerblue", marker="o"),
    "IEM": dict(color="tab:orange", marker="^"),
    "DRAGON": dict(color="tab:green", marker="D"),
}


def _kind_color(kind: str) -> str:
    return _KIND_DEFAULTS.get(str(kind).upper(), {"color": "tab:gray"})["color"]


def _kind_marker(kind: str) -> str:
    return _KIND_DEFAULTS.get(str(kind).upper(), {"marker": "s"})["marker"]


def _kind_label(kind: str) -> str:
    return "LAT subs." if str(kind).upper() == "PASS8" else str(kind).upper()


def _load_checkpoint(path: str | Path) -> dict:
    p = Path(path)
    with gzip.open(p, "rb") as f:
        return load_pickle(f)


def run_plots_from_checkpoint(
    ckpt_path: str | Path,
    *,
    outdir: str | Path | None = None,
    show: bool = False,
    only: set[str] | None = None,
    gamma_overlay: str | Path | None = None,
) -> dict[str, str]:
    """Replot the figures of one checkpoint.

    ``gamma_overlay`` is a second checkpoint of the same |l| windows and |b|
    band (e.g. a DRAGON mock run): its γ1 / γ2 are overplotted in the
    ``params_gammas`` figure, written as ``..._params_gammas_vs_glon_vs_<name>.png``.
    """
    s = _load_checkpoint(ckpt_path)
    runs = list(s["runs"])
    cfg = config_from_checkpoint(s)
    prior_kind = s.get("prior_kind") if cfg.plots.stamp_prior_nuisance else None
    cr = cfg.cr_reference
    cr_bands = dict(cr_ebr_band=cr.ebreak_band, cr_dgamma_band=cr.dgamma_band)
    cr_point = (cr.ebreak_gev, cr.dgamma)
    sed_kwargs = dict(sed_scaling=cfg.data.sed_scaling, sed_band=cfg.plots.sed_band, cr_ebr_band=cr.ebreak_band)
    nuisance_kwargs = dict(cr_ebr_band=cr.ebreak_band, nuisance_band=cfg.plots.nuisance_band)
    # Two nuisance-mode notions can exist in older checkpoints:
    # - nuisance_mode: the actual fit configuration ("indep", "common", ...)
    # - stamp_nuisance_mode: a human-friendly display tag ("Classical", ...)
    # Plot layers that depend on the actual nuisance configuration should use this.
    nuisance_mode = s.get("nuisance_mode") or s.get("stamp_nuisance_mode")
    stamp_nuisance_mode = s.get("stamp_nuisance_mode") or nuisance_mode
    scan_smooth_sigma = float(s.get("scan_smooth_sigma", 1.0))
    control_glat_lims = tuple(s.get("control_glat_lims", (4.0, 90.0)))

    glon_edges = np.asarray(s["glon_edges"], dtype=float)
    glat_lims = tuple(float(x) for x in s["glat_lims"])
    n_glon = int(glon_edges.size - 1)
    glon_tag = f"{int(glon_edges[0])}-{int(glon_edges[-1])}l_{n_glon}win"
    glat_tag = f"{int(glat_lims[0])}b{int(glat_lims[1])}"

    out = Path(outdir) if outdir is not None else Path(ckpt_path).parent
    out.mkdir(parents=True, exist_ok=True)

    if len(runs) == 1:
        stem = f"{runs[0].kind.lower()}_fit_sanity_{glat_tag}_{glon_tag}"
    else:
        stem = (
            f"overlay_{'_vs_'.join(r.kind.lower() for r in runs)}_{glat_tag}_{glon_tag}"
        )

    saved: dict[str, str] = {}

    want_all = only is None or len(only) == 0
    want_sed = want_all or "sed" in only
    want_nuisance = want_all or "nuisance" in only
    want_params = want_all or "params" in only
    want_params_gammas = want_all or "params_gammas" in only
    want_scan = want_all or "scan" in only
    want_combo = want_all or "params_plus_contour" in only
    want_heatmaps = want_all or "scan_heatmap" in only

    # 1) Per-dataset SED + (optional) control SED + nuisance.
    if want_sed or want_nuisance:
        for r in runs:
            kind_u = r.kind.upper()
            is_pass8 = kind_u == "PASS8"
            data_label = "LAT subs." if is_pass8 else kind_u
            style = SedStyle(
                data_color=_kind_color(kind_u),
                data_label=data_label,
                ebreak_color=_kind_color(kind_u),
            )
            if want_sed:
                sed_path = (
                    out / f"{r.kind.lower()}_fit_sanity_{glat_tag}_{glon_tag}.png"
                )
                plot_sed(
                    r.fitting,
                    g=(r.g_prof or r.g),
                    style=style,
                    pl_prof_name=r.pl_prof_name,
                    sbpl_prof_name=r.sbpl_prof_name,
                    prior_kind=prior_kind,
                    nuisance_mode=stamp_nuisance_mode,
                    **sed_kwargs,
                    save=sed_path,
                    show=show,
                )
                saved[f"sed_{kind_u}"] = str(sed_path.resolve())

                if r.control_fitting is not None and r.control_pl_prof_name is not None:
                    c0, c1 = control_glat_lims
                    ctrl_path = out / (
                        f"{r.kind.lower()}_control_highlat_pl_{int(c0)}b{int(c1)}_{glon_tag}.png"
                    )
                    plot_sed(
                        r.control_fitting,
                        g=None,
                        style=style,
                        pl_prof_name=r.control_pl_prof_name,
                        sbpl_prof_name=r.sbpl_prof_name,
                        show_global=False,
                        show_envelope=False,
                        ebreak_band_legend=False,
                        **sed_kwargs,
                        save=ctrl_path,
                        show=show,
                    )
                    saved[f"sed_control_{kind_u}"] = str(ctrl_path.resolve())

            if want_nuisance and (r.pl_prof_name or r.sbpl_prof_name):
                nuis_style = NuisanceStyle(
                    data_color=_kind_color(kind_u),
                    data_label=data_label,
                )
                nuis_path = out / f"{r.kind.lower()}_nuisance_{glat_tag}_{glon_tag}.png"
                if r.share_nuisance and (
                    r.g_pl_prof is not None or r.g_prof is not None
                ):
                    plot_nuisance(
                        r.fitting,
                        pl_prof_global=r.g_pl_prof,
                        sbpl_prof_global=r.g_prof,
                        style=nuis_style,
                        prior_kind=prior_kind,
                        nuisance_mode=nuisance_mode,
                        stamp_nuisance_mode=stamp_nuisance_mode,
                        **nuisance_kwargs,
                        save=nuis_path,
                        show=show,
                    )
                else:
                    plot_nuisance(
                        r.fitting,
                        pl_prof_name=r.pl_prof_name,
                        sbpl_prof_name=r.sbpl_prof_name,
                        style=nuis_style,
                        prior_kind=prior_kind,
                        nuisance_mode=nuisance_mode,
                        stamp_nuisance_mode=stamp_nuisance_mode,
                        **nuisance_kwargs,
                        save=nuis_path,
                        show=show,
                    )
                saved[f"nuisance_{kind_u}"] = str(nuis_path.resolve())

    # 2) params_vs_glon
    params_specs = [
        ParamsDataset(
            label=_kind_label(r.kind),
            fitting=r.fitting,
            sbpl_prof_name=r.sbpl_prof_name,
            color=_kind_color(r.kind),
            marker=_kind_marker(r.kind),
            glon_offset_deg=(0.0 if i == 0 else (1.0 if i == 1 else -1.0)),
        )
        for i, r in enumerate(runs)
    ]
    raw_specs = []
    for r in runs:
        if r.raw_fitting is None or r.raw_sbpl_name is None:
            continue
        if str(r.kind).upper() != "PASS8":
            continue
        raw_specs.append(
            ParamsDataset(
                label="LAT",
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
    if want_params:
        params_path = out / f"{stem}_params_vs_glon.png"
        plot_params_vs_glon(
            params_specs,
            prior_kind=prior_kind,
            nuisance_mode=stamp_nuisance_mode,
            **cr_bands,
            save=params_path,
            show=show,
            highb_text=glat_lims[1],
            dgamma_ylim=(
                (-0.40, 0.50)
                if "wsbpl" in str(s.get("nuisance_mode", "")).lower()
                else (0.0, 0.50)
            ),
        )
        saved["params"] = str(params_path.resolve())
    if want_params_gammas:
        overlay_specs, overlay_tag = [], ""
        if gamma_overlay is not None:
            o = _load_checkpoint(gamma_overlay)
            if list(o["glon_edges"]) != list(s["glon_edges"]) or tuple(o["glat_lims"]) != tuple(s["glat_lims"]):
                raise ValueError(
                    f"--gamma-overlay: {gamma_overlay} has glon_edges={list(o['glon_edges'])}, "
                    f"glat_lims={tuple(o['glat_lims'])}; {ckpt_path} has "
                    f"{list(s['glon_edges'])}, {tuple(s['glat_lims'])}."
                )
            o_cfg = config_from_checkpoint(o)
            names = [f"{r.kind} {o_cfg.data.dragon_model}" if r.kind == "DRAGON" else _kind_label(r.kind)
                     for r in o["runs"]]
            overlay_specs = [
                ParamsDataset(label=name, fitting=r.fitting, sbpl_prof_name=r.sbpl_prof_name,
                              color=_kind_color(r.kind), marker=_kind_marker(r.kind))
                for name, r in zip(names, o["runs"])
            ]
            overlay_tag = "_vs_" + "_".join(n.replace(" ", "_").replace(".", "") for n in names)
        params_gammas_path = out / f"{stem}_params_gammas_vs_glon{overlay_tag}.png"
        plot_params_gammas_vs_glon(
            params_specs,
            gamma_overlay=overlay_specs,
            prior_kind=prior_kind,
            nuisance_mode=stamp_nuisance_mode,
            **cr_bands,
            gamma_radius=cfg.profile_errors.gamma_radius,
            s_ref=cfg.model.sbpl.p0.get("s"),   # free s: its start value, the fixed s of the other runs
            save=params_gammas_path,
            show=show,
            highb_text=glat_lims[1],
            dgamma_ylim=(
                (-0.40, 0.50)
                if "wsbpl" in str(s.get("nuisance_mode", "")).lower()
                else (0.0, 0.50)
            ),
        )
        saved["params_gammas"] = str(params_gammas_path.resolve())

    # 3) Contours + combo (only if scan exists)
    if any(getattr(r, "scan", None) is not None for r in runs):
        contour_specs = []
        for r in runs:
            if r.scan is None:
                continue
            g_for_scan = r.g_prof if r.g_prof is not None else r.g
            if r.kind.upper() == "PASS8":
                contour_specs.append(
                    ContourDataset.pass8(r.scan, g=g_for_scan, label="LAT subs.")
                )
            elif r.kind.upper() == "IEM":
                contour_specs.append(
                    ContourDataset.iem(r.scan, g=g_for_scan, label="IEM")
                )
            else:
                contour_specs.append(
                    ContourDataset(label=r.kind.upper(), scan=r.scan, g=g_for_scan)
                )
        if want_scan:
            contour_path = out / f"{stem}_scan.png"
            plot_contour_plot(
                contour_specs,
                smooth_sigma=scan_smooth_sigma,
                prior_kind=prior_kind,
                nuisance_mode=nuisance_mode,
                levels=cfg.plots.contour_levels,
                cr_ref=cr_point,
                **cr_bands,
                save=contour_path,
                show=show,
                highb_text=glat_lims[1],
            )
            saved["scan"] = str(contour_path.resolve())

        if want_combo:
            combo_path = out / f"{stem}_params_plus_contour.png"
            plot_params_vs_glon_plus_contour(
                params_datasets=(params_specs + raw_specs),
                contour_datasets=contour_specs,
                smooth_sigma=scan_smooth_sigma,
                prior_kind=prior_kind,
                nuisance_mode=nuisance_mode,
                levels=cfg.plots.contour_levels,
                cr_ref=cr_point,
                **cr_bands,
                save=combo_path,
                show=show,
                highb_text=glat_lims[1],
                dgamma_ylim=(
                    (-0.40, 0.49)
                    if "wsbpl" in str(s.get("nuisance_mode", "")).lower()
                    else (0.0, 0.49)
                ),
            )
            saved["params_plus_contour"] = str(combo_path.resolve())

        if want_heatmaps:
            for ds in contour_specs:
                heatmap_path = (
                    out / f"{ds.label.lower()}_scan_heatmap_{glat_tag}_{glon_tag}.png"
                )
                plot_contour_heatmap(
                    ds,
                    prior_kind=prior_kind,
                    nuisance_mode=nuisance_mode,
                    cr_ref=cr_point,
                    **cr_bands,
                    save=heatmap_path,
                    show=show,
                )
                saved[f"scan_heatmap_{ds.label.upper()}"] = str(heatmap_path.resolve())

    return saved


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Regenerate plots from a saved main_test checkpoint."
    )
    ap.add_argument("checkpoint", help="Path to checkpoint.pkl.gz")
    ap.add_argument(
        "--outdir",
        default=None,
        help="Override output directory (defaults to the checkpoint's folder).",
    )
    ap.add_argument("--show", action="store_true", help="Call plt.show() after saving.")
    ap.add_argument(
        "--gamma-overlay",
        default=None,
        help="Checkpoint of the same windows/band whose γ1, γ2 are overplotted in params_gammas.",
    )
    ap.add_argument(
        "--only",
        action="append",
        default=None,
        choices=(
            "sed",
            "nuisance",
            "params",
            "params_gammas",
            "scan",
            "params_plus_contour",
            "scan_heatmap",
        ),
        help="Regenerate only a subset of plots. Can be passed multiple times.",
    )
    args = ap.parse_args()
    only = set(args.only) if args.only else None
    run_plots_from_checkpoint(
        args.checkpoint, outdir=args.outdir, show=bool(args.show), only=only,
        gamma_overlay=args.gamma_overlay,
    )


if __name__ == "__main__":
    main()
