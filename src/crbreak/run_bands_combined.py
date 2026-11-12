"""Compare the |b| bands of one multi-band run, in its ``<day>/combined/<name>/`` folder.

    python -m crbreak.run_bands_combined results/runs/<day>/combined/<name> [--outdir DIR]

Reads ``<day>/b<lo>-<hi>/<name>/checkpoint.pkl.gz`` for every band of the
run's ``<day>/combined/<name>/config.yaml`` (one independent analysis per
|b| band, written by :mod:`crbreak.run`) and writes, per dataset kind:

* ``<kind>_params_vs_glon_bands.png``  Ebreak and Δγ vs |l|, one series per band
* ``<kind>_gammas_vs_glon_bands.png``  γ1 and γ2 vs |l|
* ``<kind>_params_gammas_vs_glon_bands.png``  both of the above in one figure (γ1, γ2 top right)
* ``<kind>_scan_bands.png``            global Δχ²(Ebreak, Δγ) contours (68%, 95%)
* ``summary.csv``                      every plotted value, per window and global

With a single |l| window the ``*_vs_glon`` figures are replaced by
``<kind>_params_vs_glat_bands.png``, ``<kind>_gammas_vs_glat_bands.png`` and
``<kind>_params_gammas_vs_glat_bands.png`` (one point per band, vs |b|).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from crbreak.checkpoint import load_pickle
from crbreak.config.analysis import config_from_checkpoint
from crbreak.plotting import (
    ContourDataset,
    ParamsDataset,
    plot_contour_plot,
    plot_gammas_vs_glon,
    plot_params_gammas_vs_glat,
    plot_params_gammas_vs_glon,
    plot_params_vs_glat,
    plot_params_vs_glon,
)
from crbreak.plotting.plot_params_vs_glon import window_interval

PARAMS = ("Ebreak", "dgamma", "gamma1", "gamma2")
# |b| is ordinal: one light-to-dark ramp per dataset kind, plus a marker (or,
# in the γ plot, a line style) per band so a band is never identified by
# colour alone.
_CMAPS = {"PASS8": "Blues", "IEM": "Oranges", "DRAGON": "Greens"}
_MARKERS = ("o", "s", "D", "^", "v", "P")
_LINESTYLES = ("-", "--", "-.", ":")


def _load(path: Path) -> dict:
    with gzip.open(path, "rb") as f:
        return load_pickle(f)


def _band_label(band: tuple[float, float]) -> str:
    return rf"${band[0]:g}^\circ<|b|<{band[1]:g}^\circ$"


def _fmt(x) -> str:
    return "" if x is None else f"{float(x):.6g}"


def _summary_rows(kind: str, band, glon_lims, r) -> list[list[str]]:
    """Per-window rows plus one global row (shared Ebreak, Δγ) for one band."""
    ds = ParamsDataset(label="", fitting=r.fitting, sbpl_prof_name=r.sbpl_prof_name)
    pl, sbpl = r.pl_prof_name or "PL", r.sbpl_prof_name or "SBPL"
    rows = []
    for k in sorted(r.fitting.window_fits, key=lambda k: r.fitting[k].lowl):
        wf = r.fitting[k]
        chi2_pl = wf[pl].chi2 if pl in wf else None
        chi2_sb = wf[sbpl].chi2 if sbpl in wf else None
        vals = [x for p in PARAMS for x in window_interval(wf, ds, p)]
        rows.append([kind, "window", *map(_fmt, (*band, wf.lowl, wf.highl, chi2_pl, chi2_sb,
                                                  wf[sbpl].ndof if sbpl in wf else None, *vals))])
    g = r.g_prof or r.g
    prof = g.profile_errors or {}
    vals = []
    for p in PARAMS:
        v, e = g.shared_params.get(p, (None, None))
        lo, hi = prof[p][1:] if p in prof else ((v - e, v + e) if v is not None else (None, None))
        vals += [v, lo, hi]
    rows.append([kind, "global", *map(_fmt, (*band, *glon_lims, None, g.chi2, g.ndof, *vals))])
    return rows


def band_dirs(combined_dir: str | Path, bands) -> list[Path]:
    """``<day>/b<lo>-<hi>/<name>`` of each |b| band of the run folder ``<day>/combined/<name>``."""
    combined_dir = Path(combined_dir)
    return [combined_dir.parent.parent / f"b{lo:g}-{hi:g}" / combined_dir.name for lo, hi in bands]


def run_bands_combined(combined_dir: str | Path, *, outdir: str | Path | None = None) -> dict[str, str]:
    combined_dir = Path(combined_dir)
    edges = yaml.safe_load((combined_dir / "config.yaml").read_text())["sky"]["glat_edges"]
    paths = [d / "checkpoint.pkl.gz" for d in band_dirs(combined_dir, zip(edges, edges[1:]))]
    states = sorted((_load(p) for p in paths if p.exists()), key=lambda s: tuple(s["glat_lims"]))
    if len(states) < 2:
        raise FileNotFoundError(f"need at least two band checkpoints among {[str(p) for p in paths]}")
    out = Path(outdir) if outdir is not None else combined_dir
    out.mkdir(parents=True, exist_ok=True)

    cfg = config_from_checkpoint(states[0])
    cr = cfg.cr_reference
    s0 = states[0]
    stamp = dict(prior_kind=s0["prior_kind"] if cfg.plots.stamp_prior_nuisance else None,
                 nuisance_mode=s0["stamp_nuisance_mode"])
    bands = [tuple(float(b) for b in s["glat_lims"]) for s in states]
    single_window = len(s0["glon_edges"]) == 2
    offsets = np.linspace(-1.5, 1.5, len(bands))
    dgamma_ylim = (-0.40, 0.50) if "wsbpl" in cfg.nuisance.mode else (0.0, 0.50)

    saved: dict[str, str] = {}
    rows: list[list[str]] = []
    for kind in cfg.data.kinds:
        runs = [next(r for r in s["runs"] if r.kind == kind) for s in states]
        cmap = plt.get_cmap(_CMAPS.get(kind.upper(), "Greys"))
        colors = [tuple(float(c) for c in cmap(x)) for x in np.linspace(0.4, 0.95, len(bands))]
        params = [
            ParamsDataset(label=_band_label(band), fitting=r.fitting, sbpl_prof_name=r.sbpl_prof_name,
                          color=colors[i], marker=_MARKERS[i % len(_MARKERS)],
                          linestyle=_LINESTYLES[i % len(_LINESTYLES)],
                          glon_offset_deg=float(offsets[i]))
            for i, (band, r) in enumerate(zip(bands, runs))
        ]
        stem = f"{kind.lower()}_{{}}_bands.png"

        if single_window:
            path = out / stem.format("params_vs_glat")
            plt.close(plot_params_vs_glat(params, bands, pnames=("Ebreak", "dgamma"),
                                          ref_bands={"Ebreak": cr.ebreak_band, "dgamma": cr.dgamma_band},
                                          ylims={"dgamma": dgamma_ylim}, save=path, **stamp))
            saved[f"params_vs_glat_{kind}"] = str(path.resolve())

            path = out / stem.format("gammas_vs_glat")
            plt.close(plot_params_vs_glat(params, bands, pnames=("gamma1", "gamma2"), save=path, **stamp))
            saved[f"gammas_vs_glat_{kind}"] = str(path.resolve())

            path = out / stem.format("params_gammas_vs_glat")
            plt.close(plot_params_gammas_vs_glat(params, bands,
                                                 ref_bands={"Ebreak": cr.ebreak_band, "dgamma": cr.dgamma_band},
                                                 ylims={"dgamma": dgamma_ylim},
                                                 gamma_radius=cfg.profile_errors.gamma_radius,
                                                 gamma_color=cmap(0.6), save=path, **stamp))
            saved[f"params_gammas_vs_glat_{kind}"] = str(path.resolve())
        else:
            path = out / stem.format("params_vs_glon")
            plt.close(plot_params_vs_glon(params, cr_ebr_band=cr.ebreak_band, cr_dgamma_band=cr.dgamma_band,
                                          dgamma_ylim=dgamma_ylim, legend_ncol=2, save=path, **stamp))
            saved[f"params_vs_glon_{kind}"] = str(path.resolve())

            path = out / stem.format("gammas_vs_glon")
            plt.close(plot_gammas_vs_glon(params, save=path, **stamp))
            saved[f"gammas_vs_glon_{kind}"] = str(path.resolve())

            path = out / stem.format("params_gammas_vs_glon")
            plt.close(plot_params_gammas_vs_glon(params, cr_ebr_band=cr.ebreak_band, cr_dgamma_band=cr.dgamma_band,
                                                 gamma_radius=cfg.profile_errors.gamma_radius,
                                                 s_ref=cfg.model.sbpl.p0.get("s"),
                                                 dgamma_ylim=dgamma_ylim, legend_ncol=2, save=path, **stamp))
            saved[f"params_gammas_vs_glon_{kind}"] = str(path.resolve())

        contours = [
            ContourDataset(label=ds.label, scan=r.scan, g=r.g_prof or r.g,
                           line_color=ds.color, band_rgba=((*ds.color[:3], 0.35), (*ds.color[:3], 0.10)),
                           marker=ds.marker, marker_color=ds.color)
            for ds, r in zip(params, runs) if r.scan is not None
        ]
        if contours:
            path = out / stem.format("scan")
            plt.close(plot_contour_plot(contours, smooth_sigma=cfg.plots.scan_smooth_sigma,
                                        levels=cfg.plots.contour_levels[:2], label_contours=False,
                                        cr_ref=(cr.ebreak_gev, cr.dgamma), cr_ebr_band=cr.ebreak_band,
                                        cr_dgamma_band=cr.dgamma_band, save=path, **stamp))
            saved[f"scan_{kind}"] = str(path.resolve())

        for band, s, r in zip(bands, states, runs):
            rows += _summary_rows(kind, band, (s["glon_edges"][0], s["glon_edges"][-1]), r)

    path = out / "summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "scope", "lowb", "highb", "lowl", "highl", "chi2_pl", "chi2_sbpl", "ndof_sbpl",
                    *(f"{p}{s}" for p in PARAMS for s in ("", "_lo", "_hi"))])
        w.writerows(rows)
    saved["summary"] = str(path.resolve())

    for k, p in saved.items():
        print(f"[combined] {k:>22}  ->  {p}")
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("combined_dir", help="<day>/combined/<name> folder of a multi-band run (holds its config.yaml)")
    ap.add_argument("--outdir", default=None, help="default: combined_dir")
    args = ap.parse_args()
    run_bands_combined(args.combined_dir, outdir=args.outdir)


if __name__ == "__main__":
    main()
