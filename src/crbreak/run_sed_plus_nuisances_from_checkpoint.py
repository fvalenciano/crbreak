from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path

import numpy as np

from crbreak.checkpoint import load_pickle
from crbreak.config.analysis import config_from_checkpoint
from crbreak.plotting import plot_sed_plus_nuisances
from crbreak.plotting.plot_sed import SedStyle
from crbreak.plotting import NuisanceStyle


_KIND_DEFAULTS = {
    "PASS8": dict(color="cornflowerblue"),
    "IEM": dict(color="tab:orange"),
}


def _kind_color(kind: str) -> str:
    return _KIND_DEFAULTS.get(str(kind).upper(), {"color": "tab:gray"})["color"]


def _load_checkpoint(path: str | Path) -> dict:
    p = Path(path)
    with gzip.open(p, "rb") as f:
        return load_pickle(f)


def run_sed_plus_nuisances_from_checkpoint(
    ckpt_path: str | Path,
    *,
    outdir: str | Path | None = None,
    show: bool = False,
) -> dict[str, str]:
    s = _load_checkpoint(ckpt_path)
    runs = list(s["runs"])
    cfg = config_from_checkpoint(s)
    prior_kind = s.get("prior_kind") if cfg.plots.stamp_prior_nuisance else None
    cr = cfg.cr_reference
    nuisance_mode = s.get("nuisance_mode") or s.get("stamp_nuisance_mode")
    stamp_nuisance_mode = s.get("stamp_nuisance_mode") or nuisance_mode

    glon_edges = np.asarray(s["glon_edges"], dtype=float)
    glat_lims = tuple(float(x) for x in s["glat_lims"])
    n_glon = int(glon_edges.size - 1)
    glon_tag = f"{int(glon_edges[0])}-{int(glon_edges[-1])}l_{n_glon}win"
    glat_tag = f"{int(glat_lims[0])}b{int(glat_lims[1])}"

    out = Path(outdir) if outdir is not None else Path(ckpt_path).parent
    out.mkdir(parents=True, exist_ok=True)

    saved: dict[str, str] = {}
    for r in runs:
        kind_u = r.kind.upper()
        is_pass8 = kind_u == "PASS8"
        data_label = "LAT subs." if is_pass8 else kind_u
        sed_style = SedStyle(
            data_color=_kind_color(kind_u),
            data_label=data_label,
            ebreak_color=("cornflowerblue" if is_pass8 else "tab:orange"),
        )
        nuis_style = NuisanceStyle(
            data_color=_kind_color(kind_u),
            data_label=data_label,
        )
        out_path = out / f"{r.kind.lower()}_sed_plus_nuisances_{glat_tag}_{glon_tag}.png"
        plot_sed_plus_nuisances(
            r.fitting,
            g=(r.g_prof or r.g),
            sed_style=sed_style,
            nuisance_style=nuis_style,
            pl_prof_name=r.pl_prof_name,
            sbpl_prof_name=r.sbpl_prof_name,
            sed_scaling=cfg.data.sed_scaling,
            sed_band=cfg.plots.sed_band,
            nuisance_band=cfg.plots.nuisance_band,
            cr_ebr_band=cr.ebreak_band,
            cr_ebr_fixed=cr.ebreak_gev,
            prior_kind=prior_kind,
            nuisance_mode=nuisance_mode,
            save=out_path,
            show=show,
        )
        saved[f"sed_plus_nuisances_{kind_u}"] = str(out_path.resolve())

    return saved


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create a long per-window plot: SED panel + nuisance panel below it."
    )
    ap.add_argument("checkpoint", help="Path to checkpoint.pkl.gz")
    ap.add_argument(
        "--outdir",
        default=None,
        help="Override output directory (defaults to the checkpoint's folder).",
    )
    ap.add_argument("--show", action="store_true", help="Call plt.show() after saving.")
    args = ap.parse_args()
    run_sed_plus_nuisances_from_checkpoint(
        args.checkpoint, outdir=args.outdir, show=bool(args.show)
    )


if __name__ == "__main__":
    main()

