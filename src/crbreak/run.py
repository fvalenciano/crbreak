"""Run one analysis from a YAML config (every option lives in ``configs/``).

    python -m crbreak.run configs/baseline.yaml [--outdir DIR]
    python -m crbreak.run configs/baseline.yaml --print-outdir   # used by the sbatch

Each |b| band of ``sky.glat_edges`` is an independent analysis.  Outputs go to
``<parent>/<ddMonyy>/<band>/<name>`` (see :func:`default_outdir`): one band
writes straight into ``b<lo>-<hi>/<name>``; several bands write each band to
``b<lo>-<hi>/<name>`` and the merged config, logs and band comparison to
``combined/<name>`` (see :mod:`crbreak.run_bands_combined`).
"""
from __future__ import annotations

import argparse
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from crbreak.config.analysis import AnalysisConfig, dump_config, load_config
from crbreak.data_extract.diagnostics import run_bg_plot, run_ps_comparison_plot
from crbreak.fitting.pipeline import run as _run
from crbreak.run_bands_combined import band_dirs, run_bands_combined


def _auto_n_workers() -> int:
    """``$SLURM_CPUS_PER_TASK`` when running under SLURM, else ``os.cpu_count()``."""
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE"):
        v = os.environ.get(var)
        if v:
            try:
                return max(1, int(v.split(",")[0]))  # handles "4,4,4" forms
            except ValueError:
                pass
    return max(1, os.cpu_count() or 1)


def default_outdir(cfg: AnalysisConfig) -> Path:
    """``<parent>/<ddMonyy>/<band>/<kinds>[_glon<sign>][_<tag>][_v<i>]`` (first free name).

    ``<band>`` is ``b<lo>-<hi>`` with one |b| band.  With several it is ``combined``
    (merged config, logs, band comparison) and each band goes to ``b<lo>-<hi>/``
    under the same name (:func:`band_dirs`).  ``<kinds>`` is ``data.kinds`` in lower
    case (``pass8``, ``iem``, ``dragon``, ``pass8_iem``, ...).  ``_glonpos`` /
    ``_glonneg`` appear only when ``sky.glon_sign`` is not ``both``, ``_v<i>`` (i >= 2)
    only when the name is taken.  The rest of the setup is in ``config.yaml``.
    """
    day = Path(cfg.output.parent) / f"{datetime.now():%d%b%y}"
    bands = cfg.sky.bands
    sub = f"b{bands[0][0]:g}-{bands[0][1]:g}" if len(bands) == 1 else "combined"
    kinds = "_".join(k.lower() for k in cfg.data.kinds)
    sign = "" if cfg.sky.glon_sign == "both" else f"_glon{cfg.sky.glon_sign}"
    tag = f"_{cfg.output.tag}" if cfg.output.tag else ""
    base = f"{kinds}{sign}{tag}"
    i = 1
    while True:
        outdir = day / sub / (base if i == 1 else f"{base}_v{i}")
        if not any(d.exists() for d in (outdir, *band_dirs(outdir, bands))):
            return outdir
        i += 1


def _run_band(cfg: AnalysisConfig, outdir: Path, n_workers: int) -> None:
    """Fits, figures and checkpoint of one |b| band into ``outdir``."""
    saved = _run(cfg, outdir=str(outdir), n_workers=n_workers)
    print(f"\n[run] Done. {len(saved)} figure(s) saved under {outdir}:")
    for k, p in saved.items():
        print(f"  {k:>20}  ->  {p}")

    glon_lims = (cfg.sky.glon_edges[0], cfg.sky.glon_edges[-1])
    if cfg.plots.background_components:
        run_bg_plot(cfg, glon_lims=glon_lims, glat_lims=cfg.sky.glat_edges, outdir=str(outdir))
    if cfg.plots.ps_comparison:
        run_ps_comparison_plot(cfg, glon_lims=glon_lims, glat_lims=cfg.sky.glat_edges, outdir=str(outdir))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", type=Path, help="YAML config, e.g. configs/baseline.yaml")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="default: see default_outdir(); with several |b| bands this is the combined "
                         "folder <day>/combined/<name> and the bands go to <day>/b<lo>-<hi>/<name>")
    ap.add_argument("--print-outdir", action="store_true", help="print the output dir and exit")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    outdir = args.outdir or default_outdir(cfg)
    if args.print_outdir:
        print(outdir)
        return

    outdir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, outdir / "config.yaml")
    n_workers = cfg.run.n_workers or _auto_n_workers()
    print(f"[run] config: {args.config.resolve()}  (merged copy: {outdir / 'config.yaml'})")
    print(f"[run] output: {outdir.resolve()}")
    print(f"[run] n_workers={n_workers}")
    print((outdir / "config.yaml").read_text())

    bands = cfg.sky.bands
    dirs = [outdir] if len(bands) == 1 else band_dirs(outdir, bands)
    for band, band_dir in zip(bands, dirs):
        band_cfg = replace(cfg, sky=replace(cfg.sky, glat_edges=band))
        print(f"\n[run] |b| band {band[0]:g}-{band[1]:g} deg -> {band_dir}")
        _run_band(band_cfg, band_dir, n_workers)

    if len(bands) > 1:
        if cfg.run.save_checkpoint:
            run_bands_combined(outdir)
        else:
            print("[run] combined/ skipped: it is built from the band checkpoints (run.save_checkpoint=false).")


if __name__ == "__main__":
    main()
