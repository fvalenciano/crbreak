"""MC toy movie for one window, with the analysis settings of a YAML config.

    python -m crbreak.run_movie configs/baseline.yaml --glon 0 5 --n-toys 100 --fps 4
    python -m crbreak.run_movie configs/baseline.yaml --frames <movie dir>/frames --fps 4

The window's |b| band, dataset (first of ``data.kinds``), models, nuisance
prior and MC settings come from the config; the movie itself uses the
profiled PL / SBPL pair with an independent ``w`` per window.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from crbreak.config.analysis import load_config
from crbreak.data_extract.main import Data, SkyMesh
from crbreak.fitting.main import Fitting
from crbreak.fitting.models import PowerLaw, SmoothBrokenPowerLaw
from crbreak.fitting.nuisance import NuisanceModel, nuisance_kwargs_from_config
from crbreak.plotting import make_mc_toy_movie
from crbreak.plotting.plot_mc_toy_movie import write_movie_from_frames


def _outdir(parent: str, kind: str, glon: tuple[float, float], glat: tuple[float, float], prior: str) -> Path:
    stem = (
        f"{datetime.now():%d%b%y}_movie_{kind.lower()}_"
        f"{glon[0]:g}-{glon[1]:g}l_{glat[0]:g}-{glat[1]:g}b_{prior}_indep"
    )
    i = 1
    while (Path(parent) / f"{stem}_v{i}").exists():
        i += 1
    return Path(parent) / f"{stem}_v{i}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", type=Path)
    ap.add_argument("--glon", type=float, nargs=2, default=None,
                    help="|l| window (default: first window of sky.glon_edges)")
    ap.add_argument("--n-toys", type=int, default=100)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--frames", type=Path, default=None,
                    help="re-render the movie from an existing frames/ directory and exit")
    args = ap.parse_args(argv)

    if args.frames is not None:
        frames = sorted(args.frames.glob("toy_*.png"))
        print(f"[run_movie] Re-rendering {len(frames)} frames from {args.frames.resolve()} at fps={args.fps}")
        movie = write_movie_from_frames(
            frames, args.frames.parent / f"mc_toy_movie_fps{args.fps}.mp4", fps=args.fps
        )
        print(f"[run_movie] Movie: {movie.resolve()}")
        return

    cfg = load_config(args.config)
    kind = cfg.data.kinds[0]
    glon = tuple(args.glon) if args.glon else cfg.sky.glon_edges[:2]
    glat = (cfg.sky.glat_edges[0], cfg.sky.glat_edges[-1])
    outdir = _outdir(cfg.output.parent, kind, glon, glat, cfg.nuisance.prior)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[run_movie] Output directory: {outdir.resolve()}")
    print(f"[run_movie] kind={kind}  glon={glon}  glat={glat}  prior={cfg.nuisance.prior}  n_toys={args.n_toys}")

    mesh = SkyMesh(glon_lims=glon, glat_lims=glat, Nglon=1, Nglat=1,
                   glon_bins=list(glon), glat_bins=list(glat))
    data = Data(mesh)
    data.build(kind, cfg=cfg, subtraction=True)
    fitting = Fitting(data, solver=cfg.solver)

    print("[run_movie] Classical seeds ...")
    fitting.fit_per_window(
        [PowerLaw.from_config(cfg.model), SmoothBrokenPowerLaw.from_config(cfg.model)],
        do_profile_errors=False,
        progress=True,
    )

    nm_kwargs = nuisance_kwargs_from_config(cfg.nuisance)
    nm_pl = NuisanceModel(spectral_model=PowerLaw.from_config(cfg.model), share_nuisance=False, **nm_kwargs)
    nm_sbpl = NuisanceModel(
        spectral_model=SmoothBrokenPowerLaw.from_config(cfg.model), share_nuisance=False, **nm_kwargs
    )
    print("[run_movie] Profiled independent-w fits ...")
    fitting.fit_per_window(
        [nm_pl, nm_sbpl],
        wilks_pairs=[(nm_pl.name, nm_sbpl.name)],
        do_profile_errors=False,
        progress=True,
    )

    wfit = next(iter(fitting.window_fits.values()))
    result = make_mc_toy_movie(
        wfit,
        name_pl=nm_pl.name,
        name_alt=nm_sbpl.name,
        n_toys=args.n_toys,
        seed=cfg.mc.seed,
        sample_theta=cfg.mc.sample_theta,
        warm_start=cfg.mc.warm_start,
        solver=cfg.solver,
        sed_scaling=cfg.data.sed_scaling,
        outdir=outdir,
        fps=args.fps,
    )
    print(f"[run_movie] Movie: {Path(result['movie']).resolve()}")
    print(
        f"[run_movie] n_exceed={result['n_exceed']}/{len(result['deltas'])}  "
        f"delta_obs={result['delta_obs']:.3f}"
    )


if __name__ == "__main__":
    main()
