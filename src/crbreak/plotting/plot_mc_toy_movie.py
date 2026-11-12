from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from ..fitting.chi2 import Chi2Fit, ProfiledChi2Fit
from ..fitting.montecarlo import sample_pseudo_y_pl
from ..fitting.nuisance import NuisanceModel


def _fit_model(model, E, y, sigma, p0, solver):
    if isinstance(model, NuisanceModel):
        return ProfiledChi2Fit(model, solver=solver).fit(
            E, y, sigma, p0=p0, warn_on_fail=False
        )
    return Chi2Fit(model, solver=solver).fit(E, y, sigma, p0=p0)


def _predict(fit, E):
    if isinstance(fit.model, NuisanceModel) and fit.model.is_bound:
        return fit.model.predict_deformed_on_grid(E, tuple(fit.popt))
    return fit.predict(E)


def _predict_physical(fit, E):
    model = fit.model.spectral_model if isinstance(fit.model, NuisanceModel) else fit.model
    return model(E, *fit.popt)


def write_movie_from_frames(
    frame_paths: list[Path],
    movie_path: str | Path,
    fps: int,
) -> Path:
    if not frame_paths:
        raise ValueError("No frames were produced.")
    movie_path = Path(movie_path)
    movie_path.parent.mkdir(parents=True, exist_ok=True)
    if movie_path.suffix.lower() == ".mp4":
        try:
            import imageio_ffmpeg

            mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
        if not animation.FFMpegWriter.isAvailable():
            raise RuntimeError(
                "MP4 output requires ffmpeg, but matplotlib cannot find it. "
                "Run with `poetry run python -m crbreak.run_movie`, or "
                "install/activate ffmpeg."
            )
        out = movie_path
        writer = animation.FFMpegWriter(fps=fps)
    else:
        out = movie_path.with_suffix(".gif")
        writer = animation.PillowWriter(fps=fps)

    first = plt.imread(frame_paths[0])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(first)
    ax.axis("off")
    with writer.saving(fig, str(out), dpi=120):
        for frame in frame_paths:
            im.set_data(plt.imread(frame))
            writer.grab_frame()
    plt.close(fig)
    return out


def _write_movie(frame_paths: list[Path], movie_path: Path, fps: int) -> Path:
    return write_movie_from_frames(frame_paths, movie_path, fps)


def make_mc_toy_movie(
    wfit,
    *,
    name_pl: str,
    name_alt: str,
    n_toys: int,
    seed: int,
    sample_theta: bool,
    warm_start: bool,
    solver,
    outdir: str | Path,
    fps: int,
    sed_scaling: float,
    movie_name: str = "mc_toy_movie.mp4",
    n_dense: int = 500,
) -> dict[str, object]:
    """Save toy-by-toy MC frames and a movie for one window.

    Each frame shows one pseudo-dataset generated under PL, the refitted PL
    and SBPL curves, and a green tick when ``Delta chi2_toy >= Delta chi2_obs``.
    """

    out = Path(outdir)
    frame_dir = out / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    fit_pl_obs = wfit.fits[name_pl]
    fit_alt_obs = wfit.fits[name_alt]
    E = np.asarray(fit_pl_obs.E, dtype=float)
    y_obs = np.asarray(fit_pl_obs.y, dtype=float)
    sigma = np.asarray(fit_pl_obs.sigma, dtype=float)
    delta_obs = float(fit_pl_obs.chi2 - fit_alt_obs.chi2)
    w_ref = (
        np.asarray(fit_pl_obs.model.last_w, dtype=float).copy()
        if isinstance(fit_pl_obs.model, NuisanceModel)
        and fit_pl_obs.model.last_w is not None
        else None
    )

    rng = np.random.default_rng(seed)
    E_dense = np.logspace(np.log10(E.min() / 1.2), np.log10(E.max() * 1.2), n_dense)
    scale = E**sed_scaling
    scale_dense = E_dense**sed_scaling
    y_obs_scaled = y_obs * scale
    y_obs_pos = y_obs_scaled[np.isfinite(y_obs_scaled) & (y_obs_scaled > 0)]
    y_min_fixed = 0.75 * float(y_obs_pos.min()) if y_obs_pos.size else None
    y_max_fixed = 1.25 * float(y_obs_pos.max()) if y_obs_pos.size else None
    frame_paths: list[Path] = []
    deltas: list[float] = []
    n_attempt = 0
    max_attempts = max(int(n_toys) * 10, int(n_toys))

    while len(frame_paths) < int(n_toys) and n_attempt < max_attempts:
        n_attempt += 1
        y_pseudo = sample_pseudo_y_pl(
            fit_pl_obs.model,
            E,
            sigma,
            np.asarray(fit_pl_obs.popt, dtype=float),
            np.asarray(fit_pl_obs.pcov, dtype=float),
            rng=rng,
            sample_theta=sample_theta,
            w_ref=w_ref,
        )
        try:
            fit_pl = _fit_model(fit_pl_obs.model, E, y_pseudo, sigma, fit_pl_obs.popt, solver)
            p0_alt = fit_alt_obs.popt if warm_start else None
            fit_alt = _fit_model(fit_alt_obs.model, E, y_pseudo, sigma, p0_alt, solver)
        except Exception:
            continue
        if not (getattr(fit_pl, "converged", True) and getattr(fit_alt, "converged", True)):
            continue

        delta = float(fit_pl.chi2 - fit_alt.chi2)
        if not np.isfinite(delta):
            continue
        deltas.append(delta)
        exceed = delta >= delta_obs

        y_pl_dense = _predict(fit_pl, E_dense) * scale_dense
        y_alt_dense = _predict(fit_alt, E_dense) * scale_dense
        y_pl_phys_dense = _predict_physical(fit_pl, E_dense) * scale_dense
        y_alt_phys_dense = _predict_physical(fit_alt, E_dense) * scale_dense
        i = len(frame_paths)
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        ax.plot(
            E,
            y_obs_scaled,
            color="cornflowerblue",
            lw=1.6,
            ls="--",
            alpha=0.95,
            zorder=2,
        )
        ax.errorbar(
            E,
            y_pseudo * scale,
            yerr=sigma * scale,
            fmt="o",
            color="0.55",
            ecolor="black",
            markeredgecolor="black",
            markeredgewidth=0.8,
            ms=9,
            lw=0.8,
            capsize=0,
            label="pseudo-exp",
            zorder=3,
        )
        ax.errorbar(
            E,
            y_obs_scaled,
            yerr=sigma * scale,
            fmt="o",
            color="white",
            ecolor="cornflowerblue",
            markeredgecolor="cornflowerblue",
            markeredgewidth=1.7,
            ms=8,
            lw=1.0,
            capsize=0,
            label="PASS8 data",
            zorder=4,
        )
        ax.plot(
            E_dense,
            y_alt_phys_dense,
            color="forestgreen",
            lw=1.8,
            ls="--",
            alpha=0.85,
            label="SBPL physical",
        )
        ax.plot(
            E_dense,
            y_alt_dense,
            color="forestgreen",
            lw=2.4,
            label="SBPL deformed",
        )
        ax.plot(
            E_dense,
            y_pl_phys_dense,
            color="tab:red",
            lw=1.7,
            ls="--",
            alpha=0.85,
            label="PL physical",
        )
        ax.plot(
            E_dense,
            y_pl_dense,
            color="tab:red",
            lw=2.2,
            label="PL deformed",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(E_dense.min(), E_dense.max())
        if y_min_fixed is not None and y_max_fixed is not None:
            ax.set_ylim(y_min_fixed, y_max_fixed)
        ax.set_xlabel("Energy [GeV]", fontsize=20)
        s = f"{sed_scaling:g}"
        ax.set_ylabel(
            rf"$E^{{{s}}}\,J\;[\mathrm{{GeV^{{{s} - 1}}\,cm^{{-2}}\,s^{{-1}}\,sr^{{-1}}}}]$",
            fontsize=20,
        )
        ax.tick_params(
            axis="both", which="major", direction="in", labelsize=18, length=6, width=1.6
        )
        ax.tick_params(
            axis="both", which="minor", direction="in", labelsize=16, length=4, width=1.3
        )
        ax.legend(
            frameon=False,
            loc="upper left",
            fontsize=17,
            bbox_to_anchor=(0.02, 0.98),
        )
        ax.text(
            0.96,
            0.96,
            (
                f"{'✓' if exceed else '✗'} toy {i + 1}/{n_toys}\n"
                rf"$\Delta\chi^2_\mathrm{{toy}}={delta:.2f}$" "\n"
                rf"$\Delta\chi^2_\mathrm{{obs}}={delta_obs:.2f}$" "\n"
                + ("exceeds observed" if exceed else "below observed")
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=18,
            color="forestgreen" if exceed else "crimson",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                alpha=0.88,
                edgecolor="none",
            ),
        )
        ax.text(
            0.96,
            0.05,
            f"{wfit.lowl:g}<|$l$|<{wfit.highl:g}\n{wfit.lowb:g}<|$b$|<{wfit.highb:g}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=18,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                alpha=0.85,
                edgecolor="none",
            ),
        )
        fig.tight_layout()
        frame_path = frame_dir / f"toy_{i:04d}.png"
        fig.savefig(frame_path, dpi=160)
        plt.close(fig)
        frame_paths.append(frame_path)

    movie_path = _write_movie(frame_paths, out / movie_name, fps=fps)
    return {
        "movie": movie_path,
        "frames": frame_paths,
        "delta_obs": delta_obs,
        "deltas": np.asarray(deltas, dtype=float),
        "n_exceed": int(np.sum(np.asarray(deltas, dtype=float) >= delta_obs)),
        "n_attempted": n_attempt,
    }
