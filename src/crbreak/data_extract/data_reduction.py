from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy import ndarray

from .data_loader import (
    effective_area_fractional_error,
    load_bubble_template,
    load_dragon_pi0_template,
    load_hermes_template,
    load_iem_healpix_cube,
    load_isotropic_spectrum,
    load_pass8_cube,
    load_point_sources_template,
    load_unresolved_sources_template,
)
from .utils import _interp_scaled, _roi_average_intensity, pixnumrectangle_symmetric

# Every setting (files, cuts, background components, SED scaling, Aeff error)
# comes from the analysis config: ``data`` / ``paths`` / ``files`` sections.


def _roi_mask(nside: int, lowl: float, highl: float, lowb: float, highb: float, glon_sign: str) -> np.ndarray:
    """|l| ROI mask (or one side of l = 0, ``glon_sign``), symmetric in |b| (North and South bands)."""
    return pixnumrectangle_symmetric(
        nside=nside,
        lowb_North=lowb,
        highb_North=highb,
        lowb_South=-highb,
        highb_South=-lowb,
        lowl=lowl,
        highl=highl,
        glon_sign=glon_sign,
    )


def _build_background_model(
    E_GeV: np.ndarray,
    lowl: float,
    highl: float,
    lowb: float,
    highb: float,
    *,
    glon_sign: str,
    data,
    paths,
    files,
    nside_pass8: int,
    bg_components: tuple[str, ...],
) -> dict:
    data_dir = Path(paths.data_dir)
    s = data.sed_scaling
    components: dict[str, np.ndarray] = {}

    # Isotropic spectrum (MeV → GeV units)
    if "ISO" in bg_components:
        iso = load_isotropic_spectrum(data_dir / files.isotropic)
        components["ISO"] = _interp_scaled(E_GeV, 1e-3 * iso["E_MeV"], 1e3 * iso["F"], s)

    # 4FGL point sources
    if "PS" in bg_components:
        ps = load_point_sources_template(data_dir / files.point_sources)
        ipix_ps = _roi_mask(ps["nside"], lowl, highl, lowb, highb, glon_sign)
        norm_ps = ipix_ps.sum()
        PS_av = np.array([np.sum(m * ipix_ps) / norm_ps for m in ps["maps"].T])
        components["PS"] = _interp_scaled(E_GeV, ps["E_GeV"], PS_av, s)

    # Unresolved sources
    if "UPS" in bg_components:
        ups = load_unresolved_sources_template(data_dir / files.unresolved_sources, data.nside_ups)
        ipix_ups = _roi_mask(ups["nside"], lowl, highl, lowb, highb, glon_sign)
        norm_ups = ipix_ups.sum()
        UPS_av = np.array([np.sum(mU * ipix_ups) / norm_ups for mU in ups["maps"]])
        components["UPS"] = _interp_scaled(E_GeV, ups["E_GeV"], UPS_av, s)

    # Fermi bubbles
    if "BUB" in bg_components:
        bubble_cube = load_bubble_template(
            E_target_GeV=E_GeV,
            nside_out=nside_pass8,
            mask_file=Path(paths.repo_data_dir) / files.bubble_mask,
            spec_file=Path(paths.repo_data_dir) / files.bubble_spectrum,
        )
        ipix_bub = np.asarray(_roi_mask(nside_pass8, lowl, highl, lowb, highb, glon_sign), dtype=float)
        BUB_roi = (
            np.array([bubble_cube[iE] @ ipix_bub for iE in range(E_GeV.size)])
            / ipix_bub.sum()
        )
        components["BUB"] = BUB_roi * E_GeV**s

    # HERMES IC + Brems components (in the config's order: that is the sum order)
    for key, fname in files.hermes.items():
        if key in bg_components:
            E_tmp, F_tmp = load_hermes_template(
                fname=str(data_dir / fname),
                lowb=lowb,
                highb=highb,
                lowl=lowl,
                highl=highl,
                glon_sign=glon_sign,
            )
            components[key] = _interp_scaled(E_GeV, E_tmp, F_tmp, s)

    # Generic total: sum only what exists
    if len(components) == 0:
        BG_total = np.zeros_like(E_GeV, dtype=float)
    else:
        BG_total = np.sum(np.stack(list(components.values()), axis=0), axis=0)
    out = dict[str, ndarray](components)
    out["total"] = BG_total
    return out


def _prep_pass8(lowl, highl, lowb, highb, *, glon_sign, cube):
    nside = cube["nside"]
    ipix = _roi_mask(nside, lowl, highl, lowb, highb, glon_sign)

    # F_avg is in GeV⁻¹ units already per your comments
    F_avg, inv_sqrtN = _roi_average_intensity(cube["intensity"], cube["counts"], ipix)

    E_MeV = cube["E_MeV"]
    E_GeV = E_MeV * 1e-3
    counts_roi = (1.0 / inv_sqrtN) ** 2

    return E_GeV, E_MeV, F_avg, counts_roi, nside


def _prep_iem(lowl, highl, lowb, highb, *, glon_sign, cube, iem_cube):
    # 1. PASS8 exposure cube (used to derive expected counts and Omega_roi).
    nside_p8 = cube["nside"]

    # ROI mask in HEALPix (PASS8 exposure grid) -- symmetric in |b|, and in
    # |l| unless ``glon_sign`` keeps one side; same convention as PASS8.
    ipix_p8 = np.asarray(_roi_mask(nside_p8, lowl, highl, lowb, highb, glon_sign), dtype=float)

    npix_roi_p8 = ipix_p8.sum()
    exp_roi_p8 = np.array(
        [
            (cube["exposure"][i] * ipix_p8).sum() / npix_roi_p8
            for i in range(len(cube["E_MeV"]))
        ]
    )

    # 2. IEM cube reprojected to HEALPix, averaged over the same symmetric ROI.
    iem = iem_cube
    nside_iem = int(iem["nside"])
    ipix_iem = np.asarray(_roi_mask(nside_iem, lowl, highl, lowb, highb, glon_sign), dtype=float)
    norm_iem = ipix_iem.sum()
    IEM_av = np.array([np.sum(m * ipix_iem) / norm_iem for m in iem["maps_HEALPix"]])

    # 3. Interpolate exposure to IEM energy grid for the Poisson model counts.
    exp_roi_model = np.interp(iem["E_GeV"], cube["E_MeV"] * 1e-3, exp_roi_p8)
    omega_pix = 4.0 * np.pi / (12 * nside_p8**2)
    Omega_roi = omega_pix * npix_roi_p8
    counts_roi = IEM_av * exp_roi_model * Omega_roi * iem["dE_GeV"]

    return iem["E_GeV"], iem["E_MeV"], IEM_av, counts_roi, nside_iem


def _prep_dragon(lowl, highl, lowb, highb, *, glon_sign, paths, files, model):
    # DRAGON pi0 (H2 + HI) mock spectrum: its own energy grid and
    # normalisation, and no counts (legacy FDE_dragon.py).
    maps = files.dragon_models[model]
    E_GeV, F_avg = load_dragon_pi0_template(
        Path(paths.dragon_dir) / maps["H2"],
        Path(paths.dragon_dir) / maps["HI"],
        lowb=lowb,
        highb=highb,
        lowl=lowl,
        highl=highl,
        glon_sign=glon_sign,
    )
    return E_GeV, 1e3 * E_GeV, F_avg, None, None


def reduce_window(
    kind: str,  # 'PASS8', 'IEM' or 'DRAGON'
    lowl: float,
    highl: float,
    lowb: float,
    highb: float,
    *,
    glon_sign: str,
    data,
    paths,
    files,
    subtraction: bool,
    cube: dict | None = None,
    iem_cube: dict | None = None,
) -> dict:
    """
    Run the data extraction + background subtraction pipeline for one ROI.

    ``glon_sign`` is ``cfg.sky.glon_sign``: ``both`` takes ``lowl <= |l| <= highl``,
    ``pos`` / ``neg`` only the ``l > 0`` / ``l < 0`` side (data and every template).
    ``data`` / ``paths`` / ``files`` are the corresponding sections of the
    analysis config; the kind-specific cuts are ``data.pass8`` / ``data.iem`` /
    ``data.dragon``.  ``cube`` / ``iem_cube`` are the preloaded cubes
    (:func:`preload_cubes`); they are loaded here when not given.
    ``DRAGON`` is the pi0 mock spectrum of ``data.dragon_model``, with
    ``sigma = data.dragon_sigma_frac * F`` (no counts, so no Poisson term,
    Aeff term or counts cut).
    """

    if lowl > highl or lowb > highb:
        raise ValueError("Require lowl<=highl and lowb<=highb.")
    cuts = data.cuts(kind)
    if kind != "DRAGON" and (cube is None or (kind == "IEM" and iem_cube is None)):
        loaded = preload_cubes(kind, paths=paths, files=files, nside_iem=data.nside_iem)
        cube, iem_cube = loaded["cube"], loaded.get("iem_cube")

    # 1. DATA LOADING
    if kind == "PASS8":
        E_GeV, E_MeV, intensity_avg, counts_roi, nside_val = _prep_pass8(
            lowl, highl, lowb, highb, glon_sign=glon_sign, cube=cube
        )
    elif kind == "IEM":
        E_GeV, E_MeV, intensity_avg, counts_roi, nside_val = _prep_iem(
            lowl, highl, lowb, highb, glon_sign=glon_sign, cube=cube, iem_cube=iem_cube
        )
    else:  # DRAGON
        E_GeV, E_MeV, intensity_avg, counts_roi, nside_val = _prep_dragon(
            lowl, highl, lowb, highb, glon_sign=glon_sign, paths=paths, files=files, model=data.dragon_model
        )

    # 2. TRIMMING (legacy ``last_points = drop_last_bins``)
    drop = int(cuts.drop_last_bins)
    keep_slice = slice(0, -drop) if drop > 0 else slice(None)
    E_GeV = E_GeV[keep_slice]
    E_MeV = E_MeV[keep_slice]
    F_avg = intensity_avg[keep_slice]
    if counts_roi is not None:
        counts_roi = counts_roi[keep_slice]

    # 3. BACKGROUND MODEL (always built on the trimmed energy grid).
    bg = _build_background_model(
        E_GeV,
        lowl,
        highl,
        lowb,
        highb,
        glon_sign=glon_sign,
        data=data,
        paths=paths,
        files=files,
        nside_pass8=nside_val,
        bg_components=cuts.bg_components,
    )

    # 4. SCALING + ERROR BARS
    s = data.sed_scaling
    Aeff_err = effective_area_fractional_error(E_MeV, data.aeff_error)
    F_obs_scaled_total = F_avg * E_GeV**s

    if subtraction:
        F_obs_raw = F_avg - bg["total"] / E_GeV**s
    else:
        F_obs_raw = F_avg.copy()

    y = F_obs_raw * E_GeV**s

    # ``sigma_reference`` chooses the magnitude that multiplies the fractional
    # error: legacy uses the *raw* intensity, which keeps the error bars
    # well-defined even where ``F_obs_raw`` flips sign after subtraction.
    if cuts.sigma_reference == "raw":
        sigma_mag = np.abs(F_avg)
    else:  # 'subtracted'
        sigma_mag = np.abs(F_obs_raw)
    if counts_roi is None:  # DRAGON mock: a fixed fraction (legacy: the median one where F <= 0)
        frac = data.dragon_sigma_frac
        sigma = np.where(sigma_mag > 0, frac * sigma_mag, frac * np.median(sigma_mag[sigma_mag > 0])) * E_GeV**s
    else:
        frac_stat = 1.0 / np.sqrt(np.maximum(counts_roi, 1.0))
        sigma = sigma_mag * np.sqrt(Aeff_err**2 + frac_stat**2) * E_GeV**s

    # 5. ENERGY + STATISTICS CUTS
    keep = E_GeV >= float(cuts.e_min_gev)
    if counts_roi is not None:
        keep &= counts_roi >= int(data.min_counts_per_bin)
    if cuts.e_max_gev is not None:
        keep &= E_GeV < float(cuts.e_max_gev)
    if keep.sum() < 5:
        raise RuntimeError(
            f"Only {keep.sum()} bins survived cuts (kind={kind}, "
            f"min_counts={data.min_counts_per_bin}, E_min={cuts.e_min_gev} GeV, "
            f"E_max={cuts.e_max_gev} GeV)."
        )

    return {
        "E_GeV": E_GeV[keep],
        "y": y[keep],
        "sigma": sigma[keep],
        "sed_scaling": s,
        "Ssys_frac": Aeff_err[keep],
        "F_avg": F_avg[keep],
        "F_obs_raw": F_obs_raw[keep],
        "F_total_scaled": F_obs_scaled_total[keep],
        "F_bg_total": bg["total"][keep],
        "bg_components": {k: v[keep] for k, v in bg.items() if k != "total"},
        "n_counts": counts_roi[keep] if counts_roi is not None else None,
        "roi": {"lowl": lowl, "highl": highl, "lowb": lowb, "highb": highb, "glon_sign": glon_sign},
        "sigma_reference": cuts.sigma_reference,
        "kind": kind,
    }


def load_ps_scaled(
    path: str | Path,
    *,
    E_GeV: np.ndarray,
    lowl: float,
    highl: float,
    lowb: float,
    highb: float,
    glon_sign: str,
    sed_scaling: float,
) -> np.ndarray:
    """Load one PS template variant and return the SED-scaled ROI average.

    Replicates the per-file PS extraction from :func:`_build_background_model`
    so callers can load multiple variants (``files.point_source_variants``)
    and compare them on the same energy grid.

    Returns
    -------
    ps_scaled : np.ndarray
        ``E^sed_scaling``-weighted ROI-averaged PS flux on ``E_GeV``.
    """
    ps = load_point_sources_template(path)
    ipix_ps = _roi_mask(ps["nside"], lowl, highl, lowb, highb, glon_sign)
    norm_ps = ipix_ps.sum()
    PS_av = np.array([np.sum(m * ipix_ps) / norm_ps for m in ps["maps"].T])
    return _interp_scaled(E_GeV, ps["E_GeV"], PS_av, sed_scaling)


def preload_cubes(kind: str, *, paths, files, nside_iem: int) -> dict:
    """
    Eagerly load the cube(s) needed for many ``reduce_window(kind, ...)``
    calls, so callers can pass them in via ``cube=`` / ``iem_cube=`` and avoid
    repeated I/O.

    Returns
    -------
    dict with keys:
        - "cube"     : PASS8 exposure+events cube (PASS8, IEM)
        - "iem_cube" : IEM HEALPix cube (only when kind == "IEM")
    Nothing for ``DRAGON``: its maps are read per ROI.
    """
    data_dir = Path(paths.data_dir)
    if kind == "PASS8":
        return {"cube": load_pass8_cube(data_dir / files.pass8_exposure, data_dir / files.pass8_events)}
    if kind == "IEM":
        return {
            "cube": load_pass8_cube(data_dir / files.iem_exposure, data_dir / files.iem_events),
            "iem_cube": load_iem_healpix_cube(data_dir / files.iem_model, nside_iem),
        }
    if kind == "DRAGON":
        return {}
    raise ValueError(f"Unknown kind: {kind!r}. Expected 'PASS8', 'IEM' or 'DRAGON'.")
