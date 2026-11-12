from astropy.io import fits as pyfits
from astropy.wcs import WCS
import numpy as np
import healpy
from reproject.healpix import reproject_to_healpix

from pathlib import Path

from .constants import GeV_PER_JOULE, SI_TO_ASTRO
from .utils import pixnumrectangle_symmetric


# ------------------------------------------------------------------
# 1) PASS8 cube


def load_pass8_cube(exposure_path: str | Path, events_path: str | Path) -> dict:
    """
    Load PASS8 events and exposure maps and compute the intensity cube.

    Returns
    -------
    out : dict with keys
        - ``E_MeV``       : bin central energies (MeV), geometric mean of edges.
        - ``E_GeV``       : same in GeV.
        - ``E_low_MeV``   : lower edges (MeV).
        - ``E_high_MeV``  : upper edges (MeV).
        - ``dE_MeV``      : (E_high - E_low) / 2 (the convention used upstream).
        - ``n_bins``      : number of energy bins.
        - ``nside``       : HEALPix NSIDE of the data maps.
        - ``counts``      : (n_bins, npix) event counts.
        - ``exposure``    : (n_bins, npix) exposure (cm² s).
        - ``intensity``   : (n_bins, npix) differential intensity
          ``ph cm⁻² s⁻¹ sr⁻¹ MeV⁻¹``.
    """

    exp_path = str(exposure_path)
    dat_path = str(events_path)

    with pyfits.open(exp_path) as hdul:
        n_bins = hdul[1].header["TFIELDS"]
        nside = hdul[1].header["NSIDE"]

    with pyfits.open(dat_path) as hdul:
        bins = np.array(list(hdul[2].data))  # columns: idx, lo (keV), hi (keV)

    E_low_MeV = np.array(bins[:, 1] * 1e-3, float)
    E_high_MeV = np.array(bins[:, 2] * 1e-3, float)
    E_MeV = np.sqrt(E_low_MeV * E_high_MeV)
    dE_MeV = (E_high_MeV - E_low_MeV) / 2.0

    exposure = healpy.read_map(
        exp_path, field=range(n_bins), dtype=np.float64, nest=False
    )
    counts = healpy.read_map(
        dat_path, field=range(n_bins), dtype=np.float64, nest=False
    )

    npix = 12 * nside * nside
    omega_pix = 4.0 * np.pi / npix  # sr per pixel

    flux = counts / np.where(exposure > 0, exposure, np.inf)
    flux /= omega_pix  # ph cm⁻² s⁻¹ sr⁻¹ (per bin)
    flux[exposure == 0] = 0.0
    intensity = flux / dE_MeV.reshape(n_bins, 1)  # ph cm⁻² s⁻¹ sr⁻¹ MeV⁻¹

    return {
        "E_MeV": E_MeV,
        "E_GeV": E_MeV * 1e-3,
        "E_low_MeV": E_low_MeV,
        "E_high_MeV": E_high_MeV,
        "dE_MeV": dE_MeV,
        "n_bins": n_bins,
        "nside": nside,
        "counts": counts,
        "exposure": exposure,
        "intensity": intensity,
    }


# ------------------------------------------------------------------
# 2) IEM cube (gll_iem_v*.fits) and its HEALPix reprojection


def load_iem_cube(path: str | Path) -> dict:
    """
    Load the Galactic IEM cube (CAR WCS) and its ENERGIES extension.

    Returns
    -------
    out : dict with keys
        - ``E_MeV``  : (n_bins,) ENERGY column (MeV).
        - ``E_GeV``  : same in GeV.
        - ``E_edges_MeV`` : (n_bins+1,) bin edges (MeV) built from geometric means.
        - ``dE_GeV`` : (n_bins,) bin widths in GeV.
        - ``maps_CAR`` : (n_bins, NAXIS2, NAXIS1) the IEM intensity cube.
        - ``header`` : FITS header (with WCS).
        - ``wcs_2d`` : celestial WCS object.
    """
    fn = str(path)
    with pyfits.open(fn) as hdul:
        E_MeV = np.asarray(hdul["ENERGIES"].data["energy"], float)
        maps_CAR = np.asarray(hdul[0].data, float)
        header = hdul[0].header.copy()

    E_edges = np.empty(len(E_MeV) + 1)
    E_edges[1:-1] = np.sqrt(E_MeV[:-1] * E_MeV[1:])
    ratio = E_MeV[1] / E_MeV[0]
    E_edges[0] = E_MeV[0] / np.sqrt(ratio)
    E_edges[-1] = E_MeV[-1] * np.sqrt(ratio)
    dE_GeV = (E_edges[1:] - E_edges[:-1]) * 1e-3

    wcs_2d = WCS(header).celestial

    return {
        "E_MeV": E_MeV,
        "E_GeV": E_MeV * 1e-3,
        "E_edges_MeV": E_edges,
        "dE_GeV": dE_GeV,
        "maps_CAR": maps_CAR,
        "header": header,
        "wcs_2d": wcs_2d,
    }


def load_iem_healpix_cube(path: str | Path, nside_out: int) -> dict:
    """
    Reproject the IEM cube from CAR to HEALPix at ``nside_out``, in
    ``ph cm⁻² s⁻¹ sr⁻¹ GeV⁻¹`` (the native gll_iem is in MeV⁻¹).

    Returns
    -------
    out : dict with keys from :func:`load_iem_cube` plus
        ``maps_HEALPix`` : (n_bins, NPIX) reprojected cube.
        ``nside``        : int.
    """
    cube = load_iem_cube(path)
    wcs2d = cube["wcs_2d"]
    maps_CAR = cube["maps_CAR"]

    NPIX = healpy.nside2npix(nside_out)
    maps_HP = np.empty((maps_CAR.shape[0], NPIX), dtype=np.float32)
    scale = 1e3  # MeV⁻¹ -> GeV⁻¹
    for k, img in enumerate(maps_CAR):
        hp_map, _ = reproject_to_healpix(
            (img, wcs2d),
            "galactic",
            nside=nside_out,
            order="nearest-neighbor",
            nested=False,
        )
        maps_HP[k] = hp_map.astype(np.float32) * scale

    out = dict(cube)
    out["maps_HEALPix"] = maps_HP
    out["nside"] = int(nside_out)
    return out


# ------------------------------------------------------------------
# 3) HERMES / DRAGON templates


def load_hermes_template(
    fname: str,
    lowb: float,
    highb: float,
    lowl: float,
    highl: float,
    glon_sign: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ROI-averaged HERMES template spectrum.

    Parameters
    ----------
    fname : str
        Absolute path to a ``VariableMin-*`` or ``EleCutOff*`` FITS cube.
    lowb, highb : float
        Latitude band (degrees, symmetric N/S).
    lowl, highl : float
        Longitude band (degrees, symmetric in |l|).
    glon_sign : str
        ``both`` (|l|), ``pos`` (l > 0 only) or ``neg`` (l < 0 only).

    Returns
    -------
    E_GeV : ndarray  Template energies (GeV).
    F_avg : ndarray  ROI-averaged differential intensity in
                     ``J cm⁻² s⁻¹ sr⁻¹ GeV⁻¹`` divided by :data:`SI_TO_ASTRO`.
    """
    with pyfits.open(fname) as hdul:
        nside = hdul[1].header["NSIDE"]
        ipix = np.asarray(
            pixnumrectangle_symmetric(
                nside=nside,
                lowb_North=lowb,
                highb_North=highb,
                lowb_South=-highb,
                highb_South=-lowb,
                lowl=lowl,
                highl=highl,
                glon_sign=glon_sign,
            ),
            dtype=float,
        )
        norm = ipix.sum()

        E_list, F_list = [], []
        for hdu in hdul[1:]:
            E_GeV = hdu.header["ENERGY"] * GeV_PER_JOULE
            m = hdu.data.astype(float) / SI_TO_ASTRO
            F_list.append((m * ipix).sum() / norm)
            E_list.append(E_GeV)

    return np.asarray(E_list), np.asarray(F_list)


def load_dragon_pi0_template(
    h2_path: str | Path,
    hi_path: str | Path,
    lowb: float,
    highb: float,
    lowl: float,
    highl: float,
    glon_sign: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ROI-averaged DRAGON pi0 (H2 + HI, combined) spectrum for one CR
    propagation scenario.  Same NSIDE=64 HEALPix cube layout and physical
    units (``GeV⁻¹ m⁻² s⁻¹ sr⁻¹``) as the HERMES templates, so it uses the
    same ROI mask and unit conversion as :func:`load_hermes_template`.

    Parameters
    ----------
    h2_path, hi_path
        H2 and HI cubes of one DRAGON model (``files.dragon_models`` in the config).
    lowb, highb, lowl, highl, glon_sign
        ROI definition, same convention as :func:`load_hermes_template`.

    Returns
    -------
    E_GeV : ndarray  Map energies (GeV).
    F_avg : ndarray  ROI-averaged differential intensity in
                     cm⁻² s⁻¹ sr⁻¹ GeV⁻¹.
    """
    fname_h2 = str(h2_path)
    fname_hi = str(hi_path)

    with pyfits.open(fname_h2) as h_h2, pyfits.open(fname_hi) as h_hi:
        nside = h_h2[1].header["NSIDE"]
        ipix = np.asarray(
            pixnumrectangle_symmetric(
                nside=nside,
                lowb_North=lowb,
                highb_North=highb,
                lowb_South=-highb,
                highb_South=-lowb,
                lowl=lowl,
                highl=highl,
                glon_sign=glon_sign,
            ),
            dtype=float,
        )
        norm = ipix.sum()

        E_list, F_list = [], []
        for hdu_h2, hdu_hi in zip(h_h2[1:], h_hi[1:]):
            E_GeV = hdu_h2.header["ENERGY"] * GeV_PER_JOULE
            m = (hdu_h2.data.astype(float) + hdu_hi.data.astype(float)) / SI_TO_ASTRO
            F_list.append((m * ipix).sum() / norm)
            E_list.append(E_GeV)

    return np.asarray(E_list), np.asarray(F_list)


# ------------------------------------------------------------------
# 4) Isotropic spectrum


def load_isotropic_spectrum(path: str | Path) -> dict:
    """
    Load the isotropic gamma-ray background spectrum (energy in MeV).

    Returns
    -------
    out : dict with ``E_MeV``, ``F``, ``F_err``.
    """
    full = str(path)
    E_MeV = np.loadtxt(full, usecols=(0,))
    flux, err = np.loadtxt(full, usecols=(1, 2), unpack=True)
    return {"E_MeV": E_MeV, "F": flux, "F_err": err}


# ------------------------------------------------------------------
# 5) Point-source and unresolved-source templates


def load_point_sources_template(path: str | Path) -> dict:
    """
    Load the 4FGL-based point-source template cube.

    Returns
    -------
    out : dict with
        - ``nside``
        - ``npix``
        - ``pix_solid_angle``
        - ``E_GeV``  : (n_bins,) bin-centres in GeV.
        - ``E_edges_GeV`` : (n_bins+1,) edges.
        - ``maps`` : (npix, n_bins) template intensity per bin.
    """
    full = str(path)
    with pyfits.open(full) as hdul:
        nside = hdul[0].header["NSIDE"]
        npix = 12 * nside * nside
        pix_omega = 4.0 * np.pi / npix
        n_edges = hdul[0].header["HIERARCH NBIN E"] + 1
        E_min = hdul[0].header["EMIN"]
        E_max = hdul[0].header["EMAX"]
        factor = pow((E_max / E_min), 1.0 / (n_edges - 1))
        E_edges = E_min * pow(factor, np.arange(0, n_edges))
        E_cen = np.sqrt(np.array(E_edges[:-1]) * np.array(E_edges[1:]))

        maps = np.zeros((npix, n_edges - 1))
        for iE in range(n_edges - 1):
            maps[:, iE] = np.array(hdul[1].data.field(iE))

    return {
        "nside": nside,
        "npix": npix,
        "pix_solid_angle": pix_omega,
        "E_GeV": E_cen,
        "E_edges_GeV": E_edges,
        "maps": maps,
    }


def load_unresolved_sources_template(path: str | Path, nside_out: int) -> dict:
    """
    Load the unresolved-sources template (CAR projection, gridded into HEALPix).

    Returns
    -------
    out : dict with
        - ``E_GeV``, ``E_low_GeV``, ``E_high_GeV``
        - ``maps`` : list of HEALPix maps (one per energy bin).
        - ``nside``
    """
    full = str(path)
    with pyfits.open(full) as hdul:
        bins = hdul[1].data
        E_cen = np.asarray([en[1] for en in bins], float)
        E_lo = np.asarray([en[2] for en in bins], float)
        E_hi = np.asarray([en[3] for en in bins], float)
        maps_CAR = hdul[0].data  # (n_bins, 481, 960)

    Long_vals = np.arange(-180 + 0.375 / 2, 180, 0.375)
    Lat_vals = np.arange(-90, 90 + 0.375 / 2, 0.375)

    npix = healpy.nside2npix(nside_out)
    maps: list[np.ndarray] = []
    for en in range(maps_CAR.shape[0]):
        hpxmap = np.zeros(npix, dtype=float)
        for glat in range(maps_CAR.shape[1]):
            indices = healpy.pixelfunc.ang2pix(
                nside_out,
                phi=Lat_vals[glat],
                theta=Long_vals,
                lonlat=True,
            )
            hpxmap[indices] += maps_CAR[en, glat, :]
        maps.append(hpxmap)

    return {
        "E_GeV": E_cen,
        "E_low_GeV": E_lo,
        "E_high_GeV": E_hi,
        "maps": maps,
        "nside": int(nside_out),
    }


# ------------------------------------------------------------------
# 6) Fermi bubbles


def load_bubble_mask(fname: str | Path, nside_out: int) -> np.ndarray:
    """
    Read the published 1-D RING-ordered Fermi-bubble mask.

    The mask is up/down-graded to ``nside_out`` (so North and South are
    kept together) and then returned unrenormalised.
    """
    m = np.load(fname)
    nside_in = healpy.npix2nside(len(m))
    if nside_in != nside_out:
        m = healpy.ud_grade(m, nside_out, order_in="RING", order_out="RING", power=0)
    return m


def load_bubble_template(
    E_target_GeV,
    nside_out: int,
    mask_file: str | Path,
    spec_file: str | Path,
) -> np.ndarray:
    """
    Build the outer-product ``F(E) × mask(pix)`` bubble template.

    The spectrum is interpolated from Ackermann+14 Table 2 to ``E_target_GeV``
    (zeros outside the tabulated range) and multiplied by the spatial mask.

    Parameters
    ----------
    E_target_GeV : array_like
        Energies (GeV) on which the template is evaluated.
    nside_out : int
        HEALPix NSIDE of the output template.
    mask_file : str
        Path to the RING-ordered binary mask (``.npy``).
    spec_file : str
        Path to the Ackermann+14 Table 2 (4 columns).

    Returns
    -------
    bubble_intensity : ndarray of shape ``(len(E_target), NPIX)``.
    """
    bubble_mask = load_bubble_mask(mask_file, nside_out=nside_out)
    tbl = np.loadtxt(spec_file, skiprows=1)
    E_bub_GeV = tbl[:, 0]
    E2F_bub = tbl[:, 3] * 1e-7  # 10⁻⁷ GeV cm⁻² s⁻¹ sr⁻¹
    F_bub_GeV = E2F_bub / E_bub_GeV**2  # ph cm⁻² s⁻¹ sr⁻¹ GeV⁻¹

    E_target = np.asarray(E_target_GeV, float)
    F_interp = np.interp(E_target, E_bub_GeV, F_bub_GeV, left=0, right=0)
    return F_interp[:, None] * bubble_mask[None, :]


def effective_area_fractional_error(E_MeV: np.ndarray, aeff) -> np.ndarray:
    """
    Fermi-LAT effective-area fractional uncertainty as a function of energy.

    Piecewise, with the coefficients of ``aeff`` (``data.aeff_error`` in the
    config), ``x = log10(E / MeV)``:
      * E < low_below_mev:                  base + low_slope · (low_pivot − x)
      * low_below_mev < E < high_above_mev: base
      * otherwise:                          base + high_slope · (x − high_pivot)

    ``E_MeV`` is expected in MeV.  Returns a fractional (dimensionless) array.
    """
    E = np.asarray(E_MeV, float)
    x = np.log10(E)
    lo = E < aeff.low_below_mev
    Aeff_err = aeff.base + aeff.high_slope * (x - aeff.high_pivot_log10_mev)
    Aeff_err[lo] = aeff.base + aeff.low_slope * (aeff.low_pivot_log10_mev - x[lo])
    Aeff_err[(E > aeff.low_below_mev) * (E < aeff.high_above_mev)] = aeff.base
    return Aeff_err
