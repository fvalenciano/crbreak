import healpy
import numpy as np


def pixnumrectangle_symmetric(
    nside,
    lowb_North,
    highb_North,
    lowb_South,
    highb_South,
    lowl,
    highl,
    *,
    glon_sign,
    exclude_equator=True,
):
    """
    Purpose: Create a binary mask (list of 0/1) that selects all HEALPix pixels whose Galactic coordinates (ℓ, b)
    fall inside a latitude/longitude rectangle that is symmetric around the Galactic centre (ℓ = 0°).
    ``glon_sign`` picks the longitude side: ``both`` (lowl <= |ℓ| <= highl), ``pos`` (lowl <= ℓ <= highl,
    within 0-180°) or ``neg`` (lowl <= -ℓ <= highl, i.e. within 180-360°).
    """

    npix = healpy.nside2npix(nside)
    theta, phi = healpy.pix2ang(nside, np.arange(npix))  # rad
    b = np.degrees(0.5 * np.pi - theta)  # deg
    l = np.degrees(phi)
    l[l > 180] -= 360  # wrap to (−180, +180]

    # longitude:  symmetric about 0°, or one side only
    lon = {"both": np.abs(l), "pos": l, "neg": -l}[glon_sign]
    lon_sel = (lon >= lowl) & (lon <= highl)

    # latitude:   independent North / South bands
    lat_sel = ((b >= lowb_North) & (b <= highb_North)) | (  # north slice
        (b >= lowb_South) & (b <= highb_South)
    )  # south slice

    mask = lon_sel & lat_sel

    if exclude_equator:
        mask &= np.abs(b) > 1e-3  # drop |b| ≲ 0.001°

    return mask.astype(float)  # 1.0 inside, 0.0 outside


def _roi_average_intensity(
    intensity_MeV: np.ndarray,
    counts: np.ndarray,
    ipix_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ROI-averaged intensity in GeV⁻¹, 1/√(counts per bin))."""
    norm = ipix_mask.sum()
    n_bins = intensity_MeV.shape[0]
    F_avg = np.empty(n_bins, float)
    inv_sqrtN = np.empty(n_bins, float)
    for i in range(n_bins):
        F_avg[i] = 1.0e3 * np.sum(intensity_MeV[i] * ipix_mask) / norm  # MeV⁻¹ → GeV⁻¹
        n_ev = np.sum(counts[i] * ipix_mask)
        inv_sqrtN[i] = 1.0 / np.sqrt(max(n_ev, 1.0))
    return F_avg, inv_sqrtN


def _interp_scaled(
    x_target: np.ndarray, x_src: np.ndarray, y_src: np.ndarray, scaling: float
) -> np.ndarray:
    """Interpolate ``y_src * x_src**scaling`` onto ``x_target``.

    Uses :func:`numpy.interp`'s default endpoint behaviour (constant
    extrapolation: values below ``x_src[0]`` return ``fp[0]`` and values
    above ``x_src[-1]`` return ``fp[-1]``).  This matches the legacy
    ``FDE_PASS8_new_v4.py`` / ``FDE_gll_v4.py`` pipelines, which never
    passed ``left=`` / ``right=`` to ``np.interp`` for the ISO / PS / UPS /
    HERMES (IC, Brems) templates.  The Fermi-bubble template still uses
    ``left=0, right=0`` inside :func:`load_bubble_template` because the
    legacy did so explicitly there.
    """
    return np.interp(
        np.asarray(x_target, float),
        np.asarray(x_src, float),
        np.asarray(y_src, float) * np.asarray(x_src, float) ** scaling,
    )
