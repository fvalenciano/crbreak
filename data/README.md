# Data

`bubbles/` is part of the repository. `fermi/` and `dragon/` (6.3 GB) are downloaded with
`bash scripts/download_data.sh`, which checks every file against `MANIFEST.sha256`.
The paths below are the ones in `configs/baseline.yaml` (`paths`, `files`).

| Path | Size | Content |
|---|---|---|
| `fermi/pass8/events_binned_healpix.fits`, `fermi_exposure_healpix_nside256.fits` | 2 x 253 MB | *Fermi*-LAT counts and exposure, HEALPix NSIDE 256 (kind `PASS8`) |
| `fermi/iem/fermi_healpix_NSIDE256.fits`, `fermi_exposure-z105_healpix_NSIDE256.fits` | 2 x 473 MB | counts and exposure used with the interstellar emission model (kind `IEM`) |
| `fermi/iem/gll_iem_v07.fits` | 465 MB | LAT Galactic interstellar emission model |
| `fermi/iso_P8R3_CLEAN_V3_v1.txt` | 2 kB | LAT isotropic spectrum |
| `fermi/point_sources/FV_PS_4FGLv34_..._KingPSF[_PL,_LP,_PLSEC].fits` | 4 x 1057 MB | 4FGL point-source maps, NSIDE 1024: the one subtracted and three spectral variants for the comparison plot |
| `fermi/Fermi_Unresolved_Sources_Flux_Map_Mean_mSp4.fits` | 76 MB | unresolved-source flux map |
| `fermi/hermes/EleCutOff50TeV-*_nside64.fits` | 4 x 10 MB | inverse-Compton (CMB, ISRF) and bremsstrahlung (H2, HI) templates |
| `dragon/*_nside64.fits` | 6 x 10 MB | DRAGON pi0 maps (H2, HI) of three models (kind `DRAGON` and the background plot) |
| `bubbles/template_bub.npy` | 1.5 MB | Fermi-bubbles mask, NSIDE 128, from the NPTFit example data (https://github.com/bsafdi/NPTFit) |
| `bubbles/fermi_bubbles_table2.txt` | 1 kB | Fermi-bubbles spectrum, Ackermann et al. 2014, table 2 |

A run only reads what its config asks for: the baseline (`PASS8`) does not need `fermi/iem/`,
and only the point-source comparison plot (`plots.ps_comparison`) reads the three variants.
