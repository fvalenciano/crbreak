"""Bridge between the extraction output and the fitter's input.

The fit is performed in *unscaled* differential intensity (legacy
``F_obs`` / ``sigma_F``), **not** in SED units.

The extraction pipeline (``reduce_window``) returns ``sigma`` in SED-scaled
units (multiplied by ``E^sed_scaling``, stored in the same dict).  The helpers
here *undo* that so the fitter always sees the same arrays:
``(E_GeV, y_unscaled, sigma_unscaled)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def prepare_fit_data_from_dict(
    res: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(E_GeV, y_unscaled, sigma_unscaled)`` from a reduction dict.

    Notes
    -----
    - ``y_unscaled`` is taken from ``res['F_obs_raw']`` (already unscaled).
    - ``sigma_unscaled`` is obtained by undoing the SED scaling applied in
      the extraction layer: ``sigma / E^sed_scaling``.
    """
    E = np.asarray(res["E_GeV"], dtype=float)
    scale = E ** float(res["sed_scaling"])
    y_unscaled = np.asarray(res["F_obs_raw"], dtype=float)
    sigma_unscaled = np.asarray(res["sigma"], dtype=float) / scale
    return E, y_unscaled, sigma_unscaled


def fit_meta_from_extract_dict(res: dict) -> dict:
    """Minimal metadata kept next to fit arrays (optional)."""
    return {
        "roi": res.get("roi"),
        "kind": res.get("kind"),
        "Ssys_frac": res.get("Ssys_frac"),
    }


def prepare_fit_data_from_window(
    window: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(E_GeV, y_unscaled, sigma_unscaled)`` from a populated ``Window``."""
    if getattr(window, "data", None) is None:
        raise RuntimeError(
            "Window has no extracted data; call window.extract(...) first."
        )
    return prepare_fit_data_from_dict(window.data)


def fit_meta_from_window(window: Any) -> dict:
    """Return metadata for a populated ``Window`` (optional)."""
    if getattr(window, "data", None) is None:
        raise RuntimeError(
            "Window has no extracted data; call window.extract(...) first."
        )
    return fit_meta_from_extract_dict(window.data)


def coerce_reduction_like(entry: Any) -> dict:
    """Normalize various window-like inputs to a reduction-style dict.

    Accepts:
    - a ``Window`` (has ``.data`` dict),
    - a ``WindowFitResult`` (has ``.reduction`` dict),
    - a plain reduction dict returned by ``reduce_window``.
    """
    if isinstance(entry, dict):
        return entry
    if hasattr(entry, "data") and isinstance(entry.data, dict):
        return entry.data
    if hasattr(entry, "reduction") and isinstance(entry.reduction, dict):
        return entry.reduction
    raise TypeError(f"Cannot interpret {type(entry).__name__} as a reduction dict.")


def prepare_fit_entry(entry: Any) -> dict:
    """Return a normalized fit-entry dict with arrays ready for fitting.

    Output keys
    -----------
    - ``E_GeV``: ndarray
    - ``y``: ndarray (unscaled)
    - ``sigma``: ndarray (unscaled)
    - ``meta``: dict (optional metadata)

    Notes
    -----
    If ``entry`` already looks like a per-window fit result (e.g. it has
    ``E_GeV/y/sigma`` attributes), this function reuses those arrays and does
    *not* re-derive them from the reduction dict.
    """
    # Fast path 1: already-prepared object (e.g. WindowFitResult).
    if all(hasattr(entry, k) for k in ("E_GeV", "y", "sigma")):
        return {
            "E_GeV": np.asarray(getattr(entry, "E_GeV"), dtype=float),
            "y": np.asarray(getattr(entry, "y"), dtype=float),
            "sigma": np.asarray(getattr(entry, "sigma"), dtype=float),
            "meta": dict(getattr(entry, "meta", {}) or {}),
        }

    # Fast path 2: plain dict that already carries the fit arrays
    # (used by montecarlo.py to pass synthetic ``y_pseudo`` to GlobalChi2Fit).
    if isinstance(entry, dict) and all(k in entry for k in ("E_GeV", "y", "sigma")):
        return {
            "E_GeV": np.asarray(entry["E_GeV"], dtype=float),
            "y": np.asarray(entry["y"], dtype=float),
            "sigma": np.asarray(entry["sigma"], dtype=float),
            "meta": dict(entry.get("meta", {}) or {}),
        }

    # Otherwise derive from a reduction-like dict (raw F_obs/sigma_obs/...).
    res = coerce_reduction_like(entry)
    E, y, sigma = prepare_fit_data_from_dict(res)
    return {"E_GeV": E, "y": y, "sigma": sigma, "meta": fit_meta_from_extract_dict(res)}
