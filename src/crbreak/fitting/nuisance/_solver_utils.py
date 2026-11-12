from __future__ import annotations

import numpy as np


def _zip_bounds(
    lo: np.ndarray, hi: np.ndarray
) -> list[tuple[float | None, float | None]]:
    """``minimize(L-BFGS-B)``-style bounds.  ``inf`` is mapped to ``None``."""
    out: list[tuple[float | None, float | None]] = []
    for a, b in zip(lo, hi):
        a = None if not np.isfinite(a) else float(a)
        b = None if not np.isfinite(b) else float(b)
        out.append((a, b))
    return out


def _fd_hessian(fun, x: np.ndarray, *, eps: float) -> np.ndarray:
    """
        Symmetric Hessian of ``fun`` at ``x`` (``H[i,j] = ∂²fun/∂xi∂xj``).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n), dtype=float)
    f0 = float(fun(x))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = eps
        f1 = float(fun(x + ei))
        f2 = float(fun(x - ei))
        H[i, i] = (f1 - 2.0 * f0 + f2) / (eps * eps)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = eps
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = eps
            fpp = float(fun(x + ei + ej))
            fpm = float(fun(x + ei - ej))
            fmp = float(fun(x - ei + ej))
            fmm = float(fun(x - ei - ej))
            Hij = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
            H[i, j] = Hij
            H[j, i] = Hij
    return H


def _pcov_from_hessian(H: np.ndarray, *, ridge: float) -> np.ndarray:
    """`H_chi2 ≈ 2*H_-2logL
    """
    H = 0.5 * (np.asarray(H, dtype=float) + np.asarray(H, dtype=float).T)
    n = H.shape[0]
    lam = ridge
    for _ in range(6):
        try:
            return 2.0 * np.linalg.inv(H + lam * np.eye(n))
        except np.linalg.LinAlgError:
            lam *= 10.0
    return 2.0 * np.linalg.pinv(H + lam * np.eye(n))
