"""Small helper to stamp run configuration onto plots.

Requested convention:

    **Prior**: box/gaussian
    **Nuisances**: indep/common/...

Callers pass ``prior_kind=None`` to leave a plot unstamped (the pipeline does
so unless ``plots.stamp_prior_nuisance`` is on in the analysis config).
"""

from __future__ import annotations


def _norm(x: str | None) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    return s or None


def stamp_prior_nuisance(
    ax,
    *,
    prior_kind: str | None,
    nuisance_mode: str | None,
    fontsize: int = 15,
    xy: tuple[float, float] = (0.98, 0.98),
    ha: str = "right",
    va: str = "top",
    color: str = "black",
    alpha: float = 0.95,
) -> None:
    """Write the Prior/Nuisance tag onto an Axes.

    If either field is missing/None, the stamp is skipped.
    """
    pk = _norm(prior_kind)
    nm = _norm(nuisance_mode)
    if pk is None or nm is None:
        return

    # Use mathtext for bold labels (no bbox / no frame).
    txt = r"$\bf{Prior}$: " + f"{pk}" + "\n" + r"$\bf{Nuisances}$: " + f"{nm}"
    if str(color) in {"0.15", "0.2", "0.25", "0.3"}:
        color = "black"
    ax.text(
        float(xy[0]),
        float(xy[1]),
        txt,
        transform=ax.transAxes,
        ha=str(ha),
        va=str(va),
        fontsize=int(fontsize),
        color=color,
        alpha=float(alpha),
    )
