from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

import numpy as np


class SpectralModel(ABC):
    """Abstract base for spectral models.

    Subclasses set ``name`` and ``param_names`` and implement
    :meth:`__call__`, :meth:`default_p0`, and :meth:`default_bounds`.
    ``E0`` is the reference energy of the normalisation ``K``.
    """

    name: str = "MODEL"
    param_names: tuple[str, ...] = ()

    def __init__(self, *, E0: float):
        self.E0 = float(E0)

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    @abstractmethod
    # This calls the model, make it callable:
    def __call__(self, E: np.ndarray, *params: float) -> np.ndarray: ...

    @abstractmethod
    def default_p0(self, E: np.ndarray, y: np.ndarray) -> tuple[float, ...]: ...

    @abstractmethod
    def default_bounds(self) -> tuple[Sequence[float], Sequence[float]]: ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(E0={self.E0}, params={self.param_names})"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _seed_K(
    E: np.ndarray,
    y: np.ndarray,
    E0: float,
    *,
    gamma_guess: float,
) -> float:
    """Robust starting value for the normalisation ``K`` at ``E0``.

    The model parameter ``K`` is the differential intensity at the
    reference energy ``E0`` (1 GeV by convention).  Using ``max(y)``
    underestimates ``K`` by orders of magnitude when the data only covers
    ``E >> E0``; we extrapolate the lowest-energy observed point with the
    nominal slope ``gamma_guess`` instead:

    .. math::
        K_0 \\approx y_{\\min E} \\cdot (E_{\\min} / E_0)^{\\gamma_\\mathrm{guess}}

    Returns 1.0 if the data has no finite positive points.
    """
    E = np.asarray(E, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(E) & np.isfinite(y) & (y > 0)
    if not finite.any():
        return 1.0
    Ef, yf = E[finite], y[finite]
    i_min = int(np.argmin(Ef))
    return float(yf[i_min] * (Ef[i_min] / float(E0)) ** float(gamma_guess))


def _checked_start_and_bounds(
    model: SpectralModel,
    p0: Mapping[str, float],
    bounds: Mapping[str, Sequence[float]],
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """``p0`` covers every parameter but ``K`` (seeded from the data);
    ``bounds`` covers every parameter."""
    names = set(model.param_names)
    if set(p0) != names - {"K"} or set(bounds) != names:
        raise ValueError(
            f"{model.name}: p0 needs {sorted(names - {'K'})} and bounds needs "
            f"{sorted(names)}; got p0={sorted(p0)}, bounds={sorted(bounds)}."
        )
    return (
        {k: float(v) for k, v in p0.items()},
        {k: (float(lo), float(hi)) for k, (lo, hi) in bounds.items()},
    )


# ----------------------------------------------------------------------
# Concrete models
# ----------------------------------------------------------------------


class PowerLaw(SpectralModel):
    """Simple power law.

    .. math::
        F(E) = K \\, (E/E_0)^{-\\gamma}

    ``p0`` gives the starting ``gamma`` (``K`` is seeded from the data with
    that slope) and ``bounds`` the ``(lo, hi)`` range of every parameter.
    """

    name = "PL"
    param_names = ("K", "gamma")

    def __init__(self, *, E0: float, p0: Mapping[str, float], bounds: Mapping[str, Sequence[float]]):
        super().__init__(E0=E0)
        self.p0, self.bounds = _checked_start_and_bounds(self, p0, bounds)

    @classmethod
    def from_config(cls, model_cfg) -> "PowerLaw":
        """Build from the ``model`` section of the analysis config."""
        return cls(E0=model_cfg.e0, p0=model_cfg.pl.p0, bounds=model_cfg.pl.bounds)

    def __call__(self, E, K, gamma):
        return K * (np.asarray(E, dtype=float) / self.E0) ** (-gamma)

    def default_p0(self, E, y):
        gamma = self.p0["gamma"]
        return (_seed_K(E, y, self.E0, gamma_guess=gamma), gamma)

    def default_bounds(self):
        return tuple(zip(*(self.bounds[n] for n in self.param_names)))


class SmoothBrokenPowerLaw(SpectralModel):
    """Smooth broken power law (legacy ``sbknpow``).

    .. math::
        F(E) = K \\, (E/E_0)^{-\\gamma_1}
               \\Bigl[ 1 + (E/E_\\mathrm{br})^{|\\Delta\\gamma|/s} \\Bigr]^{s}

    By default, uses the legacy convention
    :math:`\\Delta\\gamma = \\gamma_1 - \\gamma_2 \\ge 0`, so positive
    ``dgamma`` produces hardening above ``E_break``.  Set
    ``signed_dgamma=True`` to allow negative values, where negative
    ``dgamma`` produces softening above ``E_break``; the ``dgamma`` range
    then becomes ``[-hi, hi]`` of ``bounds["dgamma"]``.

    The smoothing parameter ``s`` is fixed at ``s_fixed``, or is a free
    parameter (last in ``param_names``) when ``s_fixed`` is ``None``.
    ``Ebreak`` is fixed at ``ebreak_fixed`` when given, and then leaves
    ``param_names``.  ``p0`` gives the starting ``gamma1``, ``dgamma``,
    ``Ebreak``, ``s`` (``K`` is seeded from the data with slope ``gamma1``)
    and ``bounds`` the ``(lo, hi)`` range of every parameter; the entries of
    a fixed ``Ebreak`` / ``s`` are ignored.
    """

    name = "SBPL"
    param_names = ("K", "gamma1", "dgamma", "Ebreak")

    def __init__(
        self,
        *,
        E0: float,
        s_fixed: float | None,
        p0: Mapping[str, float],
        bounds: Mapping[str, Sequence[float]],
        signed_dgamma: bool = False,
        ebreak_fixed: float | None = None,
    ):
        super().__init__(E0=E0)
        self.s_fixed = None if s_fixed is None else float(s_fixed)
        self.ebreak_fixed = None if ebreak_fixed is None else float(ebreak_fixed)
        self.signed_dgamma = bool(signed_dgamma)
        names = (("K", "gamma1", "dgamma")
                 + (("Ebreak",) if self.ebreak_fixed is None else ())
                 + (("s",) if self.s_fixed is None else ()))
        if names != type(self).param_names:
            self.param_names = names
        unused = {"Ebreak", "s"} - set(names)
        self.p0, self.bounds = _checked_start_and_bounds(
            self,
            {k: v for k, v in p0.items() if k not in unused},
            {k: v for k, v in bounds.items() if k not in unused},
        )

    @classmethod
    def from_config(cls, model_cfg, *, signed_dgamma: bool = False) -> "SmoothBrokenPowerLaw":
        """Build from the ``model`` section of the analysis config."""
        return cls(
            E0=model_cfg.e0,
            s_fixed=model_cfg.sbpl_s_fixed,
            ebreak_fixed=model_cfg.sbpl_ebreak_fixed,
            p0=model_cfg.sbpl.p0,
            bounds=model_cfg.sbpl.bounds,
            signed_dgamma=signed_dgamma,
        )

    @property
    def fixed(self) -> dict[str, float]:
        """Fixed ``Ebreak`` (as :attr:`FrozenSpectralModel.fixed`, for the fit tables)."""
        eb = getattr(self, "ebreak_fixed", None)  # absent in checkpoints older than the option
        return {} if eb is None else {"Ebreak": eb}

    def __call__(self, E, K, g1, dgamma, *rest):
        # ``rest`` is the free subset of (Ebreak, s), in ``param_names`` order.
        extra = dict(zip(self.param_names[3:], rest))
        Ebr = extra["Ebreak"] if "Ebreak" in extra else self.ebreak_fixed
        E = np.asarray(E, dtype=float)
        s = float(np.clip(extra["s"] if "s" in extra else self.s_fixed, 1e-6, None))
        dg = float(dgamma)
        if self.signed_dgamma:
            if abs(dg) < 1e-12:
                return K * (E / self.E0) ** (-g1)
            bend = (1.0 + (E / Ebr) ** (abs(dg) / s)) ** (np.sign(dg) * s)
        else:
            bend = (1.0 + (E / Ebr) ** (abs(dg) / s)) ** s
        return K * (E / self.E0) ** (-g1) * bend

    def default_p0(self, E, y):
        g1 = self.p0["gamma1"]
        p0 = {**self.p0, "K": _seed_K(E, y, self.E0, gamma_guess=g1)}
        return tuple(p0[n] for n in self.param_names)

    def default_bounds(self):
        bounds = dict(self.bounds)
        if self.signed_dgamma:
            hi = bounds["dgamma"][1]
            bounds["dgamma"] = (-hi, hi)
        return tuple(zip(*(bounds[n] for n in self.param_names)))


# ----------------------------------------------------------------------
# Parameter-freezing wrapper
# ----------------------------------------------------------------------


class FrozenSpectralModel(SpectralModel):
    """Wrap an existing :class:`SpectralModel` with some parameters fixed.

    The frozen instance is callable with only the **free** parameters; the
    fixed values are spliced back in their native order at evaluation time
    so any downstream code (``Chi2Fit``, ``envelope``, ``FitResult.params``)
    keeps working unchanged.

    Useful for the legacy "fixedshape" SBPL fits (``Ebreak`` and ``dgamma``
    pinned to a Cosmic-Ray reference value) where each window then only
    fits ``K`` and ``gamma1``.

    For an SBPL, the high-energy slope ``gamma2 = gamma1 - dgamma`` can
    also be frozen even though it is not a native parameter: ``gamma1``
    then leaves the free set and is rebuilt as ``gamma2 + dgamma`` at
    evaluation time (used for the 1-D profile errors on ``gamma2``).

    Examples
    --------
    >>> sbpl = SmoothBrokenPowerLaw()
    >>> frozen = FrozenSpectralModel(sbpl, Ebreak=26.4, dgamma=0.14)
    >>> frozen.param_names
    ('K', 'gamma1')
    >>> FrozenSpectralModel(sbpl, gamma2=2.4).param_names
    ('K', 'dgamma', 'Ebreak')
    """

    def __init__(self, model: SpectralModel, **fixed: float):
        derived = (
            {"gamma2"}
            if {"gamma1", "dgamma"} <= set(model.param_names)
            else set()
        )
        unknown = set(fixed) - set(model.param_names) - derived
        if unknown:
            raise ValueError(
                f"Cannot freeze unknown parameter(s) {sorted(unknown)} for "
                f"{model.name} (known: {model.param_names})."
            )
        if "gamma2" in fixed and "gamma1" in fixed:
            raise ValueError(
                "Cannot freeze both gamma1 and gamma2 -- that also fixes dgamma; "
                "freeze gamma1 and dgamma instead."
            )
        if len(fixed) >= len(model.param_names):
            raise ValueError(
                "Cannot freeze ALL parameters of the model -- nothing to fit."
            )
        super().__init__(E0=model.E0)
        self._inner = model
        self._fixed = {n: float(v) for n, v in fixed.items()}
        # Native parameters removed from the free set (gamma2 removes gamma1).
        self._dropped = set(self._fixed) | (
            {"gamma1"} if "gamma2" in self._fixed else set()
        )
        self.name = f"{model.name}|fix={','.join(sorted(self._fixed))}"
        self.param_names = tuple(
            p for p in model.param_names if p not in self._dropped
        )

    @property
    def fixed(self) -> dict[str, float]:
        return dict(self._fixed)

    def _expand(self, free_params: Sequence[float]) -> tuple[float, ...]:
        if len(free_params) != len(self.param_names):
            raise TypeError(
                f"{self.name}: expected {len(self.param_names)} free params "
                f"({self.param_names}), got {len(free_params)}."
            )
        if "gamma2" in self._fixed:
            by_name = {**dict(zip(self.param_names, free_params)), **self._fixed}
            by_name["gamma1"] = by_name["gamma2"] + by_name["dgamma"]
            return tuple(by_name[p] for p in self._inner.param_names)
        it = iter(free_params)
        return tuple(
            self._fixed[p] if p in self._fixed else next(it)
            for p in self._inner.param_names
        )

    def __call__(self, E, *free_params):
        return self._inner(E, *self._expand(free_params))

    def default_p0(self, E, y):
        full = self._inner.default_p0(E, y)
        return tuple(
            v
            for n, v in zip(self._inner.param_names, full)
            if n not in self._dropped
        )

    def default_bounds(self):
        lo, hi = self._inner.default_bounds()
        new_lo = tuple(
            b for n, b in zip(self._inner.param_names, lo) if n not in self._dropped
        )
        new_hi = tuple(
            b for n, b in zip(self._inner.param_names, hi) if n not in self._dropped
        )
        return new_lo, new_hi
