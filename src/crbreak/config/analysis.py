"""Analysis configuration, read once from a YAML file (see ``configs/``).

A config may start with ``extends: <other>.yaml`` (path relative to itself)
and then list only the keys it changes; sections are merged key by key.  The
merged result must match the dataclasses below exactly: a missing or unknown
key is an error, so no value is ever silently defaulted in the code.
"""
from __future__ import annotations

import types
import typing
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path

import yaml

NUISANCE_MODES = (
    "indep", "common", "control", "control_common", "control_wsbpl", "control_common_wsbpl",
)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _increasing(edges: tuple[float, ...]) -> bool:
    return len(edges) >= 2 and all(a < b for a, b in zip(edges, edges[1:]))


GLON_SIGNS = ("both", "pos", "neg")


@dataclass(frozen=True)
class Sky:
    glon_edges: tuple[float, ...]
    glon_sign: str                  # both: |l| windows; pos / neg: only the l > 0 / l < 0 side
    glat_edges: tuple[float, ...]   # each |b| band is an independent analysis
    control_glat_lims: tuple[float, float]

    def __post_init__(self):
        _check(_increasing(self.glon_edges), f"sky.glon_edges must increase: {self.glon_edges}")
        _check(self.glon_sign in GLON_SIGNS, f"sky.glon_sign must be one of {GLON_SIGNS}: {self.glon_sign!r}")
        _check(_increasing(self.glat_edges), f"sky.glat_edges must increase: {self.glat_edges}")

    @property
    def bands(self) -> tuple[tuple[float, float], ...]:
        return tuple(zip(self.glat_edges, self.glat_edges[1:]))


REPO_ROOT = Path(__file__).resolve().parents[3]
BG_COMPONENTS = ("ISO", "PS", "UPS", "BUB", "IC_cmb", "IC_isrf", "Brems_H2", "Brems_HI")


@dataclass(frozen=True)
class Paths:
    data_dir: str        # Fermi data and templates
    dragon_dir: str      # DRAGON pi0 runs
    repo_data_dir: str   # templates shipped with the repo

    def __post_init__(self):
        # Relative paths are relative to the repo root; store them absolute.
        for f in fields(self):
            p = Path(getattr(self, f.name))
            object.__setattr__(self, f.name, str(p if p.is_absolute() else REPO_ROOT / p))


@dataclass(frozen=True)
class Files:
    pass8_exposure: str                      # under paths.data_dir ...
    pass8_events: str
    iem_exposure: str
    iem_events: str
    iem_model: str
    isotropic: str
    point_sources: str
    point_source_variants: dict[str, str]
    unresolved_sources: str
    hermes: dict[str, str]
    bubble_mask: str                         # ... under paths.repo_data_dir
    bubble_spectrum: str
    dragon_models: dict[str, dict[str, str]]  # ... under paths.dragon_dir

    def __post_init__(self):
        _check(set(self.hermes) <= set(BG_COMPONENTS), f"files.hermes keys must be in {BG_COMPONENTS}")


@dataclass(frozen=True)
class KindCuts:
    bg_components: tuple[str, ...]
    e_min_gev: float
    e_max_gev: float | None
    drop_last_bins: int
    sigma_reference: str

    def __post_init__(self):
        _check(set(self.bg_components) <= set(BG_COMPONENTS),
               f"bg_components must be a subset of {BG_COMPONENTS}: {self.bg_components}")
        _check(self.sigma_reference in {"raw", "subtracted"}, "sigma_reference: raw | subtracted")


@dataclass(frozen=True)
class AeffError:
    base: float
    low_below_mev: float
    low_slope: float
    low_pivot_log10_mev: float
    high_above_mev: float
    high_slope: float
    high_pivot_log10_mev: float


@dataclass(frozen=True)
class Data:
    kinds: tuple[str, ...]
    pass8_raw_comparison: bool
    sed_scaling: float
    min_counts_per_bin: int
    nside_iem: int
    nside_ups: int
    aeff_error: AeffError
    pass8: KindCuts
    iem: KindCuts
    dragon: KindCuts
    dragon_model: str             # kind DRAGON: key of files.dragon_models (pi0 H2 + HI maps)
    dragon_sigma_frac: float      # kind DRAGON: sigma = dragon_sigma_frac * F (legacy FDE_dragon.py)

    def __post_init__(self):
        _check(len(self.kinds) > 0 and set(self.kinds) <= {"PASS8", "IEM", "DRAGON"},
               f"data.kinds must be a non-empty subset of [PASS8, IEM, DRAGON]: {self.kinds}")

    def cuts(self, kind: str) -> KindCuts:
        return {"PASS8": self.pass8, "IEM": self.iem, "DRAGON": self.dragon}[kind]


@dataclass(frozen=True)
class ModelParams:
    p0: dict[str, float]                       # start values (K is seeded from the data)
    bounds: dict[str, tuple[float, float]]     # (lo, hi) for every parameter


@dataclass(frozen=True)
class CRReference:
    ebreak_gev: float                      # CR break of the fixed-shape fits and plot markers
    dgamma: float
    ebreak_band_gev: tuple[float, float]   # plotted band = ebreak_band_factor * ebreak_band_gev
    ebreak_band_factor: float
    dgamma_band: tuple[float, float]

    @property
    def ebreak_band(self) -> tuple[float, float]:
        lo, hi = self.ebreak_band_gev
        return (lo * self.ebreak_band_factor, hi * self.ebreak_band_factor)


@dataclass(frozen=True)
class Model:
    e0: float
    sbpl_s_fixed: float | None        # SBPL smoothness; None: s is free (sbpl.p0.s, sbpl.bounds.s)
    sbpl_ebreak_fixed: float | None   # [GeV] None: Ebreak free; else fixed in every SBPL fit
    pl: ModelParams
    sbpl: ModelParams

    def __post_init__(self):
        if self.sbpl_s_fixed is None:
            _check("s" in self.sbpl.p0 and "s" in self.sbpl.bounds,
                   "model.sbpl_s_fixed: null (free s) needs model.sbpl.p0.s and model.sbpl.bounds.s")


@dataclass(frozen=True)
class Nuisance:
    classical_only: bool
    mode: str
    prior: str
    prior_sigma: float
    prior_width: float
    prior_tail: float
    basis: str
    n_nodes: int
    bspline_degree: int
    enforce_s_bounds: bool
    control_enforce_s_bounds: bool
    max_dw: float
    safety_net: bool

    def __post_init__(self):
        _check(self.mode in NUISANCE_MODES, f"nuisance.mode must be one of {NUISANCE_MODES}")
        _check(self.prior in {"box_gauss", "gaussian"}, "nuisance.prior: box_gauss | gaussian")
        _check(self.basis in {"bspline", "piecewise_linear"}, "nuisance.basis: bspline | piecewise_linear")


@dataclass(frozen=True)
class MC:
    do_windows: bool
    do_global: bool
    n_toys_window: int
    n_toys_global_free: int
    n_toys_global_fixed: int
    batch_size: int
    store_deltachi2: bool
    seed: int
    sample_theta: bool
    warm_start: bool


@dataclass(frozen=True)
class Solver:
    chi2_maxfev: int             # classical per-window curve_fit
    global_max_nfev: int         # classical global least_squares
    global_x_scale: str
    profiled_max_iter: int       # L-BFGS-B outer loop of every profiled fit
    hessian_eps: float           # finite-difference step of the profiled Hessian (pcov)
    hessian_ridge: float


@dataclass(frozen=True)
class Scan:
    enabled: bool
    n_ebreak: int
    n_dgamma: int
    ebreak_range: tuple[float, float]
    dgamma_range: tuple[float, float]


@dataclass(frozen=True)
class ProfileErrors:
    params: tuple[str, ...]
    delta_chi2: float
    ebreak_grid: tuple[float, float, int]
    dgamma_grid: tuple[float, float, int]
    gamma_radius: float
    gamma_n_points: int

    def __post_init__(self):
        _check(set(self.params) <= {"Ebreak", "dgamma", "gamma1", "gamma2"},
               f"profile_errors.params: subset of [Ebreak, dgamma, gamma1, gamma2]: {self.params}")


@dataclass(frozen=True)
class BreakPolish:
    radius_ebreak: float
    radius_dgamma: float
    n_ebreak: int
    n_dgamma: int
    min_improvement: float


@dataclass(frozen=True)
class SedBand:
    n_samples: int                 # draws from the fit covariance
    q: tuple[float, float]         # percentiles of the drawn curves
    seed: int


@dataclass(frozen=True)
class NuisanceBand:
    n_samples: int                 # posterior draws of w
    n_samples_sed_plus_nuisances: int
    seed: int
    hessian_eps: float             # finite-difference step of the w Hessian
    hessian_ridge: float


@dataclass(frozen=True)
class Plots:
    scan_smooth_sigma: float
    stamp_prior_nuisance: bool     # write "Prior / Nuisances" on the figures
    contour_levels: tuple[float, ...]
    sed_band: SedBand
    nuisance_band: NuisanceBand
    background_components: bool
    dragon_model: str | None
    dragon_norm_n_bins: int
    ps_comparison: bool


@dataclass(frozen=True)
class Run:
    n_workers: int | None
    save_checkpoint: bool


@dataclass(frozen=True)
class Output:
    parent: str
    tag: str


@dataclass(frozen=True)
class AnalysisConfig:
    paths: Paths
    files: Files
    sky: Sky
    data: Data
    model: Model
    cr_reference: CRReference
    nuisance: Nuisance
    mc: MC
    solver: Solver
    scan: Scan
    profile_errors: ProfileErrors
    break_polish: BreakPolish
    plots: Plots
    run: Run
    output: Output

    def __post_init__(self):
        _check(not (self.scan.enabled and self.model.sbpl_ebreak_fixed is not None),
               "scan.enabled needs a free Ebreak: set scan.enabled: false with model.sbpl_ebreak_fixed")
        _check(self.data.dragon_model in self.files.dragon_models,
               f"data.dragon_model must be a key of files.dragon_models: {self.data.dragon_model!r}")
        if self.nuisance.mode.startswith("control"):
            c0, c1 = self.sky.control_glat_lims
            for lo, hi in self.sky.bands:
                _check(hi <= c0 or lo >= c1,
                       f"|b| band [{lo:g}, {hi:g}] overlaps sky.control_glat_lims "
                       f"[{c0:g}, {c1:g}] (nuisance.mode={self.nuisance.mode})")


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("extends", None)
    return _merge(_read(path.parent / parent), raw) if parent else raw


def _coerce(tp, value, where: str):
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        (tp,) = [a for a in args if a is not type(None)]
        return _coerce(tp, value, where)
    if origin is dict:
        _check(isinstance(value, dict), f"{where}: expected a mapping, got {value!r}")
        key_t, val_t = args
        return {_coerce(key_t, k, where): _coerce(val_t, v, f"{where}.{k}") for k, v in value.items()}
    if origin is tuple:
        _check(isinstance(value, (list, tuple)), f"{where}: expected a list, got {value!r}")
        elem_types = [args[0]] * len(value) if args[-1] is Ellipsis else list(args)
        _check(len(elem_types) == len(value), f"{where}: expected {len(elem_types)} items, got {value!r}")
        return tuple(_coerce(t, v, f"{where}[{i}]") for i, (t, v) in enumerate(zip(elem_types, value)))
    ok = {
        bool: isinstance(value, bool),
        int: isinstance(value, int) and not isinstance(value, bool),
        float: isinstance(value, (int, float)) and not isinstance(value, bool),
        str: isinstance(value, str),
    }[tp]
    _check(ok, f"{where}: expected {tp.__name__}, got {value!r}")
    return float(value) if tp is float else value


def _build(cls, raw, where: str):
    _check(isinstance(raw, dict), f"{where}: expected a section, got {raw!r}")
    names = [f.name for f in fields(cls)]
    missing, unknown = set(names) - raw.keys(), raw.keys() - set(names)
    _check(not missing and not unknown,
           f"{where}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
    hints = typing.get_type_hints(cls)
    return cls(**{
        n: (_build(hints[n], raw[n], f"{where}.{n}") if is_dataclass(hints[n])
            else _coerce(hints[n], raw[n], f"{where}.{n}"))
        for n in names
    })


def load_config(path: str | Path) -> AnalysisConfig:
    """Read ``path`` (resolving ``extends``) and validate it."""
    path = Path(path)
    return _build(AnalysisConfig, _read(path), path.name)


def config_from_checkpoint(state: dict) -> AnalysisConfig:
    """Config a checkpoint was made with (its ``config`` entry).

    Checkpoints written before the YAML configs existed have no ``config``;
    they get ``configs/baseline.yaml``, whose plot references (SED scaling,
    CR bands, band settings) are the values those runs used.  Keys added
    after a checkpoint was written are filled in (:data:`_ADDED_KEYS`).
    """
    if "config" in state:
        raw = state["config"]
        baseline = _read(REPO_ROOT / "configs" / "baseline.yaml")
        for (section, key), value in _ADDED_KEYS.items():
            if key not in raw[section]:
                value = baseline[section][key] if value is _FROM_BASELINE else value
                raw = _merge(raw, {section: {key: value}})
        return _build(AnalysisConfig, raw, "checkpoint config")
    return load_config(REPO_ROOT / "configs" / "baseline.yaml")


# Keys added after checkpoints were written.  ``sky.glon_sign`` gets the value
# those runs used (|l| windows); the ``data.dragon*`` keys are only read by
# kind DRAGON, which such checkpoints never ran, so they come from baseline.yaml;
# ``model.sbpl_ebreak_fixed`` is None (Ebreak was always free).
_FROM_BASELINE = object()
_ADDED_KEYS = {
    ("sky", "glon_sign"): "both",
    ("data", "dragon"): _FROM_BASELINE,
    ("data", "dragon_model"): _FROM_BASELINE,
    ("data", "dragon_sigma_frac"): _FROM_BASELINE,
    ("model", "sbpl_ebreak_fixed"): None,
}


def config_to_dict(cfg: AnalysisConfig) -> dict:
    """Plain ``dict`` (lists instead of tuples) for YAML / checkpoint metadata."""
    def plain(x):
        if isinstance(x, dict):
            return {k: plain(v) for k, v in x.items()}
        return [plain(v) for v in x] if isinstance(x, (list, tuple)) else x

    return plain(asdict(cfg))


def dump_config(cfg: AnalysisConfig, path: str | Path) -> None:
    """Write the fully merged config (no ``extends``) next to a run's outputs."""
    Path(path).write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False))
