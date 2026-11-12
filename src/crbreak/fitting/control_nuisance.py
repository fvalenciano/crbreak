from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..data_extract.main import Data, SkyMesh
from .global_fit import GlobalFitResult
from .main import Fitting
from .models import SpectralModel
from .nuisance import NuisanceModel, nuisance_kwargs_from_config


class ControlNuisanceBuilder:
    def __init__(
        self,
        *,
        kind: str,
        signal_fitting: Fitting,
        mesh: SkyMesh,
        cfg,
    ) -> None:
        """Control windows: same |l| bins, |b| in ``cfg.sky.control_glat_lims``;
        their nuisance models use ``cfg.nuisance`` with
        ``control_enforce_s_bounds``."""
        self.kind = kind
        self.signal_fitting = signal_fitting
        self.mesh = mesh
        self.cfg = cfg
        self.control_glat_lims = cfg.sky.control_glat_lims
        self.nm_kwargs = {
            **nuisance_kwargs_from_config(cfg.nuisance),
            "enforce_s_bounds": cfg.nuisance.control_enforce_s_bounds,
        }
        self._control_fitting: Fitting | None = None
        self.allsky_globals: dict[str, GlobalFitResult] = {}

    @property
    def control_fitting(self) -> Fitting:
        if self._control_fitting is None:
            c0, c1 = self.control_glat_lims
            print(
                f"[fit_test] Control windows: |b|=({c0:g},{c1:g}), same |l| bins"
            )
            control_mesh = SkyMesh(
                glon_lims=self.mesh.glon_lims,
                glat_lims=self.control_glat_lims,
                Nglon=self.mesh.Nglon,
                Nglat=1,
                glon_bins=self.mesh.glon_bins,
            )
            control_data = Data(control_mesh)
            control_data.build(self.kind, cfg=self.cfg, subtraction=True)
            self._control_fitting = Fitting(control_data, solver=self.signal_fitting.solver)
        return self._control_fitting

    def _nuisance(
        self,
        spectral_model_factory: Callable[[], SpectralModel],
        *,
        share_nuisance: bool,
    ) -> NuisanceModel:
        return NuisanceModel(
            spectral_model=spectral_model_factory(),
            share_nuisance=share_nuisance,
            **self.nm_kwargs,
        )

    def fit_by_glon(
        self,
        spectral_model_factory: Callable[[], SpectralModel],
        fit_name: str,
        *,
        progress: bool = True,
        refit_control: bool = False,
    ) -> dict[tuple[float, float], np.ndarray]:
        self.control_fitting.fit_per_window(
            [spectral_model_factory()], do_profile_errors=False, progress=False
        )
        nm_control = self._nuisance(spectral_model_factory, share_nuisance=False)
        self.control_fitting.fit_per_window(
            [nm_control], do_profile_errors=False, progress=progress
        )
        signal_seed = self._nuisance(spectral_model_factory, share_nuisance=False)
        self.signal_fitting.fit_per_window(
            [signal_seed], do_profile_errors=False, progress=False
        )
        w_by_glon = {}
        for key, wf_low in self.signal_fitting.window_fits.items():
            wf_high = self.control_fitting.window_fits[key]
            w_low = np.asarray(wf_low.fits[fit_name].model.last_w, dtype=float)
            w_high = np.asarray(wf_high.fits[fit_name].model.last_w, dtype=float)
            g_pair = GlobalFitResult.fit(
                self._nuisance(spectral_model_factory, share_nuisance=True),
                {"low": wf_low, "high": wf_high},
                shared=(),
                solver=self.signal_fitting.solver,
                w_shared0=0.5 * (w_low + w_high),
            )
            if progress:
                Fitting._print_box_prior_diagnostic(
                    "[control pair]",
                    f"{fit_name} shared w for l=({wf_low.lowl:g},{wf_low.highl:g})",
                    g_pair.model,
                    g_pair.w_shared,
                )
            w_by_glon[tuple(float(x) for x in wf_low.glon_bin)] = np.asarray(
                g_pair.w_shared, dtype=float
            ).copy()
        if refit_control:
            nm_control_fixed = self._nuisance(
                spectral_model_factory, share_nuisance=False
            )
            nm_control_fixed.w_shared_by_glon = w_by_glon
            self.control_fitting.fit_per_window(
                [nm_control_fixed], do_profile_errors=False, progress=False
            )
        return w_by_glon

    def fit_allsky(
        self,
        spectral_model_factory: Callable[[], SpectralModel],
        fit_name: str,
        *,
        progress: bool = True,
        refit_control: bool = False,
    ) -> dict[tuple[float, float], np.ndarray]:
        self.control_fitting.fit_per_window(
            [spectral_model_factory()], do_profile_errors=False, progress=False
        )
        nm_control = self._nuisance(spectral_model_factory, share_nuisance=False)
        self.control_fitting.fit_per_window(
            [nm_control], do_profile_errors=False, progress=progress
        )
        signal_seed = self._nuisance(spectral_model_factory, share_nuisance=False)
        self.signal_fitting.fit_per_window(
            [signal_seed], do_profile_errors=False, progress=False
        )
        windows = {
            **{f"low:{k}": wf for k, wf in self.signal_fitting.window_fits.items()},
            **{f"high:{k}": wf for k, wf in self.control_fitting.window_fits.items()},
        }
        w0 = np.mean(
            [
                np.asarray(wf.fits[fit_name].model.last_w, dtype=float)
                for wf in windows.values()
            ],
            axis=0,
        )
        g = GlobalFitResult.fit(
            self._nuisance(spectral_model_factory, share_nuisance=True),
            windows,
            shared=(),
            solver=self.signal_fitting.solver,
            w_shared0=w0,
        )
        if progress:
            Fitting._print_box_prior_diagnostic(
                "[control common]",
                f"{fit_name} all-sky shared w",
                g.model,
                g.w_shared,
            )
        self.allsky_globals[fit_name] = g
        w_shared = np.asarray(g.w_shared, dtype=float).copy()
        w_by_glon = {
            tuple(float(x) for x in wf.glon_bin): w_shared.copy()
            for wf in self.signal_fitting.window_fits.values()
        }
        if refit_control:
            nm_control_fixed = self._nuisance(
                spectral_model_factory, share_nuisance=False
            )
            nm_control_fixed.w_shared_by_glon = w_by_glon
            self.control_fitting.fit_per_window(
                [nm_control_fixed], do_profile_errors=False, progress=False
            )
        return w_by_glon
