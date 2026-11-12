from __future__ import annotations

from typing import Any, Tuple
import numpy as np

from .data_reduction import preload_cubes, reduce_window


class Window:
    def __init__(self, glon_bin: Tuple[float, float], glat_bin: Tuple[float, float]):
        self.glon_bin = glon_bin
        self.glat_bin = glat_bin

        self.lowl = float(glon_bin[0])
        self.highl = float(glon_bin[1])
        self.lowb = float(glat_bin[0])
        self.highb = float(glat_bin[1])

        self.data: dict | None = None  # populated by extract()

    def extract(self, kind: str, **reduce_kwargs: Any) -> dict:
        """Run :func:`reduce_window` on this ROI (``reduce_kwargs``: ``glon_sign``,
        ``data``, ``paths``, ``files``, ``subtraction`` and the preloaded cubes)."""
        if self.data is not None:
            raise RuntimeError(
                "Window already extracted; create a new Window to re-run."
            )

        self.data = reduce_window(
            kind=kind,
            lowl=self.lowl,
            highl=self.highl,
            lowb=self.lowb,
            highb=self.highb,
            **reduce_kwargs,
        )
        return self.data


class SkyMesh:
    def __init__(
        self,
        glon_lims: Tuple[float, float],
        glat_lims: Tuple[float, float],
        Nglon: int = 9,
        Nglat: int = 1,
        glon_bins: np.ndarray | None = None,
        glat_bins: np.ndarray | None = None,
    ):
        self.glon_lims = glon_lims
        self.glat_lims = glat_lims

        if glon_bins is None:
            self.glon_bins = np.linspace(glon_lims[0], glon_lims[1], int(Nglon) + 1)
        else:
            self.glon_bins = np.asarray(glon_bins, dtype=float)
            if self.glon_bins.ndim != 1 or self.glon_bins.size < 2:
                raise ValueError("glon_bins must be a 1D array with at least 2 edges.")
            if not np.all(np.diff(self.glon_bins) > 0):
                raise ValueError("glon_bins must be strictly increasing.")

        if glat_bins is None:
            self.glat_bins = np.linspace(glat_lims[0], glat_lims[1], int(Nglat) + 1)
        else:
            self.glat_bins = np.asarray(glat_bins, dtype=float)
            if self.glat_bins.ndim != 1 or self.glat_bins.size < 2:
                raise ValueError("glat_bins must be a 1D array with at least 2 edges.")
            if not np.all(np.diff(self.glat_bins) > 0):
                raise ValueError("glat_bins must be strictly increasing.")

        self.Nglon = int(self.glon_bins.size - 1)
        self.Nglat = int(self.glat_bins.size - 1)

        self.glon_centers = 0.5 * (self.glon_bins[1:] + self.glon_bins[:-1])
        self.glat_centers = 0.5 * (self.glat_bins[1:] + self.glat_bins[:-1])


class Data:
    def __init__(self, sky_mesh: SkyMesh):
        self.sky_mesh = sky_mesh
        self.Nglon = sky_mesh.Nglon
        self.Nglat = sky_mesh.Nglat
        self.data_dict: dict[str, Window] = {}

    def build(self, kind: str, *, cfg, subtraction: bool) -> dict[str, Window]:
        """Extract every window of the mesh with the analysis config ``cfg``
        (``sky.glon_sign`` and its ``data``, ``paths`` and ``files`` sections).

        Returns
        -------
        dict[key, Window] where key = "iijj" (i = glon idx, j = glat idx).
        """
        reduce_kwargs = dict(
            glon_sign=cfg.sky.glon_sign,
            data=cfg.data,
            paths=cfg.paths,
            files=cfg.files,
            subtraction=subtraction,
            **preload_cubes(kind, paths=cfg.paths, files=cfg.files, nside_iem=cfg.data.nside_iem),
        )

        data_dict: dict[str, Window] = {}
        for i in range(self.Nglon):
            for j in range(self.Nglat):
                key = f"{i:02d}{j:02d}"
                glon_bin = (self.sky_mesh.glon_bins[i], self.sky_mesh.glon_bins[i + 1])
                glat_bin = (self.sky_mesh.glat_bins[j], self.sky_mesh.glat_bins[j + 1])

                window = Window(glon_bin=glon_bin, glat_bin=glat_bin)
                window.extract(kind=kind, **reduce_kwargs)

                data_dict[key] = window

        self.data_dict = data_dict
        return data_dict
