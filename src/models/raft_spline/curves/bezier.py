from __future__ import annotations

import math
from typing import Union, List

import numpy as np
import torch as th
from numba import jit

has_scipy_special = True
try:
    from scipy import special
except ImportError:
    has_scipy_special = False

from . import CurveBase


class BezierCurves(CurveBase):
    # Each ctrl point lives in R^2
    CTRL_DIM: int = 2

    def __init__(self, bezier_params: th.Tensor):
        super().__init__(bezier_params)

        # math.comb is only available in python 3.8 or higher
        self.use_math_comb = hasattr(math, 'comb')
        if not self.use_math_comb:
            assert has_scipy_special
            assert hasattr(special, 'comb')

    def comb(self, n: int, k:int):
        if self.use_math_comb:
            return math.comb(n, k)
        return special.comb(n, k)

    @classmethod
    def create_from_specification(cls, batch_size: int, n_ctrl_pts: int,
                                  height: int, width: int, device: th.device) -> BezierCurves:
        assert batch_size > 0
        assert n_ctrl_pts > 1
        assert height > 0
        assert width > 0
        params = th.zeros(batch_size, cls.CTRL_DIM * (n_ctrl_pts - 1),
                          height, width, device=device)
        return cls(params)

    @classmethod
    def from_2view(cls, flow_tensor: th.Tensor) -> BezierCurves:
        # This function has been written to visualize 2-view predictions for our paper.
        batch_size, channel_size, height, width = flow_tensor.shape
        assert channel_size == 2 == cls.CTRL_DIM
        return cls(flow_tensor)

    @classmethod
    def create_from_voxel_grid(cls, voxel_grid: th.Tensor, downsample_factor: int=8,
                               bezier_degree: int=2) -> BezierCurves:
        assert isinstance(downsample_factor, int)
        assert downsample_factor >= 1
        batch, _, ht, wd = voxel_grid.shape
        assert ht % 8 == 0
        assert wd % 8 == 0
        ht, wd = ht//downsample_factor, wd//downsample_factor
        n_ctrl_pts = bezier_degree + 1
        return cls.create_from_specification(batch_size=batch, n_ctrl_pts=n_ctrl_pts, height=ht,
                                             width=wd, device=voxel_grid.device)

    @staticmethod
    def _get_binom_coeffs(degree: int):
        n = degree
        k = np.arange(degree) + 1
        return special.binom(n, k)

    @staticmethod
    @jit(nopython=True)
    def _get_time_coeffs(timestamps: np.ndarray, degree: int):
        assert timestamps.min() >= 0
        assert timestamps.max() <= 1
        assert timestamps.ndim == 1
        # I would like to check ensure float64 dtype but have not found a way to check in jit
        #assert timestamps.dtype == np.dtype('float64')

        num_ts = timestamps.size
        out = np.zeros((num_ts, degree))
        for t_idx in range(num_ts):
            for d_idx in range(degree):
                time = timestamps[t_idx]
                i = d_idx + 1
                out[t_idx, d_idx] = (1 - time)**(degree - i)*time**i
        return out

    def _compute_flow_from_timestamps(self, timestamps: Union[List[float], np.ndarray]):
        if isinstance(timestamps, list):
            timestamps = np.asarray(timestamps)
        else:
            assert isinstance(timestamps, np.ndarray)
        assert timestamps.dtype == 'float64'
        assert timestamps.size > 0
        assert np.min(timestamps) >= 0
        assert np.max(timestamps) <= 1

        degree = self.degree
        binom_coeffs = self._get_binom_coeffs(degree)
        time_coeffs = self._get_time_coeffs(timestamps, degree)
        # poly coeffs: time, degree
        polynomial_coeffs = np.einsum('j,ij->ij', binom_coeffs, time_coeffs)
        polynomial_coeffs = th.from_numpy(polynomial_coeffs).float().to(device=self.device)

        # params: batch, dim, degree, height, width
        params = self._param_view()
        # flow: timestamps, batch, dim, height, width
        flow = th.einsum('bdphw,tp->tbdhw', params, polynomial_coeffs)
        return flow
