from typing import Union, List

import torch as th
import numpy as np

from . import CurveBase


class PolynomialCurves(CurveBase):
    # Each ctrl point lives in R^2
    CTRL_DIM: int = 2

    def __init__(self, bezier_params: th.Tensor):
        super().__init__(bezier_params)

    @classmethod
    def create_from_specification(cls, batch_size: int, degree: int,
                                  height: int, width: int, device: th.device):
        assert batch_size > 0
        assert degree >= 1
        assert height > 0
        assert width > 0
        params = th.zeros(batch_size, cls.CTRL_DIM * degree,
                          height, width, device=device)
        return cls(params)

    @classmethod
    def from_2view(cls, flow_tensor: th.Tensor):
        raise NotImplementedError

    @classmethod
    def create_from_voxel_grid(cls, voxel_grid: th.Tensor, downsample_factor: int=8,
                               bezier_degree: int=2):
        assert isinstance(downsample_factor, int)
        assert downsample_factor >= 1
        batch, _, ht, wd = voxel_grid.shape
        assert ht % 8 == 0
        assert wd % 8 == 0
        ht, wd = ht//downsample_factor, wd//downsample_factor
        return cls.create_from_specification(batch_size=batch, degree=bezier_degree, height=ht,
                                             width=wd, device=voxel_grid.device)

    def _compute_flow_from_timestamps(self, timestamps: Union[List[float], np.ndarray]):
        if isinstance(timestamps, list):
            timestamps = th.Tensor(timestamps)
        elif isinstance(timestamps, np.ndarray):
            timestamps = th.from_numpy(timestamps)
        else:
            assert isinstance(timestamps, th.Tensor)

        assert th.min(timestamps) >= 0
        assert th.max(timestamps) <= 1

        params = self._param_view()
        timestamps = timestamps.float().to(params.device)

        k_idx = th.arange(1, self.degree+1, device=params.device)
        basis = timestamps[:, None] ** k_idx[None, :]

        raw_coeff_x = params[:, 0] # [b, n, h, w]
        raw_coeff_y = params[:, 1] # [b, n, h, w]

        flows = th.cat(
            [
                th.sum(raw_coeff_x[:, None] * basis[None, :, :, None, None], dim=2, keepdim=True),
                th.sum(raw_coeff_y[:, None] * basis[None, :, :, None, None], dim=2, keepdim=True)
            ],
            dim=2,
        )

        return flows.permute(1, 0, 2, 3, 4)
