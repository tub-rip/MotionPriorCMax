from typing import Union, List

import torch as th
import numpy as np

from . import CurveBase
from ..utils import cvx_upsample


class LearnedCurves(CurveBase):
    # Each ctrl point lives in R^2
    CTRL_DIM: int = 2

    def __init__(self, bezier_params: th.Tensor, basis_network):
        super().__init__(bezier_params)
        self.basis_network = basis_network

    def cpu(self):
        return type(self)(self._params.cpu(), self.basis_network)

    def detach(self, clone: bool=False, cpu: bool=False):
        params = self._params.detach()
        if cpu:
            return type(self)(params.cpu(), self.basis_network)
        if clone:
            params = params.clone()
        return type(self)(params, self.basis_network)

    def create_upsampled(self, mask: th.Tensor):
        """ Upsample params [N, dim, H/8, W/8] -> [N, dim, H, W] using convex combination """
        up_params = cvx_upsample(self._params, mask)
        return type(self)(up_params, self.basis_network)

    @classmethod
    def create_from_specification(cls, batch_size: int, degree: int,
                                  height: int, width: int, device: th.device,
                                  basis_network):
        assert batch_size > 0
        assert degree >= 1
        assert height > 0
        assert width > 0
        params = th.zeros(batch_size, cls.CTRL_DIM * degree,
                          height, width, device=device)
        return cls(params, basis_network)

    @classmethod
    def from_2view(cls, flow_tensor: th.Tensor):
        raise NotImplementedError

    @classmethod
    def create_from_voxel_grid(cls, voxel_grid: th.Tensor, downsample_factor: int=8,
                               bezier_degree: int=2, basis_network = None):
        assert isinstance(downsample_factor, int)
        assert downsample_factor >= 1
        batch, _, ht, wd = voxel_grid.shape
        assert ht % 8 == 0
        assert wd % 8 == 0
        ht, wd = ht//downsample_factor, wd//downsample_factor
        return cls.create_from_specification(batch_size=batch, degree=bezier_degree, height=ht,
                                             width=wd, device=voxel_grid.device,
                                             basis_network=basis_network)

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
        self.basis_network.to(params.device)
        basis = self.basis_network(timestamps[..., None])

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
