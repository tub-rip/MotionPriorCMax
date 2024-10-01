from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np
import torch as th
from torchvision import transforms

from ..utils import cvx_upsample

INTERP = transforms.InterpolationMode.BICUBIC


class CurveBase(ABC):
    def __init__(self, params) -> None:
        assert params.ndim == 4
        # bezier_params: batch, ctrl_dim*(n_ctrl_pts - 1), height, width
        self._params = params

        # some helpful meta-data:
        self.batch, channels, self.ht, self.wd = self._params.shape
        assert channels % 2 == 0
        # P0 is always zeros as it corresponds to the pixel locations.
        # Consequently, we only compute P1, P2, ...
        self.n_ctrl_pts = channels // self.CTRL_DIM + 1
        assert self.n_ctrl_pts > 0

    @property
    def device(self):
        return self._params.device

    @property
    def dtype(self):
        return self._params.dtype

    def create_upsampled(self, mask: th.Tensor):
        """ Upsample params [N, dim, H/8, W/8] -> [N, dim, H, W] using convex combination """
        up_params = cvx_upsample(self._params, mask)
        return type(self)(up_params)

    def detach(self, clone: bool=False, cpu: bool=False):
        params = self._params.detach()
        if cpu:
            return type(self)(params.cpu())
        if clone:
            params = params.clone()
        return type(self)(params)

    def detach_(self, cpu: bool=False) -> None:
        # Detaches the bezier parameters in-place!
        self._params = self._params.detach()
        if cpu:
            self._params = self._params.cpu()

    def cpu(self):
        return type(self)(self._params.cpu())

    def cpu_(self) -> None:
        # Puts the bezier parameters to CPU in-place!
        self._params = self._params.cpu()

    @property
    def requires_grad(self):
        return self._params.requires_grad

    @property
    def batch_size(self):
        return self._params.shape[0]

    @property
    def degree(self):
        return self.n_ctrl_pts - 1

    @property
    def dim(self):
        return self._params.shape[1]

    @property
    def height(self):
        return self._params.shape[-2]

    @property
    def width(self):
        return self._params.shape[-1]

    def get_params(self) -> th.Tensor:
        return self._params

    def _param_view(self) -> th.Tensor:
        return self._params.view(self.batch, self.CTRL_DIM, self.degree, self.ht, self.wd)

    def delta_update_params(self, delta_bezier: th.Tensor) -> None:
        assert delta_bezier.shape == self._params.shape
        self._params = self._params + delta_bezier

    def get_flow_from_reference(self, time: Union[float, int, List[float], np.ndarray]) -> th.Tensor:
        params = self._param_view()
        batch, dim, degree, height, width = params.shape
        time_is_scalar = isinstance(time, int) or isinstance(time, float)
        if time_is_scalar:
            assert time >= 0.0
            assert time <= 1.0
            if time == 1:
                P_end = params[:, :, -1, ...]
                return P_end
            if time == 0:
                return th.zeros((batch, dim, height, width), dtype=self.dtype, device=self.device)
            time = np.array([time], dtype='float64')
        elif isinstance(time, list):
            time = np.asarray(time, dtype='float64')
        else:
            assert isinstance(time, np.ndarray)
        assert time.dtype == 'float64'
        assert time.size > 0
        assert np.min(time) >= 0
        assert np.max(time) <= 1

        # flow is coords1 - coords0
        # flows: timestamps, batch, dim, height, width
        flows = self._compute_flow_from_timestamps(timestamps=time)
        if time_is_scalar:
            assert flows.shape[0] == 1
            return flows[0]
        return flows

    @classmethod
    @abstractmethod
    def create_from_specification(cls, batch_size: int, n_ctrl_pts: int,
                                  height: int, width: int, device: th.device):
        pass

    @classmethod
    @abstractmethod
    def from_2view(cls, flow_tensor: th.Tensor):
        pass

    @classmethod
    @abstractmethod
    def create_from_voxel_grid(cls, voxel_grid: th.Tensor, downsample_factor: int=8,
                               bezier_degree: int=2):
        pass

    @abstractmethod
    def _compute_flow_from_timestamps(self, timestamps: Union[List[float], np.ndarray]):
        pass
