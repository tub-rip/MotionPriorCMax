from typing import List, Optional, Any
from collections.abc import Mapping
import math
from functools import wraps

import torch
import torch as th
from torchmetrics import Metric
import torch.nn.functional as F

def _obj_has_function(obj, func_name: str):
    return hasattr(obj, func_name) and callable(getattr(obj, func_name))

def _data_to_cpu(input_: Any):
    if input_ is None:
        return input_
    if isinstance(input_, torch.Tensor):
        return input_.cpu()
    if isinstance(input_, dict):
        return {k: _data_to_cpu(v) for k, v in input_.items()}
    if isinstance(input_, list):
        return [_data_to_cpu(x) for x in input_]
    if isinstance(input_, tuple):
        return (_data_to_cpu(x) for x in input_)
    assert _obj_has_function(input_, 'cpu')
    return input_.cpu()

def to_cpu(func):
    ''' Move stuff to cpu '''

    @wraps(func)
    def inner(*args, **kwargs):
        output = func(*args, **kwargs)
        output = _data_to_cpu(output)
        return output

    return inner

def detach_tensors(cpu: bool=False):
    ''' Detach tensors and optionally move to cpu. Only detaches torch.Tensor instances '''
    # Decorator factory to enable decorator arguments: https://stackoverflow.com/a/50538967
    def allow_detach(key: str, value, train: bool):
        if train and key == 'loss':
            return False
        return isinstance(value, torch.Tensor)
    def detach_tensor(input_tensor: torch.Tensor, cpu: bool):
        assert isinstance(input_tensor, torch.Tensor)
        if cpu:
            return input_tensor.detach().cpu()
        return input_tensor.detach()
    def decorator(func):
        train = 'train' in func.__name__

        @wraps(func)
        def inner(*args, **kwargs):
            output = func(*args, **kwargs)
            if isinstance(output, Mapping):
                return {k: (detach_tensor(v, cpu) if allow_detach(k, v, train) else v) for k, v in output.items()}
            assert isinstance(output, torch.Tensor)
            if train:
                # Do not detach because this will be the loss function of the training hook, which must not be detached.
                return output
            return detach_tensor(output, cpu)
        return inner
    return decorator

def predictions_from_lin_assumption(source: th.Tensor, target_timestamps: List[float]) -> List[th.Tensor]:
    assert max(target_timestamps) <= 1
    assert 0 <= min(target_timestamps)

    output = list()
    for target_ts in target_timestamps:
        output.append(target_ts * source)
    return output

def reduce_ev_repr(ev_repr: torch.Tensor) -> torch.Tensor:
    # This function is useful to reduce the overhead of moving an event representation
    # to CPU for visualization.
    # For now simply sum up the time dimension to reduce the memory.
    assert isinstance(ev_repr, torch.Tensor)
    assert ev_repr.ndim == 4
    assert ev_repr.is_cuda
    return torch.sum(ev_repr, dim=1)
    
def epe_masked(source: th.Tensor, target: th.Tensor, valid_mask: Optional[th.Tensor] = None) -> Optional[th.Tensor]:
    # source: (N, C, *),
    # target: (N, C, *), source.shape == target.shape
    # valid_mask: (N, *), where the channel dimension is missing. I.e. valid_mask.ndim == target.ndim - 1
    assert source.ndim > 2
    assert source.shape == target.shape

    epe = th.sqrt(th.square(source - target).sum(1))
    if valid_mask is not None:
        assert valid_mask.shape[0] == target.shape[0]
        assert valid_mask.ndim == target.ndim - 1
        assert valid_mask.dtype == th.bool
        assert epe.shape == valid_mask.shape
        denominator = valid_mask.sum()
        if denominator == 0:
            return None
        return epe[valid_mask].sum() / denominator
    return th.mean(epe)

def epe_masked_multi(source_lst: List[th.Tensor], target_lst: List[th.Tensor], valid_mask_lst: Optional[List[th.Tensor]] = None) -> Optional[th.Tensor]:
    # source_lst: [(N, C, *), ...], M evaluation/predictions
    # target_lst: [(N, C, *), ...], source_lst[*].shape == target_lst[*].shape
    # valid_mask_lst: [(N, *), ...], where the channel dimension is missing. I.e. valid_mask_lst[*].ndim == target_lst[*].ndim - 1

    num_preds = len(source_lst)
    assert num_preds > 0
    assert len(target_lst) == num_preds, len(target_lst)
    if valid_mask_lst is not None:
        assert len(valid_mask_lst) == num_preds, len(valid_mask_lst)
    else:
        valid_mask_lst = [None]*num_preds
    epe_sum = 0
    denominator = 0
    for source, target, valid_mask in zip(source_lst, target_lst, valid_mask_lst):
        epe = epe_masked(source, target, valid_mask)
        if epe is not None:
            epe_sum += epe
            denominator += 1
    if denominator == 0:
        return None
    epe_sum: th.Tensor
    return epe_sum / denominator

def ae_masked_multi(source_lst: List[th.Tensor], target_lst: List[th.Tensor], valid_mask_lst: Optional[List[th.Tensor]]=None, degrees: bool=True) -> th.Tensor:
    # source_lst: [(N, C, *), ...], M evaluation/predictions
    # target_lst: [(N, C, *), ...], source_lst[*].shape == target_lst[*].shape
    # valid_mask_lst: [(N, *), ...], where the channel dimension is missing. I.e. valid_mask_lst[*].ndim == target_lst[*].ndim - 1

    num_preds = len(source_lst)
    assert num_preds > 0
    assert len(target_lst) == num_preds, len(target_lst)
    if valid_mask_lst is not None:
        assert len(valid_mask_lst) == num_preds, len(valid_mask_lst)
    else:
        valid_mask_lst = [None]*num_preds
    ae_sum = 0
    for source, target, valid_mask in zip(source_lst, target_lst, valid_mask_lst):
        ae_sum += ae_masked(source, target, valid_mask, degrees)
    ae_sum: th.Tensor
    return ae_sum / num_preds


def ae_masked(source: th.Tensor, target: th.Tensor, valid_mask: Optional[th.Tensor]=None, degrees: bool=True) -> th.Tensor:
    # source: (N, C, *),
    # target: (N, C, *), source.shape == target.shape
    # valid_mask: (N, *), where the channel dimension is missing. I.e. valid_mask.ndim == target.ndim - 1
    assert source.ndim > 2
    assert source.shape == target.shape

    shape = list(source.shape)
    extension_shape = shape
    extension_shape[1] = 1
    extension = th.ones(extension_shape, device=source.device)

    source_ext = th.cat((source, extension), dim=1)
    target_ext = th.cat((target, extension), dim=1)

    # according to https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf

    nominator = th.sum(source_ext * target_ext, dim=1)
    denominator = th.linalg.norm(source_ext, dim=1) * th.linalg.norm(target_ext, dim=1)

    tmp = th.div(nominator, denominator)

    # Somehow this seems necessary
    tmp[tmp > 1.0] = 1.0
    tmp[tmp < -1.0] = -1.0

    ae = th.acos(tmp)
    if degrees:
        ae = ae/math.pi*180

    if valid_mask is not None:
        assert valid_mask.shape[0] == target.shape[0]
        assert valid_mask.ndim == target.ndim - 1
        assert valid_mask.dtype == th.bool
        assert ae.shape == valid_mask.shape
        ae_masked = ae[valid_mask].sum() / valid_mask.sum()
        return ae_masked
    return th.mean(ae)

def n_pixel_error_masked(source: th.Tensor, target: th.Tensor, valid_mask: Optional[th.Tensor], n_pixels: float):
    # source: (N, C, *),
    # target: (N, C, *), source.shape == target.shape
    # valid_mask: (N, *), where the channel dimension is missing. I.e. valid_mask.ndim == target.ndim - 1
    assert source.ndim > 2
    assert source.shape == target.shape

    if valid_mask is not None:
        assert valid_mask.shape[0] == target.shape[0]
        assert valid_mask.ndim == target.ndim - 1
        assert valid_mask.dtype == th.bool

        num_valid = th.sum(valid_mask)
        assert num_valid > 0

    gt_flow_magn = th.linalg.norm(target, dim=1)
    error_magn = th.linalg.norm(source - target, dim=1)

    if valid_mask is not None:
        rel_error = th.zeros_like(error_magn)
        rel_error[valid_mask] = error_magn[valid_mask] / th.clip(gt_flow_magn[valid_mask], min=1e-6)
    else:
        rel_error = error_magn / th.clip(gt_flow_magn, min=1e-6)

    error_map = (error_magn > n_pixels) & (rel_error >= 0.05)

    if valid_mask is not None:
        error = error_map[valid_mask].sum() / num_valid
    else:
        error = th.mean(error_map.float())

    error *= 100
    return error

def calculate_flow_error(
    flow_gt: th.Tensor,
    flow_pred: th.Tensor,
    event_mask: th.Tensor = None,
    time_scale: th.Tensor = None,
) -> dict:
    """Calculate flow error.
    Args:
        flow_gt (th.Tensor) ... [B x 2 x H x W]
        flow_pred (th.Tensor) ... [B x 2 x H x W]
        event_mask (th.Tensor) ... [B x 1 x W x H]. Optional.
        time_scale (th.Tensor) ... [B x 1]. Optional. This will be multiplied.
            If you want to get error in 0.05 ms, time_scale should be
            `0.05 / actual_time_period`.

    Retuns:
        errors (dict) ... Key containers 'AE', 'EPE', '1/2/3PE'. all float.

    """
    # Only compute error over points that are valid in the GT (not inf or 0).
    flow_mask = th.logical_and(
        th.logical_and(~th.isinf(flow_gt[:, [0], ...]), ~th.isinf(flow_gt[:, [1], ...])),
        th.logical_and(th.abs(flow_gt[:, [0], ...]) > 0, th.abs(flow_gt[:, [1], ...]) > 0),
    )  # B, H, W
    if event_mask is None:
        total_mask = flow_mask
    else:
        if len(event_mask.shape) == 3:
            event_mask = event_mask[:, None]
        total_mask = th.logical_and(event_mask, flow_mask)
    gt_masked = flow_gt * total_mask  # b, 2, H, W
    pred_masked = flow_pred * total_mask
    n_points = th.sum(total_mask, dim=(1, 2, 3)) + 1e-5  # B, 1

    errors = {}
    # Average endpoint error.
    if time_scale is not None:
        time_scale = time_scale.reshape(len(gt_masked), 1, 1, 1)
        gt_masked = gt_masked * time_scale
        pred_masked = pred_masked * time_scale
    endpoint_error = th.linalg.norm(gt_masked - pred_masked, dim=1)
    errors["EPE"] = th.mean(th.sum(endpoint_error, dim=(1, 2)) / n_points)
    errors["1PE"] = th.mean(th.sum(endpoint_error > 1, dim=(1, 2)) / n_points)
    errors["2PE"] = th.mean(th.sum(endpoint_error > 2, dim=(1, 2)) / n_points)
    errors["3PE"] = th.mean(th.sum(endpoint_error > 3, dim=(1, 2)) / n_points)

    # Angular error
    u, v = pred_masked[:, 0, ...], pred_masked[:, 1, ...]
    u_gt, v_gt = gt_masked[:, 0, ...], gt_masked[:, 1, ...]
    cosine_similarity = (1.0 + u * u_gt + v * v_gt) / (th.sqrt(1 + u * u + v * v) * th.sqrt(1 + u_gt * u_gt + v_gt * v_gt))
    cosine_similarity = th.clamp(cosine_similarity, -1, 1)
    errors["AE"] = th.mean(th.sum(th.acos(cosine_similarity), dim=(1, 2)) / n_points)
    errors["AE"] = errors["AE"] * (180.0 / th.pi)
    return errors

def calculate_trajectory_flow_error(
    flows_gt: th.Tensor,
    flows_pred: th.Tensor,
    mask: th.Tensor = None
) -> dict:
    """Calculate flow error.
    Args:
        flow_gt (th.Tensor) ... [B x M x 2 x H x W]
        flow_pred (th.Tensor) ... [B x M x 2 x H x W]
        mask [th.Tensor]: [B x M x H x W]
    Retuns:
        errors (dict) ... Key containers 'AE', 'EPE', '1/2/3PE'. all float.
    """
    b, m, _, h, w = flows_gt.shape
    flows_gt = flows_gt.reshape(-1, 2, h, w)
    flows_pred = flows_pred.reshape(-1, 2, h, w)

    if mask is not None:
        mask = mask.reshape(-1, h, w)[:, None]

    errors = calculate_flow_error(flows_gt, flows_pred, mask)
    return {f'T{k}': v for k, v in errors.items()}

class InputPadder:
    """ Pads input tensor such that the last two dimensions are divisible by min_size """
    def __init__(self, min_size: int=8, no_top_padding: bool=False):
        assert min_size > 0
        self.min_size = min_size
        self.no_top_padding = no_top_padding
        self._pad = None

    def requires_padding(self, input_tensor: torch.Tensor):
        ht, wd = input_tensor.shape[-2:]
        answer = False
        answer &= ht % self.min_size == 0
        answer &= wd % self.min_size == 0
        return answer

    def pad(self, input_tensor: torch.Tensor):
        ht, wd = input_tensor.shape[-2:]
        pad_ht = (((ht // self.min_size) + 1) * self.min_size - ht) % self.min_size
        pad_wd = (((wd // self.min_size) + 1) * self.min_size - wd) % self.min_size
        if self.no_top_padding:
            # Pad only bottom instead of top
            # RAFT uses this for KITTI
            pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]
        else:
            # RAFT uses this for SINTEL (as default too)
            pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        if self._pad is None:
            self._pad = pad
        else:
            assert self._pad == pad
        return F.pad(input_tensor, self._pad, mode='replicate')

    def unpad(self, input_tensor: torch.Tensor):
        ht, wd = input_tensor.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return input_tensor[..., c[0]:c[1], c[2]:c[3]]
    
class EPE(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("epe", default=th.tensor(0, dtype=th.float64), dist_reduce_fx="sum")
        self.add_state("total", default=th.tensor(0, dtype=th.int64), dist_reduce_fx="sum")

    def update(self, source: th.Tensor, target: th.Tensor, valid_mask: Optional[th.Tensor]=None):
        # source (prediction): (N, C, *),
        # target (ground truth): (N, C, *), source.shape == target.shape
        # valid_mask: (N, *), where the channel dimension is missing. I.e. valid_mask.ndim == target.ndim - 1

        epe = epe_masked(source, target, valid_mask)
        if epe is not None:
            self.epe += epe.double()
            self.total += 1

    def compute(self):
        assert self.total > 0
        return (self.epe / self.total).float()

class EPE_MULTI(Metric):
    def __init__(self, dist_sync_on_step=False, min_traj_len=None, max_traj_len=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("epe", default=th.tensor(0, dtype=th.float64), dist_reduce_fx="sum")
        self.add_state("total", default=th.tensor(0, dtype=th.int64), dist_reduce_fx="sum")
        self.min_traj_len = min_traj_len
        self.max_traj_len = max_traj_len

    @staticmethod
    def compute_traj_len(target: List[th.Tensor]):
        target_stack = th.stack(target, dim=0)
        diff = target_stack[1:] - target_stack[:-1]
        return diff.square().sum(dim=2).sqrt().sum(dim=0)

    def get_true_mask(self, target: List[th.Tensor], device: th.device):
        valid_shape = (target[0].shape[0],) + target[0].shape[2:]
        return th.ones(valid_shape, dtype=th.bool, device=device)

    def update(self, source: List[th.Tensor], target: List[th.Tensor], valid_mask: Optional[List[th.Tensor]]=None):
        # source_lst: [(N, C, *), ...], M evaluation/predictions
        # target_lst: [(N, C, *), ...], source_lst[*].shape == target_lst[*].shape
        # valid_mask_lst: [(N, *), ...], where the channel dimension is missing. I.e. valid_mask_lst[*].ndim == target_lst[*].ndim - 1

        if self.min_traj_len is not None or self.max_traj_len is not None:
            traj_len = self.compute_traj_len(target=target)
            valid_len = self.get_true_mask(target=target, device=target[0].device)
            if self.min_traj_len is not None:
                valid_len &= (traj_len >= self.min_traj_len)
            if self.max_traj_len is not None:
                valid_len &= (traj_len <= self.max_traj_len)
            if valid_mask is None:
                valid_mask = [valid_len.clone() for _ in range(len(target))]
            else:
                valid_mask = [valid_mask[idx] & valid_len for idx in range(len(target))]

        epe = epe_masked_multi(source, target, valid_mask)
        if epe is not None:
            self.epe += epe.double()
            self.total += 1

    def compute(self):
        assert self.total > 0
        return (self.epe / self.total).float()
    
class AE(Metric):
    def __init__(self, degrees: bool=True, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.degrees = degrees

        self.add_state("ae", default=th.tensor(0, dtype=th.float64), dist_reduce_fx="sum")
        self.add_state("total", default=th.tensor(0, dtype=th.int64), dist_reduce_fx="sum")

    def update(self, source: th.Tensor, target: th.Tensor, valid_mask: Optional[th.Tensor]=None):
        # source (prediction): (N, C, *),
        # target (ground truth): (N, C, *), source.shape == target.shape
        # valid_mask: (N, *), where the channel dimension is missing. I.e. valid_mask.ndim == target.ndim - 1

        self.ae += ae_masked(source, target, valid_mask, degrees=self.degrees).double()
        self.total += 1

    def compute(self):
        assert self.total > 0
        return (self.ae / self.total).float()

class AE_MULTI(Metric):
    def __init__(self, degrees: bool=True, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.degrees = degrees

        self.add_state("ae", default=th.tensor(0, dtype=th.float64), dist_reduce_fx="sum")
        self.add_state("total", default=th.tensor(0, dtype=th.int64), dist_reduce_fx="sum")

    def update(self, source: List[th.Tensor], target: List[th.Tensor], valid_mask: Optional[List[th.Tensor]]=None):
        # source_lst: [(N, C, *), ...], M evaluation/predictions
        # target_lst: [(N, C, *), ...], source_lst[*].shape == target_lst[*].shape
        # valid_mask_lst: [(N, *), ...], where the channel dimension is missing. I.e. valid_mask_lst[*].ndim == target_lst[*].ndim - 1

        self.ae += ae_masked_multi(source, target, valid_mask, degrees=self.degrees).double()
        self.total += 1

    def compute(self):
        assert self.total > 0
        return (self.ae / self.total).float()
    
class NPE(Metric):
    def __init__(self, n_pixels: float, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        assert n_pixels > 0
        self.n_pixels = n_pixels

        self.add_state("npe", default=th.tensor(0, dtype=th.float64), dist_reduce_fx="sum")
        self.add_state("total", default=th.tensor(0, dtype=th.int64), dist_reduce_fx="sum")

    def update(self, source: th.Tensor, target: th.Tensor, valid_mask: Optional[th.Tensor]=None):
        # source (prediction): (N, C, *),
        # target (ground truth): (N, C, *), source.shape == target.shape
        # valid_mask: (N, *), where the channel dimension is missing. I.e. valid_mask.ndim == target.ndim - 1

        self.npe += n_pixel_error_masked(source, target, valid_mask, self.n_pixels).double()
        self.total += 1

    def compute(self):
        assert self.total > 0
        return (self.npe / self.total).float()
    
class FLOW_METRICS_MULTI(Metric):
    def __init__(self, dist_sync_on_step=False, min_traj_len=None, max_traj_len=None,
                 threshold=1):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total", default=th.tensor(0, dtype=th.int64), dist_reduce_fx="sum")
        self.add_state("T3PE", default=th.tensor(0, dtype=th.float32), dist_reduce_fx="sum")
        self.add_state("TEPE", default=th.tensor(0, dtype=th.float32), dist_reduce_fx="sum")
        self.add_state("TAE", default=th.tensor(0, dtype=th.float32), dist_reduce_fx="sum")
        self.keys_multi = ['T3PE', 'TEPE', 'TAE']
        self.keys_step = []

        self.num_steps = 6

        for i in range(self.num_steps):
            key = f"EPE_STEP{str(i).zfill(2)}"
            self.add_state(key, default=th.tensor(0, dtype=th.float32), dist_reduce_fx="sum")
            self.keys_step.append(key)

        self.min_traj_len = min_traj_len
        self.max_traj_len = max_traj_len

    @staticmethod
    def compute_traj_len(target: List[th.Tensor]):
        target_stack = th.stack(target, dim=0)
        diff = target_stack[1:] - target_stack[:-1]
        return diff.square().sum(dim=2).sqrt().sum(dim=0)

    def get_true_mask(self, target: List[th.Tensor], device: th.device):
        valid_shape = (target[0].shape[0],) + target[0].shape[2:]
        return th.ones(valid_shape, dtype=th.bool, device=device)

    def update(self, source: List[th.Tensor], target: List[th.Tensor], valid_mask: Optional[List[th.Tensor]]=None):
        # source_lst: [(N, C, *), ...], M evaluation/predictions
        # target_lst: [(N, C, *), ...], source_lst[*].shape == target_lst[*].shape
        # valid_mask_lst: [(N, *), ...], where the channel dimension is missing. I.e. valid_mask_lst[*].ndim == target_lst[*].ndim - 1

        if self.min_traj_len is not None or self.max_traj_len is not None:
            traj_len = self.compute_traj_len(target=target)
            valid_len = self.get_true_mask(target=target, device=target[0].device)
            if self.min_traj_len is not None:
                valid_len &= (traj_len >= self.min_traj_len)
            if self.max_traj_len is not None:
                valid_len &= (traj_len <= self.max_traj_len)
            if valid_mask is None:
                valid_mask = [valid_len.clone() for _ in range(len(target))]
            else:
                valid_mask = [valid_mask[idx] & valid_len for idx in range(len(target))]

        if valid_mask is not None:
            valid_mask = th.stack(valid_mask)

        assert len(source) == self.num_steps

        source = th.stack(source)
        target = th.stack(target)
        errors = calculate_trajectory_flow_error(target, source, valid_mask)

        for metric in self.keys_multi:
            val = getattr(self, metric)
            setattr(self, metric, val + errors[metric])

        for i, metric in enumerate(self.keys_step):
            val = getattr(self, metric)
            mask = valid_mask[i][:, None] if valid_mask is not None else None
            errors = calculate_flow_error(target[i], source[i], mask)
            update_val = errors['EPE']
            setattr(self, metric, val + update_val)

        self.total += 1

    def compute(self):
        total = self.total
        assert total > 0

        metrics_multi = {key: (getattr(self, key) / total).float() for key in self.keys_multi} 
        metrics_step = {key: (getattr(self, key) / total).float() for key in self.keys_step} 
        return {**metrics_multi, **metrics_step}