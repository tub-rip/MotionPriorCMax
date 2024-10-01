from pathlib import Path
from typing import Optional

import torch.nn.functional as F
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset

from ..utils.representation import VoxelGrid, norm_voxel_grid
from ..utils.keys import DataLoading, DataSetType

class EVIMO2_Datasubset(Dataset):
    def __init__(self,
                 train_or_val_path: Path,
                 data_augm: bool,
                 num_bins_context: int,
                 flow_every_n_ms: int,
                 load_voxel_grid: bool=True,
                 extended_voxel_grid: bool=True,
                 normalize_voxel_grid: bool=False,
                 downsample: bool=False,
                 photo_augm: bool=False,
                 return_img: bool=True,
                 return_ev: bool=True,
                 flow_time: int=500,
                 provide_raw_events: bool=False,
                 polarity_aware_batching: bool=False,
                 ):
        assert train_or_val_path.is_dir()
        assert(num_bins_context == 41)

        self.nbins_context2corr = {
            6: 4,
            11: 7,
            21: 13,
            41: 25,
        }
        self.nbins_context2deltatime = {
            6: 100000,
            11: 50000,
            21: 25000,
            41: 12500,
        }

        # Save output dimensions
        self.original_height = 480
        self.original_width = 640

        self.resize_height = 384
        self.resize_width = 512

        self.return_ev = True
        self.train_or_val_path = train_or_val_path

        crop_height = 352
        crop_width = 480

        self.crop_shape = (crop_height, crop_width)

        self.flow_duration = flow_time

        if downsample:
            crop_height = crop_height // 2
            crop_width = crop_width // 2
        
        self.crop_height = crop_height
        self.crop_width = crop_width

        self.normalize_voxel_grid: Optional[norm_voxel_grid] = \
                            norm_voxel_grid if normalize_voxel_grid else None
        self.num_bins_context = num_bins_context
        self.num_bins_correlation = self.nbins_context2corr[num_bins_context]

        self.provide_raw_events = provide_raw_events
        self.polarity_aware_batching = polarity_aware_batching
        self.t_bins = torch.linspace(0, 1, self.num_bins_context+1) \
                                        if provide_raw_events else None

        self.init_seq()

    
    def init_seq(self):
        assert self.train_or_val_path.is_dir()
        # Flow data
        # self.flow_path = self.train_or_val_path / 'dataset_multiflow.h5'
        self.flow_path = self.train_or_val_path / 'dataset_multiflow_10steps_vis.h5'

        with h5py.File(str(self.flow_path), 'r') as h5f:
            self.flow_time = np.asarray(h5f['time'])

        # Event data
        ev_dir = self.train_or_val_path
        assert ev_dir.is_dir()
        self.xy_path = ev_dir / 'dataset_events_xy.npy'
        self.p_path = ev_dir / 'dataset_events_p.npy'
        self.t_path = ev_dir / 'dataset_events_t.npy'

        ### Voxel Grid Saving
        self.num_bins_total = self.num_bins_context + self.nbins_context2corr[self.num_bins_context] - 1

        # time search, prev is 0.4 before the current flow
        prev_time = self.flow_time - 0.4
        next_time = self.flow_time + self.flow_duration / 1000
        self.evt = np.load(self.t_path)

        self.flow2evt = np.searchsorted(self.evt, self.flow_time, side='left')
        self.prev2evt = np.searchsorted(self.evt, prev_time, side='left')
        self.next2evt = np.searchsorted(self.evt, next_time, side='left')

        # compute start index to save 0.4 sec at the beiginning
        self.start_index = 0
        for tid in range(len(self.flow_time)):
            if (self.flow_time[tid] - self.evt[0]) > 0.4:
                self.start_index = tid
                break

        self.length = len(self.flow_time) - self.start_index

    def get_num_bins_context(self):
        return self.num_bins_context

    def get_num_bins_correlation(self):
        return self.num_bins_correlation

    def get_num_bins_total(self):
        return self.num_bins_total

    def _voxel_grid_bin_idx_for_reference(self) -> int:
        return self.get_num_bins_correlation() - 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        index = index + self.start_index

        xy = np.load(self.xy_path, mmap_mode='r')
        p = np.load(self.p_path, mmap_mode='r')
        t = np.load(self.t_path, mmap_mode='r')

        prev_start = self.prev2evt[index]
        next = self.next2evt[index]

        xs, ys, ts, ps = np.array(xy[prev_start:next, 0]).astype('int32'),  \
                            np.array(xy[prev_start:next, 1]).astype('int32'),  \
                            np.array(t[prev_start:next] * 1e6).astype('int'),  \
                            np.array(p[prev_start:next]).astype('int')
        
        xs = torch.tensor(xs)
        ys = torch.tensor(ys)
        ts = torch.tensor(ts)
        ps = 1 - torch.tensor(ps) 
        
        height = self.original_height
        width = self.original_width

        voxel_grid = VoxelGrid(self.num_bins_total, height, width)
        voxel_grid = voxel_grid.convert(x=xs, y=ys, pol=ps, time=ts).contiguous()

        if voxel_grid is not None and self.normalize_voxel_grid is not None:
            voxel_grid = self.normalize_voxel_grid(voxel_grid)

        num_flow_steps = int(self.flow_duration // 50)
        with h5py.File(self.flow_path, 'r') as h5f:
            gt_flow = np.asarray(h5f['multiflow'][index])[:num_flow_steps]
            gt_flow_ts = np.linspace(0, 1, 1+num_flow_steps)[1:]
            id_mask = torch.tensor(h5f['obj_id_mask'][index])

        flow_valid = (~np.isnan(gt_flow[:, 0, ...])) & (~np.isnan(gt_flow[:, 1, ...]))
        flow_valid = torch.tensor(flow_valid, dtype=torch.bool)
        gt_flow[np.isnan(gt_flow)] = 0.
        gt_flow = torch.tensor(gt_flow)

        gt_flow = F.interpolate(gt_flow, 
                                size=[self.resize_height, self.resize_width], mode='bilinear', align_corners=False)

        flow_valid = F.interpolate(flow_valid.float().unsqueeze(1), 
                                size=[self.resize_height, self.resize_width], mode='nearest')\
                                        .squeeze(1).bool()
        id_mask = F.interpolate(id_mask.float()[None, None], 
                                size=[self.resize_height, self.resize_width], mode='nearest')\
                                .squeeze()
        y_scale = self.resize_height / height
        x_scale = self.resize_width / width
        gt_flow[:, 0, ...] *= x_scale
        gt_flow[:, 1, ...] *= y_scale
        voxel_grid = F.interpolate(voxel_grid.unsqueeze(0), size=(self.resize_height, self.resize_width), mode='bilinear', align_corners=False).squeeze(0)

        out = {
            DataLoading.BIN_META: {
                'bin_idx_for_reference': self._voxel_grid_bin_idx_for_reference(),
                'nbins_context': self.get_num_bins_context(),
                'nbins_correlation': self.get_num_bins_correlation(),
                'nbins_total': self.get_num_bins_total(),
            },
            DataLoading.FLOW: gt_flow,
            DataLoading.FLOW_TIMESTAMPS: gt_flow_ts,
            DataLoading.DATASET_TYPE: DataSetType.EVIMO2,
            DataLoading.EV_REPR: voxel_grid,
            DataLoading.FLOW_VALID: flow_valid,
            DataLoading.ID_MASK: id_mask
        }

        if self.provide_raw_events:
            all_events = torch.stack((ys, xs, ts, ps), dim=1)
            ts_start = ts[-1] - (self.flow_duration * 1e3)
            ts_end = ts[-1]
            events = all_events[all_events[:, 2] > ts_start].float()
            events[:, 2] = (events[:, 2] - ts_start) / (ts_end - ts_start)
            timestamps = events[:, 2].contiguous()
            bin_indices = torch.searchsorted(self.t_bins, timestamps) - 1
            bin_indices[bin_indices == -1] = 0
            events = torch.cat((events, bin_indices[:, None]), dim=1)

            if self.polarity_aware_batching:
                ev_data = {
                    DataLoading.POS_EVENTS: events[events[:, 3] == 1],
                    DataLoading.NEG_EVENTS: events[events[:, 3] == 0]
                }
            else:
                ev_data = {DataLoading.EVENTS: events}

            out.update(ev_data)

            out.update({DataLoading.X_SCALE: x_scale,
                        DataLoading.Y_SCALE: y_scale})
        return out
