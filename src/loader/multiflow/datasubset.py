from pathlib import Path
from typing import Optional, List

import h5py
from torch.utils.data import Dataset

from .sample import Sample
from ..utils.representation import norm_voxel_grid
from ..utils.keys import DataLoading, DataSetType

NUM_EVENTS_MAX = 23542180

class Datasubset(Dataset):
    def __init__(self,
                 train_or_val_path: Path,
                 num_bins_context: int,
                 flow_every_n_ms: int,
                 load_voxel_grid: bool=True,
                 extended_voxel_grid: bool=True,
                 normalize_voxel_grid: bool=False,
                 downsample: bool=False,
                 return_img: bool=True,
                 return_ev: bool=True,
                 provide_raw_events: bool=False,
                 polarity_aware_batching: bool=False,
                 cap_num_events: bool =False,
                 prediction_time: int=500):
        assert train_or_val_path.is_dir()
        assert train_or_val_path.name in ('train', 'test')
        assert 100 <= prediction_time <= 500

        self.provide_raw_events = provide_raw_events
        self.polarity_aware_batching = polarity_aware_batching

        # Save output dimensions
        original_height = 384
        original_width = 512

        crop_height = 352
        crop_width = 480
        self.crop_shape = (crop_height, crop_width)

        self.return_img = return_img
        if not self.return_img:
            raise NotImplementedError
        self.return_ev = return_ev

        if downsample:
            crop_height = crop_height // 2
            crop_width = crop_width // 2

        self.delta_ts_flow_ms = flow_every_n_ms
        self.prediction_time_ms = prediction_time
        self.normalize_voxel_grid: Optional[norm_voxel_grid] = norm_voxel_grid if normalize_voxel_grid else None

        sample_list: List[Sample] = list()
        for sample_path in train_or_val_path.iterdir():
            if not sample_path.is_dir():
                continue
            event_path = sample_path / 'events' /'events.h5'

            with h5py.File(event_path, 'r') as events:
                if len(events['t']) < NUM_EVENTS_MAX or not cap_num_events:
                    sample_list.append(
                        Sample(sample_path, original_height, original_width,
                               num_bins_context, load_voxel_grid, extended_voxel_grid,
                               downsample, prediction_time)
                    )
        self.sample_list = sample_list

    def get_num_bins_context(self):
        return self.sample_list[0].num_bins_context

    def get_num_bins_correlation(self):
        return self.sample_list[0].num_bins_correlation

    def get_num_bins_total(self):
        return self.sample_list[0].num_bins_total

    def _voxel_grid_bin_idx_for_reference(self) -> int:
        return self.sample_list[0].voxel_grid_bin_idx_for_reference()

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        sample = self.sample_list[index]

        voxel_grid = sample.get_voxel_grid() if self.return_ev else None
        if voxel_grid is not None and self.normalize_voxel_grid is not None:
            voxel_grid = self.normalize_voxel_grid(voxel_grid)

        gt_flow_dict = sample.get_flow_gt(self.delta_ts_flow_ms)
        gt_flow = gt_flow_dict['flow']
        gt_flow_ts = gt_flow_dict['timestamps']

        imgs_with_ts = sample.get_images()
        imgs = imgs_with_ts['images']
        img_ts = imgs_with_ts['timestamps']

        events = sample.get_events_context() if self.provide_raw_events else None

        ts_end = gt_flow_ts[-1] # hacky, for prediction time < 500ms

        # normalize image timestamps from 0 to 1
        assert len(img_ts) == 2
        ts_start = img_ts[0]
        #ts_end = img_ts[1]
        assert ts_end > ts_start
        img_ts = [(x - ts_start)/(ts_end - ts_start) for x in img_ts]
        assert img_ts[0] == 0
        #assert img_ts[1] == 1

        # we assume that img_ts[0] refers to reference time and img_ts[1] to the final target time
        gt_flow_ts = [(x - ts_start)/(ts_end - ts_start) for x in gt_flow_ts]
        assert gt_flow_ts[-1] == 1
        assert len(gt_flow_ts) == len(gt_flow)

        if self.spatial_augmentor is not None:
            data_to_augment = {'flow': gt_flow, 'images': imgs}

            if self.provide_raw_events is not None:
                data_to_augment['events'] = events

            if voxel_grid is not None:
                data_to_augment['ev_repr'] = voxel_grid

            voxel_grid, gt_flow, _, imgs, events = self.spatial_augmentor(**data_to_augment)

        if self.photo_augmentor is not None:
            imgs = self.photo_augmentor(imgs)

        out = {
            DataLoading.BIN_META: {
                'bin_idx_for_reference': self._voxel_grid_bin_idx_for_reference(),
                'nbins_context': self.get_num_bins_context(),
                'nbins_correlation': self.get_num_bins_correlation(),
                'nbins_total': self.get_num_bins_total(),
            },
            DataLoading.FLOW: gt_flow,
            DataLoading.FLOW_TIMESTAMPS: [ts for ts in gt_flow_ts],
            DataLoading.IMG: imgs,
            DataLoading.IMG_TIMESTAMPS: img_ts,
            DataLoading.DATASET_TYPE: DataSetType.MULTIFLOW2D,
        }
        if voxel_grid is not None:
            out.update({DataLoading.EV_REPR: voxel_grid})

        if self.provide_raw_events:
            if self.polarity_aware_batching:
                ev_data = {
                    DataLoading.POS_EVENTS: events[events[:, 3] == 1],
                    DataLoading.NEG_EVENTS: events[events[:, 3] == 0]
                }
            else:
                ev_data = {DataLoading.EVENTS: events}
            
            out.update(ev_data)

        return out