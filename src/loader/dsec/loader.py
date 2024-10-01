import math
from pathlib import Path
from typing import Dict, Tuple
import weakref
from torch.utils.data import DataLoader

import cv2
import h5py
from numba import jit
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import imageio
import hdf5plugin
import pandas as pd

from .utils import VoxelGrid, flow_16bit_to_float

TRAIN_SEQS = ['zurich_city_04_d', 'zurich_city_02_a', 'interlaken_00_f', 'zurich_city_11_a',
              'zurich_city_04_b', 'zurich_city_02_d', 'interlaken_00_d', 'zurich_city_04_c',
              'zurich_city_07_a', 'zurich_city_04_f', 'zurich_city_06_a', 'zurich_city_11_b',
              'interlaken_00_c', 'zurich_city_02_b', 'interlaken_00_e', 'zurich_city_04_a',
              'zurich_city_05_a', 'zurich_city_02_e', 'zurich_city_03_a', 'interlaken_00_g',
              'zurich_city_08_a', 'zurich_city_04_e', 'thun_00_a', 'zurich_city_02_c']
VAL_SEQS = ['zurich_city_05_b', 'zurich_city_11_c']

class DatasetProvider:
    def __init__(self, dataset_path: str, split: str = 'train', num_bins: int = 15,
                 polarity_aware_batching: bool = False, norm_type=None,
                 quantile: int = 0):
        dataset_path = Path(dataset_path)
        assert dataset_path.is_dir(), f"Invalid dataset path: {dataset_path}"
        self.name_mapper_test = []

        seq_names = TRAIN_SEQS if split == 'train' else VAL_SEQS if split == 'val' else []

        sequences = [
            Sequence(child, split, num_bins,
                     polarity_aware_batching=polarity_aware_batching,
                     norm_type=norm_type, quantile=quantile)
            for child in sorted(dataset_path.iterdir()) if child.name in seq_names
        ]

        self.dataset = torch.utils.data.ConcatDataset(sequences)

    def get_name_mapping_test(self):
        return self.name_mapper_test
    
class Sequence(Dataset):
    def __init__(self, seq_path: Path, phase: str = 'train', num_bins: int = 15, 
                 timestamp_path: str = None,
                 polarity_aware_batching=False, norm_type=None, quantile=0):
        assert num_bins >= 1, "Number of bins should be at least 1"
        assert seq_path.is_dir(), f"{seq_path} is not a valid directory"
        
        self.name = seq_path.name
        self.phase = phase
        self.num_bins = num_bins
        self.polarity_aware_batching = polarity_aware_batching
        self.height, self.width = 480, 640  # Constants for output dimensions
        self.t_bins = np.linspace(0, 1, self.num_bins + 1)

        # Event representation
        self.voxel_grid = VoxelGrid((self.num_bins, self.height, self.width), norm_type=norm_type, quantile=quantile)
        
        # Event data paths
        self.event_slicer, self.rectify_ev_map = self._initialize_event_data(seq_path)

        self.delta_t_us = 1e5  # 100ms per frame

        # Load and compute timestamps and indices
        if self.phase == 'train':
            self._load_training_data(seq_path)
        elif self.phase == 'val':
            self._load_validation_data(seq_path)
        elif self.phase == 'test':
            self._load_test_data(seq_path, timestamp_path)
        else:
            raise ValueError(f"Invalid phase: {self.phase}")
        
        self._finalizer = weakref.finalize(self, self._close_h5_file)

    def _initialize_event_data(self, seq_path: Path):
        """Load event data and rectify map from h5 files."""
        ev_dir = seq_path / 'events/left'
        h5f = h5py.File(ev_dir / 'events.h5', 'r')
        rectify_map = h5py.File(ev_dir / 'rectify_map.h5', 'r')['rectify_map'][()]

        return EventSlicer(h5f), rectify_map

    def _load_training_data(self, seq_path: Path):
        """Load timestamps and paths for training phase."""
        timestamps_images = np.loadtxt(seq_path / 'images/timestamps.txt', dtype='int64')
        image_indices = np.arange(len(timestamps_images))

        starttime_flow = timestamps_images[::2][1:-1]
        endtime_flow = starttime_flow + self.delta_t_us
        self.timestamps_flow = np.stack((starttime_flow, endtime_flow), axis=1)
        self.indices = image_indices[::2][1:-1]

        keep_i = self.timestamps_flow[:, 1] < self.event_slicer.t_final
        self.timestamps_flow, self.indices = self.timestamps_flow[keep_i], self.indices[keep_i]
        self.paths_to_forward_flow = [
            seq_path / 'flow/forward' / f'{str(i).zfill(6)}.png' for i in self.indices
        ]

    def _load_validation_data(self, seq_path: Path):
        """Load timestamps and paths for validation phase."""
        self.timestamps_flow = np.loadtxt(seq_path / 'flow/forward_timestamps.txt', 
                                          delimiter=',', skiprows=1, dtype='int64')
        keep_i = self.timestamps_flow[:, 0] > self.event_slicer.t_offset
        self.timestamps_flow = self.timestamps_flow[keep_i]

        flow_files = [f for f, keep in zip(sorted(os.listdir(seq_path / 'flow/forward')), keep_i) if keep]
        self.paths_to_forward_flow = [seq_path / 'flow/forward' / f for f in flow_files]
        self.indices = [int(f.split('.')[0]) for f in flow_files]

    def _load_test_data(self, seq_path: Path, timestamp_path: str):
        """Load timestamps for test phase."""
        if timestamp_path is None:
            raise ValueError("Test timestamp path cannot be None")

        df = pd.read_csv(timestamp_path)
        self.timestamps_flow = np.stack((df['from_timestamp_us'].to_numpy(), df['to_timestamp_us'].to_numpy()), axis=1)
        self.indices = df['file_index']
        self.paths_to_forward_flow = None

    @staticmethod
    def _close_h5_file(h5f):
        h5f.close()

    def events_to_voxel_grid(self, p, t, x, y, device='cpu'):
        """Convert events to voxel grid."""
        t = (t - t[0]).astype('float32')
        t = t / t[-1]
        x, y, pol = map(lambda v: v.astype('float32'), [x, y, p])
        event_data = {'p': torch.from_numpy(pol), 't': torch.from_numpy(t), 'x': torch.from_numpy(x), 'y': torch.from_numpy(y)}
        return self.voxel_grid.convert(event_data)

    def get_data_sample(self, index, flip=None):
        """Get a sample from the dataset."""
        t_start, t_end = self.timestamps_flow[index]
        file_index = self.indices[index]

        output = {
            'name': f'{self.name}_{str(file_index).zfill(6)}',
            'timestamp': torch.tensor([t_start, t_end]),
            'file_index': torch.tensor(file_index, dtype=torch.int64)
        }

        event_data = self.event_slicer.get_events(t_start, t_end)
        x_rect, y_rect = self.rectify_events(event_data['x'], event_data['y']).T

        # Normalize time and bin indices
        t = (event_data['t'] - event_data['t'].min()) / (event_data['t'].max() - event_data['t'].min())
        bin_indices = np.clip(np.searchsorted(self.t_bins, t) - 1, 0, None)
        events = np.column_stack((y_rect, x_rect, t, event_data['p'], bin_indices))

        mask = (0 <= events[:, 0]) & (events[:, 0] < self.height) & (0 <= events[:, 1]) & (events[:, 1] < self.width)
        events = events[mask].astype('float32')

        if self.polarity_aware_batching:
            output['pos_events'] = torch.from_numpy(events[events[:, 3] == 1])
            output['neg_events'] = torch.from_numpy(events[events[:, 3] == 0])
        else:
            output['events'] = torch.from_numpy(events)

        output['voxel'] = self.events_to_voxel_grid(event_data['p'], t, x_rect, y_rect)

        if self.paths_to_forward_flow is not None:
            flow_path = Path(self.paths_to_forward_flow[index])
            if flow_path.exists():
                flow_16bit = imageio.imread(flow_path, format='PNG-FI').astype(np.float32)
                flow = np.zeros((2, self.height, self.width), dtype=np.float32)
                flow[0] = (flow_16bit[..., 1] - 2**15) / 128.
                flow[1] = (flow_16bit[..., 0] - 2**15) / 128.
                valid = flow_16bit[..., 2].astype(bool)
                output['forward_flow'] = torch.from_numpy(flow)
                output['flow_valid'] = torch.from_numpy(valid)

        return output

    def __len__(self):
        return len(self.timestamps_flow)

    def rectify_events(self, x, y):
        """Rectify event coordinates."""
        return self.rectify_ev_map[y, x]

    def __getitem__(self, idx):
        return self.get_data_sample(idx)

    @staticmethod
    def get_disparity_map(filepath: Path):
        """Load disparity map."""
        assert filepath.is_file(), f"{filepath} is not a valid file"
        return cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH).astype('float32') / 256

    @staticmethod
    def load_flow(flowfile: Path):
        """Load optical flow from PNG file."""
        assert flowfile.exists(), f"Flow file {flowfile} not found"
        assert flowfile.suffix == '.png', "Flow file must be a PNG"
        flow_16bit = imageio.imread(str(flowfile), format='PNG-FI')
        return flow_16bit_to_float(flow_16bit)

class EventSlicer:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f

        self.events = dict()
        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = self.h5f['events/{}'.format(dset_str)]

        # This is the mapping from milliseconds to event index:
        # It is defined such that
        # (1) t[ms_to_idx[ms]] >= ms*1000
        # (2) t[ms_to_idx[ms] - 1] < ms*1000
        # ,where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
        #
        # As an example, given 't' and 'ms':
        # t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
        # ms:   0       1       2       3       4       5       6       7       8       9
        #
        # we get
        #
        # ms_to_idx:
        #       0       2       2       3       3       3       5       5       8       9
        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')

        self.t_offset = int(h5f['t_offset'][()])
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_final_time_us(self):
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        # We assume that the times are top-off-day, hence subtract offset:
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            # Cannot guarantee window size anymore
            return None

        events = dict()
        time_array_conservative = np.asarray(self.events['t'][t_start_ms_idx:t_end_ms_idx], dtype='int64')
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_start_us, t_end_us)
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        # Again add t_offset to get gps time
        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])
            assert events[dset_str].size == events['t'].size
        return events


    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us/1000)
        window_end_ms = math.ceil(ts_end_us/1000)
        return window_start_ms, window_end_ms

    @staticmethod
    @jit(nopython=True)
    def get_time_indices_offsets(
            time_array: np.ndarray,
            time_start_us: int,
            time_end_us: int) -> Tuple[int, int]:
        """Compute index offset of start and end timestamps in microseconds
        Parameters
        ----------
        time_array:     timestamps (in us) of the events
        time_start_us:  start timestamp (in us)
        time_end_us:    end timestamp (in us)
        Returns
        -------
        idx_start:  Index within this array corresponding to time_start_us
        idx_end:    Index within this array corresponding to time_end_us
        such that (in non-edge cases)
        time_array[idx_start] >= time_start_us
        time_array[idx_end] >= time_end_us
        time_array[idx_start - 1] < time_start_us
        time_array[idx_end - 1] < time_end_us
        this means that
        time_start_us <= time_array[idx_start:idx_end] < time_end_us
        """

        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:
            # This can happen in extreme corner cases. E.g.
            # time_array[0] = 1016
            # time_array[-1] = 1984
            # time_start_us = 1990
            # time_end_us = 2000

            # Return same index twice: array[x:x] is empty.
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]

def pad_events(events, max_length):
    padded = torch.zeros((max_length, 6), dtype=events.dtype)
    padded[:len(events), :5] = events
    padded[:len(events), 5] = 1
    return padded

def sequence_collate_fn(batch):
    batched_data = {
        'events': [], 
        'timestamp': [], 
        'voxel': [], 
        'file_index': []
    }

    event_types = ('events',) if 'events' in batch[0] else ('pos_events', 'neg_events')
    max_events = {et: 0 for et in event_types}
    all_have_gt_flow = all('forward_flow' in sample for sample in batch)

    for sample in batch:
        for et in event_types:
            max_events[et] = max(max_events[et], len(sample[et]))

    if len(event_types) > 1:
        batched_data['num_pos_events'] = max_events['pos_events']

    if all_have_gt_flow:
        batched_data['forward_flow'] = []
        batched_data['flow_valid'] = []

    for sample in batch:
        if len(event_types) == 1:
            padded_events = pad_events(sample['events'], max_events['events'])
        else:
            pos_padded = pad_events(sample['pos_events'], max_events['pos_events'])
            neg_padded = pad_events(sample['neg_events'], max_events['neg_events'])
            padded_events = torch.cat((pos_padded, neg_padded), dim=0)

        batched_data['events'].append(padded_events)
        batched_data['timestamp'].append(sample['timestamp'])
        batched_data['voxel'].append(sample['voxel'])
        batched_data['file_index'].append(sample['file_index'])

        if all_have_gt_flow:
            batched_data['forward_flow'].append(sample['forward_flow'])
            batched_data['flow_valid'].append(sample['flow_valid'])

    batched_data['events'] = torch.stack(batched_data['events'], dim=0)
    batched_data['timestamp'] = torch.stack(batched_data['timestamp'], dim=0)
    batched_data['voxel'] = torch.stack(batched_data['voxel'], dim=0)
    batched_data['file_index'] = torch.stack(batched_data['file_index'], dim=0)

    if all_have_gt_flow:
        batched_data['forward_flow'] = torch.stack(batched_data['forward_flow'], dim=0)
        batched_data['flow_valid'] = torch.stack(batched_data['flow_valid'], dim=0)

    return batched_data

def get_dataloaders(data_path, num_workers, batch_size, polarity_aware_batching=False,
                    norm_type=None, quantile=0, num_bins=15, collate_fn=sequence_collate_fn):
    """Create dataloaders for train and validation splits."""
    
    def create_loader(split, shuffle):
        provider = DatasetProvider(
            dataset_path=data_path, split=split, num_bins=num_bins,
            polarity_aware_batching=polarity_aware_batching, norm_type=norm_type, quantile=quantile
        )
        return DataLoader(provider.dataset, batch_size=batch_size, num_workers=num_workers,
                          collate_fn=collate_fn, shuffle=shuffle)

    # Create both train and validation loaders
    train_loader = create_loader('train', shuffle=True)
    val_loader = create_loader('val', shuffle=False)
    
    return train_loader, val_loader