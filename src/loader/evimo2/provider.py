import copy
import os
from typing import Dict, Any
from pathlib import Path

import torch

from .datasubset import EVIMO2_Datasubset
from ..utils.provider import DatasetProviderBase
from ..utils.generic import check_key_and_bool

class EVIMO2Provider(DatasetProviderBase):
    def __init__(self,
                 dataset_params: Dict[str, Any],
                 nbins_context: int):
        dataset_path = Path(dataset_params['path'])

        return_ev_key = 'return_ev'
        if return_ev_key in dataset_params:
            return_ev = dataset_params[return_ev_key]
        base_args = {
            'num_bins_context': nbins_context,
            'load_voxel_grid': dataset_params['load_voxel_grid'],
            'normalize_voxel_grid': dataset_params['normalize_voxel_grid'],
            'extended_voxel_grid': dataset_params['extended_voxel_grid'],
            'flow_every_n_ms': dataset_params['flow_every_n_ms'],
            'downsample': dataset_params['downsample'],
            'photo_augm': dataset_params['photo_augm'],
            return_ev_key: return_ev,
        }

        splits = ["imo"]
        train_subsets = []
        for split in splits:
            train_path = dataset_path / split / 'train'
            val_path = dataset_path / split / 'eval'

            assert dataset_path.is_dir(), str(dataset_path)
            assert train_path.is_dir(), str(train_path)
            assert val_path.is_dir(), str(val_path)

            val_subsets = []
            val_test_args = copy.deepcopy(base_args)
            val_test_args.update({'data_augm': False})
            val_test_args.update({'provide_raw_events': check_key_and_bool(dataset_params, 'provide_raw_events_val')})
            for dir in os.listdir(val_path):
                seq_path = val_path / dir
                one_subset = EVIMO2_Datasubset(seq_path, **val_test_args, flow_time=dataset_params['flow_time'])
                val_subsets.append(one_subset)

        self.val_dataset = torch.utils.data.ConcatDataset(val_subsets)
        self.nbins_context = val_subsets[-1].get_num_bins_context()
        self.nbins_correlation = val_subsets[-1].get_num_bins_correlation()

    def get_train_dataset(self):
        raise NotImplementedError

    def get_val_dataset(self):
        return self.val_dataset

    def get_test_dataset(self):
        raise NotImplementedError

    def get_nbins_context(self):
        return self.nbins_context

    def get_nbins_correlation(self):
        return self.nbins_correlation
