from typing import Dict, Any, Union

import torch
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.loader.evimo2.provider import EVIMO2Provider
from src.loader.multiflow.provider import MultiFlowProvider
from src.loader.utils.keys import DataLoading
from torch.utils.data._utils.collate import default_collate

def pad_events(events, max_length):
    if isinstance(events, np.ndarray):
        events = torch.from_numpy(events)
    padded = torch.zeros((max_length, 6), dtype=events.dtype)
    padded[:len(events), :5] = events
    padded[:len(events), 5] = 1
    return padded

def event_collate_fn(batch, event_types):
    batched_data = {DataLoading.EVENTS: [], DataLoading.NUM_POS_EVENTS: -1}
    max_events = {et: 0 for et in event_types}

    for sample in batch:
        for et in event_types:
            if len(sample[et]) > max_events[et]:
                max_events[et] = len(sample[et])

    if len(event_types) > 1:
        batched_data[DataLoading.NUM_POS_EVENTS] = max_events[DataLoading.POS_EVENTS]

    for sample in batch:
        if len(event_types) == 1:
            padded = pad_events(sample[DataLoading.EVENTS],
                                max_events[DataLoading.EVENTS])
        else:
            pos_padded = pad_events(sample[DataLoading.POS_EVENTS],
                                    max_events[DataLoading.POS_EVENTS])
            neg_padded = pad_events(sample[DataLoading.NEG_EVENTS],
                                    max_events[DataLoading.NEG_EVENTS])
            padded = torch.cat((pos_padded, neg_padded), dim=0)
        batched_data[DataLoading.EVENTS].append(padded)

    batched_data[DataLoading.EVENTS] = torch.stack(batched_data[DataLoading.EVENTS], dim=0)
    return batched_data

def scales_collate_fn(batch, scale_types):
    batched_data = {}

    for stype in scale_types:
        vals = [sample[stype] for sample in batch]
        assert len(set(vals)) == 1, 'Expecting uniform scale across samples'
        batched_data[stype] = vals[0]

    return batched_data

def sequence_collate_fn(batch):
    if DataLoading.EVENTS in batch[0]:
        event_types = set((DataLoading.EVENTS,))
    elif DataLoading.POS_EVENTS in batch[0] and DataLoading.NEG_EVENTS in batch[0]:
        event_types = set((DataLoading.POS_EVENTS, DataLoading.NEG_EVENTS))
    else:
        event_types = set()

    if DataLoading.X_SCALE in batch[0]:
        scale_types = set((DataLoading.X_SCALE, DataLoading.Y_SCALE))
    else:
        scale_types = set()

    no_default_types = event_types.union(scale_types)
    others_batch, event_batch, scale_batch = [], [], []

    for sample in batch:
        event_batch.append({key: value for key, value in sample.items() if key in event_types})
        scale_batch.append({key: value for key, value in sample.items() if key in scale_types})
        others_batch.append({key: value for key, value in sample.items() if key not in no_default_types})

    others_collated = default_collate(others_batch)
    events_collated = event_collate_fn(event_batch, event_types) if len(event_types) > 0 else {}
    scales_collated = scales_collate_fn(scale_batch, scale_types)

    return {**others_collated, **events_collated, **scales_collated}

class DataModule(pl.LightningDataModule):
    MULTIFLOW2D_REGEN_STR = 'multiflow_regen'
    EVIMO2 = "evimo2"

    def __init__(self,
                 config: Union[Dict[str, Any], DictConfig],
                 batch_size_train: int = 1,
                 batch_size_val: int = 1):
        super().__init__()
        dataset_params = config['dataset']
        dataset_type = dataset_params['name']
        num_workers = config['hardware']['num_workers']

        assert dataset_type in {self.MULTIFLOW2D_REGEN_STR, self.EVIMO2}
        self.dataset_type = dataset_type

        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val

        assert self.batch_size_train >= 1
        assert self.batch_size_val >= 1

        self.num_workers = num_workers
        assert self.num_workers >= 0

        nbins_context = config['model']['num_bins']['context']

        if dataset_type == self.MULTIFLOW2D_REGEN_STR:
            dataset_provider = MultiFlowProvider(dataset_params, nbins_context)
        elif dataset_type == self.EVIMO2:
            dataset_provider = EVIMO2Provider(dataset_params, nbins_context)
        else:
            raise NotImplementedError

        self.val_dataset = dataset_provider.get_val_dataset()
        self.test_dataset = None

        self.nbins_context = dataset_provider.get_nbins_context()
        self.nbins_correlation = dataset_provider.get_nbins_correlation()

        assert self.nbins_context == nbins_context
        # Fill in nbins_correlation here because it can depend on the dataset.
        if 'correlation' in config['model']['num_bins']:
            nbins_correlation = config['model']['num_bins']['correlation']
            if nbins_correlation is None:
                config['model']['num_bins']['correlation'] = self.nbins_correlation
            else:
                assert nbins_correlation == self.nbins_correlation

    def val_dataloader(self):
        assert self.val_dataset is not None, f'No validation data found for {self.dataset_type} dataset'
        return DataLoader(
                dataset=self.val_dataset,
                batch_size=self.batch_size_val,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=sequence_collate_fn)

    def test_dataloader(self):
        assert self.test_dataset is not None, f'No test data found for {self.dataset_type} dataset'
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False)

    def get_nbins_context(self):
        return self.nbins_context

    def get_nbins_correlation(self):
        return self.nbins_correlation
