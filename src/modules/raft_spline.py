from typing import Dict, Any, Union

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from torchmetrics import MetricCollection

from .utils import InputPadder, EPE, EPE_MULTI, AE, AE_MULTI, NPE, FLOW_METRICS_MULTI, \
    to_cpu, predictions_from_lin_assumption, reduce_ev_repr
from src.models.raft_spline.raft import RAFTSpline
from src.models.raft_spline.curves import CurveBase
from src.loader.utils.keys import DataLoading, DataSetType

class RAFTSplineModule(pl.LightningModule):
    def __init__(self, config: Union[Dict[str, Any], DictConfig]):
        super(RAFTSplineModule, self).__init__()

        self.num_iter_train = config['model']['num_iter']['train']
        self.num_iter_test = config['model']['num_iter']['test']

        min_size = 32 if config['model']['type'] == "UNET_RNB" else 8
        self._input_padder = InputPadder(min_size=min_size, no_top_padding=False)

        self.curve_type = config['model']['curve_type']

        if self.curve_type == 'LEARNED':
            num_basis = config['model']['bezier_degree']

            self.basis_network = torch.nn.Sequential(torch.nn.Linear(1, 64),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(64, 64),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(64, num_basis)
            )
            print(id(self.basis_network))
        else:
            self.basis_network = None

        if config['model']['type'] == "ERAFTPP":
            self.net = RAFTSpline(config['model'], self.basis_network)
        else:
            raise ValueError

        self.use_images = config['model']['use_boundary_images']
        self.use_events = config['model']['use_events']

        self.train_params = config['training']
        #self.train_with_multi_loss = self.train_params['multi_loss']

        single_metrics = MetricCollection({
            'epe': EPE(),
            'ae': AE(degrees=True),
            '1pe': NPE(1),
            '2pe': NPE(2),
            '3pe': NPE(3),
        })

        multi_metrics = MetricCollection({
            'epe_multi': EPE_MULTI(),
            'ae_multi': AE_MULTI(degrees=True),
            'flow_metrics': FLOW_METRICS_MULTI()
        })

        self.train_single_metrics = single_metrics.clone(prefix='train/')
        self.train_multi_metrics = multi_metrics.clone(prefix='train/')

        self.val_single_metrics = single_metrics.clone(prefix='val/')
        self.val_multi_metrics = multi_metrics.clone(prefix='val/')

        self.val_single_metrics_masked = single_metrics.clone(prefix='val/masked_')
        self.val_multi_metrics_masked = multi_metrics.clone(prefix='val/masked_')

        self.val_multi_metrics_ev_masked = multi_metrics.clone(prefix='val/ev_masked_')

        # To evaluate a pseudo-linear prediction with the multi metrics:
        self.train_epe_multi_lin = EPE_MULTI()
        self.train_ae_multi_lin = AE_MULTI(degrees=True)
        self.val_epe_multi_lin = EPE_MULTI()
        self.val_ae_multi_lin = AE_MULTI(degrees=True)

        self.last_batch_cache = None
        self.best_epe = 1e6

    def forward(self, voxel_grid, images, iters, test_mode: bool):
        return self.net(voxel_grid=voxel_grid, images=images, iters=iters, test_mode=test_mode)

    @to_cpu
    def validation_step(self, batch, batch_idx):
        # forward_flow: (N, 2, 480, 640), float32
        forward_flow_gt = batch[DataLoading.FLOW]
        # forward_flow_valid: (N, 480, 640), bool
        forward_flow_gt_valid = batch[DataLoading.FLOW_VALID] if DataLoading.FLOW_VALID in batch else None
        # event_representation: (N, 2*num_bins-1, 480, 640), float32
        ev_repr = batch[DataLoading.EV_REPR]
        if self.use_images:
            images = batch[DataLoading.IMG]
        else:
            images = None

        dataset_type = batch[DataLoading.DATASET_TYPE][0]

        output = dict()

        if dataset_type == DataSetType.MULTIFLOW2D or dataset_type == DataSetType.EVIMO2:
            nbins_context = batch[DataLoading.BIN_META]['nbins_context'][0]
            nbins_corr = batch[DataLoading.BIN_META]['nbins_correlation'][0]
            nbins_total = ev_repr.shape[1]
            assert nbins_total == batch[DataLoading.BIN_META]['nbins_total'][0] == nbins_context + nbins_corr - 1
            ev_repr_previous = ev_repr[:, 0:nbins_corr, ...]
            ev_repr_current = ev_repr[:, -nbins_corr:, ...]

            forward_flow_ts = batch[DataLoading.FLOW_TIMESTAMPS]

            requires_padding = self._input_padder.requires_padding(ev_repr)
            if requires_padding:
                ev_repr = self._input_padder.pad(ev_repr)
                ev_repr_previous = self._input_padder.pad(ev_repr_previous)
                ev_repr_current = self._input_padder.pad(ev_repr_current)
                if images is not None:
                    images = [self._input_padder.pad(x) for x in images]

            bezier_low, bezier_up = self(
                voxel_grid=ev_repr if self.use_events else None,
                images=images,
                iters=self.num_iter_test,
                test_mode=True)
            bezier_up: CurveBase

            forward_flow_preds = list()
            timestamp_eval_lst = list()

            if dataset_type == DataSetType.EVIMO2:
                timestamp_eval_lst = forward_flow_ts[0].cpu().numpy()
                for ts in timestamp_eval_lst:
                    forward_flow_pred = bezier_up.get_flow_from_reference(ts)
                    if requires_padding:
                        forward_flow_pred = self._input_padder.unpad(forward_flow_pred)
                    forward_flow_preds.append(forward_flow_pred)
                # change to list
                forward_flow_gt = [forward_flow_gt[:, ss, ...] for ss \
                                        in range(forward_flow_gt.shape[1])]
            else:
                for timestamp_batch in forward_flow_ts:
                    # check that timestamps are essentially the same
                    ts_mean_diff = (timestamp_batch[1:] - timestamp_batch[:-1]).abs().mean().item()
                    assert 0 <= ts_mean_diff < 0.001, ts_mean_diff
                    timestamp = timestamp_batch[0].item()

                    timestamp_eval_lst.append(timestamp)

                    forward_flow_pred = bezier_up.get_flow_from_reference(timestamp)
                    if requires_padding:
                        forward_flow_pred = self._input_padder.unpad(forward_flow_pred)
                    forward_flow_preds.append(forward_flow_pred)

            if requires_padding and images is not None:
                images = [self._input_padder.unpad(x) for x in images]

            val_single_metrics = self.val_single_metrics(forward_flow_preds[-1], forward_flow_gt[-1])
            self.log_dict(val_single_metrics, logger=True, on_epoch=True, sync_dist=True)
            val_multi_metrics = self.val_multi_metrics(forward_flow_preds, forward_flow_gt)
            self.log_dict(val_multi_metrics, logger=True, on_epoch=True, sync_dist=True)

            event_mask = torch.abs(ev_repr).any(dim=1) > 0

            val_single_metrics_masked = self.val_single_metrics_masked(forward_flow_preds[-1], forward_flow_gt[-1], event_mask)
            self.log_dict(val_single_metrics_masked, logger=True, on_epoch=True, sync_dist=True)

            flow_valid_mask = batch[DataLoading.FLOW_VALID] if DataLoading.FLOW_VALID in batch else None

            if flow_valid_mask is not None:
                assert flow_valid_mask.shape[1] == len(forward_flow_preds)
                eval_masks_ev = [(event_mask & flow_valid_mask[:, sid, ...]) \
                                                for sid, _ in enumerate(forward_flow_preds)]
                eval_masks = [(flow_valid_mask[:, sid, ...]) \
                                                for sid, _ in enumerate(forward_flow_preds)]
                
            else:
                eval_masks_ev = [event_mask for _ in forward_flow_preds]
                eval_masks = None

            val_multi_metrics_ev_masked = self.val_multi_metrics_ev_masked(forward_flow_preds, forward_flow_gt, eval_masks_ev)
            self.log_dict(val_multi_metrics_ev_masked, logger=True, on_epoch=True, sync_dist=True)

            val_multi_metrics_masked = self.val_multi_metrics_masked(forward_flow_preds, forward_flow_gt, eval_masks)
            self.log_dict(val_multi_metrics_masked, logger=True, on_epoch=True, sync_dist=True)

            # evaluate against linearity assumption (both space and time)
            forward_flow_preds_lin = predictions_from_lin_assumption(forward_flow_preds[-1], timestamp_eval_lst)
            epe_multi_lin = self.val_epe_multi_lin(forward_flow_preds_lin, forward_flow_gt)
            ae_multi_lin = self.val_ae_multi_lin(forward_flow_preds_lin, forward_flow_gt)

            self.log('val/epe_multi_lin', epe_multi_lin, logger=True, sync_dist=True)
            self.log('val/ae_multi_lin', ae_multi_lin, logger=True, sync_dist=True)

            output.update({
                'pred': forward_flow_preds[-1], # list of pred  [(N, 2, H, W), ...] (M times)
                'gt': forward_flow_gt, # list of gt -> [(N, 2, H, W), ...] (M times)
                'all_pred': forward_flow_preds
                #'gt': forward_flow_gt[-1],
            })
        else:
            raise NotImplementedError

        output.update({
            'bezier_prediction': bezier_up,
        })
        if self.use_events:
            output.update({
                'ev_repr_reduced': reduce_ev_repr(ev_repr_current),
                'ev_repr_reduced_m1': reduce_ev_repr(ev_repr_previous),
            })
        if images is not None:
            output.update({'images': images})
        return output

    def configure_optimizers(self):
        lr = self.train_params['learning_rate']
        weight_decay = self.train_params['weight_decay']

        params = list()
        params += list(self.net.parameters())

        if self.basis_network is not None:
            params += list(self.basis_network.parameters())

        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        scheduler_params = self.train_params['lr_scheduler']
        if not scheduler_params['use']:
            return optimizer

        total_steps = scheduler_params['total_steps']
        assert total_steps is not None
        assert total_steps > 0
        pct_start = scheduler_params['pct_start']
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            total_steps=total_steps+100,
            pct_start=pct_start,
            cycle_momentum=False,
            anneal_strategy='linear')
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "step",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": 'learning_rate',
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}