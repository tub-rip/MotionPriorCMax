import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from .. import models
from .. import utils
from ..losses import TrajectoryLossBase
from ..utils.metrics import ErrorBase

class TrajectoryNet(pl.LightningModule):
    def __init__(self, error_calculator: ErrorBase,
                 loss_calculator: TrajectoryLossBase,
                 image_shape, lr, num_bins, num_basis, patch_size,
                 model_type, basis_type, num_train_steps, skip_frames=1,
                 monitor_metric=None):
        super(TrajectoryNet, self).__init__()
        assert loss_calculator.is_needing_offsets is not None
        self.save_hyperparameters(ignore=['error_calculator', 'loss_calculator',
                                          'num_train_steps'])
        self.error_calculator = error_calculator
        self.loss_calculator = loss_calculator
        self.num_train_steps = num_train_steps
        self.anchor_time = 0

        if self.hparams.model_type == 'default':
            self.model = models.UNet(self.hparams.num_bins,
                                     2 * self.hparams.num_basis)
        else:
            raise ValueError

        utils.initialize_weights(self.model)

        bt = self.hparams.basis_type

        if bt == 'learned' or bt == 'learned_2d':
            n_out = num_basis if bt == 'learned' else 2 * num_basis

            self.basis_network = nn.Sequential(
                nn.Linear(1, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 64),
                nn.LeakyReLU(),
                nn.Linear(64, n_out)
            )
            utils.initialize_weights(self.basis_network)
        else:
            self.basis_network = None

        object_mask = utils.get_optical_flow_tile_mask(self.hparams.image_shape,
                                                       self.hparams.patch_size)
        self.register_buffer('object_mask', object_mask, persistent=False)
        self.best_target_metric = 1e6

    def compute_basis(self, coeffs, times):
        """Calculate trajectory values for given coefficients at n_t
        given times.

        Args:
            coeffs: [b, s, 2, n, k]
            times: [1, n_t]

        Raises:
            ValueError: _description_

        Returns:
            [b, n, k, n_t, 2]
        """
        if self.hparams.basis_type == "dct":
            K = self.hparams.num_basis
            T = 1
            k_idx = torch.arange(1, K+1, device=coeffs.device)
            A = (2 * times[..., None] + 1) * k_idx[None, None, :]
            in_cos = np.pi / (2.0 * T) * A
            basis = np.sqrt(2.0 / T) * torch.cos(in_cos)

        elif self.hparams.basis_type == "learned":
            basis = self.basis_network(times[..., None]) # [b, m, k]

        elif self.hparams.basis_type == "polynomial":
            k_idx = torch.arange(1, self.hparams.num_basis+1, device=coeffs.device)
            basis = times[..., None] ** k_idx[None, None, :]
        else:
            raise ValueError

        raw_coeff_y = coeffs[..., 0, :, :] # [b, n, k]
        raw_coeff_x = coeffs[..., 1, :, :] # [b, n, k]

        y_product = basis[..., None, :, :] * raw_coeff_y[..., None, :] # [n, m, k]
        x_product = basis[..., None, :, :] * raw_coeff_x[..., None, :] # [n, m, k]

        coords = torch.stack([
            torch.sum(y_product, dim=-1),
            torch.sum(x_product, dim=-1)
        ], dim=-1)

        return torch.sum(coords, dim=1) # Sum trajectories of all scales

    def calculate_coords(self, coeffs, pixel_positions, times, add_offsets):
        offsets = pixel_positions[None, :, None, :]

        traj_at_tanchor = self.compute_basis(coeffs, torch.full((1,), self.anchor_time,
                                                                device=coeffs.device,
                                                                dtype=coeffs.dtype))
        trajectories = self.compute_basis(coeffs, times) # [b, n, len(times), 2]
        trajectories = trajectories - traj_at_tanchor
        if add_offsets:
            trajectories = trajectories + offsets
        return trajectories.permute(0, 2, 1, 3).contiguous()

    def calculate_trajectories_at_t(self, coeff_grid, times, mask, add_offsets):
        if len(coeff_grid.shape) == 4:
            coeff_grid = coeff_grid[:, None]

        coeffs, pixel_positions, _ = utils.coeffs_grid_to_list(coeff_grid, mask,
                                                               num_coeffs=self.hparams.num_basis)
        return self.calculate_coords(coeffs, pixel_positions, times, add_offsets)

    def calculate_flow(self, coeff_grid, mask):
        coeffs, pixel_positions, _ = utils.coeffs_grid_to_list(coeff_grid, mask,
                                                            num_coeffs=self.hparams.num_basis)
        ts = torch.tensor(self.anchor_time, device=self.device,
                        dtype=coeffs.dtype)[None]

        if self.hparams.skip_frames == 1:
            te = torch.tensor(1, device=self.device,
                            dtype=coeffs.dtype)[None]
        else:
            te = torch.tensor(1 / self.hparams.skip_frames, device=self.device,
                            dtype=coeffs.dtype)[None]

        traj_at_ts = self.compute_basis(coeffs, ts)
        traj_at_te = self.compute_basis(coeffs, te)
        traj_flow = traj_at_te - traj_at_ts
        flow_pred, _ = utils.dense_flow_from_traj(traj_flow[..., 0, :], pixel_positions,
                                                    self.hparams.patch_size,
                                                    self.hparams.image_shape)
        return flow_pred

    def step(self, batch, mask=None):
        if mask is None:
            mask = self.object_mask

        voxel = batch['voxel']
        coeff_grid = self(voxel)

        if len(coeff_grid.shape) == 4:
            coeff_grid = coeff_grid[:, None]

        reconstruction_times = self.loss_calculator.get_reconstruction_times(coeff_grid.device)
        trajectories = self.calculate_trajectories_at_t(coeff_grid, reconstruction_times,
                                                        mask, self.loss_calculator.is_needing_offsets)

        loss, log_data, misc_data = self.loss_calculator.calc(trajectories,
                                                                reconstruction_times,
                                                                batch)
        misc_data['coeff_grid'] = coeff_grid

        return loss, log_data, misc_data

    def training_step(self, batch, batch_idx):
        loss, log_data, _ = self.step(batch)
        self.log('train_losses/total', loss, on_epoch=True, sync_dist=True,
                 batch_size=batch['voxel'].shape[0])
        log_data = {f'train_losses/{key}': value for key, value in log_data.items()}
        self.log_dict(log_data, on_epoch=True, sync_dist=True,
                      batch_size=batch['voxel'].shape[0])
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, log_data, misc_data = self.step(batch)
        coeff_grid = misc_data['coeff_grid']

        self.log('val_losses/total', loss, on_epoch=True, sync_dist=True,
                 batch_size=batch['voxel'].shape[0])
        log_name = 'val_losses'
        log_data = {f'{log_name}/{key}': value for key, value in log_data.items()}

        # Calculate errors
        with torch.no_grad():
            if 'gt_flows' in batch:
                n_gt_steps = batch['gt_flows'].shape[1]
                predictions = {'flows': self.calculate_traj_flows(coeff_grid, n_gt_steps)}
            else:
                predictions = {'flow': self.calculate_flow(coeff_grid, self.object_mask)}

            errors = self.error_calculator.run(predictions, batch)

            for key, value in errors.items():
                log_data[f'{log_name}/{key}'] = value

            self.log_dict(log_data, on_epoch=True, sync_dist=True,
                          batch_size=coeff_grid.shape[0],
                          add_dataloader_idx=False)
        return loss
    
    def predict_step(self, batch, batch_idx):
        loss, log_data, misc_data = self.step(batch)
        coeff_grid = misc_data['coeff_grid']
        flow_pred = self.calculate_flow(coeff_grid, self.object_mask)
        return flow_pred, misc_data

    def forward(self, event_voxel):
        return self.model(event_voxel)

    def on_validation_epoch_end(self):
        self.best_target_metric = self.error_calculator.log_best(self.best_target_metric,
                                                                 self.trainer, self.logger,
                                                                 target_metric=self.hparams.monitor_metric)

    def configure_optimizers(self):
        params = list()
        params += list(self.model.parameters())

        if self.basis_network is not None:
            params += list(self.basis_network.parameters())
        return torch.optim.AdamW(params, lr=self.hparams.lr)