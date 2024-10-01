import torch
import numpy as np
from pytorch_lightning import loggers, Callback

from .visualization import visualize_optical_flow, normalize_images

N_SAMPLES = 5

class ImageLoggingBase(Callback):
    def __init__(self, train_loader, val_loader, patch_size,
                 image_shape) -> None:
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_night_loader = None
        self.patch_size = patch_size
        self.image_shape = image_shape

    def log_images(self, pl_module, phase):
        if phase == 'train':
            split = 'train/'
            dataloader = self.train_loader
        elif phase == 'val_night':
            split = 'val_night/'
            dataloader = self.val_night_loader
        elif phase == 'val':
            split = 'val/'
            dataloader = self.val_loader 
        else:
            split = f'{phase}/'
            dataloader = self.val_loader
        dataset = dataloader.dataset 

        dataset_size = len(dataset)
        indices = np.linspace(0, dataset_size - 1, N_SAMPLES, dtype=int)
        step = pl_module.current_epoch

        with torch.no_grad():
            for i, data_idx in enumerate(indices):
                sample = dataset[data_idx]
                batch = dataloader.collate_fn([sample])

                for k, v in batch.items():
                    if type(v) == torch.Tensor:
                        batch[k] = v.to(pl_module.device)

                self.log_all(batch, pl_module, step, split, phase, i)
                self.log_dataset_specific(batch, pl_module, step, split, phase, i)

    def log_dataset_specific(self, batch, pl_module, step, split, phase, i):
        raise NotImplementedError

    def log_all(self, batch, pl_module, step, split, phase, index):
        
        # IWE
        _, _, metadata = pl_module.step(batch)

        if 'iwes' in metadata:
            iwes = metadata['iwes']
            name = f'{str(index).zfill(2)}_{split}' + '2_iwe'
            iwe = iwes.cpu().detach().numpy()[0, 0][None]
            log_iwe(iwe, pl_module.logger, name, step)

        # 2-View Flow
        flow = pl_module.predict_step(batch, 0)
        rgb_flow, color_wheel = visualize_optical_flow(flow[0])

        log_image(rgb_flow, pl_module.logger,
                    name=f'{str(index).zfill(2)}_{split}' + '4_flow',
                    step=step)
    
        if phase == 'fit_start':
            log_image(color_wheel, pl_module.logger, 'color_wheel', step=step)

        # Event Image (unwarped)
        if hasattr(pl_module.loss_calculator, 'imager'):
            events = batch['events']
            event_image = pl_module.loss_calculator.imager.create_iwe(events,
                                                method='bilinear_vote', sigma=1)
            event_image = event_image.cpu().detach().numpy()

            log_iwe(event_image, pl_module.logger,
                    name=f'{str(index).zfill(2)}_{split}' + '0_unwarped',
                    step=step)

class DsecImageLoggingCallback(ImageLoggingBase):
    def __init__(self, train_loader, val_loader,
                 val_night_loader, patch_size, image_shape):
        super().__init__(train_loader, val_loader, patch_size, image_shape)
        self.val_night_loader = val_night_loader

    def log_dataset_specific(self, batch, pl_module, step, split, phase, index):
        # GT Flow
        if 'forward_flow' in batch:
            gt_flow_rgb, _ = visualize_optical_flow(batch['forward_flow'][0])
            log_image(gt_flow_rgb, pl_module.logger,
                        name=f'{str(index).zfill(2)}_{split}' + '3_gt_flow',
                        step=step)

        # GT IWE
        if pl_module.hparams.basis_type == "polynomial" \
                and 'forward_flow' in batch and pl_module.hparams.num_basis == 1:
            object_mask = batch['flow_valid'][0]
            object_mask = object_mask.to(pl_module.device)

            flow = batch['forward_flow']
            coeff_grid = flow

            if len(coeff_grid.shape) == 4:
                coeff_grid = coeff_grid[:, None]

            times = pl_module.loss_calculator.get_reconstruction_times(coeff_grid.device)
            times[0] = 0
            trajectories = pl_module.calculate_trajectories_at_t(coeff_grid, times, object_mask, True)
            _, _, meta_data = pl_module.loss_calculator.calc(trajectories, times, batch)

            iwes = meta_data['iwes']
            iwe = iwes.cpu().detach().numpy()[0, 0][None]
            name = f'{str(index).zfill(2)}_{split}' + '1_gt_iwe'
            log_iwe(iwe, pl_module.logger, name, step)

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_images(pl_module, 'train')

    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_images(pl_module, 'val')
        if self.val_night_loader is not None:
            self.log_images(pl_module, 'val_night')

    def on_fit_start(self, trainer, pl_module):
        self.log_images(pl_module, 'fit_start')

def log_image(image, logger, name, step):
    if type(logger) == loggers.TensorBoardLogger:
        logger.experiment.add_image(name, image, step)
    elif type(logger) == loggers.WandbLogger:
        image = image.transpose(1, 2, 0)
        logger.log_image(key=name, images=[image], step=step)

def log_iwe(iwe, logger, name, step):
    if len(iwe.shape) == 4:
        log_iwe(iwe[:, 0], logger, f'{name}_pos', step)
        log_iwe(iwe[:, 1], logger, f'{name}_neg', step)
    else:
        image = normalize_images(iwe, invert=True)
        log_image(image, logger, name, step)