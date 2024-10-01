import argparse
import os
import sys
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.loader import get_dsec_dataloaders
from src.modules.trajectory_net import TrajectoryNet
from src.utils.metrics import ErrorCalculatorFactory
from src.losses import LossFactory
import src.utils as utils

torch.set_float32_matmul_precision('high')

def get_dataloaders(dataset_name, data_config):
    if dataset_name == 'DSEC':
        return get_dsec_dataloaders(**data_config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_image_callback(dataset_name, train_loader, val_loader, patch_size, image_shape):
    if dataset_name == 'DSEC':
        return utils.DsecImageLoggingCallback(
            train_loader, val_loader, val_loader, patch_size, image_shape
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def propagate_config(config):
    image_shape = (config['common']['height'], config['common']['width'])
    config['model']['image_shape'] = image_shape
    config['loss']['image_shape'] = image_shape

    num_bins = config['common']['num_bins']
    config['model']['num_bins'] = num_bins
    config['data']['num_bins'] = num_bins
    if config['loss']['loss_name'] == 'FOCUS':
        config['loss']['num_bins'] = num_bins

    pab = config['common']['polarity_aware_batching']
    config['data']['polarity_aware_batching'] = pab
    config['loss']['polarity_aware_batching'] = pab

    patch_size = config['common']['patch_size']
    config['model']['patch_size'] = patch_size

    return config

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='config/exe/flow_training/debug.yaml', help="Path to the config file.")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help="GPUs to use.")
    parser.add_argument('--ckp_path', type=str, default=None, help="Checkpoint path to resume training.")

    args = parser.parse_args()

    # Load and propagate the configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = propagate_config(config)

    project_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    save_dir = os.path.join(project_root, 'outputs/flow_training')
    os.makedirs(save_dir, exist_ok=True)

    # Setup data loaders, loss, and metrics
    data_config = config['data']
    dataset_name = data_config.pop('dataset')

    loss_config = config['loss']
    loss_name = loss_config.pop('loss_name')

    train_loader, val_loader = get_dataloaders(dataset_name, data_config)
    monitor_metric = 'val_losses/EPE'

    try:
        logger = loggers.WandbLogger(project='MotionPriorCM', save_dir=save_dir)
    except ImportError:
        print("Wandb is not available, using TensorBoardLogger instead.")
        logger = loggers.TensorBoardLogger(save_dir=save_dir, name="MotionPriorCM")

    error_calculator = ErrorCalculatorFactory.get_error_calculator(dataset_name)
    loss_calculator = LossFactory.get_loss_calculator(loss_name, loss_config)

    # Model setup
    if args.ckp_path is None:
        model = TrajectoryNet(
            error_calculator, loss_calculator,
            num_train_steps=len(train_loader), monitor_metric=monitor_metric,
            **config['model']
        )
    else:
        model = TrajectoryNet.load_from_checkpoint(
            args.ckp_path, error_calculator=error_calculator, loss_calculator=loss_calculator,
            num_train_steps=len(train_loader), monitor_metric=monitor_metric,
            **config['model']
        )

    # Multi-GPU support
    checkpoint_dir = logger.experiment.dir if type(logger) == loggers.WandbLogger else logger.log_dir
    checkpoint_dir = os.path.join(checkpoint_dir, 'ckpt') if type(checkpoint_dir) == str else None
    filename = f"model-{{epoch:02d}}-{monitor_metric}={{" + f"{monitor_metric}:.2f}}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir, monitor=monitor_metric, mode='min', save_top_k=5,
        filename=filename, auto_insert_metric_name=False
    )

    # Setup callbacks
    callbacks = [checkpoint_callback]

    if type(logger) == loggers.WandbLogger:
        callbacks.append(get_image_callback(dataset_name, train_loader, val_loader,
                                            model.hparams.patch_size, model.hparams.image_shape))

    # Trainer configuration
    num_sanity_val_steps = 0 if args.ckp_path is None else None

    trainer = pl.Trainer(
        **config['trainer'], logger=logger, log_every_n_steps=200, devices=args.gpus,
        accelerator='gpu', callbacks=callbacks, num_sanity_val_steps=num_sanity_val_steps
    )

    trainer.fit(model, train_loader, val_loader)
    print('Training complete!')

if __name__ == '__main__':
    main()
