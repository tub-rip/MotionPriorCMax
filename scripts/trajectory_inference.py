import os
import sys
from pathlib import Path
from typing import Dict, Any

import hydra
import torch
import hdf5plugin
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import CSVLogger

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.modules.data_loading import DataModule
from src.modules.raft_spline import RAFTSplineModule

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

torch.set_float32_matmul_precision('high')

@hydra.main(config_path='../config/exe/trajectory_inference', config_name='val', version_base='1.3')
def main(config: DictConfig) -> None:
    print_configuration(config)
    validate_config(config)

    data_module = setup_data_module(config)
    print_bin_info(data_module)

    logger = CSVLogger(save_dir='./validation_logs')
    module = setup_model(config)
    
    trainer = setup_trainer(config, logger)
    
    results = run_validation(trainer, module, data_module, config)
    print_metrics(results)

def print_configuration(config: DictConfig) -> None:
    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------\n')

def validate_config(config: DictConfig) -> None:
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    assert isinstance(config.hardware.gpus, int), 'No more than 1 GPU supported'
    assert config.batch_size > 0, 'Batch size must be positive'

def setup_data_module(config: DictConfig) -> DataModule:
    OmegaConf.set_struct(config['dataset'], False)
    config['dataset']['data_augm'] = False
    OmegaConf.set_struct(config['dataset'], True)
    return DataModule(config, batch_size_train=config.batch_size, batch_size_val=config.batch_size)

def print_bin_info(data_module: DataModule) -> None:
    num_bins_context = data_module.get_nbins_context()
    num_bins_corr = data_module.get_nbins_correlation()
    print(f'Number of bins:\n\tContext: {num_bins_context}\n\tCorrelation: {num_bins_corr}')

def setup_model(config: DictConfig) -> RAFTSplineModule:
    if config['model']['type'] == 'ERAFT':
        return RAFTSplineModule(config)
    else:
        ckpt_path = Path(config.checkpoint)
        return RAFTSplineModule.load_from_checkpoint(str(ckpt_path), config=config)

def setup_trainer(config: DictConfig, logger: CSVLogger) -> pl.Trainer:
    callbacks = [ModelSummary(max_depth=2)]
    return pl.Trainer(
        accelerator='gpu',
        callbacks=callbacks,
        default_root_dir=None,
        devices=[config.hardware.gpus],
        logger=logger,
        log_every_n_steps=100,
        precision=32,
    )

def run_validation(trainer: pl.Trainer, module: RAFTSplineModule, data_module: DataModule, config: DictConfig) -> Dict[str, Any]:
    ckpt_path = str(Path(config.checkpoint)) if config['model']['type'] != 'ERAFT' else None
    with torch.inference_mode():
        results = trainer.validate(model=module, datamodule=data_module, ckpt_path=ckpt_path, verbose=False)
    return results[0]

def print_metrics(results: Dict[str, Any]) -> None:
    metrics = ['val/masked_TEPE', 'val/masked_TAE', 'val/masked_T3PE']
    max_length = max(len(metric) for metric in metrics)
    print("\nValidation Results:")
    print("-" * (max_length + 20))
    for metric in metrics:
        print(f"{metric:<{max_length}} : {results[metric]:.6f}")
    print("-" * (max_length + 20))

if __name__ == '__main__':
    main()