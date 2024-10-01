import argparse
from datetime import datetime
from pathlib import Path
import os
import sys

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.loader import Sequence, sequence_collate_fn
from src import utils
from src.models import UNet

torch.set_float32_matmul_precision('high')

GPU = 0

def propagate_config(config):
    image_shape = (config['common']['height'], config['common']['width'])
    config['model'].update({
        'image_shape': image_shape,
        'num_bins': config['common']['num_bins']
    })
    config['data']['num_bins'] = config['common']['num_bins']
    return config

def scale_optical_flow(flow, max_flow_magnitude):
    u, v = flow[0, :, :], flow[1, :, :]
    magnitude = torch.sqrt(u**2 + v**2)
    exceed_indices = magnitude > max_flow_magnitude

    u[exceed_indices] = (u[exceed_indices] / magnitude[exceed_indices]) * max_flow_magnitude
    v[exceed_indices] = (v[exceed_indices] / magnitude[exceed_indices]) * max_flow_magnitude

    return torch.stack([u, v], dim=0)

def save_flow(file_path: Path, flow: np.ndarray):
    """Save the optical flow as a 16-bit PNG."""
    height, width = flow.shape[1], flow.shape[2]
    flow_16bit = np.zeros((height, width, 3), dtype=np.uint16)
    flow_16bit[..., 1] = (flow[0] * 128 + 2**15).astype(np.uint16)  # y-component
    flow_16bit[..., 0] = (flow[1] * 128 + 2**15).astype(np.uint16)  # x-component
    imageio.imwrite(str(file_path), flow_16bit, format='PNG-FI')

def load_model(config, device):
    """Load the pre-trained UNet model."""
    model = UNet(config['model']['num_bins'], 2 * config['model']['num_basis'])
    model.load_state_dict(torch.load(config['model']['ckpt_path'], map_location=device, weights_only=True))
    return model.to(device).eval()

def process_sequence(seq, config, device, model, mask, run_out_dir, timestamp_dir, dataset_dir):
    """Process an individual sequence."""
    seq_dir = Path(dataset_dir) / seq
    timestamp_path = Path(timestamp_dir) / f'{seq}.csv'
    flow_out_dir = Path(run_out_dir) / 'flow' / seq
    flow_out_dir.mkdir(parents=True, exist_ok=True)

    dataset = Sequence(seq_dir, 'test', 15, timestamp_path=timestamp_path,
                       polarity_aware_batching=config['data']['polarity_aware_batching'])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=sequence_collate_fn)

    for sample in tqdm(loader, desc=f"Processing {seq}", leave=False):
        process_sample(sample, device, model, mask, flow_out_dir, config)

def process_sample(sample, device, model, mask, flow_out_dir, config):
    """Process a single sample (100ms)."""
    sample['events'] = sample['events'].to(device)
    sample['voxel'] = sample['voxel'].to(device)

    coeff_grid = model(sample['voxel'])

    # If the coefficients grid has an extra dimension, ensure consistent shape
    if len(coeff_grid.shape) == 4:
        coeff_grid = coeff_grid[:, None]

    coeffs, pixel_positions, _ = utils.coeffs_grid_to_list(coeff_grid, mask, num_coeffs=1)

    # Compute trajectory and optical flow
    traj_at_t0 = utils.compute_basis(coeffs, torch.zeros(1, device=device), config['model']['num_basis'],
                                     config['model']['basis_type'])
    traj_at_t1 = utils.compute_basis(coeffs, torch.ones(1, device=device), config['model']['num_basis'],
                                     config['model']['basis_type'])
    traj_flow = traj_at_t1 - traj_at_t0
    flow, _ = utils.dense_flow_from_traj(traj_flow[..., 0, :], pixel_positions,
                                         config['model']['patch_size'], config['model']['image_shape'])
    flow = flow.squeeze().cpu()
    flow = scale_optical_flow(flow, 60).numpy()

    # Save flow
    file_index = sample['file_index'].item()
    file_name = f'{str(file_index).zfill(6)}.png'
    save_flow(flow_out_dir / file_name, flow)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = propagate_config(config)

    device = torch.device(f'cuda:{GPU}')
    model = load_model(config, device)

    mask = utils.get_optical_flow_tile_mask(config['model']['image_shape'], config['model']['patch_size'])

    timestamp_dir = Path(__file__).resolve().parent / '../config/misc/dsec_test_timestamps'
    test_seqs = [p.stem for p in timestamp_dir.glob('*.csv')]
    dataset_dir = Path(config['data']['root_dir']) / 'test'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_out_dir = Path(config['output_dir']) / timestamp

    with torch.no_grad():
        for seq in tqdm(test_seqs, desc="Processing sequences"):
            process_sequence(seq, config, device, model, mask, run_out_dir, timestamp_dir, dataset_dir)

    print('Done.')

if __name__ == '__main__':
    main()
