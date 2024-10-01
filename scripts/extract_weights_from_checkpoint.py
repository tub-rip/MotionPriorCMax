import os
import argparse

import torch
from collections import OrderedDict

def save_model_state_dict(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    state_dict = checkpoint['state_dict']
    model_state_dict = OrderedDict()
    
    for key, value in state_dict.items():
        if key.startswith('model.'):
            model_state_dict[key.replace('model.', '')] = value
    
    base_dir = os.path.dirname(ckpt_path)
    output_path = os.path.join(base_dir, 'model.pth')
    
    torch.save(model_state_dict, output_path)
    
    print(f"Model saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert checkpoint to model state dict.")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to the checkpoint file.")
    
    args = parser.parse_args()
    
    save_model_state_dict(args.ckpt_path)
