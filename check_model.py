import torch
import os

model_path = 'models/best_convnext_tiny.pth'
try:
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"Type of checkpoint: {type(checkpoint)}")
    if isinstance(checkpoint, dict):
        print(f"Keys: {checkpoint.keys()}")
        if 'model_state_dict' in checkpoint:
            print("Found model_state_dict key.")
            weights = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            print("Found state_dict key.")
            weights = checkpoint['state_dict']
        else:
            # Maybe it is the state dict
            weights = checkpoint
            print("Assuming dictionary is state_dict")
    else:
        print("Checkpoint is likely a direct model object (not recommended) or other format.")
        weights = None

    if weights:
        print(f"Number of keys in weights: {len(weights)}")
        # Check potential size savings
        param_size = 0
        for k, v in weights.items():
            param_size += v.nelement() * v.element_size()
        print(f"Estimated parameter size: {param_size / (1024*1024):.2f} MB")
        
except Exception as e:
    print(f"Error loading: {e}")
