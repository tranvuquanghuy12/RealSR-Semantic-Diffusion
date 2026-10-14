import torch
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('base')

opt = {
    'model': 'latent_diffusion',
    'dist': False,
    'gpu_ids': [0],
    'is_train': True,
    'num_classes': 10,
    'train': {
        'weight_decay_G': 0,
        'lr_G': 1e-4,
        'beta1_G': 0.9,
        'beta2_G': 0.999,
        'lr_scheme': 'MultiStepLR',
        'lr_steps': [10000, 20000],
        'lr_gamma': 0.5
    },
    'path': {
        'models': 'mock_models',
        'training_state': 'mock_state'
    }
}

from models import create_model

def run_mock():
    print("Initializing Multi-Task Latent Diffusion Model...")
    model = create_model(opt)
    
    batch_size = 2
    # Mock Dual Inputs (LR Reference and LR Source)
    dummy_L_R = torch.randn(batch_size, 3, 32, 32)
    dummy_L_S = torch.randn(batch_size, 3, 32, 32)
    
    # Mock High-Res Reference (for Semantic Guidance during training)
    dummy_H_R = torch.randn(batch_size, 3, 128, 128)
    
    # Mock Segmentation Mask Ground Truth (e.g., 10 classes)
    # Mask is integer labels, shape (B, H, W) same as HR image
    dummy_M_S = torch.randint(0, 10, (batch_size, 128, 128))
    
    data = {
        'L_R': dummy_L_R,
        'L_S': dummy_L_S,
        'H_R': dummy_H_R,
        'M_S': dummy_M_S
    }
    
    print("Feeding dual inputs and mask into model...")
    model.feed_data(data)
    
    print("Running 1 step of optimization (forward + backward)...")
    model.optimize_parameters(1)
    
    losses = model.get_current_losses()
    print("========================================")
    print("🚀 Mock training step finished SUCCESSFULY!")
    print(f"📉 Loss Diffusion (l_diff): {losses.get('l_diff')}")
    print(f"📉 Loss Segmentation (l_seg): {losses.get('l_seg')}")
    print(f"📉 Total Loss (l_total): {losses.get('l_total')}")
    print("========================================")

if __name__ == "__main__":
    run_mock()
