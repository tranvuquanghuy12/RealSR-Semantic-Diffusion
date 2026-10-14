import torch
import torch.nn.functional as F
import torch.nn as nn
from ldm_v2 import FrequencyLoss

def test_bandpass_filter():
    print("Testing Band-Pass Filter in FrequencyLoss...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize loss
    loss_fn = FrequencyLoss(loss_weight=1.0, min_radius=0.05, max_radius=0.95).to(device)
    
    # Create mock inputs (B, C, H, W)
    B, C, H, W = 2, 3, 64, 64
    pred = torch.randn(B, C, H, W, requires_grad=True).to(device)
    target = torch.randn(B, C, H, W).to(device)
    
    # Forward pass
    loss = loss_fn(pred, target)
    print(f"Loss value: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    
    print(f"Gradient computed: {pred.grad is not None}")
    if pred.grad is not None:
        print(f"Gradient shape: {pred.grad.shape}")
        print(f"Gradient sum (absolute): {torch.sum(torch.abs(pred.grad)):.6f}")
        
    print("Test passed successfully!")

if __name__ == '__main__':
    test_bandpass_filter()
