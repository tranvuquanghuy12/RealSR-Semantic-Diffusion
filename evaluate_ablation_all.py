import os
import glob
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from models import create_model
from utils.util import calculate_psnr
from test_train import SimpleDatasetA

def evaluate_model_on_1_image(opt_variant, checkpoint_dir, image_path):
    opt = {
        'model': 'latent_diffusion',
        'dist': False,
        'gpu_ids': [0],
        'is_train': False,
        'num_classes': 10,
        'path': {
            'models': checkpoint_dir
        }
    }
    opt.update(opt_variant)
    model = create_model(opt)
    
    # Load epoch 2 weights
    try:
        model.load_network(f"{checkpoint_dir}/epoch_2_UNet.pth", model.unet)
        model.load_network(f"{checkpoint_dir}/epoch_2_SharedEncoder.pth", model.shared_encoder)
        if opt_variant.get('use_semantic_guidance', True):
            model.load_network(f"{checkpoint_dir}/epoch_2_SemanticEncoder.pth", model.semantic_encoder)
        if opt_variant.get('use_multitask', True):
            model.load_network(f"{checkpoint_dir}/epoch_2_SemanticDecoder.pth", model.semantic_decoder)
            model.load_network(f"{checkpoint_dir}/epoch_2_AdaptingLayer.pth", model.adapting_layer)
    except Exception as e:
        pass
        
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = SimpleDatasetA([image_path], transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for i, data in enumerate(dataloader):
        model.feed_data(data)
        model.test()
        
        visuals = model.get_current_visuals()
        fake_H_R = model.fake_H_R.detach()[0].float().cpu()
        real_H_R = visuals['H_R']
        
        # Denormalize
        fake_H_R = torch.clamp((fake_H_R * 0.5 + 0.5) * 255.0, 0, 255).numpy().astype(np.uint8)
        real_H_R = torch.clamp((real_H_R * 0.5 + 0.5) * 255.0, 0, 255).numpy().astype(np.uint8)
        
        fake_H_R = np.transpose(fake_H_R, (1, 2, 0))
        real_H_R = np.transpose(real_H_R, (1, 2, 0))
        
        # Resize to match GT just in case
        fake_img = Image.fromarray(fake_H_R)
        gt_img = Image.fromarray(real_H_R)
        fake_img = fake_img.resize(gt_img.size, Image.BICUBIC)
        fake_H_R = np.array(fake_img)
        
        psnr = calculate_psnr(fake_H_R, real_H_R)
        return psnr

if __name__ == "__main__":
    test_image = '/home/ptitdemo/TVQH_TLU/Real-SR-master/Data/CPDM_Dataset/test/A/image_109_tooth_3_normal.jpg'
    
    print("=========================================")
    print("      ABLATION STUDY RESULTS (Epoch 2)   ")
    print("=========================================")
    
    # 1. Base Model
    psnr_base = evaluate_model_on_1_image(
        {'use_semantic_guidance': False, 'use_multitask': False},
        './ablation_base',
        test_image
    )
    print(f"[1] Base LDM (No Semantic, No Multitask)  : PSNR = {psnr_base:.4f} dB")
    
    # 2. Semantic Model
    psnr_sem = evaluate_model_on_1_image(
        {'use_semantic_guidance': True, 'use_multitask': False},
        './ablation_semantic',
        test_image
    )
    print(f"[2] LDM + Semantic (No Multitask)       : PSNR = {psnr_sem:.4f} dB")
    
    # 3. Full Model
    psnr_full = evaluate_model_on_1_image(
        {'use_semantic_guidance': True, 'use_multitask': True},
        './ablation_full',
        test_image
    )
    print(f"[3] Full Proposed Model (Semantic + Multi): PSNR = {psnr_full:.4f} dB")
    print("=========================================")
