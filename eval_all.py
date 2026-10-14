import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import logging
import argparse

from models import create_model
from utils.util import calculate_psnr, calculate_ssim

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('base')

class IsprsEvalDataset(Dataset):
    def __init__(self, lst_path, mask_lst_path=None, transform=None):
        self.image_paths = []
        with open(lst_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 0 and os.path.exists(parts[0]):
                    self.image_paths.append(parts[0])
                    
        self.mask_paths = []
        if mask_lst_path:
            with open(mask_lst_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 1 and os.path.exists(parts[1]):
                        self.mask_paths.append(parts[1])
        
        self.image_paths = self.image_paths[:100] # evaluate on 100 images for speed
        if self.mask_paths:
            self.mask_paths = self.mask_paths[:100]
            
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        hr_img = Image.open(path).convert('RGB')
        hr_img = hr_img.resize((256, 256), Image.BICUBIC)
        hr_tensor = self.transform(hr_img)
        
        lr_tensor = F.interpolate(hr_tensor.unsqueeze(0), scale_factor=0.25, mode='bicubic', align_corners=False).squeeze(0)
        
        mask_tensor = None
        if self.mask_paths:
            m_path = self.mask_paths[idx]
            m_img = Image.open(m_path)
            m_img = m_img.resize((256, 256), Image.NEAREST)
            m_np = np.array(m_img)
            
            from data.SSSR_dataset import SSSRDataset
            arr_2d = SSSRDataset.color_to_class(m_np)
            
            mask_tensor = torch.from_numpy(arr_2d).long()
        
        return {'LQ': lr_tensor, 'GT': hr_tensor, 'msk': mask_tensor, 'img': hr_tensor}

def calculate_iou(pred, target, num_classes=7, ignore_index=255):
    pred = pred.view(-1)
    target = target.view(-1)
    
    valid_mask = target != ignore_index
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return ious

def evaluate(model_type, ckpt_dir, step, output_dir, has_dino=False, has_freq=False, has_semantic=True):
    print(f"🚀 Bắt đầu Evaluation cho model {model_type}...")
    
    opt = {
        'model': model_type,
        'dist': False,
        'gpu_ids': [0],
        'is_train': False,
        'num_classes': 7,
        'use_multitask': True,
        'use_semantic_guidance': has_semantic,
        'freq_loss_weight': 0.1 if has_freq else 0.0,
        'dino_model': 'dinov2_vits14' if has_dino else None,
        'path': {
            'models': ckpt_dir,
        }
    }
    
    model = create_model(opt)
    
    try:
        model.load_network(f"{ckpt_dir}/{step}_UNet.pth", model.unet)
        model.load_network(f"{ckpt_dir}/{step}_SharedEncoder.pth", model.shared_encoder)
        model.load_network(f"{ckpt_dir}/{step}_SemanticEncoder.pth", model.semantic_encoder)
        model.load_network(f"{ckpt_dir}/{step}_SemanticDecoder.pth", model.semantic_decoder)
        model.load_network(f"{ckpt_dir}/{step}_AdaptingLayer.pth", model.adapting_layer)
        if has_semantic:
            model.load_network(f"{ckpt_dir}/{step}_CondProj.pth", model.cond_proj)
        print(f"✅ Đã load thành công toàn bộ weights từ mốc {step}.")
    except Exception as e:
        print(f"⚠️ Lỗi khi load weights: {e}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
        
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = IsprsEvalDataset(
        '../datasets/kaggle_isprs/val_SR.lst', 
        mask_lst_path='../datasets/kaggle_isprs/val_SS.lst',
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    total_psnr = 0.0
    total_ssim = 0.0
    total_ious = []
    
    for i, data in enumerate(dataloader):
        # Fake SS and SR tasks
        model.feed_data(data, task="SR")
        model.test()
        
        hr_gt = (model.real_H_R.squeeze().cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0
        hr_pred = (model.fake_H_R.squeeze().cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0
        
        hr_gt = np.clip(hr_gt * 255.0, 0, 255).astype(np.uint8)
        hr_pred = np.clip(hr_pred * 255.0, 0, 255).astype(np.uint8)
        
        psnr = calculate_psnr(hr_gt, hr_pred)
        ssim = calculate_ssim(hr_gt, hr_pred)
        
        total_psnr += psnr
        total_ssim += ssim
        
        # Calculate mIoU
        if hasattr(model, 'msk_P') and data['msk'] is not None and model.msk_P is not None:
            mask_pred = torch.argmax(model.msk_P, dim=1) # (1, H, W)
            ious = calculate_iou(mask_pred, data['msk'].to(model.device))
            total_ious.append(ious)
            
    avg_psnr = total_psnr / len(dataset)
    avg_ssim = total_ssim / len(dataset)
    
    avg_miou = 0.0
    if total_ious:
        valid_ious = []
        for c in range(7):
            cls_ious = [iou_list[c] for iou_list in total_ious if not np.isnan(iou_list[c])]
            if cls_ious:
                valid_ious.append(np.mean(cls_ious))
        avg_miou = np.mean(valid_ious)
    
    print("\n========================================")
    print(f"🏆 KẾT QUẢ ĐÁNH GIÁ: {model_type}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average mIoU: {avg_miou:.4f}")
    print("========================================\n")
    return avg_psnr, avg_ssim, avg_miou

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_v2', action='store_true')
    parser.add_argument('--run_baseline', action='store_true')
    parser.add_argument('--run_v2_1', action='store_true')
    args = parser.parse_args()
    
    if args.run_baseline:
        evaluate(
            model_type='latent_diffusion', 
            ckpt_dir='./wandb/offline-run-20260821_181859-timpeuvr/local_files/models', 
            step='25000', 
            output_dir='../test_results/baseline',
            has_dino=True, has_freq=False, has_semantic=False
        )
        
    if args.run_v2:
        evaluate(
            model_type='ldm_v2', 
            ckpt_dir='./wandb/offline-run-20260821_181859-cxae2xuv/local_files/models', 
            step='25000', 
            output_dir='../test_results/ours_v2',
            has_dino=True, has_freq=True, has_semantic=True
        )

    if hasattr(args, 'run_v2_1') and args.run_v2_1:
        evaluate(
            model_type='ldm_v2', 
            ckpt_dir='./wandb/offline-run-20260823_150510-u8t5qeao/local_files/models', 
            step='best', 
            output_dir='../test_results/ours_v2_1',
            has_dino=True, has_freq=True, has_semantic=True
        )
