import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('base')

from models import create_model

# 1. Configuration for the test
opt = {
    'model': 'latent_diffusion',
    'dist': False,
    'gpu_ids': [0],
    'is_train': True,
    'num_classes': 10,
    'use_semantic_guidance': True,
    'use_multitask': True,
    'train': {
        'weight_decay_G': 0,
        'lr_G': 1e-4,
        'beta1_G': 0.9,
        'beta2_G': 0.999,
        'lr_scheme': 'MultiStepLR',
        'lr_steps': [50, 100],
        'lr_gamma': 0.5
    },
    'path': {
        'models': './ablation_full',
        'training_state': './ablation_full_state'
    }
}

os.makedirs(opt['path']['models'], exist_ok=True)
os.makedirs(opt['path']['training_state'], exist_ok=True)

# 2. Simple Custom Dataset
class SimpleDatasetA(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Mock segmentation mask (all zeros for test, 10 classes)
        _, h, w = image.shape
        mask = torch.zeros((h, w), dtype=torch.long)
            
        # We return the same image for L_R, L_S, and H_R for this test
        return {
            'L_R': image,
            'L_S': image,
            'H_R': image,
            'M_S': mask
        }

def main():
    print("🚀 Bắt đầu quá trình test train 10 Epochs với 1000 ảnh từ tập A")
    
    # Get 1000 images from Set A
    dataset_dir = '/home/ptitdemo/TVQH_TLU/Real-SR-master/Data/CPDM_Dataset/train/A/'
    all_images = sorted(glob.glob(os.path.join(dataset_dir, '*.jpg')))
    selected_images = all_images[:20]
    
    if len(selected_images) == 0:
        print(f"❌ Không tìm thấy ảnh nào trong {dataset_dir}")
        return
        
    print(f"✅ Đã chọn {len(selected_images)} ảnh.")
    
    # Transforms: Resize to 256x256 and convert to Tensor
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = SimpleDatasetA(selected_images, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # Initialize Model
    model = create_model(opt)
    
    epochs = 2
    total_steps = 0
    
    for epoch in range(epochs):
        print(f"\\n--- Epoch {epoch+1}/{epochs} ---")
        for i, data in enumerate(dataloader):
            total_steps += 1
            
            # Feed data and optimize
            model.feed_data(data)
            model.optimize_parameters(total_steps)
            
            if (i + 1) % 50 == 0:
                losses = model.get_current_losses()
                print(f"Step [{i+1}/{len(dataloader)}] - Loss Diff: {losses.get('l_diff'):.4f} - Loss Seg: {losses.get('l_seg'):.4f} - Total: {losses.get('l_total'):.4f}")
                
        # Save model after each epoch
        model.save(f'epoch_{epoch+1}')
        print(f"✅ Đã lưu checkpoint epoch {epoch+1}")
        
    print("\\n🎉 Đã hoàn thành quá trình train 10 epochs!")

if __name__ == '__main__':
    main()
