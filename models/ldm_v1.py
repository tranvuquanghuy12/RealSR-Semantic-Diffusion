import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from .base_model import BaseModel

logger = logging.getLogger('base')

class SimpleEncoder(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, out_channels, 3, 1, 1), nn.ReLU(True)
        )
    def forward(self, x):
        return self.conv(x)

class SimpleDecoder(nn.Module):
    def __init__(self, in_channels=256, num_classes=7):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, num_classes, 3, 1, 1)
        )
    def forward(self, x, target_size=None):
        out = self.up(x)
        if target_size is not None:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=True)
        return out

class LDMV1(BaseModel):
    """
    Ours V1 Architecture: 
    - SimpleEncoder / SimpleDecoder
    - No FFT Loss
    - Train-Test mismatch on Semantic Encoder (uses HR directly during test in the original V1 flawed design).
    """
    def __init__(self, opt):
        super(LDMV1, self).__init__(opt)
        
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1
            
        self.num_classes = opt.get('num_classes', 7)
        self.use_multitask = opt.get('use_multitask', True)
        
        self.shared_encoder = SimpleEncoder(out_channels=256).to(self.device)
        self.semantic_encoder = SimpleEncoder(out_channels=256).to(self.device)
        
        self.adapting_layer = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        ).to(self.device)
        
        # V1 did not use cond_proj properly for U-Net, but we will provide one so the code runs
        self.cond_proj = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1),
            nn.ReLU(True),
        ).to(self.device)
        
        self.semantic_decoder = SimpleDecoder(in_channels=256, num_classes=self.num_classes).to(self.device)
        
        self.vae = AutoencoderKL(
            in_channels=3, out_channels=3,
            down_block_types=("DownEncoderBlock2D",) * 4,
            up_block_types=("UpDecoderBlock2D",) * 4,
            block_out_channels=(64, 128, 256, 256),
            latent_channels=4,
        ).to(self.device)

        self.unet = UNet2DConditionModel(
            sample_size=64,
            in_channels=4, out_channels=4,
            cross_attention_dim=512,
            down_block_types=("CrossAttnDownBlock2D",) * 3 + ("DownBlock2D",),
            up_block_types=("UpBlock2D",) + ("CrossAttnUpBlock2D",) * 3,
            block_out_channels=(128, 256, 512, 512),
        ).to(self.device)

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        self.network = nn.ModuleDict({
            'shared_encoder':   self.shared_encoder,
            'semantic_encoder': self.semantic_encoder,
            'adapting_layer':   self.adapting_layer,
            'cond_proj':        self.cond_proj,
            'semantic_decoder': self.semantic_decoder,
            'vae':              self.vae,
            'unet':             self.unet,
        })

        if self.is_train:
            self.vae.eval()
            self.unet.train()
            self.shared_encoder.train()
            self.semantic_encoder.train()
            self.adapting_layer.train()
            self.cond_proj.train()
            self.semantic_decoder.train()

            train_opt = opt['train']
            wd = train_opt.get('weight_decay_G', 0)

            optim_params = [
                p for k, p in self.network.named_parameters()
                if 'vae' not in k and p.requires_grad
            ]

            self.optimizer = torch.optim.Adam(
                optim_params,
                lr=train_opt['lr_G'],
                weight_decay=wd,
                betas=(train_opt.get('beta1_G', 0.9), train_opt.get('beta2_G', 0.999))
            )
            self.optimizers.append(self.optimizer)

            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.MultiStepLR(
                            optimizer, train_opt['lr_steps'], train_opt['lr_gamma']
                        )
                    )

            self.log_dict = OrderedDict()
            self.criterion_diff = nn.MSELoss().to(self.device)
            self.criterion_seg  = nn.CrossEntropyLoss(ignore_index=255).to(self.device)

    def feed_data(self, data, task="SR", need_GT=True, **kwargs):
        self.task = task
        if task == "SR":
            self.real_H_R = data.get('GT').to(self.device)
            self.L_R      = data.get('LQ').to(self.device)
            self.L_S      = self.L_R
            self.mask_S   = None
            self.var_H    = self.real_H_R
            self.var_L    = self.L_R
            self.fake_H   = F.interpolate(self.L_R, scale_factor=4, mode='bilinear', align_corners=False)
        elif task == "SS":
            img = data.get('img').to(self.device)
            self.real_H_R = img
            self.L_R      = F.interpolate(img, scale_factor=0.25, mode='bilinear', align_corners=False)
            self.L_S      = self.L_R
            self.mask_S   = data.get('msk').to(self.device)
            self.var_H    = self.real_H_R
            self.var_L    = self.L_R
            self.fake_H   = self.real_H_R

    def optimize_parameters(self, step, task="SR"):
        self.optimizer.zero_grad()
        loss = torch.tensor(0.0, device=self.device)
        b = self.L_R.size(0)

        feat_R = self.shared_encoder(self.L_R)
        feat_S = self.shared_encoder(self.L_S)

        cond_feat = self.cond_proj(feat_R)
        attn_guidance = cond_feat.view(b, 512, -1).permute(0, 2, 1)

        if task == "SR" or not self.use_multitask:
            with torch.no_grad():
                latents = self.vae.encode(self.real_H_R).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (b,), device=self.device).long()
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            
            noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=attn_guidance).sample
            loss_diff = self.criterion_diff(noise_pred, noise)
            
            loss = loss + loss_diff
            self.log_dict['l_diff'] = loss_diff.item()

        if task == "SS" and self.use_multitask:
            sem_feat = self.semantic_encoder(self.real_H_R)
            adapted_S = self.adapting_layer(feat_S)
            if adapted_S.shape[2:] != sem_feat.shape[2:]:
                sem_feat = F.interpolate(sem_feat, size=adapted_S.shape[2:], mode='bilinear', align_corners=True)
            combined = adapted_S + sem_feat
            
            self.msk_P = self.semantic_decoder(combined, target_size=self.mask_S.shape[-2:])
            loss_seg = self.criterion_seg(self.msk_P, self.mask_S.long())
            
            loss = loss + loss_seg
            self.log_dict['l_seg'] = loss_seg.item()

        if loss > 0:
            loss.backward()
            self.optimizer.step()

    def test(self):
        self.vae.eval()
        self.unet.eval()
        self.shared_encoder.eval()
        self.semantic_encoder.eval()
        self.adapting_layer.eval()
        self.cond_proj.eval()
        self.semantic_decoder.eval()

        b = self.L_R.size(0)
        with torch.no_grad():
            fake_H_val = F.interpolate(self.L_R, scale_factor=4, mode='bilinear', align_corners=False)
            feat_R = self.shared_encoder(self.L_R)
            feat_S = self.shared_encoder(fake_H_val)
            
            cond_feat = self.cond_proj(feat_R)
            attn_guidance = cond_feat.view(b, 512, -1).permute(0, 2, 1)
            
            scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
            scheduler.set_timesteps(50)
            hr_h, hr_w = self.L_R.shape[2] * 4, self.L_R.shape[3] * 4
            latent_shape = (b, 4, hr_h // 8, hr_w // 8)
            latents = torch.randn(latent_shape, device=self.device)
            
            for t in scheduler.timesteps:
                noise_pred = self.unet(latents, t, encoder_hidden_states=attn_guidance).sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample
                
            self.fake_H_R = self.vae.decode(latents / self.vae.config.scaling_factor).sample
            if self.fake_H_R.shape[-2:] != (hr_h, hr_w):
                self.fake_H_R = F.interpolate(self.fake_H_R, size=(hr_h, hr_w), mode='bilinear', align_corners=True)
            
            # V1 Mismatch: Uses upsampled L_R (self.fake_H) for SemanticEncoder during test, instead of SR output!
            # (In V2, we use self.fake_H_R here).
            sem_feat = self.semantic_encoder(fake_H_val)
            
            adapted_S = self.adapting_layer(feat_S)
            combined = adapted_S + sem_feat
            
            self.msk_P = self.semantic_decoder(combined)
            if self.msk_P.shape[-2:] != (hr_h, hr_w):
                self.msk_P = F.interpolate(self.msk_P, size=(hr_h, hr_w), mode='bilinear', align_corners=True)
            self.fake_H_R = torch.clamp(self.fake_H_R, 0, 1)
            
        self.vae.train()
        self.unet.train()
        self.shared_encoder.train()
        self.semantic_encoder.train()
        self.adapting_layer.train()
        self.cond_proj.train()
        self.semantic_decoder.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.L_R.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H_R.detach()[0].float().cpu()
        if hasattr(self, 'msk_P') and self.msk_P is not None:
            out_dict['mask_pred'] = self.msk_P.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H_R.detach()[0].float().cpu()
            if self.mask_S is not None:
                out_dict['mask_GT'] = self.mask_S.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.unet)
        if isinstance(self.unet, nn.DataParallel):
            net_struc_str = f'{self.unet.__class__.__name__} - {self.unet.module.__class__.__name__}'
        else:
            net_struc_str = f'{self.unet.__class__.__name__}'
        logger.info(f'Network structure: {net_struc_str}, with parameters: {n:,d}')

    def save(self, iter_label, mylogger=None):
        self.save_network(self.unet,             'UNet',           iter_label)
        self.save_network(self.shared_encoder,   'SharedEncoder',  iter_label)
        self.save_network(self.semantic_encoder, 'SemanticEncoder',iter_label)
        self.save_network(self.adapting_layer,   'AdaptingLayer',  iter_label)
        self.save_network(self.cond_proj,        'CondProj',       iter_label)
        self.save_network(self.semantic_decoder, 'SemanticDecoder',iter_label)

    def load(self):
        load_path_network = self.opt['path'].get('pretrain_model_G', None)
        if load_path_network is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_network))
            # Just a placeholder if we need it
            pass
