import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from .base_model import BaseModel
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    pass

logger = logging.getLogger('base')


# =============================================================================
# CẢI TIẾN 1: DINO v2 ADAPTER - Thay thế SimpleEncoder
# =============================================================================
# Toán học:
#   DINOv2 là ViT (Vision Transformer) được pretrain với mục tiêu
#   Self-Supervised Learning (SSL):
#     L_DINO = -sum_k [ q_k * log(p_k) ]
#   trong đó q_k là phân phối xác suất từ teacher network (EMA),
#   p_k là phân phối từ student network.
#
#   Kết quả: DINOv2 học được đặc trưng ngữ nghĩa (semantic features)
#   cực kỳ phong phú trên 142M ảnh mà KHÔNG cần nhãn.
#
#   Ta ĐÓNG BĂNG (freeze) toàn bộ DINOv2 và chỉ train Adapter nhẹ:
#     f_adapter = LayerNorm(Linear(f_dino))
#   => Tiết kiệm VRAM, tận dụng tri thức pretrained.
# =============================================================================
class DinoV2Adapter(nn.Module):
    """
    Wraps a frozen DINOv2 backbone + a small trainable adapter.
    Input : (B, 3, H, W) - RGB image, bất kỳ kích thước nào
    Output: (B, out_channels, H/patch, W/patch) - spatial feature map
             với patch_size=14 của DINOv2-small
    """
    def __init__(self, out_channels=256, dino_model='dinov2_vits14', freeze_backbone=True, use_lora=False):
        super().__init__()
        self.patch_size = 14
        self.use_lora = use_lora

        # Load DINOv2 pretrained (torch.hub)
        try:
            self.backbone = torch.hub.load('facebookresearch/dinov2', dino_model, pretrained=True)
            self.embed_dim = self.backbone.embed_dim  # 384 for vits14
        except Exception as e:
            logger.warning(f"Could not load DINOv2 from hub: {e}. Using fallback SimpleEncoder.")
            self.backbone = None
            self.embed_dim = 384

        if freeze_backbone and self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        if self.use_lora and self.backbone is not None:
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["qkv", "proj", "fc1", "fc2"],
                lora_dropout=0.1,
                bias="none"
            )
            self.backbone = get_peft_model(self.backbone, config)
            # PEFT automatically sets requires_grad=True for LoRA parameters

        # Adapter nhỏ: project từ embed_dim của DINO xuống out_channels
        self.adapter = nn.Sequential(
            nn.Linear(self.embed_dim, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )

        # Fallback nếu không load được DINOv2
        self.fallback_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, out_channels, 3, 1, 1), nn.ReLU(True)
        )

    def forward(self, x):
        if self.backbone is None:
            return self.fallback_conv(x)

        B, C, H, W = x.shape

        # DINOv2 yêu cầu ảnh chia hết cho patch_size=14
        # Padding nếu cần
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        H_pad, W_pad = x.shape[2], x.shape[3]
        h_patches = H_pad // self.patch_size
        w_patches = W_pad // self.patch_size

        # Lấy patch tokens từ DINOv2 (bỏ qua [CLS] token đầu tiên)
        if self.use_lora:
            dino_out = self.backbone.forward_features(x)
        else:
            with torch.no_grad():
                dino_out = self.backbone.forward_features(x)
                
        # patch_tokens: (B, h_patches*w_patches, embed_dim)
        patch_tokens = dino_out['x_norm_patchtokens']

        # Adapter: (B, h_patches*w_patches, embed_dim) -> (B, h_patches*w_patches, out_channels)
        feat = self.adapter(patch_tokens)

        # Reshape thành spatial feature map: (B, out_channels, h_patches, w_patches)
        feat = feat.permute(0, 2, 1).reshape(B, -1, h_patches, w_patches)

        return feat


# =============================================================================
# CẢI TIẾN 2: DPT-STYLE SEMANTIC DECODER - Thay thế SimpleDecoder
# =============================================================================
# Toán học (Dense Prediction Transformer - DPT):
#   Thay vì chỉ dùng TransposedConv liên tục, DPT kết hợp đặc trưng
#   ở nhiều tầng độ phân giải khác nhau (feature pyramid fusion):
#
#     F_out = Upsample(F4) ⊕ Upsample(F3) ⊕ Upsample(F2) ⊕ F1
#
#   Sau đó qua một vài lớp conv để đưa ra bản đồ nhãn:
#     M_S = Conv(F_out)  ∈ R^(H_HR × W_HR × K)
#
#   Trong đó K là số lớp nhãn, và ta lấy argmax để ra nhãn cuối.
#
#   Lý do DPT tốt hơn SimpleDecoder:
#   - Tận dụng thông tin đa tầng (multi-scale context).
#   - Giữ được chi tiết cục bộ (local) lẫn ngữ cảnh toàn cục (global).
# =============================================================================
class DPTSemanticDecoder(nn.Module):
    """
    Simplified DPT-style decoder.
    Input : (B, in_channels, H_feat, W_feat)
    Output: (B, num_classes, H_target, W_target)
    """
    def __init__(self, in_channels=256, num_classes=7):
        super().__init__()

        # Refinement blocks ở mỗi bước upsample
        def _make_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, 1, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(True),
                nn.Conv2d(out_c, out_c, 3, 1, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(True),
            )

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                  _make_block(in_channels, 128))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                  _make_block(128, 64))
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                  _make_block(64, 32))
        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                  _make_block(32, 16))
        self.up5 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                  _make_block(16, 16))

        # Head phân loại cuối cùng
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x, target_size=None):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.head(x)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        return x


# =============================================================================
# CẢI TIẾN 3: HIGH-FREQUENCY LOSS (FFT Loss)
# =============================================================================
# Toán học:
#   Standard MSE loss tối ưu hóa trên không gian pixel:
#     L_mse = ||ŷ - y||²₂
#   Nhưng MSE bị "mù" trước các thành phần tần số cao (cạnh, đường nét).
#
#   High-Frequency Loss dùng biến đổi Fourier rời rạc (DFT):
#     F(u,v) = Σ_x Σ_y f(x,y) · e^(-j2π(ux/M + vy/N))
#
#   Ta tính loss trên SPECTRUM (biên độ của FFT):
#     L_freq = ||A(ŷ) - A(y)||₁
#   trong đó A(·) = |FFT(·)| là biên độ phổ.
#
#   Lý do dùng L1 thay vì L2 trên phổ:
#   - L2 sẽ overpenalize các thành phần tần số cao (sparse),
#     khiến mạng né tránh việc tái tạo chúng.
#   - L1 robust hơn với outliers, phù hợp với phổ thưa.
#
#   Tổng Loss SR:
#     L_SR = L_diff + λ_freq · L_freq
#   với λ_freq = 0.1 (mặc định).
# =============================================================================
class FrequencyLoss(nn.Module):
    def __init__(self, loss_weight=0.1, min_radius=0.05, max_radius=0.95):
        super().__init__()
        self.loss_weight = loss_weight
        self.min_radius = min_radius
        self.max_radius = max_radius

    def forward(self, pred, target):
        """
        pred, target: (B, C, H, W) tensor trong khoảng [-1, 1] hoặc [0, 1]
        """
        # Biến đổi FFT 2D (torch.fft.fft2 trả về complex tensor)
        pred_fft   = torch.fft.fft2(pred,   norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        
        # Dịch mốc tần số (0) ra tâm ma trận
        pred_fft = torch.fft.fftshift(pred_fft, dim=(-2, -1))
        target_fft = torch.fft.fftshift(target_fft, dim=(-2, -1))

        # Lấy BIÊN ĐỘ (amplitude) của phổ
        pred_amp   = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)

        # Tạo mask băng thông (Band-Pass Filter)
        B, C, H, W = pred.shape
        Y, X = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        radius = torch.sqrt(X**2 + Y**2).to(pred.device)
        
        # Tạo mask: 1 ở dải trung-cao, 0 ở lõi tần số thấp (<=0.05) và viền cực đại (>=0.95)
        mask = (radius >= self.min_radius) & (radius <= self.max_radius)
        mask = mask.unsqueeze(0).unsqueeze(0).float() # shape: (1, 1, H, W)
        
        # Áp dụng mask
        pred_amp = pred_amp * mask
        target_amp = target_amp * mask

        # L1 loss trên biên độ phổ đã lọc
        loss = F.l1_loss(pred_amp, target_amp)
        return self.loss_weight * loss


class TaskWeights(nn.Module):
    def __init__(self, num_tasks=2):
        super().__init__()
        # Khởi tạo log_vars = 0 -> weight = 1
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

# =============================================================================
# MÔ HÌNH CHÍNH: LatentDiffusionModel (đã được cải tiến)
# =============================================================================
class LatentDiffusionModel(BaseModel):
    """
    Kiến trúc Hybrid đề xuất (đúng với hình vẽ trong paper + LDM):

    LUỒNG DỮ LIỆU (sửa lại đúng bản gốc paper):

    Train:
      L_R, L_S → SharedEncoder(DINOv2) → feat_R, feat_S
      feat_R → cond_proj → attn_guidance (điều kiện cho U-Net)
      VAE.encode(H_R) → z_0 → add_noise → z_t
      UNet(z_t, t, encoder_hidden_states=attn_guidance) → noise_pred → L_diff
      noise_pred → FrequencyLoss vs noise_high_freq → L_freq
      H_R → SemanticEncoder(DINOv2) → sem_feat  [Teacher Forcing]
      feat_S → adapting_layer → adapted_S
      adapted_S ⊕ sem_feat → SemanticDecoder(DPT) → mask_pred → L_seg

    Test (Inference):
      feat_R → cond_proj → attn_guidance
      randn(z_shape) → DDIMScheduler(50 steps) → z_0_clean
      VAE.decode(z_0_clean) → H_tilde_R  [Ảnh SR sinh ra]
      H_tilde_R → SemanticEncoder(DINOv2) → sem_feat  [Dùng ảnh SR thật]
      feat_S → adapting_layer → adapted_S
      adapted_S ⊕ sem_feat → SemanticDecoder(DPT) → M_tilde_S
    """
    def __init__(self, opt):
        super(LatentDiffusionModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1

        self.num_classes        = opt.get('num_classes', 7)
        self.use_semantic_guidance = opt.get('use_semantic_guidance', True)
        self.use_multitask      = opt.get('use_multitask', True)
        self.use_dynamic_weight = opt.get('use_dynamic_weight', False)
        self.use_bidirectional_fusion = opt.get('use_bidirectional_fusion', True)
        self.freq_loss_weight   = opt.get('freq_loss_weight', 0.1)

        if self.use_dynamic_weight:
            self.task_weights = TaskWeights(num_tasks=2).to(self.device)

        if self.use_bidirectional_fusion:
            # Chiếu đặc trưng phân vùng (num_classes channels) sang không gian cross-attention (512-dim)
            self.sem_proj = nn.Sequential(
                nn.Conv2d(self.num_classes, 512, kernel_size=1),
                nn.ReLU(True),
            ).to(self.device)

        # ------------------------------------------------------------------
        # CẢI TIẾN 1: Dùng DINOv2Adapter thay cho SimpleEncoder
        # out_channels=256 để tương thích với phần còn lại của pipeline
        # ------------------------------------------------------------------
        use_lora = opt.get('use_lora', False)
        self.shared_encoder   = DinoV2Adapter(out_channels=256, use_lora=use_lora).to(self.device)
        # Chỉ áp dụng LoRA cho Shared Encoder để tiết kiệm VRAM
        self.semantic_encoder = DinoV2Adapter(out_channels=256, use_lora=False).to(self.device)

        # Adapting Layer μ: Biến đặc trưng SR (256-ch) thành đặc trưng
        # phù hợp cho cộng element-wise với semantic features (256-ch)
        # Dùng GroupNorm thay LayerNorm vì LayerNorm yêu cầu shape cố định,
        # còn GroupNorm hoạt động đúng với bất kỳ H,W nào.
        self.adapting_layer = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.GroupNorm(32, 256),   # 32 groups, 256 channels → 8 ch/group
            nn.GELU(),
        ).to(self.device)

        # Projection để đưa feat_R (256-ch) thành cross-attention dim (512)
        # cho U-Net: feat_R là ĐIỀU KIỆN DUY NHẤT của LDM (đúng theo paper)
        self.cond_proj = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1),
            nn.ReLU(True),
        ).to(self.device)

        # ------------------------------------------------------------------
        # CẢI TIẾN 2: DPT Decoder thay cho SimpleDecoder
        # ------------------------------------------------------------------
        self.semantic_decoder = DPTSemanticDecoder(
            in_channels=256, num_classes=self.num_classes
        ).to(self.device)

        # ------------------------------------------------------------------
        # Latent Diffusion Model (VAE + UNet + Scheduler) - giữ nguyên
        # ------------------------------------------------------------------
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
            cross_attention_dim=512,       # khớp với cond_proj output
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
            self.vae.eval()    # VAE được giữ cố định (frozen)
            self.unet.train()
            self.shared_encoder.train()
            self.semantic_encoder.train()
            self.adapting_layer.train()
            self.cond_proj.train()
            self.semantic_decoder.train()

            train_opt = opt['train']
            wd = train_opt.get('weight_decay_G', 0)

            # Không train VAE và backbone DINOv2 (đã frozen trong DinoV2Adapter)
            optim_params = [
                p for k, p in self.network.named_parameters()
                if 'vae' not in k and p.requires_grad
            ]
            if getattr(self, 'use_dynamic_weight', False):
                optim_params.extend(self.task_weights.parameters())

            self.optimizer = torch.optim.AdamW(
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

            # Loss functions
            self.criterion_diff = nn.MSELoss().to(self.device)
            self.criterion_seg  = nn.CrossEntropyLoss(ignore_index=255).to(self.device)

            # CẢI TIẾN 3: High-Frequency Loss
            self.criterion_freq = FrequencyLoss(loss_weight=self.freq_loss_weight).to(self.device)

    # ------------------------------------------------------------------
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
            self.real_H_R = data.get('img').to(self.device)
            self.L_R      = data.get('img').to(self.device)
            self.L_S      = data.get('img').to(self.device)
            self.mask_S   = data.get('msk').to(self.device)
            self.var_H    = self.real_H_R
            self.var_L    = self.L_R
            self.fake_H   = self.real_H_R

    # ------------------------------------------------------------------
    def optimize_parameters(self, step, task="SR"):
        self.optimizer.zero_grad()
        loss = torch.tensor(0.0, device=self.device)
        b    = self.L_R.size(0)

        # ============================================================
        # BƯỚC 1: Shared Encoder → feat_R (và feat_S nếu cần)
        # Toán học: feat_R = E_shared(L_R) ∈ R^(B × 256 × h × w)
        # ============================================================
        feat_R = self.shared_encoder(self.L_R)
        feat_S = self.shared_encoder(self.L_S)

        # ============================================================
        # BƯỚC 2: Tạo điều kiện cho U-Net từ feat_R
        # Toán học:
        #   c = cond_proj(feat_R) ∈ R^(B × 512 × h × w)
        #   attn_guidance = flatten(c) ∈ R^(B × (h*w) × 512)
        # Đây là điều kiện DUY NHẤT của LDM - đúng với paper gốc!
        # ============================================================
        cond_feat   = self.cond_proj(feat_R)
        attn_guidance = cond_feat.view(b, 512, -1).permute(0, 2, 1)  # (B, seq_len1, 512)

        # ------------------------------------------------------------------
        # CẢI TIẾN 6: Bidirectional Feature Fusion (Semantic-Guided SR)
        # ------------------------------------------------------------------
        if getattr(self, 'use_bidirectional_fusion', False):
            # 1. Sinh bản đồ phân vùng từ feat_R
            sem_logits = self.semantic_decoder(feat_R) # (B, num_classes, H_out, W_out)
            
            # 2. Downsample sem_logits về cùng độ phân giải với feat_R để giảm độ dài chuỗi
            sem_logits_small = F.interpolate(sem_logits, size=feat_R.shape[-2:], mode='bilinear', align_corners=False)
            
            # 3. Chiếu sang không gian 512-dim
            sem_cond = self.sem_proj(sem_logits_small) # (B, 512, H_feat, W_feat)
            sem_guidance = sem_cond.view(b, 512, -1).permute(0, 2, 1) # (B, seq_len2, 512)
            
            # 4. Concatenate 2 chuỗi token: Ảnh (DINOv2) + Ngữ nghĩa (Semantic)
            attn_guidance = torch.cat([attn_guidance, sem_guidance], dim=1) # (B, seq_len1 + seq_len2, 512)

        # ============================================================
        # BƯỚC 3: SR Task - Quá trình Diffusion (chỉ khi task == "SR")
        # Toán học:
        #   z_0 = VAE_enc(H_R) · s_vae    [Nén vào latent space]
        #   z_t = sqrt(ᾱ_t)·z_0 + sqrt(1-ᾱ_t)·ε,  ε ~ N(0,I)
        #   ε_θ = UNet(z_t, t, c=attn_guidance)
        #   L_diff = ||ε_θ - ε||²
        #
        # High-Freq Loss:
        #   L_freq = λ · ||A(FFT(ε_θ)) - A(FFT(ε))||₁
        #
        # Tổng: L_SR = L_diff + L_freq
        # ============================================================
        if task == "SR" or not self.use_multitask:
            with torch.no_grad():
                latents   = self.vae.encode(self.real_H_R).latent_dist.sample()
                latents   = latents * self.vae.config.scaling_factor

            noise          = torch.randn_like(latents)
            timesteps      = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (b,), device=self.device
            ).long()
            noisy_latents  = self.noise_scheduler.add_noise(latents, noise, timesteps)
            noise_pred     = self.unet(
                noisy_latents, timesteps,
                encoder_hidden_states=attn_guidance
            ).sample

            # L_diff: MSE loss trên noise prediction
            loss_diff = self.criterion_diff(noise_pred, noise)

            # CẢI TIẾN 3: L_freq - High-Frequency Loss trên PIXEL SPACE
            # (Không tính trên latent noise vì noise không có ý nghĩa tần số)
            # Decode latent để lấy ảnh dự đoán trong pixel space
            with torch.no_grad():
                # Tính z_0 dự đoán từ noise_pred (x0-parameterization)
                alpha_prod_t = self.noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(self.device)
                pred_latent_x0 = (noisy_latents - (1 - alpha_prod_t).sqrt() * noise_pred) / alpha_prod_t.sqrt()
                pred_latent_x0 = pred_latent_x0.clamp(-1, 1)
                pred_pixel = self.vae.decode(pred_latent_x0 / self.vae.config.scaling_factor).sample
                gt_pixel   = self.vae.decode(latents / self.vae.config.scaling_factor).sample
            loss_freq = self.criterion_freq(pred_pixel, gt_pixel)

            loss_sr = loss_diff + loss_freq

            if getattr(self, 'use_dynamic_weight', False):
                # Anchored Dynamic Weighting: SR branch fixed at 1.0
                loss_sr_weighted = loss_sr
                self.log_dict['l_sr_weighted'] = loss_sr_weighted.item()
            else:
                loss_sr_weighted = loss_sr

            loss = loss + loss_sr_weighted
            self.log_dict['l_diff'] = loss_diff.item()
            self.log_dict['l_freq'] = loss_freq.item()

        # ============================================================
        # BƯỚC 4: Semantic Segmentation Task (chỉ khi task == "SS")
        # Toán học:
        #   [Teacher Forcing] sem_feat = E_sem(H_R)   ← Ảnh thật lúc train
        #   adapted_S = μ(feat_S)
        #   combined  = adapted_S ⊕ sem_feat            [Element-wise add]
        #   M̃_S = D_sem(combined) ∈ R^(B × K × H × W)
        #   L_seg = CE(M̃_S, M_S)                       [Cross Entropy]
        # ============================================================
        if task == "SS" and self.use_multitask:
            sem_feat  = self.semantic_encoder(self.real_H_R)
            adapted_S = self.adapting_layer(feat_S)

            # Align kích thước (phòng khi có lệch do padding/patch)
            if adapted_S.shape[2:] != sem_feat.shape[2:]:
                sem_feat = F.interpolate(sem_feat, size=adapted_S.shape[2:],
                                         mode='bilinear', align_corners=True)
            combined_feat = adapted_S + sem_feat

            # Lấy kích thước mask target để upsample đầu ra đúng
            target_size = self.mask_S.shape[-2:] if self.mask_S is not None else None
            pred_mask_S = self.semantic_decoder(combined_feat, target_size=target_size)

            if self.mask_S is not None:
                loss_seg = self.criterion_seg(pred_mask_S, self.mask_S)
                if getattr(self, 'use_dynamic_weight', False):
                    precision_seg = torch.exp(-self.task_weights.log_vars[1])
                    loss_seg_weighted = precision_seg * loss_seg + self.task_weights.log_vars[1]
                    self.log_dict['l_seg_weighted'] = loss_seg_weighted.item()
                else:
                    loss_seg_weighted = loss_seg
                loss = loss + loss_seg_weighted
                self.log_dict['l_seg'] = loss_seg.item()

            self.msk_P = pred_mask_S

        loss.backward()
        self.optimizer.step()
        self.log_dict['l_total'] = loss.item()

    # ------------------------------------------------------------------
    def get_current_log(self):
        return self.log_dict

    # ------------------------------------------------------------------
    def test(self):
        """
        Inference mode.
        LUỒNG ĐÚNG THEO PAPER:
        1. feat_R → điều kiện cho LDM → sinh H̃_R
        2. H̃_R → Semantic Encoder → gộp với feat_S → sinh M̃_S
        """
        self.unet.eval()
        self.vae.eval()
        self.shared_encoder.eval()
        self.semantic_encoder.eval()
        self.semantic_decoder.eval()
        self.adapting_layer.eval()
        self.cond_proj.eval()

        with torch.no_grad():
            b = self.L_R.size(0)

            if self.task == "SS":
                sem_feat  = self.semantic_encoder(self.real_H_R)
                feat_S    = self.shared_encoder(self.L_S)
                adapted_S = self.adapting_layer(feat_S)

                if adapted_S.shape[2:] != sem_feat.shape[2:]:
                    sem_feat = F.interpolate(sem_feat, size=adapted_S.shape[2:],
                                             mode='bilinear', align_corners=True)
                combined_feat = adapted_S + sem_feat
                target_size = self.mask_S.shape[-2:] if self.mask_S is not None else None
                self.fake_M_S = self.semantic_decoder(combined_feat, target_size=target_size)
                self.msk_P    = self.fake_M_S
                
                self.unet.train()
                self.shared_encoder.train()
                self.semantic_encoder.train()
                self.semantic_decoder.train()
                self.adapting_layer.train()
                self.cond_proj.train()
                return

            # BƯỚC 1: Lấy điều kiện từ L_R qua Shared Encoder
            feat_R      = self.shared_encoder(self.L_R)
            feat_S      = self.shared_encoder(self.L_S)
            cond_feat   = self.cond_proj(feat_R)
            attn_guidance = cond_feat.view(b, 512, -1).permute(0, 2, 1)

            # CẢI TIẾN 6: Bidirectional Feature Fusion (Semantic-Guided SR) trong lúc Test
            if getattr(self, 'use_bidirectional_fusion', False):
                sem_logits = self.semantic_decoder(feat_R)
                sem_logits_small = F.interpolate(sem_logits, size=feat_R.shape[-2:], mode='bilinear', align_corners=False)
                sem_cond = self.sem_proj(sem_logits_small)
                sem_guidance = sem_cond.view(b, 512, -1).permute(0, 2, 1)
                attn_guidance = torch.cat([attn_guidance, sem_guidance], dim=1)

            # BƯỚC 2: Sinh ảnh SR bằng DDIM (50 bước khử nhiễu)
            # Toán học DDIM:
            #   z_{t-1} = sqrt(ᾱ_{t-1}) · pred_z0 + sqrt(1-ᾱ_{t-1}) · ε_θ
            hr_h, hr_w = self.L_R.shape[2] * 4, self.L_R.shape[3] * 4
            latent_shape = (b, 4, hr_h // 8, hr_w // 8)
            latents = torch.randn(latent_shape, device=self.device)

            ddim_scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
            ddim_scheduler.set_timesteps(50)

            for t in ddim_scheduler.timesteps:
                noise_pred = self.unet(latents, t, encoder_hidden_states=attn_guidance).sample
                latents    = ddim_scheduler.step(noise_pred, t, latents).prev_sample

            # Giải mã từ latent space ra pixel space
            latents       = (1 / self.vae.config.scaling_factor) * latents
            H_tilde_R     = self.vae.decode(latents).sample
            if H_tilde_R.shape[-2:] != (hr_h, hr_w):
                H_tilde_R = F.interpolate(H_tilde_R, size=(hr_h, hr_w), mode='bilinear', align_corners=True)
            H_tilde_R     = (H_tilde_R / 2 + 0.5).clamp(0, 1)
            self.fake_H   = H_tilde_R
            self.fake_H_R = H_tilde_R

            # BƯỚC 3: Semantic Segmentation (dùng H̃_R thật sự từ LDM - đúng paper!)
            if self.use_multitask:
                # Dùng ảnh SR vừa sinh ra (không phải L_R upsampled nữa)
                sem_feat  = self.semantic_encoder(H_tilde_R)
                adapted_S = self.adapting_layer(feat_S)

                if adapted_S.shape[2:] != sem_feat.shape[2:]:
                    sem_feat = F.interpolate(sem_feat, size=adapted_S.shape[2:],
                                             mode='bilinear', align_corners=True)
                combined_feat   = adapted_S + sem_feat
                self.fake_M_S   = self.semantic_decoder(combined_feat, target_size=(hr_h, hr_w))
                self.msk_P      = self.fake_M_S
            else:
                self.fake_M_S = None
                self.msk_P    = None

        # Chuyển lại train mode
        self.unet.train()
        self.shared_encoder.train()
        self.semantic_encoder.train()
        self.semantic_decoder.train()
        self.adapting_layer.train()
        self.cond_proj.train()

    # ------------------------------------------------------------------
    def get_current_visuals(self, need_GT=True, **kwargs):
        out_dict = OrderedDict()
        out_dict['L_R'] = self.L_R.detach()[0].float().cpu()
        out_dict['L_S'] = self.L_S.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H_R.detach()[0].float().cpu()
        if hasattr(self, 'fake_H'):
            out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if hasattr(self, 'msk_P') and self.msk_P is not None:
            out_dict['SS'] = self.msk_P.detach()[0].float().cpu()
        return out_dict

    # ------------------------------------------------------------------
    def save(self, iter_label, mylogger=None):
        self.save_network(self.unet,             'UNet',           iter_label)
        self.save_network(self.shared_encoder,   'SharedEncoder',  iter_label)
        self.save_network(self.semantic_encoder, 'SemanticEncoder',iter_label)
        self.save_network(self.adapting_layer,   'AdaptingLayer',  iter_label)
        self.save_network(self.cond_proj,        'CondProj',       iter_label)
        self.save_network(self.semantic_decoder, 'SemanticDecoder',iter_label)
        if getattr(self, 'use_dynamic_weight', False):
            self.save_network(self.task_weights, 'TaskWeights', iter_label)

    # ------------------------------------------------------------------
    def load(self):
        load_path_G = self.opt['path'].get('pretrain_model_G', None)
        if load_path_G is not None:
            import os
            base_dir  = os.path.dirname(load_path_G)
            basename  = os.path.basename(load_path_G)
            iter_step = basename.split('_')[0]
            prefix    = os.path.join(base_dir, iter_step)
            strict    = self.opt['path'].get('strict_load', True)

            logger.info(f'Loading model for LDM from [{prefix}]')
            self.load_network(prefix + '_UNet.pth',           self.unet,             strict)
            self.load_network(prefix + '_SharedEncoder.pth',  self.shared_encoder,   strict)
            self.load_network(prefix + '_SemanticEncoder.pth',self.semantic_encoder, strict)
            self.load_network(prefix + '_AdaptingLayer.pth',  self.adapting_layer,   strict)
            self.load_network(prefix + '_CondProj.pth',       self.cond_proj,        strict)
            self.load_network(prefix + '_SemanticDecoder.pth',self.semantic_decoder, strict)
            if getattr(self, 'use_dynamic_weight', False):
                task_weights_path = prefix + '_TaskWeights.pth'
                if os.path.exists(task_weights_path):
                    self.load_network(task_weights_path, self.task_weights, strict)
