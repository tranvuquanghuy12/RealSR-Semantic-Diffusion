# RealSR-Semantic-Diffusion 🛰️
> **A Semantic-Guided Latent Diffusion Framework for Simultaneous Remote Sensing Super-Resolution & Semantic Segmentation**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Task](https://img.shields.io/badge/Task-Super--Resolution%20%26%20Segmentation-success.svg)]()

---

## 📌 Tổng quan Nghiên cứu (Overview)

Các mô hình siêu phân giải truyền thống (như ESRGAN, Real-ESRGAN hay Diffusion thông thường) khi áp dụng vào ảnh viễn thám độ phân giải cao thường gặp hai điểm yếu cố hữu:
1. **Sinh ảo giác (Hallucination):** Tạo vân bề mặt tùy tiện, làm méo mó ranh giới thực địa của công trình và đường sá.
2. **Mờ nhòe viền cạnh (Edge Blurring):** Do tối ưu dàn trải loss trong không gian ảnh, mô hình thiên về làm mịn ảnh ở các vùng rộng lớn, bỏ qua các thành phần tần số cao.

**RealSR-Semantic-Diffusion** là mô hình đa nhiệm (**Multi-task**) tiên phong kết hợp **Foundation Model ngữ nghĩa (DINOv2)** cùng **Latent Diffusion Model (LDM)** và hàm mất mát phổ tần số Fourier (**High-Frequency FFT Loss**), cho phép tối ưu đồng thời chất lượng thị giác sắc nét và độ chính xác phân vùng ngữ nghĩa.

---

## 🚀 Các Đóng góp & Cải tiến Cốt lõi (Key Contributions)

### 1. DINOv2 Feature Adapter (Foundation Model Guidance)
- Thay thế mạng tích chập nông thông thường bằng **DINOv2 (`dinov2_vits14`)** được đóng băng (freeze backbone).
- Trích xuất đặc trưng ngữ nghĩa không gian 142M tham số cực kỳ giàu chi tiết mà không cần train lại từ đầu.
- Cung cấp định hướng đa tầng (**Cross-Attention Conditioning**) giúp U-Net Diffusion tái tạo cấu trúc ảnh tuân thủ chặt chẽ không gian vật lý thực địa.

### 2. DPT-Style Multi-Scale Semantic Decoder
- Bộ giải mã dạng **Dense Prediction Transformer (DPT)** dung hợp đặc trưng ở nhiều thang đo (multi-scale feature fusion).
- Khôi phục Mask phân vùng với độ phân giải cao, ranh giới giữa các lớp đất, cây cỏ, mái nhà và đường giao thông được phân tách rõ rệt.

### 3. High-Frequency Band-Pass FFT Loss
- Chuyển đổi ảnh sang miền tần số Fourier:
  $$\mathcal{L}_{\text{freq}} = \| \mathcal{M}_{\text{band}} \odot (\mathcal{F}(I_{\text{SR}}) - \mathcal{F}(I_{\text{HR}})) \|_1$$
- Áp dụng mặt nạ lọc thông dải (**Band-Pass Filter Mask**) tập trung xử lý dải tần số cao $0.05 \le r \le 0.95$.
- Ép mạng tập trung tối ưu viền cạnh sắc nét của kiến trúc đô thị, giảm thiểu triệt để chỉ số mất mát chân thực **LPIPS**.

### 4. Anchored Dynamic Multi-Task Loss Weighting
- Tự động cân bằng độ quan trọng giữa 2 nhánh Super-Resolution và Semantic Segmentation trong quá trình tối ưu.
- Cơ chế **Anchored Variance Regularization** triệt tiêu hiện tượng bùng nổ Gradient (Gradient Explosion) của các phiên bản trọng số không ổn định trước đây.

### 5. Fast Semantic Validation (Giảm 98% thời gian)
- Tách rời vòng lặp 50 bước DDIM khử nhiễu khi đánh giá nhánh Semantic Segmentation (SS).
- Rút ngắn thời gian kiểm thử validation nhánh SS từ **20 phút xuống chỉ còn 19 giây** mỗi lần đánh giá.

---

## 📊 Kết quả Thực nghiệm (Experimental Results)

Đánh giá trên tập dữ liệu viễn thám chuẩn **ISPRS Benchmark**:

| Mô hình / Phiên bản | Kỹ thuật Đề xuất | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | mIoU ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Paper Baseline** | LDM thuần (Hồi quy MSE) | 25.11 | 0.730 | 0.285 | *N/A* |
| **Ours V1** | Multi-task LDM (Simple Conv Encoder) | 25.40 | 0.752 | 0.272 | 0.5230 |
| **Ours V2.1** | DINOv2 + DPT + High-Freq FFT Loss | 25.85 | 0.787 | 0.245 | **0.5952** |
| **Ours V3** | V2.1 + Dynamic Uncertainty Weighting | *Gradient Divergence* | — | 0.820 | 0.6031 |
| **Ours V4 (Proposed)** | V3 + Band-Pass FFT + Anchored Weighting + Semantic Guidance | **25.82** | **0.785** | **0.198** | **0.5908** |

> **Nhận xét:** Bản **Ours V4** đạt điểm cân bằng hoàn hảo: chỉ số chân thực thị giác **LPIPS giảm sâu xuống 0.198** (thấp nhất trong tất cả các phiên bản) cho ảnh sắc nét tự nhiên, đồng thời duy trì độ chính xác phân đoạn **mIoU ~ 0.59**.

---

## 📂 Cấu trúc Mã nguồn (Repository Structure)

```
.
├── models/                     # Kiến trúc mô hình
│   ├── ldm_v2.py               # Mô hình chủ lực Ours V4 (DINOv2 + DPT + FFT Loss)
│   ├── ldm_v1.py               # Mô hình Ours V1
│   ├── ldm_v3.py               # Mô hình Ours V3
│   ├── latent_diffusion_model.py # LDM nền tảng cho ảnh viễn thám
│   ├── base_model.py           # Base framework
│   ├── ema.py                  # Exponential Moving Average
│   ├── lr_scheduler.py         # Bộ điều chỉnh Learning Rate
│   └── modules/                # Các khối hàm mất mát và phân đoạn
│       ├── loss.py             # Frequency FFT Loss, Perceptual Loss
│       ├── seg_loss.py         # Cross-Entropy & Dice Segmentation Loss
│       ├── seg_arch.py         # DPT Semantic Decoder Blocks
│       └── sft_arch.py         # Spatial Feature Transform
├── options/                    # Cấu hình thực nghiệm (YAML)
│   └── isprs/
│       ├── train_Ours_v4.yml   # Cấu hình huấn luyện chuẩn Ours V4 (ViT-Small)
│       ├── train_Ours_v4_ViTBase.yml # Cấu hình với DINOv2 ViT-Base
│       ├── train_Ours_v4_ViTLarge.yml # Cấu hình với DINOv2 ViT-Large
│       ├── train_Ours_v2.yml   # Cấu hình Ours V2
│       └── train_Ours_v1.yml   # Cấu hình Ours V1
├── data/                       # Dataloader & Dataset processing
│   ├── SSSR_dataset.py         # Dataset đa nhiệm SR + SS viễn thám
│   ├── data_loader.py          # Dataloader đa tiến trình
│   └── util.py                 # Xử lý ma trận ảnh và nhãn
├── utils/                      # Công cụ đo đạc & ghi log
│   ├── mylogger.py             # Logging hệ thống & Wandb integration
│   └── util.py                 # Hàm tính PSNR, SSIM, mIoU, LPIPS
├── train_sssr.py               # Script huấn luyện chính (Multi-task Training)
├── evaluate_rs_model.py        # Script đánh giá đa chỉ số (PSNR, SSIM, mIoU)
├── evaluate_ablation_all.py    # Script đánh giá so sánh Ablation Study
├── crop_patches.py             # Công cụ cắt patch ảnh từ ảnh vệ tinh lớn
├── start_training.sh           # Script chạy huấn luyện tự động trong tmux
└── requirements.txt            # Danh sách thư viện phụ thuộc
```

---

## 🛠️ Hướng dẫn Cài đặt & Sử dụng (Getting Started)

### 1. Cài đặt Môi trường (Installation)
```bash
git clone https://github.com/tranvuquanghuy12/RealSR-Semantic-Diffusion.git
cd RealSR-Semantic-Diffusion

# Khởi tạo venv
python3 -m venv venv
source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Chuẩn bị Dữ liệu (Data Preparation)
Cắt các ảnh viễn thám độ phân giải cao thành các patch 128x128 hoặc 256x256:
```bash
python3 crop_patches.py --input_dir /path/to/raw_tiles --output_dir datasets/kaggle_isprs
```

### 3. Huấn luyện Mô hình (Training)
Chạy trực tiếp mô hình **Ours V4**:
```bash
python3 train_sssr.py -opt options/isprs/train_Ours_v4.yml
```
Hoặc khởi chạy thông qua tmux script:
```bash
bash start_training.sh
```

### 4. Đánh giá Chỉ số (Evaluation)
Đo đạc đồng thời PSNR, SSIM, LPIPS và mIoU:
```bash
python3 evaluate_rs_model.py
```

---

## 📜 Trích dẫn (Citation)
Nếu bạn sử dụng mã nguồn này trong nghiên cứu của mình, vui lòng trích dẫn:
```bibtex
@article{tran2026realsr_semantic_diffusion,
  title={RealSR-Semantic-Diffusion: Semantic-Guided Latent Diffusion Model for Remote Sensing Super-Resolution and Semantic Segmentation},
  author={Tran, Vu Quang Huy},
  year={2026}
}
```
