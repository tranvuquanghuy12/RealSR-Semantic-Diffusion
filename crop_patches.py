import cv2

# Đọc ảnh Low-Res
img = cv2.imread('/home/ptitdemo/TVQH_TLU/Real-SR-master/Ảnh_1_DauVao_LowRes.png')
# Nếu LR bị nhỏ quá (300x300), ta upscale lên 1200x1200 để cắt cho to (bằng NN để giữ nguyên độ mờ)
img = cv2.resize(img, (1200, 1200), interpolation=cv2.INTER_NEAREST)

if img is not None:
    # 1. Xe cộ quá nhỏ (Ví dụ lấy góc phần tư dưới)
    patch1 = img[800:1100, 200:500]
    cv2.imwrite('/home/ptitdemo/TVQH_TLU/Real-SR-master/Patch1_XeCoNho.png', patch1)

    # 2. Ảnh nhiễu mờ (Lấy vùng trung tâm)
    patch2 = img[400:700, 400:700]
    cv2.imwrite('/home/ptitdemo/TVQH_TLU/Real-SR-master/Patch2_NhieuMo.png', patch2)

    # 3. Ranh giới không rõ (Lấy góc trên)
    patch3 = img[100:400, 700:1000]
    cv2.imwrite('/home/ptitdemo/TVQH_TLU/Real-SR-master/Patch3_RanhGoiKhongRo.png', patch3)
    print("Xong!")
