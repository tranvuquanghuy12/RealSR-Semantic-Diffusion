#!/bin/bash
# =============================================================================
# Script khởi động training 3 phiên bản trong tmux
# - Session 1: paper_baseline  (Bản gốc - LDM không có cải tiến)
# - Session 2: ours_v1         (Đã train 60k - CHỈ HIỂN THỊ kết quả, không train lại)
# - Session 3: ours_v2         (Bản cải tiến: DINOv2 + DPT + FFT Loss)
#
# Cách dùng: bash start_training.sh
# =============================================================================

CODES_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$CODES_DIR/venv/bin/activate"

echo "======================================================"
echo "  Bắt đầu training 3 phiên bản"
echo "  Dataset: ISPRS Kaggle"
echo "  Target:  50 epochs = 25,000 iterations"
echo "======================================================"

# Hàm xóa checkpoint cũ (chỉ giữ checkpoint mới nhất)
cleanup_fn='
cleanup_old_ckpts() {
    STATE_DIR="$1"
    MODELS_DIR="$2"
    # Lấy iteration mới nhất
    LATEST=$(ls "$STATE_DIR"/*.state 2>/dev/null | sort -t/ -k2 -V | tail -1)
    if [ -z "$LATEST" ]; then return; fi
    LATEST_ITER=$(basename "$LATEST" .state)
    # Xóa tất cả state cũ hơn
    for f in "$STATE_DIR"/*.state; do
        ITER=$(basename "$f" .state)
        if [ "$ITER" != "$LATEST_ITER" ]; then
            rm -f "$f"
            echo "[Cleanup] Xóa $f"
        fi
    done
    # Xóa tất cả model file cũ hơn
    for f in "$MODELS_DIR"/*_*.pth; do
        ITER=$(basename "$f" | cut -d_ -f1)
        if [ "$ITER" != "$LATEST_ITER" ]; then
            rm -f "$f"
            echo "[Cleanup] Xóa $f"
        fi
    done
}
'

# ==============================================================
# SESSION 1: Paper Baseline (LDM thuần, không DINOv2/DPT/FFT)
# ==============================================================
echo "[1/3] Tạo tmux session: paper_baseline"
tmux new-session -d -s paper_baseline -x 220 -y 50
tmux send-keys -t paper_baseline "
cd $CODES_DIR && source $VENV
export PYTHONPATH=$CODES_DIR:\$PYTHONPATH
export WANDB_MODE=offline
echo '=== PAPER BASELINE: bắt đầu train ==='
python3 train_sssr.py -opt options/isprs/train_Paper_Baseline.yml 2>&1 | tee -a train_paper_baseline.log
echo '=== PAPER BASELINE: HOÀN THÀNH ==='
" Enter

# ==============================================================
# SESSION 2: Ours V1 (LDM v1 + SimpleEncoder/Decoder)
# ==============================================================
echo "[2/3] Tạo tmux session: ours_v1 (Bắt đầu train)"
tmux new-session -d -s ours_v1 -x 220 -y 50
tmux send-keys -t ours_v1 "
cd $CODES_DIR && source $VENV
export PYTHONPATH=$CODES_DIR:\$PYTHONPATH
export WANDB_MODE=offline
echo '=== OURS V1: bắt đầu train ==='
python3 train_sssr.py -opt options/isprs/train_Ours_v1.yml 2>&1 | tee -a train_ours_v1.log
echo '=== OURS V1: HOÀN THÀNH ==='
" Enter

# ==============================================================
# SESSION 3: Ours V2 (DINOv2 + DPT + FFT Loss)
# ==============================================================
echo "[3/3] Tạo tmux session: ours_v2"
tmux new-session -d -s ours_v2 -x 220 -y 50
tmux send-keys -t ours_v2 "
cd $CODES_DIR && source $VENV
export PYTHONPATH=$CODES_DIR:\$PYTHONPATH
export WANDB_MODE=offline
echo '=== OURS V2 (DINOv2+DPT+FFT): bắt đầu train ==='
echo 'Lưu ý: DINOv2 sẽ download lần đầu (~350MB)'
python3 train_sssr.py -opt options/isprs/train_Ours_v2.yml 2>&1 | tee -a train_ours_v2.log
echo '=== OURS V2: HOÀN THÀNH ==='
" Enter

echo ""
echo "======================================================"
echo "  Tất cả sessions đã khởi động!"
echo ""
echo "  Xem sessions:   tmux ls"
echo "  Vào session 1:  tmux attach -t paper_baseline"
echo "  Vào session 2:  tmux attach -t ours_v1"
echo "  Vào session 3:  tmux attach -t ours_v2"
echo "  Thoát session:  Ctrl+B rồi D"
echo ""
echo "  Log files:"
echo "    train_paper_baseline.log"
echo "    train_ours_v2.log"
echo "======================================================"
