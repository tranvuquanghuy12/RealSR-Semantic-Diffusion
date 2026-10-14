#!/bin/bash
CODES_DIR="/home/ptitdemo/TVQH_TLU/Real-SR-master/Real-SR-master/codes"
VENV="$CODES_DIR/venv/bin/activate"

echo "======================================================"
echo "  Bắt đầu training OURS V1"
echo "  Dataset: ISPRS Kaggle"
echo "  Target:  30 epochs = 15,000 iterations"
echo "======================================================"

tmux new-session -d -s ours_v1_train -x 220 -y 50
tmux send-keys -t ours_v1_train "
cd $CODES_DIR && source $VENV
export PYTHONPATH=$CODES_DIR:\$PYTHONPATH
export WANDB_MODE=offline
echo '=== OURS V1: bắt đầu train ==='
python3 train_sssr.py -opt options/isprs/train_Ours_v1.yml 2>&1 | tee -a train_ours_v1.log
echo '=== OURS V1: HOÀN THÀNH ==='
" Enter

echo "Đã tạo tmux session: ours_v1_train"
echo "Xem log bằng: tmux attach -t ours_v1_train"
