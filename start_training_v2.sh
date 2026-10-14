#!/bin/bash
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "======================================================"
echo "  Bắt đầu training OURS V2"
echo "  Dataset: ISPRS Kaggle"
echo "  Target:  50 epochs = 25,000 iterations"
echo "======================================================"

# Run in background via tmux
tmux new-session -d -s ours_v2_train "python3 train_sssr.py -opt options/isprs/train_MTL_kaggle_LDM.yml"

echo "Đã tạo tmux session: ours_v2_train"
echo "Xem log bằng: tmux attach -t ours_v2_train"
