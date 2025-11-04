#!/bin/sh
#SBATCH --job-name=seg_0
#SBATCH -p tron
#SBATCH --qos=high  
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --output=./siyuan_out_log/R1_%j.out
#SBATCH --error=./siyuan_out_log/R1_%j_error.out
#SBATCH --account=nexus

if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
conda activate python39

python train.py --cfg configs/lodge/finedance_fea139.yaml --cfg_assets configs/data/assets.yaml  


wait

exit 0
