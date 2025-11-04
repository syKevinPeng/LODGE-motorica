#!/bin/sh
#SBATCH --job-name=LODGE_R1
#SBATCH -p tron
#SBATCH --qos=high  
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxa5000:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --mem=128GB
#SBATCH --output=./siyuan_out_log/R1_%j.out
#SBATCH --error=./siyuan_out_log/R1_%j_error.out
#SBATCH --account=nexus
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peng2000@umd.edu

if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
conda activate python39

# Don't use srun - PyTorch Lightning will handle process spawning
python train.py --cfg configs/lodge/finedance_fea139.yaml --cfg_assets configs/data/assets.yaml  


wait

exit 0
