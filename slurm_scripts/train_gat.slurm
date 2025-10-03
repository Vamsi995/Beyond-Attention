#!/bin/bash
#SBATCH --job-name="salisettigpushared"
#SBATCH --output="salisettigpu.%j.%N.out"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=1
#SBATCH --account=csb193
#SBATCH --time 10:00:00

export TORCH_EXTENSIONS_DIR="/expanse/lustre/projects/csb193/salisetti"

source /expanse/lustre/projects/csb193/salisetti/Beyond-Attention/VirtualEnv/sptemp/bin/activate

module reset
module load cuda11.7/toolkit

cd /expanse/lustre/projects/csb193/salisetti/Beyond-Attention
torchrun --nproc_per_node=4 weather2k_forecasting.py