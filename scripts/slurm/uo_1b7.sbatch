#!/bin/bash
#SBATCH --job-name=uo_qwen3_1b7
#SBATCH --partition=spgpu
#SBATCH --account=YOUR_ACCOUNT       # edit: e.g. huterer0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/uo_1b7_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiamingp@umich.edu

# ---- environment ----
module load cuda
conda activate quant           # edit: your env name

REPO=/path/to/llm-quant-experiments   # edit: your cluster path
cd $REPO

mkdir -p logs results/figures

# ---- run ----
python scripts/run_uo_detection.py \
    --config configs/qwen3_1b7.yaml \
    --output results/uo_candidates_1b7.json

echo "Done: $(date)"
