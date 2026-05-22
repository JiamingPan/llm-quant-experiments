#!/bin/bash
#SBATCH --job-name=uo_qwen3_8b
#SBATCH --partition=spgpu
#SBATCH --account=YOUR_ACCOUNT       # edit: e.g. huterer0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/uo_8b_%j.out
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
    --config configs/qwen3_8b.yaml \
    --output results/uo_candidates_8b.json

echo "Done: $(date)"
