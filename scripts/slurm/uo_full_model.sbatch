#!/bin/bash
#SBATCH --job-name=uo_full
#SBATCH --partition=spgpu
#SBATCH --account=huterer0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/jiamingp/llm-quant-experiments/logs/uo_full_model_%j.out
#SBATCH --error=/home/jiamingp/llm-quant-experiments/logs/uo_full_model_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiamingp@umich.edu

set -euo pipefail

PROJECT_DIR=/home/jiamingp/llm-quant-experiments
PYTHON=/home/jiamingp/.conda/envs/quant/bin/python

cd "${PROJECT_DIR}"
mkdir -p logs results/figures

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_DIR}/.mplconfig"
export HF_HOME=/scratch/huterer_root/huterer0/jiamingp/hf_cache
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "${MPLCONFIGDIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

"${PYTHON}" scripts/run_uo_v1.py --config configs/uo_v1.yaml
"${PYTHON}" scripts/run_uo_full_model.py --K 20 --ruleset B
