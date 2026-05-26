#!/bin/bash
#SBATCH --job-name=jupyter_awq
#SBATCH --partition=spgpu
#SBATCH --account=huterer0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/jiamingp/llm-quant-experiments/logs/jupyter_awq_%j.out
#SBATCH --error=/home/jiamingp/llm-quant-experiments/logs/jupyter_awq_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiamingp@umich.edu

set -euo pipefail

PROJECT_DIR=/home/jiamingp/llm-quant-experiments
NOTEBOOK_DIR=${PROJECT_DIR}/notebooks
INFO_FILE=${PROJECT_DIR}/jupyter_info_awq.txt
PYTHON=/home/jiamingp/.conda/envs/quant/bin/python
JUPYTER=/home/jiamingp/.conda/envs/quant/bin/jupyter

cd "${PROJECT_DIR}"
mkdir -p logs results
unset XDG_RUNTIME_DIR

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME=/scratch/huterer_root/huterer0/jiamingp/hf_cache
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export MPLCONFIGDIR=${PROJECT_DIR}/.mplconfig
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${MPLCONFIGDIR}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing Python env: ${PYTHON}" >&2
  exit 1
fi

if [[ ! -x "${JUPYTER}" ]]; then
  echo "Jupyter is not installed in the quant env. Install it with:" >&2
  echo "${PYTHON} -m pip install notebook ipykernel" >&2
  exit 1
fi

"${PYTHON}" - <<'PY'
import sys
import torch
import awq
import weight_handles
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("awq:", awq.__file__)
print("weight_handles:", weight_handles.__file__)
PY

nport=$(shuf -i 8000-9000 -n1)
ip=$(hostname -i | awk '{print $1}')
hn=$(hostname)
me=$(whoami)

cat > "${INFO_FILE}" <<EOT
Jupyter notebook running on Great Lakes host ${hn}.

Copy this command into your local Mac terminal:
ssh -N -L ${nport}:${ip}:${nport} ${me}@greatlakes.arc-ts.umich.edu

Then open:
http://localhost:${nport}

Notebook directory:
${NOTEBOOK_DIR}

Suggested notebook:
${NOTEBOOK_DIR}/awq_qwen3_1b7_smoke.ipynb
EOT

echo "Wrote connection info to ${INFO_FILE}"
cat "${INFO_FILE}"

cd "${NOTEBOOK_DIR}"
exec "${JUPYTER}" notebook \
  --no-browser \
  --port="${nport}" \
  --ip=0.0.0.0 \
  --NotebookApp.token='' \
  --NotebookApp.password=''
