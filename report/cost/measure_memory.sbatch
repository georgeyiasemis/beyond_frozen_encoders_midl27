#!/bin/bash
#SBATCH --job-name=fm_cost_mem
#SBATCH --account=aifo
#SBATCH --partition=a100
#SBATCH --qos=four_a100_qos
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/home/g.yiasemis/projects/cvpr_ct_challenge/midl_paper_v2/report/cost/measure_memory_%j.out
set -euo pipefail
cd /home/g.yiasemis/projects/cvpr_ct_challenge/aifofomo
source /home/g.yiasemis/miniconda3/etc/profile.d/conda.sh
conda activate aifofomo-fm-probe
export PYTHONPATH="/home/g.yiasemis/projects/cvpr_ct_challenge/aifofomo:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv
python /home/g.yiasemis/projects/cvpr_ct_challenge/midl_paper_v2/report/cost/measure_memory.py --out /home/g.yiasemis/projects/cvpr_ct_challenge/midl_paper_v2/report/cost/memory.csv
