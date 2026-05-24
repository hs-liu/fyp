#!/bin/bash
#SBATCH --job-name=download_llama
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/download_llama_%j.log
#SBATCH --error=logs/download_llama_%j.err
source /vol/bitbucket/hl2622/fyp_venv/bin/activate
export HF_TOKEN =$(cat /vol/bitbucket/hl2622/.secrets/hf_token)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    local_dir='/vol/bitbucket/hl2622/fyp/models/llama-3.1-8b',
    ignore_patterns=['*.bin'],
)
print('Done!')
"

