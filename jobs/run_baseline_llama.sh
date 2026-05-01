#!/bin/bash
#SBATCH --job-name=medqa_llama
#SBATCH --output=/vol/bitbucket/hl2622/fyp/logs/llama_%j.log
#SBATCH --error=/vol/bitbucket/hl2622/fyp/logs/llama_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00 

source /vol/bitbucket/hl2622/fyp_venv/bin/activate
export HF_TOKEN=$(cat /vol/bitbucket/hl2622/.secrets/hf_token)
export HF_HOME=/vol/bitbucket/hl2622/.cache/huggingface
export XET_HOME=/vol/bitbucket/hl2622/xet_cache
export HF_DATASETS_CACHE=/vol/bitbucket/hl2622/huggingface_cache/datasets
cd /vol/bitbucket/hl2622/fyp
python scripts/evaluate_baseline_llama.py