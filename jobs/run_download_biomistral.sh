#!/bin/bash
#SBATCH --job-name=dl_biomistral
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/download_biomistral_%j.log
#SBATCH --error=logs/download_biomistral_%j.err

source /vol/bitbucket/hl2622/fyp_venv/bin/activate
export HF_TOKEN=$(cat /vol/bitbucket/hl2622/.secrets/hf_token)
export HF_HOME=/vol/bitbucket/hl2622/.cache/huggingface
export XET_HOME=/vol/bitbucket/hl2622/xet_cache
export HF_DATASETS_CACHE=/vol/bitbucket/hl2622/huggingface_cache/datasets
mkdir -p $XET_HOME

cd /vol/bitbucket/hl2622/fyp
python scripts/downloads/download_biomistral.py