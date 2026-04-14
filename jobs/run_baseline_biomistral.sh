#!/bin/bash
#SBATCH --job-name=medqa_biomistral
#SBATCH --output=/vol/bitbucket/hl2622/fyp/logs/biomistral_%j.log
#SBATCH --error=/vol/bitbucket/hl2622/fyp/logs/biomistral_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

source /vol/bitbucket/hl2622/fyp_venv/bin/activate
export HF_HOME=/vol/bitbucket/hl2622/huggingface_cache
export TRANSFORMERS_CACHE=/vol/bitbucket/hl2622/huggingface_cache
export HF_DATASETS_CACHE=/vol/bitbucket/hl2622/huggingface_cache
export HF_TOKEN=$(cat /vol/bitbucket/hl2622/.secrets/hf_token)

cd /vol/bitbucket/hl2622/fyp
python scripts/evaluate_baseline_biomistral.py
