#!/bin/bash
#SBATCH --job-name=uq_biomistral
#SBATCH --output=/vol/bitbucket/hl2622/fyp/src/logs/uq_exp/biomistral_%j.log
#SBATCH --error=/vol/bitbucket/hl2622/fyp/src/logs/uq_exp/biomistral_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

source /vol/bitbucket/hl2622/fyp_venv/bin/activate
export HF_TOKEN=$(cat /vol/bitbucket/hl2622/.secrets/hf_token)
export HF_HOME=/vol/bitbucket/hl2622/.cache/huggingface
export XET_HOME=/vol/bitbucket/hl2622/xet_cache
export HF_DATASETS_CACHE=/vol/bitbucket/hl2622/huggingface_cache/datasets
cd /vol/bitbucket/hl2622/fyp/src
export PYTHONPATH=/vol/bitbucket/hl2622/fyp
python scripts/uq_experiments/uq_experiment_biomistral.py