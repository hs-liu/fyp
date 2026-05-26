#!/bin/bash
#SBATCH --job-name=eda_medqa
#SBATCH --partition=a30
#SBATCH --output=/vol/bitbucket/hl2622/fyp/src/logs/eda/eda_medqa_%j.log
#SBATCH --error=/vol/bitbucket/hl2622/fyp/src/logs/eda/eda_medqa_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

source /vol/bitbucket/hl2622/fyp_venv/bin/activate
export HF_HOME=/vol/bitbucket/hl2622/huggingface_cache
export HF_DATASETS_CACHE=/vol/bitbucket/hl2622/huggingface_cache
export XET_HOME=/vol/bitbucket/hl2622/xet_cache
export HF_TOKEN=$(cat /vol/bitbucket/hl2622/.secrets/hf_token)

cd /vol/bitbucket/hl2622/fyp/src
export PYTHONPATH=/vol/bitbucket/hl2622/fyp
python scripts/eda/eda_medqa.py