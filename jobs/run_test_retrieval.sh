#!/bin/bash
#SBATCH --job-name=test_retrieval
#SBATCH --output=/vol/bitbucket/hl2622/fyp/logs/test_retrieval_%j.log
#SBATCH --error=/vol/bitbucket/hl2622/fyp/logs/test_retrieval_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

source /vol/bitbucket/hl2622/fyp_venv/bin/activate
export HF_HOME=/vol/bitbucket/hl2622/huggingface_cache
cd /vol/bitbucket/hl2622/fyp
python scripts/rag/test_retrieval.py
