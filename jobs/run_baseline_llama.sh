#!/bin/bash
#SBATCH --job-name=medqa_llama
#SBATCH --output=/vol/bitbucket/hl2622/fyp/logs/llama_%j.log
#SBATCH --error=/vol/bitbucket/hl2622/fyp/logs/llama_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00 

source /vol/bitbucket/hl2622/fyp_venv/bin/activate
export HF_TOKEN=$(cat /vol/bitbucket/hl2622/.secrets/hf_token)
cd /vol/bitbucket/hl2622/fyp
python scripts/evaluate_baseline_llama.py