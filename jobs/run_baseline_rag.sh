#!/bin/bash
#SBATCH --job-name=medqa_rag
#SBATCH --output=/vol/bitbucket/hl2622/fyp/logs/groq_rag_%j.log
#SBATCH --error=/vol/bitbucket/hl2622/fyp/logs/groq_rag_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:30:00 

source /vol/bitbucket/hl2622/fyp_venv/bin/activate
export GROQ_API_KEY=$(cat /vol/bitbucket/hl2622/.secrets/groq_api_key)
cd /vol/bitbucket/hl2622/fyp
python scripts/evaluate_baseline_rag.py