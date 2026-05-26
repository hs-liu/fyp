#!/bin/bash
#SBATCH --job-name=abl_llama
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/vol/bitbucket/hl2622/fyp/src/logs/albation/llama_%j.log
#SBATCH --error=/vol/bitbucket/hl2622/fyp/src/logs/albation/llama_%j.err

export HF_HOME=/vol/bitbucket/hl2622/huggingface_cache
export XET_HOME=/vol/bitbucket/hl2622/xet_cache
export HF_DATASETS_CACHE=/vol/bitbucket/hl2622/huggingface_cache/datasets

cd /vol/bitbucket/hl2622/fyp/src
source /vol/bitbucket/hl2622/fyp_venv/bin/activate

python3 scripts/albation_experiments/albation_exp.py --model llama --mode kg_only
python3 scripts/albation_experiments/albation_exp.py --model llama --mode textbook
python3 scripts/albation_experiments/albation_exp.py --model llama --mode pubmed
python3 scripts/albation_experiments/albation_exp.py --model llama --mode no_classifier
