#!/bin/bash
#SBATCH --job-name=link_corpus_to_graph
#SBATCH --output=/vol/bitbucket/hl2622/fyp/logs/corpus_%j.log
#SBATCH --error=/vol/bitbucket/hl2622/fyp/logs/corpus_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

source /vol/bitbucket/hl2622/fyp_venv/bin/activate
export HF_HOME=/vol/bitbucket/hl2622/huggingface_cache
export HF_DATASETS_CACHE=/vol/bitbucket/hl2622/huggingface_cache
cd /vol/bitbucket/hl2622/fyp
python scripts/link_corpus_to_graph.py