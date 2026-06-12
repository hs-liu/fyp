#!/bin/bash
#SBATCH --job-name=retrieve_context
#SBATCH --output=/vol/bitbucket/hl2622/fyp/src/logs/analysis/retrieve_context_%j.log
#SBATCH --error=/vol/bitbucket/hl2622/fyp/src/logs/analysis/retrieve_context_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=04:00:00

source /vol/bitbucket/hl2622/fyp_venv/bin/activate

cd /vol/bitbucket/hl2622/fyp/src
export PYTHONPATH=/vol/bitbucket/hl2622/fyp
python scripts/analysis/retrieve_examples.py