#!/bin/bash
#SBATCH --job-name=albation_analysis
#SBATCH --output=/vol/bitbucket/hl2622/fyp/logs/analysis/albation_analysis_%j.log
#SBATCH --error=/vol/bitbucket/hl2622/fyp/logs/analysis/albation_analysis_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=04:00:00

source /vol/bitbucket/hl2622/fyp_venv/bin/activate

cd /vol/bitbucket/hl2622/fyp
export PYTHONPATH=/vol/bitbucket/hl2622/fyp
python scripts/analysis/albation_analysis.py