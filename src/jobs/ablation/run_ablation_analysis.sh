#!/bin/bash
#SBATCH --job-name=ablation_analysis
#SBATCH --output=/vol/bitbucket/hl2622/fyp/src/logs/analysis/ablation_analysis_%j.log
#SBATCH --error=/vol/bitbucket/hl2622/fyp/src/logs/analysis/ablation_analysis_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=04:00:00

source /vol/bitbucket/hl2622/fyp_venv/bin/activate

cd /vol/bitbucket/hl2622/fyp/src
export PYTHONPATH=/vol/bitbucket/hl2622/fyp
python scripts/analysis/ablation_analysis.py