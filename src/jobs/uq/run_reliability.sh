#!/bin/bash
#SBATCH --job-name=uq_reliability
#SBATCH --output=/vol/bitbucket/hl2622/fyp/src/logs/analysis/uq_reliability_%j.log
#SBATCH --error=/vol/bitbucket/hl2622/fyp/src/logs/analysis/uq_reliability_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

source /vol/bitbucket/hl2622/fyp_venv/bin/activate
cd /vol/bitbucket/hl2622/fyp/src
export PYTHONPATH=/vol/bitbucket/hl2622/fyp
python scripts/analysis/reliability_analysis.py