#!/bin/bash
#SBATCH --job-name=mcnemar_test
#SBATCH --output=/vol/bitbucket/hl2622/fyp/src/logs/analysis/mcnemar_test_%j.log
#SBATCH --error=/vol/bitbucket/hl2622/fyp/src/logs/analysis/mcnemar_test_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=04:00:00

source /vol/bitbucket/hl2622/fyp_venv/bin/activate

cd /vol/bitbucket/hl2622/fyp/src
export PYTHONPATH=/vol/bitbucket/hl2622/fyp
python scripts/analysis/mcnemar_test.py