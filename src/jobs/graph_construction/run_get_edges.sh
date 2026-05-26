#!/bin/bash
#SBATCH --job-name=graph_construction
#SBATCH --output=/vol/bitbucket/hl2622/fyp/src/logs/get_edges_%j.log
#SBATCH --error=/vol/bitbucket/hl2622/fyp/src/logs/get_edges_%j.err
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00

source /vol/bitbucket/hl2622/fyp_venv/bin/activate
export UMLS_API_KEY=$(cat /vol/bitbucket/hl2622/.secrets/umls_api_key)

cd /vol/bitbucket/hl2622/fyp/src
export PYTHONPATH=/vol/bitbucket/hl2622/fyp
python scripts/get_edges.py