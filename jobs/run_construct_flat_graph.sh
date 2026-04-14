#!/bin/bash
#SBATCH --job-name=flat_graph_construction
#SBATCH --output=/vol/bitbucket/hl2622/fyp/logs/flat_graph_%j.log
#SBATCH --error=/vol/bitbucket/hl2622/fyp/logs/flat_graph_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=04:00:00

source /vol/bitbucket/hl2622/fyp_venv/bin/activate
export UMLS_API_KEY=$(cat /vol/bitbucket/hl2622/.secrets/umls_api_key)

cd /vol/bitbucket/hl2622/fyp
python scripts/construct_flat_graph.py