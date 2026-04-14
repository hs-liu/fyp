#!/bin/bash
#SBATCH --job-name=filter_graph_new
#SBATCH --output=/vol/bitbucket/hl2622/fyp/logs/filter_graph_new_%j.log
#SBATCH --error=/vol/bitbucket/hl2622/fyp/logs/filter_graph_new_%j.err
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00

source /vol/bitbucket/hl2622/fyp_venv/bin/activate
export UMLS_API_KEY=$(cat /vol/bitbucket/hl2622/.secrets/umls_api_key)

cd /vol/bitbucket/hl2622/fyp
python scripts/new_filtered_graph.py