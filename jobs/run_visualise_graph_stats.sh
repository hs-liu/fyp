#!/bin/bash
#SBATCH --job-name=graph_stats
#SBATCH --output=/vol/bitbucket/hl2622/fyp/logs/graph_stats_%j.log
#SBATCH --error=/vol/bitbucket/hl2622/fyp/logs/graph_stats_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00

source /vol/bitbucket/hl2622/fyp_venv/bin/activate

cd /vol/bitbucket/hl2622/fyp
python scripts/knowledge_graph/visualise_graph_stats.py