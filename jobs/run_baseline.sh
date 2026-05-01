#!/bin/bash
#SBATCH --job-name=medqa_baseline
#SBATCH --output=/vol/bitbucket/hl2622/fyp/logs/baseline_%j.log
#SBATCH --error=/vol/bitbucket/hl2622/fyp/logs/baseline_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

echo "========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Job started: $(date)"
echo "Running on:  $(hostname)"
echo "GPU:         $CUDA_VISIBLE_DEVICES"
echo "========================================="

# ── Activate Environment ─────────────────────
source /vol/bitbucket/hl2622/fyp_venv/bin/activate

# ── Set Cache ────────────────────────────────
export HF_HOME=/vol/bitbucket/hl2622/huggingface_cache
export HF_DATASETS_CACHE=/vol/bitbucket/hl2622/huggingface_cache
export GEMINI_API_KEY='AIzaSyD-iEL8MCWrGf1L-yIWneDxbNxwE1rTTr0'
# ── Run Script ───────────────────────────────
cd /vol/bitbucket/hl2622/fyp
python scripts/evaluate_baseline.py

echo "========================================="
echo "Job finished: $(date)"
echo "========================================="
