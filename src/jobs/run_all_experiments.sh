#!/bin/bash
#!/bin/bash
#SBATCH --job-name=llama_experiment
#SBATCH --output=/vol/bitbucket/hl2622/fyp/src/logs/init_exp/exp_llama_%j.log
#SBATCH --error=/vol/bitbucket/hl2622/fyp/src/logs/init_exp/exp_llama_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

source /vol/bitbucket/hl2622/fyp_venv/bin/activate
export HF_TOKEN=$(cat /vol/bitbucket/hl2622/.secrets/hf_token)
export HF_HOME=/vol/bitbucket/hl2622/.cache/huggingface
export XET_HOME=/vol/bitbucket/hl2622/xet_cache
export HF_DATASETS_CACHE=/vol/bitbucket/hl2622/huggingface_cache/datasets
cd /vol/bitbucket/hl2622/fyp/src
export PYTHONPATH=/vol/bitbucket/hl2622/fyp

echo "=== RERUNNING ALL EXPERIMENTS WITH FIXED ENTITY LINKING ==="
echo "Started: $(date)"

# ── Baselines (raw model + MedRAG) ────────────────────────
# These don't use retrieval pipeline so don't need rerunning
# Skip raw model and MedRAG

# ── MedHireRAG ────────────────────────────────────────────
echo "Submitting MedHireRAG jobs..."
sbatch src/jobs/experiment/run_experiment_biomistral.sh
sbatch src/jobs/experiment/run_experiment_llama.sh
sbatch src/jobs/experiment/run_experiment_qwen.sh

# ── Ablation ──────────────────────────────────────────────
echo "Submitting ablation jobs..."
sbatch src/jobs/ablation/run_ablation_biomistral.sh
sbatch src/jobs/ablation/run_ablation_llama.sh
sbatch src/jobs/ablation/run_ablation_qwen.sh

# ── UQ ────────────────────────────────────────────────────
echo "Submitting UQ jobs..."
sbatch src/jobs/uq/run_uq_biomistral.sh
sbatch src/jobs/uq/run_uq_llama.sh
sbatch src/jobs/uq/run_uq_qwen.sh

# ── Robustness ────────────────────────────────────────────
echo "Submitting robustness jobs..."
sbatch src/jobs/ablation/run_robustness_analysis.sh

echo "All jobs submitted. Check with: squeue -u hl2622"
echo "Finished submitting: $(date)"