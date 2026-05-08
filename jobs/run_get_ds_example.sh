#!/bin/bash
#SBATCH --job-name=dataset_example
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/dataset/dataset_example_%j.log
#SBATCH --error=logs/dataset/dataset_example_%j.err
source /vol/bitbucket/hl2622/fyp_venv/bin/activate
export HF_TOKEN =$(cat /vol/bitbucket/hl2622/.secrets/hf_token)
python3 -c "
import datasets
dataset = datasets.load_dataset("bigbio/med_qa", "med_qa_en_source", trust_remote_code=True)
test_ds = list(dataset["test"])[:N_TEST]
print(test_ds[1])
"

