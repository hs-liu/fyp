# MedHireUQRAG

**Boosting Local LLM
Performance on Medical QA via
Uncertainty-Aware Hierarchical
Graph-based Retrieval-Augmented
Generation.**

> Imperial College London — Final Year Project 

---

## Overview

MedHireUQRAG is a RAG framework that combines three-level hierarchical retrieval with uncertainty quantification for safe, deployable clinical decision support. The system retrieves evidence from medical textbooks and PubMed, guided by the UMLS knowledge graph, and assigns calibrated confidence labels to each prediction to enable human-in-the-loop review.

```
Query
  │
  ▼
L1: UMLS Knowledge Graph Traversal
  │
  ▼
L2: Textbook Retrieval (125k chunks)
  │
  │  (L2 content enriches L3 query)
  ▼
L3: PubMed Retrieval (500k chunks)
  │
  ▼
LLM Inference
  │
  ▼
Uncertainty Quantification
  │
  ▼
Output: Answer + Confidence Label (Very Low → Very High)
```

---

## Key Results

| Model | No RAG | MedRAG | MedHireRAG | MedHireUQRAG (T=0.7) |
|-------|--------|--------|------------|----------------------|
| BioMistral-7B | 41.5% | 42.0% | 47.0% | 46.5% (N=20) |
| Llama-3.1-8B | 60.0% | 54.0% | 58.5% | 54.5% (N=10) |
| Qwen2.5-7B | 58.5% | 56.0% | 58.5% | 58.5% (N=10) |

**UQ Selective Accuracy (Very High confidence only, T=0.7):**

| Model | Config | Coverage | Selective Accuracy | Calibration Gap |
|-------|--------|----------|--------------------|-----------------|
| BioMistral-7B | T=0.7, N=20 | 24% | 64.6% | +17.6% |
| Llama-3.1-8B | T=0.7, N=10 | 62% | 71.0% | +12.5% |

---

## Repository Structure

```
fyp/
├── src/scripts/
│   ├── baselines/                 
│   │   ├── evaluate_baseline_biomistral.py
│   │   ├── evaluate_baseline_llama.py
│   │   ├── evaluate_baseline_qwen.py
│   │   ├── evaluate_baseline_biomistral_medrag.py
│   │   ├── evaluate_baseline_llama_medrag.py
│   │   ├── evaluate_baseline_qwen_medrag.py
│   │   └── baseline_utils.py
│   ├── rag/                       
│   │   ├── retrieval_pipeline.py  
│   │   └── retrieval_utils.py     
│   ├── rag_experiments/                       
│   │   ├── experiment_biomistral.py  
│   │   ├── experiment_llama.py
│   │   └── retrieval_qwen.py   
│   ├── uq_experiments/                         
│   │   ├── uq_utils.py             
│   │   ├── uq_experiment_biomistral.py
│   │   ├── uq_experiment_llama.py
│   │   └── uq_experiment_qwen.py
│   ├── eda/                        
│   │   ├── eda_medqa.py
│   │   ├── eda_test_set.py
│   │   ├── eda_umls.py
│   │   ├── eda_textbook.py
│   │   └── eda_pubmed.py
│   ├── knowledge_graph/                        
│   │   ├── construct_umls_graph.py
│   │   ├── link_corpus_to_graph.py
│   │   ├── new_filtered_graph.py
│   │   ├── visualise_embedded_graph.py
│   │   └── visualise_graph_stats.py
│   ├── downloads/                        
│   │   └── download_models.py
│   └── analysis/                   
│       ├── ablation_analysis.py
|       ├── analysis_medhirerag_vs_medhireuqrag.py
│       ├── analysis_medrag_vs_medhirerag.py
│       ├── analysis_raw_medrag_medhirerag_medhireuqrag.py
│       ├── analysis_raw_medrag_medhirerag.py
│       ├── analysis_raw_vs_medhirerag.py
│       ├── analysis_raw_vs_medrag.py
│       ├── analysis_sensitivity.py
│       ├── error_analysis.py
│       └── uq_summary_analysis.py
├── src/jobs/                           
│   ├── ablation/                 
│   ├── baseline/   
│   ├── download/   
│   ├── eda/   
│   ├── experiment/ 
│   ├── graph_construction/
│   └── uq/     
├── src/results/                           
│   ├── ablation/                 
│   ├── analysis/   
│   ├── appendix/   
│   ├── baseline/   
│   ├── eda/ 
│   ├── medhirerag/
│   └── UQ/      
├── corpus/                         
├── MedRAG/                        k
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-enabled GPU (NVIDIA A30 24GB recommended)
- SLURM cluster or local GPU for inference
- HuggingFace account with access to gated models (Llama-3.1)
- UMLS licence (free, register at [https://uts.nlm.nih.gov](https://uts.nlm.nih.gov))

### 1. Clone Repository

```bash
git clone --recurse-submodules https://github.com/hs-liu/fyp.git
cd fyp
```

### 2. Create Virtual Environment

```bash
python3 -m venv fyp_venv
source fyp_venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the project root with your HF_TOKEN, HF_HOME, XET_HOME, HF_DATASET_CACHE and PYTHONPATH

Replace all the paths stored in ./src/jobs bash scripts with your own path

### 4. Download Models

Download all three models using the provided script:

```bash
sbatch src/jobs/download/run_download_models.sh
```

> **Note:** Llama-3.1-8B requires you to accept the licence on HuggingFace before downloading. Visit [https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and request access.

Expected model sizes:

| Model | Size |
|-------|------|
| BioMistral-7B | ~14GB |
| Llama-3.1-8B | ~16GB |
| Qwen2.5-7B | ~15GB |

### 5. Build Retrieval Index

The retrieval index (UMLS graph, corpus, embeddings) must be built before running RAG experiments. These files are not tracked in git due to size.

```bash
# Step 1: Build UMLS knowledge graph
# Requires UMLS 2025AB download from https://www.nlm.nih.gov/research/umls/
# Place MRCONSO.RRF, MRREL.RRF, MRSTY.RRF in data/umls/
sabtch src/jobs/graph_construction/run_construct_flat_graph.sh

# Step 2: Filter the current graph to be more clinical focused 
sabtch src/jobs/graph_construction/run_filter_graph_new.sh

# Step 3: Link corpus chunks to the graph
sabtch src/jobs/graph_construction/run_link_corpus_to_graph.py
```

Expected files after setup:

```
src/data/
├── umls_graph_filtered_new.pkl   
├── corpus_linked.parquet        
└── corpus_embeddings.npy          
```

---

## Running Experiments

All scripts support checkpointing — if interrupted, re-running will resume from where it stopped.

### Baseline: Raw Model

```bash
sabtch src/jobs/baseline/run_baseline_biomistral.sh
sabtch src/jobs/baseline/run_baseline_llama.sh
sabtch src/jobs/baseline/run_baseline_qwen.sh
```

### Baseline: MedRAG

```bash
sabtch src/jobs/baseline/run_baseline_biomistral_medrag.sh
sabtch src/jobs/baseline/run_baseline_llama_medrag.sh
sabtch src/jobs/baseline/run_baseline_qwen_medrag.sh
```

### MedHireUQRAG component 1: MedHireRAG

```bash
sabtch src/jobs/baseline/run_baseline_biomistral_medrag.sh
sabtch src/jobs/baseline/run_baseline_llama_medrag.sh
sabtch src/jobs/baseline/run_baseline_qwen_medrag.sh
```

### MedHireUQRAG component 2: UQ

Temperature and sample size are configured at the top of each script (`TEMPERATURE`, `N_SAMPLES`):

```bash
# Best calibration config (recommended for deployment)
# BioMistral: T=0.7, N=20
sabtch src/jobs/uq/run_uq_biomistral.sh

# Llama: T=0.7, N=10
sabtch src/jobs/uq/run_uq_llama.sh

# Qwen: T=0.7, N=10
sabtch src/jobs/uq/run_uq_qwen.sh
```

### Ablation Study

```bash
# --model: biomistral | llama | qwen
# --mode:  kg_only | textbook | pubmed | both

sabtch src/jobs/ablation/run_ablation_biomistral.sh
sabtch src/jobs/ablation/run_ablation_llama.sh
sabtch src/jobs/ablation/run_ablation_qwen.sh
```
Run other ablation scripts for pairwise comparisons.


## Analysis

Run all analysis scripts after experiments are complete:

```bash
# Ablation analysis
sabtch src/jobs/ablation/run_ablation_analysis.sh

# UQ configuration analysis + calibration curves 
sabtch src/jobs/uq/run_uq_analysis.sh

# Reliability metrics: ECE, AUC 
sabtch src/jobs/uq/run_reliability.sh

# Robustness
sabtch src/jobs/ablation/run_robustness_analysis.sh


# EDA plots 
sabtch src/jobs/eda/run_eda_umls.sh
sabtch src/jobs/eda/run_eda_textbook.sh
sabtch src/jobs/eda/run_eda_pubmed.sh
sabtch src/jobs/eda/run_eda_medqa.sh
```

Output plots are saved to `graphs/` and results to `results/`.

---

## Confidence Labels

MedHireUQRAG assigns a confidence label to each prediction based on the consistency score (fraction of N samples agreeing with the majority vote answer):

| Consistency Score | Confidence Label | Recommended Policy |
|-------------------|------------------|--------------------|
| ≥ 0.9 | **Very High** | Auto-answer |
| [0.7, 0.9) | **High** | Flag for clinician review |
| [0.5, 0.7) | **Medium** | Defer to clinician |
| [0.3, 0.5) | **Low** | Defer to clinician |
| < 0.3 | **Very Low** | Defer to clinician |

> The human review policy is validated for BioMistral-7B (T=0.7, N=20) and Llama-3.1-8B (T=0.7, N=10) only. Qwen2.5-7B exhibits insufficient calibration signal for selective abstention.

---

## Results Files

All experiment results are saved as CSV to `results/`:

```
results/
├── baselines/
│   ├── results_local_biomistral.csv                    
│   ├── results_llama_local_no_rag.csv
│   ├── results_qwen_norag.csv
│   ├── results_biomistral_medrag.csv
│   ├── results_llama_medrag.csv                  
│   └── results_qwen_medrag.csv
├── medhirerag/
│   ├── results_biomistral.csv                
│   ├── results_llama.csv
│   └── results_qwen.csv
├── UQ/
│   ├── results_biomistral_medhireuqrag_*.csv 
│   ├── results_llama_medhireuqrag_*.csv
│   └── results_qwen_medhireuqrag_*.csv
└── ablation/
    ├── ablation_biomistral_kg_only.csv
    ├── ablation_biomistral_textbook.csv
    └── ...
```
