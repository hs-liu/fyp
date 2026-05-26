# MedHireUQRAG

**Hierarchical Graph-Guided Retrieval-Augmented Generation with Uncertainty Quantification for Medical Question Answering**

> Imperial College London — Final Year Project (2024/25)

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
| BioMistral-7B | 41.5% | — | 47.0% | 46.5% (N=20) |
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
fyp/src
├── scripts/
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
├── jobs/                           
│   ├── ablation/                 
│   │   ├── run
│   │   ├── evaluate_baseline_llama.py
│   │   ├── evaluate_baseline_qwen.py
│   │   ├── evaluate_baseline_biomistral_medrag.py
│   │   ├── evaluate_baseline_llama_medrag.py
│   │   ├── evaluate_baseline_qwen_medrag.py
│   │   └── baseline_utils.py annotations
│   └── corpus_embeddings.npy       # 625k × 768 embeddings
├── corpus/                         # Raw textbook chunks + FAISS index
├── results/                        # Experiment results (CSV)
├── graphs/                         # Generated plots
├── jobs/                           # SLURM job scripts
├── notebooks/                      # Exploratory notebooks
├── MedRAG/                         # MedRAG submodule
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
pip install --upgrade pip
pip install -r requirements.txt

# Additional packages not in requirements.txt
pip install sentence-transformers faiss-cpu networkx scipy scikit-learn \
            matplotlib python-dotenv pyarrow
```

### 3. Environment Variables

Create a `.env` file in the project root:

```bash
cat > .env << 'EOF'
HF_TOKEN=your_huggingface_token_here
EOF
```

### 4. Download Models

Download all three models using the provided script:

```bash
bash scripts/setup/download_models.sh
```

Or download individually:

```bash
# BioMistral-7B (no gate — public)
huggingface-cli download BioMistral/BioMistral-7B \
    --local-dir models/biomistral-7b \
    --token $HF_TOKEN

# Llama-3.1-8B (requires HF access request)
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir models/llama-3.1-8b \
    --token $HF_TOKEN

# Qwen2.5-7B (no gate — public)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir models/qwen2.5-7b \
    --token $HF_TOKEN
```

> **Note:** Llama-3.1-8B requires you to accept the licence on HuggingFace before downloading. Visit [https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and request access.

Expected model sizes:

| Model | Size |
|-------|------|
| BioMistral-7B | ~14GB |
| Llama-3.1-8B-Instruct | ~16GB |
| Qwen2.5-7B-Instruct | ~15GB |

### 5. Build Retrieval Index

The retrieval index (UMLS graph, corpus, embeddings) must be built before running RAG experiments. These files are not tracked in git due to size.

```bash
# Step 1: Build UMLS knowledge graph
# Requires UMLS 2025AB download from https://www.nlm.nih.gov/research/umls/
# Place MRCONSO.RRF, MRREL.RRF, MRSTY.RRF in data/umls/
python3 scripts/data/build_umls_graph.py

# Step 2: Link corpus chunks to CUIs
python3 scripts/data/link_corpus.py

# Step 3: Generate corpus embeddings (GPU required, ~2 hours)
python3 scripts/data/generate_embeddings.py
```

Expected files after setup:

```
data/
├── umls_graph_filtered_new.pkl   # ~2GB  — 2.6M nodes, 22M edges
├── corpus_linked.parquet          # ~633MB — 625k chunks with CUI annotations
└── corpus_embeddings.npy          # ~1.8GB — 625k × 768 float32 embeddings
```

---

## Running Experiments

All scripts support checkpointing — if interrupted, re-running will resume from where it stopped.

### Baseline: No RAG

```bash
python3 scripts/baselines/baseline_biomistral.py
python3 scripts/baselines/baseline_llama.py
python3 scripts/baselines/baseline_qwen.py
```

### Baseline: MedRAG

```bash
python3 scripts/baselines/baseline_biomistral_medrag.py
python3 scripts/baselines/baseline_llama_medrag.py
python3 scripts/baselines/baseline_qwen_medrag.py
```

### MedHireRAG

```bash
python3 scripts/rag/baseline_biomistral_myrag.py
python3 scripts/rag/baseline_llama_myrag.py
python3 scripts/rag/baseline_qwen_myrag.py
```

### MedHireUQRAG

Temperature and sample size are configured at the top of each script (`TEMPERATURE`, `N_SAMPLES`):

```bash
# Best calibration config (recommended for deployment)
# BioMistral: T=0.7, N=20
python3 scripts/uq/uq_biomistral_myrag.py

# Llama: T=0.7, N=10
python3 scripts/uq/uq_llama_myrag.py

# Qwen: T=0.7, N=10
python3 scripts/uq/uq_qwen_myrag.py
```

### Ablation Study

```bash
# --model: biomistral | llama | qwen
# --mode:  kg_only | textbook | pubmed | both

python3 scripts/ablation/ablation_experiment.py --model biomistral --mode kg_only
python3 scripts/ablation/ablation_experiment.py --model llama --mode textbook
python3 scripts/ablation/ablation_experiment.py --model qwen --mode pubmed
python3 scripts/ablation/ablation_experiment.py --model llama --mode both
```

### SLURM (Cluster)

```bash
# Submit individual jobs
sbatch jobs/job_biomistral_myrag.sh
sbatch jobs/job_llama_myrag.sh
sbatch jobs/job_qwen_myrag.sh

# Submit all ablation jobs
for model in biomistral llama qwen; do
    for mode in kg_only textbook pubmed both; do
        sbatch jobs/job_ablation.sh $model $mode
    done
done
```

---

## Analysis

Run all analysis scripts after experiments are complete:

```bash
# Ablation progression plots (Section 5.3)
python3 scripts/analysis/analysis_ablation_progressive.py

# UQ configuration analysis + calibration curves (Section 5.5)
python3 scripts/analysis/analysis_uq_configs.py

# Reliability metrics: ECE, AUC (Section 5.5)
python3 scripts/analysis/analysis_reliability.py

# All four methods comparison (Section 5.6)
python3 scripts/analysis/analysis_all_four.py

# Error analysis — all pairwise comparisons
for f in scripts/analysis/error_analysis/error_*.py; do
    python3 "$f"
done

# EDA plots (Chapter 3)
python3 scripts/eda/eda_medqa.py
python3 scripts/eda/eda_test_set.py
python3 scripts/eda/eda_umls.py
python3 scripts/eda/eda_textbook.py
python3 scripts/eda/eda_pubmed.py
```

Output plots are saved to `graphs/` and results to `results/analysis/`.

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
├── results_biomistral.csv                    # No RAG
├── results_llama_local_no_rag.csv
├── results_qwen_norag.csv
├── results_llama_medrag.csv                  # MedRAG
├── results_qwen_medrag.csv
├── medhirerag/
│   ├── results_biomistral.csv                # MedHireRAG
│   ├── results_llama.csv
│   └── results_qwen.csv
├── UQ/
│   ├── results_biomistral_medhireuqrag_*.csv # MedHireUQRAG
│   ├── results_llama_medhireuqrag_*.csv
│   └── results_qwen_medhireuqrag_*.csv
└── ablation/
    ├── ablation_biomistral_kg_only.csv
    ├── ablation_biomistral_textbook.csv
    └── ...
```

UQ result CSVs contain the following columns:

| Column | Description |
|--------|-------------|
| `id` | Question index (0–199) |
| `question` | Question text |
| `ground_truth` | Correct answer (A–E) |
| `greedy_answer` | Deterministic answer |
| `greedy_correct` | Whether greedy answer is correct |
| `uq_answer` | Majority vote answer |
| `uq_correct` | Whether majority vote is correct |
| `uq_consistency` | Fraction of samples agreeing with majority |
| `uq_entropy` | Predictive entropy across answer options |
| `confidence_label` | Very Low / Low / Medium / High / Very High |
| `n_valid` | Number of valid samples |

---

## Dependencies

```
pandas==2.2.2
datasets==2.20.0
transformers==4.44.2
openai==1.57.0
accelerate==0.33.0
torch==2.4.0
ipykernel
httpx==0.27.2
sentence-transformers
faiss-cpu
networkx
scipy
scikit-learn
matplotlib
python-dotenv
pyarrow
```

---

## Hardware Requirements

| Task | GPU | VRAM | Time (est.) |
|------|-----|------|-------------|
| Embedding generation | A30 | 24GB | ~2 hours |
| BioMistral inference (200 samples) | A30 | 24GB | ~30 min |
| Llama-3.1-8B inference | A30 | 24GB | ~45 min |
| Qwen2.5-7B inference | A30 | 24GB | ~45 min |
| UQ (N=20, T=0.7) | A30 | 24GB | ~6 hours |

---

## Citation

If you use this work, please cite:

```bibtex
@misc{liu2025medhireuqrag,
  author = {Liu, Hao Sheng},
  title  = {MedHireUQRAG: Hierarchical Graph-Guided Retrieval-Augmented
             Generation with Uncertainty Quantification for Medical QA},
  year   = {2025},
  school = {Imperial College London},
  note   = {BEng Computing Final Year Project}
}
```

---

## Acknowledgements

- [MedRAG](https://github.com/gzxiong/MedRAG) — flat RAG baseline (included as submodule)
- [UMLS](https://www.nlm.nih.gov/research/umls/) — knowledge graph (requires free licence)
- [MedQA](https://github.com/jind11/MedQA) — evaluation benchmark
- [BioMistral](https://huggingface.co/BioMistral/BioMistral-7B) — domain-specific medical LLM
- [MedCPT](https://huggingface.co/ncbi/MedCPT-Query-Encoder) — medical query encoder
- Imperial College London — GPU cluster infrastructure (SLURM, NVIDIA A30)