# scripts/analysis/error_analysis.py
import os
import pandas as pd

RESULTS_DIR = "./results"
OUTPUT_DIR  = "./results/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

lines = []
def log(s=""): print(s); lines.append(s)

# ── Focus: Llama-3.1-8B (primary) ─────────────────────────
model_name = "Llama-3.1-8B"

df_raw    = pd.read_csv(f"{RESULTS_DIR}/baseline/results_llama_local_no_rag.csv")
df_rag    = pd.read_csv(f"{RESULTS_DIR}/rerun/results_llama.csv")
df_medrag = pd.read_csv(f"{RESULTS_DIR}/baseline/results_llama_medrag.csv")

# Columns: id, question, ground_truth, raw_answer, model_answer, is_correct
# Rename for clarity before merging
df_raw    = df_raw.rename(columns={
    "model_answer": "answer_raw",
    "is_correct":   "correct_raw"
})
df_rag    = df_rag.rename(columns={
    "model_answer": "answer_rag",
    "is_correct":   "correct_rag"
})
df_medrag = df_medrag.rename(columns={
    "model_answer": "answer_medrag",
    "is_correct":   "correct_medrag"
})

# Merge all three on id
df = df_raw[["id", "question", "ground_truth",
             "answer_raw", "correct_raw"]].merge(
     df_rag[["id", "answer_rag", "correct_rag"]], on="id").merge(
     df_medrag[["id", "answer_medrag", "correct_medrag"]], on="id")

df["correct_raw"]    = df["correct_raw"].fillna(False).astype(bool)
df["correct_rag"]    = df["correct_rag"].fillna(False).astype(bool)
df["correct_medrag"] = df["correct_medrag"].fillna(False).astype(bool)

# ── Comparison 1: MedHireRAG vs Raw Model ─────────────────
df["helped_vs_raw"] = (~df["correct_raw"]) & (df["correct_rag"])
df["hurt_vs_raw"]   =  (df["correct_raw"]) & (~df["correct_rag"])

# ── Comparison 2: MedHireRAG vs MedRAG ────────────────────
df["helped_vs_medrag"] = (~df["correct_medrag"]) & (df["correct_rag"])
df["hurt_vs_medrag"]   =  (df["correct_medrag"]) & (~df["correct_rag"])

log(f"{'='*70}")
log(f"ERROR ANALYSIS — {model_name}")
log(f"{'='*70}")
log(f"\nMedHireRAG vs Raw Model:")
log(f"  Helped: {df['helped_vs_raw'].sum()}")
log(f"  Hurt:   {df['hurt_vs_raw'].sum()}")
log(f"  Net:    {df['helped_vs_raw'].sum() - df['hurt_vs_raw'].sum():+d}")
log(f"\nMedHireRAG vs MedRAG:")
log(f"  Helped: {df['helped_vs_medrag'].sum()}")
log(f"  Hurt:   {df['hurt_vs_medrag'].sum()}")
log(f"  Net:    {df['helped_vs_medrag'].sum() - df['hurt_vs_medrag'].sum():+d}")

# ── Print HURT cases (MedHireRAG vs Raw) ──────────────────
log(f"\n{'─'*70}")
log("HURT CASES — MedHireRAG vs Raw Model")
log("(answered correctly without RAG, wrong with MedHireRAG)")
log(f"{'─'*70}")

for i, (_, row) in enumerate(df[df["hurt_vs_raw"]].iterrows(), 1):
    log(f"\n[HURT {i}]")
    log(f"Q:  {row['question'][:500]}")
    log(f"GT: {row['ground_truth']}")
    log(f"Raw answer:      {row['answer_raw']}")
    log(f"MedHireRAG:      {row['answer_rag']}")
    log(f"MedRAG:          {row['answer_medrag']}")
    log("-" * 50)

# ── Print HELPED cases (MedHireRAG vs Raw) ────────────────
log(f"\n{'─'*70}")
log("HELPED CASES — MedHireRAG vs Raw Model")
log("(answered incorrectly without RAG, correct with MedHireRAG)")
log(f"{'─'*70}")

for i, (_, row) in enumerate(df[df["helped_vs_raw"]].iterrows(), 1):
    log(f"\n[HELPED {i}]")
    log(f"Q:  {row['question'][:500]}")
    log(f"GT: {row['ground_truth']}")
    log(f"Raw answer:      {row['answer_raw']}")
    log(f"MedHireRAG:      {row['answer_rag']}")
    log(f"MedRAG:          {row['answer_medrag']}")
    log("-" * 50)

# ── Print MedHireRAG helped but MedRAG failed ─────────────
log(f"\n{'─'*70}")
log("MedHireRAG SUCCEEDED, MedRAG FAILED")
log("(key cases showing KG-guided retrieval advantage)")
log(f"{'─'*70}")

kg_advantage = df[df["helped_vs_medrag"] & ~df["correct_medrag"]]
for i, (_, row) in enumerate(kg_advantage.iterrows(), 1):
    log(f"\n[KG ADVANTAGE {i}]")
    log(f"Q:  {row['question'][:500]}")
    log(f"GT: {row['ground_truth']}")
    log(f"Raw answer:      {row['answer_raw']}")
    log(f"MedRAG:          {row['answer_medrag']}")
    log(f"MedHireRAG:      {row['answer_rag']}")
    log("-" * 50)

# ── Save ───────────────────────────────────────────────────
with open(f"{OUTPUT_DIR}/error_analysis_llama.txt", "w") as f:
    f.write("\n".join(lines))
print(f"Saved → {OUTPUT_DIR}/error_analysis_llama.txt")

# ── Optional: BioMistral for KG-specific example ──────────
log(f"\n{'='*70}")
log("BioMistral-7B — KG advantage cases only")
log(f"{'='*70}")

df_bm_raw    = pd.read_csv(f"{RESULTS_DIR}/baseline/results_local_biomistral.csv")
df_bm_rag    = pd.read_csv(f"{RESULTS_DIR}/rerun/results_biomistral.csv")
df_bm_medrag = pd.read_csv(f"{RESULTS_DIR}/baseline/results_biomistral_medrag.csv")

df_bm_raw    = df_bm_raw.rename(columns={
    "model_answer": "answer_raw", "is_correct": "correct_raw"})
df_bm_rag    = df_bm_rag.rename(columns={
    "model_answer": "answer_rag", "is_correct": "correct_rag"})
df_bm_medrag = df_bm_medrag.rename(columns={
    "model_answer": "answer_medrag", "is_correct": "correct_medrag"})

df_bm = df_bm_raw[["id", "question", "ground_truth",
                    "answer_raw", "correct_raw"]].merge(
        df_bm_rag[["id", "answer_rag", "correct_rag"]], on="id").merge(
        df_bm_medrag[["id", "answer_medrag", "correct_medrag"]], on="id")

for col in ["correct_raw", "correct_rag", "correct_medrag"]:
    df_bm[col] = df_bm[col].fillna(False).astype(bool)

df_bm["kg_advantage"] = (~df_bm["correct_medrag"]) & (df_bm["correct_rag"])

for i, (_, row) in enumerate(df_bm[df_bm["kg_advantage"]].head(5).iterrows(), 1):
    log(f"\n[BM KG ADVANTAGE {i}]")
    log(f"Q:  {row['question'][:500]}")
    log(f"GT: {row['ground_truth']}")
    log(f"Raw:       {row['answer_raw']}")
    log(f"MedRAG:    {row['answer_medrag']}")
    log(f"MedHireRAG:{row['answer_rag']}")
    log("-" * 50)

with open(f"{OUTPUT_DIR}/error_analysis_full.txt", "w") as f:
    f.write("\n".join(lines))
print(f"Saved → {OUTPUT_DIR}/error_analysis_full.txt")
print("Done!")