# scripts/analysis/mcnemar_test.py
import os
import pandas as pd
import numpy as np
from scipy.stats import chi2

RESULTS_DIR = "./results"
OUTPUT_DIR  = "./results/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

lines = []
def log(s=""): print(s); lines.append(s)

def mcnemar_test(b, c):
    """
    McNemar's test for paired binary outcomes.
    b = helped (condition A wrong, condition B correct)
    c = hurt   (condition A correct, condition B wrong)
    Returns: chi2_stat, p_value
    """
    if b + c == 0:
        return np.nan, np.nan
    # With continuity correction
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value   = 1 - chi2.cdf(chi2_stat, df=1)
    return chi2_stat, p_value

def load_and_merge(raw_path, comparison_path,
                   raw_col="is_correct",
                   comp_col="is_correct"):
    df_raw  = pd.read_csv(os.path.join(RESULTS_DIR, raw_path))
    df_comp = pd.read_csv(os.path.join(RESULTS_DIR, comparison_path))

    df_raw[raw_col]   = df_raw[raw_col].fillna(False).astype(bool)
    df_comp[comp_col] = df_comp[comp_col].fillna(False).astype(bool)

    df = df_raw[["id", raw_col]].merge(
         df_comp[["id", comp_col]], on="id",
         suffixes=("_raw", "_comp"))
    return df

# ── File paths ─────────────────────────────────────────────
COMPARISONS = [
    # ── MedHireRAG vs Raw Model ────────────────────────────
    ("BioMistral: MedHireRAG vs Raw",
     "baseline/results_local_biomistral.csv",
     "medhirerag/results_biomistral.csv"),

    ("Llama: MedHireRAG vs Raw",
     "baseline/results_llama_local_no_rag.csv",
     "rerun/results_llama.csv"),

    ("Qwen: MedHireRAG vs Raw",
     "baseline/results_qwen_norag.csv",
     "medhirerag/results_qwen.csv"),

    # ── MedHireRAG vs MedRAG ──────────────────────────────
    ("BioMistral: MedHireRAG vs MedRAG",
     "baseline/results_biomistral_medrag.csv",
     "medhirerag/results_biomistral.csv"),
     
    ("Llama: MedHireRAG vs MedRAG",
     "baseline/results_llama_medrag.csv",
     "rerun/results_llama.csv"),
     
    ("Qwen: MedHireRAG vs MedRAG",
     "baseline/results_qwen_medrag.csv",
     "medhirerag/results_qwen.csv"),

    # ── Ablation: Textbook vs Graph-only ──────────────────
    ("BioMistral: Textbook vs Graph",
     "ablation/ablation_biomistral_kg_only.csv",
     "ablation/ablation_biomistral_textbook.csv"),

    ("Llama: Textbook vs Graph",
     "rerun/ablation/results_llama_kg_only.csv",
     "rerun/ablation/results_llama_textbook.csv"),

    ("Qwen: Textbook vs Graph",
     "ablation/ablation_qwen_kg_only.csv",
     "ablation/ablation_qwen_textbook.csv"),

    # ── Ablation: PubMed vs Graph-only ────────────────────
    ("BioMistral: PubMed vs Graph",
     "ablation/ablation_biomistral_kg_only.csv",
     "ablation/ablation_biomistral_pubmed.csv"),

    ("Llama: PubMed vs Graph",
     "rerun/ablation/results_llama_kg_only.csv",
     "rerun/ablation/results_llama_pubmed.csv"),

    ("Qwen: PubMed vs Graph",
     "ablation/ablation_qwen_kg_only.csv",
     "ablation/ablation_qwen_pubmed.csv"),

    # ── Ablation: MedHireRAG vs Graph-only ────────────────
    ("BioMistral: MedHireRAG vs Graph",
     "ablation/ablation_biomistral_kg_only.csv",
     "medhirerag/results_biomistral.csv"),

    ("Llama: MedHireRAG vs Graph",
     "rerun/ablation/results_llama_kg_only.csv",
     "rerun/results_llama.csv"),

    ("Qwen: MedHireRAG vs Graph",
     "ablation/ablation_qwen_kg_only.csv",
     "medhirerag/results_qwen.csv"),

    # ── Ablation: MedHireRAG vs Textbook-only ─────────────
    ("BioMistral: MedHireRAG vs Textbook",
     "ablation/ablation_biomistral_textbook.csv",
     "medhirerag/results_biomistral.csv"),

    ("Llama: MedHireRAG vs Textbook",
     "rerun/ablation/results_llama_textbook.csv",
     "rerun/results_llama.csv"),

    ("Qwen: MedHireRAG vs Textbook",
     "ablation/ablation_qwen_textbook.csv",
     "medhirerag/results_qwen.csv"),

    # ── Ablation: MedHireRAG vs PubMed-only ───────────────
    ("BioMistral: MedHireRAG vs PubMed",
     "ablation/ablation_biomistral_pubmed.csv",
     "medhirerag/results_biomistral.csv"),

    ("Llama: MedHireRAG vs PubMed",
     "rerun/ablation/results_llama_pubmed.csv",
     "rerun/results_llama.csv"),
     
    ("Qwen: MedHireRAG vs PubMed",
     "ablation/ablation_qwen_pubmed.csv",
     "medhirerag/results_qwen.csv"),

    # ── MedRAG vs Raw Model ───────────────────────────────
    ("BioMistral: MedRAG vs Raw",
     "baseline/results_local_biomistral.csv",
     "baseline/results_biomistral_medrag.csv"),
    ("Llama: MedRAG vs Raw",
     "baseline/results_llama_local_no_rag.csv",
     "baseline/results_llama_medrag.csv"),
    ("Qwen: MedRAG vs Raw",
     "baseline/results_qwen_norag.csv",
     "baseline/results_qwen_medrag.csv"),
]

log("=" * 70)
log("McNEMAR'S TEST — PAIRED SIGNIFICANCE")
log("=" * 70)
log(f"\n{'Comparison':<45} {'b':>5} {'c':>5} {'Δpp':>7} {'χ²':>7} {'p':>8} {'sig':>5}")
log("-" * 70)

results = []
for label, raw_file, comp_file in COMPARISONS:
    raw_path  = os.path.join(RESULTS_DIR, raw_file)
    comp_path = os.path.join(RESULTS_DIR, comp_file)

    if not os.path.exists(raw_path):
        log(f"  [MISSING] {raw_path}")
        continue
    if not os.path.exists(comp_path):
        log(f"  [MISSING] {comp_path}")
        continue

    try:
        df_raw  = pd.read_csv(raw_path)
        df_comp = pd.read_csv(comp_path)

        # Standardise correct column
        for df in [df_raw, df_comp]:
            if "is_correct" not in df.columns and "model_answer" in df.columns:
                df["is_correct"] = df["model_answer"] == df["ground_truth"]
            df["is_correct"] = df["is_correct"].fillna(False).astype(bool)

        df = df_raw[["id", "is_correct"]].merge(
             df_comp[["id", "is_correct"]], on="id",
             suffixes=("_raw", "_comp"))

        # Discordant pairs
        b = ((~df["is_correct_raw"]) & (df["is_correct_comp"])).sum()  # helped
        c = ((df["is_correct_raw"])  & (~df["is_correct_comp"])).sum() # hurt

        # Accuracy difference
        acc_raw  = df["is_correct_raw"].mean() * 100
        acc_comp = df["is_correct_comp"].mean() * 100
        delta    = acc_comp - acc_raw

        chi2_stat, p_val = mcnemar_test(b, c)

        if np.isnan(p_val):
            sig = "N/A"
        elif p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "n.s."

        p_str = f"{p_val:.3f}" if not np.isnan(p_val) else "N/A"
        log(f"{label:<45} {b:>5} {c:>5} {delta:>+7.1f} "
            f"{chi2_stat:>7.2f} {p_str:>8} {sig:>5}")

        results.append({
            "comparison": label,
            "helped": b, "hurt": c,
            "delta_pp": delta,
            "chi2": chi2_stat,
            "p_value": p_val,
            "significant": sig
        })

    except Exception as e:
        log(f"  [ERROR] {label}: {e}")

# ── Summary table ──────────────────────────────────────────
log(f"\n{'='*70}")
log("SIGNIFICANCE SUMMARY")
log(f"{'='*70}")
sig_results = [r for r in results if r["significant"] not in ["n.s.", "N/A"]]
ns_results  = [r for r in results if r["significant"] == "n.s."]

log(f"\nSignificant (p<0.05): {len(sig_results)}")
for r in sig_results:
    log(f"  {r['comparison']}: Δ={r['delta_pp']:+.1f}pp, p={r['p_value']:.3f} {r['significant']}")

log(f"\nNot significant (p≥0.05): {len(ns_results)}")
for r in ns_results:
    log(f"  {r['comparison']}: Δ={r['delta_pp']:+.1f}pp, p={r['p_value']:.3f}")

# ── Save ───────────────────────────────────────────────────
pd.DataFrame(results).to_csv(
    f"{OUTPUT_DIR}/mcnemar_results.csv", index=False)
with open(f"{OUTPUT_DIR}/mcnemar_results.txt", "w") as f:
    f.write("\n".join(lines))
print(f"\nSaved → {OUTPUT_DIR}/mcnemar_results.txt")
print("Done!")