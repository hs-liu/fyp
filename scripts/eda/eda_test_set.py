# eda_medqa_testset.py
"""
EDA on the 200 MedQA test samples used in experiments.
Covers: answer distribution, question length, domain distribution,
option length, and difficulty proxy.
"""
import os
import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR  = "./results/appendix"
GRAPHS_DIR   = "./graphs/appendix/medqa_testset"
SUMMARY_PATH = f"{RESULTS_DIR}/eda_medqa_testset_500_summary.txt"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR,  exist_ok=True)

# ── Load domain labels from classifier results ─────────────
DOMAIN_FILES = {
    "BioMistral": "./results/results_biomistral_myrag_v5_6.csv",
    "Llama":      "./results/results_llama_myrag_v4.csv",
    "Qwen":       "./results/results_qwen_myrag.csv",
}

lines = []
def log(s=""): print(s); lines.append(s)

log("=" * 60)
log("MEDQA TEST SET EDA (500 samples)")
log("=" * 60)

# ── Load dataset ───────────────────────────────────────────
print("Loading MedQA...")
dataset = datasets.load_dataset(
    "bigbio/med_qa", "med_qa_en_source", trust_remote_code=True
)
test_ds = list(dataset["test"])[:500]

# Build dataframe
rows = []
for i, s in enumerate(test_ds):
    options    = s.get("options", [])
    answer_idx = s["answer_idx"]

    # Find the correct answer text
    ans_text = ""
    for opt in options:
        if opt.get("key") == answer_idx:
            ans_text = opt.get("value", "")
            break

    rows.append({
        "id":          i,
        "question":    s["question"],
        "answer_idx":  answer_idx,
        "q_length":    len(s["question"].split()),
        "q_chars":     len(s["question"]),
        "n_options":   len(options),
        "opt_lengths": [len(opt.get("value", "").split()) for opt in options],
        "ans_length":  len(ans_text.split()),
    })

df = pd.DataFrame(rows)
df["mean_opt_length"] = df["opt_lengths"].apply(np.mean)

log(f"\nTotal samples:     {len(df)}")
log(f"Unique answers:    {df['answer_idx'].nunique()}")
log(f"\nAnswer distribution:")
for ans, cnt in df["answer_idx"].value_counts().sort_index().items():
    log(f"  {ans}: {cnt} ({cnt/len(df)*100:.1f}%)")

log(f"\nQuestion length (words):")
log(f"  Mean:   {df['q_length'].mean():.1f}")
log(f"  Median: {df['q_length'].median():.1f}")
log(f"  Std:    {df['q_length'].std():.1f}")
log(f"  Min:    {df['q_length'].min()}")
log(f"  Max:    {df['q_length'].max()}")

# ── Load domain labels ─────────────────────────────────────
# Use majority vote across all three model classifier results
domain_dfs = []
for model_name, fpath in DOMAIN_FILES.items():
    if os.path.exists(fpath):
        d = pd.read_csv(fpath)[["id","domain"]].dropna()
        d = d.rename(columns={"domain": f"domain_{model_name}"})
        domain_dfs.append(d)

if domain_dfs:
    domain_merged = domain_dfs[0]
    for d in domain_dfs[1:]:
        domain_merged = domain_merged.merge(d, on="id", how="outer")

    # Majority vote domain per question
    domain_cols = [c for c in domain_merged.columns if c.startswith("domain_")]
    def majority_domain(row):
        vals = [row[c] for c in domain_cols if pd.notna(row[c]) and row[c] != ""]
        if not vals:
            return "Unknown"
        return pd.Series(vals).mode()[0]

    domain_merged["domain"] = domain_merged.apply(majority_domain, axis=1)
    df = df.merge(domain_merged[["id","domain"]], on="id", how="left")
    df["domain"] = df["domain"].fillna("Unknown")

    log(f"\nDomain distribution (majority vote across models):")
    domain_counts = df["domain"].value_counts()
    for d, cnt in domain_counts.items():
        log(f"  {d:<35} {cnt:>3} ({cnt/len(df)*100:.1f}%)")
else:
    df["domain"] = "Unknown"
    log("\n  [No domain files found]")

# ══════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════

""" # Plot 1: Answer distribution
fig, ax = plt.subplots(figsize=(8, 6))
ans_counts = df["answer_idx"].value_counts().sort_index()
bars = ax.bar(ans_counts.index, ans_counts.values,
              color="steelblue", edgecolor="white", width=0.6)
for bar, val in zip(bars, ans_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
            f"{val}\n({val/len(df)*100:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xlabel("Answer option", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Answer Option Distribution — 500 Test Samples",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0, ans_counts.max() * 1.2)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/01_answer_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"\nSaved → {GRAPHS_DIR}/01_answer_distribution.png")
 """
# Plot 2: Question length distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df["q_length"], bins=30, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(df["q_length"].mean(), color="red", linestyle="--", linewidth=2,
           label=f"Mean = {df['q_length'].mean():.1f}")
ax.axvline(df["q_length"].median(), color="orange", linestyle="--", linewidth=2,
           label=f"Median = {df['q_length'].median():.1f}")
ax.set_xlabel("Question length (words)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Question Length Distribution",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=11)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/02_question_length.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"Saved → {GRAPHS_DIR}/02_question_length.png")

# Plot 3: Domain distribution
if df["domain"].nunique() > 1:
    fig, ax = plt.subplots(figsize=(12, 7))
    domain_counts = df[df["domain"] != "Unknown"]["domain"].value_counts()
    bars = ax.barh(domain_counts.index[::-1], domain_counts.values[::-1],
                   color="steelblue", edgecolor="white")
    ax.set_xlim(0, domain_counts.max() * 1.18)
    for bar, val in zip(bars, domain_counts.values[::-1]):
        ax.text(bar.get_width() + domain_counts.max()*0.01,
                bar.get_y() + bar.get_height()/2,
                f"{val} ({val/len(df)*100:.1f}%)",
                va="center", ha="left", fontsize=9, fontweight="bold")
    ax.set_xlabel("Number of questions", fontsize=12)
    ax.set_title("Domain Distribution",
                 fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/03_domain_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved → {GRAPHS_DIR}/03_domain_distribution.png")

""" # Plot 4: Question length by domain
if df["domain"].nunique() > 1:
    domains_to_plot = [d for d in df["domain"].value_counts().index
                       if d != "Unknown"][:8]
    data = [df[df["domain"]==d]["q_length"].tolist() for d in domains_to_plot]

    fig, ax = plt.subplots(figsize=(13, 7))
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    colors = plt.cm.Set2(np.linspace(0, 1, len(domains_to_plot)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticks(range(1, len(domains_to_plot)+1))
    ax.set_xticklabels([d[:25] for d in domains_to_plot],
                       rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Question length (words)", fontsize=12)
    ax.set_title("Question Length by Domain — 200 Test Samples",
                 fontsize=13, fontweight="bold", pad=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/04_question_length_by_domain.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved → {GRAPHS_DIR}/04_question_length_by_domain.png")

# Plot 5: Mean option length distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df["mean_opt_length"], bins=25, color="orange", edgecolor="white", alpha=0.85)
ax.axvline(df["mean_opt_length"].mean(), color="red", linestyle="--", linewidth=2,
           label=f"Mean = {df['mean_opt_length'].mean():.1f}")
ax.axvline(df["mean_opt_length"].median(), color="blue", linestyle="--", linewidth=2,
           label=f"Median = {df['mean_opt_length'].median():.1f}")
ax.set_xlabel("Mean option length (words)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Answer Option Length Distribution — 200 Test Samples",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=11)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/05_option_length.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"Saved → {GRAPHS_DIR}/05_option_length.png")

# Plot 6: Question length vs answer correctness proxy
# Use No RAG results as difficulty proxy — wrong = harder question
norag_file = "./results/results_llama_local_no_rag.csv"
if os.path.exists(norag_file):
    df_norag = pd.read_csv(norag_file)
    df_norag["is_correct"] = df_norag["is_correct"].fillna(False).astype(bool)
    df = df.merge(df_norag[["id","is_correct"]], on="id", how="left")

    fig, ax = plt.subplots(figsize=(10, 6))
    correct   = df[df["is_correct"]==True]["q_length"]
    incorrect = df[df["is_correct"]==False]["q_length"]
    ax.hist(correct,   bins=25, alpha=0.7, color="steelblue",
            label=f"Correct (n={len(correct)})", edgecolor="white")
    ax.hist(incorrect, bins=25, alpha=0.7, color="lightcoral",
            label=f"Incorrect (n={len(incorrect)})", edgecolor="white")
    ax.axvline(correct.mean(),   color="steelblue",  linestyle="--", linewidth=2,
               label=f"Correct mean = {correct.mean():.1f}")
    ax.axvline(incorrect.mean(), color="lightcoral", linestyle="--", linewidth=2,
               label=f"Incorrect mean = {incorrect.mean():.1f}")
    ax.set_xlabel("Question length (words)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Question Length vs Correctness (Llama No RAG)\n"
                 "Longer questions ≈ harder?",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/06_length_vs_correctness.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved → {GRAPHS_DIR}/06_length_vs_correctness.png")

    log(f"\nQuestion length vs correctness (Llama No RAG):")
    log(f"  Correct   — mean={correct.mean():.1f}, median={correct.median():.1f}")
    log(f"  Incorrect — mean={incorrect.mean():.1f}, median={incorrect.median():.1f}")
 """
# ── Summary ────────────────────────────────────────────────
log(f"\n{'='*60}")
log("SUMMARY")
log(f"{'='*60}")
log(f"  Total samples:         500")
log(f"  Q length mean:         {df['q_length'].mean():.1f} words")
log(f"  Q length range:        {df['q_length'].min()}–{df['q_length'].max()} words")
log(f"  Mean option length:    {df['mean_opt_length'].mean():.1f} words")
log(f"  Unique domains:        {df['domain'].nunique()}")
log(f"  Most common domain:    {df['domain'].value_counts().index[0]} "
    f"({df['domain'].value_counts().iloc[0]/len(df)*100:.1f}%)")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines))
log(f"\nSaved → {SUMMARY_PATH}")
print("Done!")