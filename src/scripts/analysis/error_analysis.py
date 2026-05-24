# analysis_error.py
"""
Analysis 3: Error analysis — where RAG helps vs hurts vs no change.
Outputs:
  - results/analysis/error_analysis.txt
  - graphs/analysis/error_*.png
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR  = "./results"
GRAPHS_DIR   = "./graphs/analysis"
SUMMARY_PATH = "./results/analysis/error_analysis.txt"
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs("./results/analysis", exist_ok=True)

lines = []
def log(s=""):
    print(s)
    lines.append(s)

# ── Model pairs (no RAG vs my RAG) ────────────────────────
pairs = {
    "BioMistral": (
        "results_biomistral.csv",
        "results_biomistral_myrag_v5_6.csv",
    ),
    "Llama": (
        "results_llama_local_no_rag.csv",
        "results_llama_myrag_v4.csv",
    ),
    "Qwen": (
        "results_qwen_norag.csv",
        "results_qwen_myrag.csv",
    ),
}

def load(fname):
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    df = pd.read_csv(path)
    df["is_correct"] = df["is_correct"].fillna(False).astype(bool)
    return df

log("=" * 60)
log("ERROR ANALYSIS: RAG HELPS vs HURTS vs NO CHANGE")
log("=" * 60)

summary_rows = []

for model_name, (norag_file, rag_file) in pairs.items():
    df_norag = load(norag_file)
    df_rag   = load(rag_file)
    if df_norag is None or df_rag is None:
        continue

    df = df_norag[["id","question","ground_truth","is_correct"]].merge(
        df_rag[["id","is_correct","domain","source_route"]],
        on="id", suffixes=("_norag","_rag")
    )

    rag_helped  = df[(df["is_correct_rag"]==True)  & (df["is_correct_norag"]==False)]
    rag_hurt    = df[(df["is_correct_rag"]==False) & (df["is_correct_norag"]==True)]
    both_right  = df[(df["is_correct_rag"]==True)  & (df["is_correct_norag"]==True)]
    both_wrong  = df[(df["is_correct_rag"]==False) & (df["is_correct_norag"]==False)]

    n = len(df)
    log(f"\n{'='*40}")
    log(f"Model: {model_name}")
    log(f"{'='*40}")
    log(f"  Total questions:  {n}")
    log(f"  RAG helped:       {len(rag_helped):>3} ({len(rag_helped)/n*100:.1f}%)")
    log(f"  RAG hurt:         {len(rag_hurt):>3} ({len(rag_hurt)/n*100:.1f}%)")
    log(f"  Both correct:     {len(both_right):>3} ({len(both_right)/n*100:.1f}%)")
    log(f"  Both wrong:       {len(both_wrong):>3} ({len(both_wrong)/n*100:.1f}%)")
    log(f"  Net RAG effect:   {(len(rag_helped)-len(rag_hurt)):+d} questions")

    summary_rows.append({
        "model":       model_name,
        "rag_helped":  len(rag_helped),
        "rag_hurt":    len(rag_hurt),
        "both_right":  len(both_right),
        "both_wrong":  len(both_wrong),
        "net":         len(rag_helped) - len(rag_hurt),
    })

    # Domain breakdown of where RAG helped/hurt
    if "domain" in df_rag.columns:
        log(f"\n  Domain breakdown — RAG helped:")
        if len(rag_helped) > 0:
            for domain, cnt in rag_helped["domain"].value_counts().items():
                log(f"    {domain:<30} {cnt}")
        log(f"\n  Domain breakdown — RAG hurt:")
        if len(rag_hurt) > 0:
            for domain, cnt in rag_hurt["domain"].value_counts().items():
                log(f"    {domain:<30} {cnt}")

    # Sample questions where RAG helped
    log(f"\n  Sample questions where RAG HELPED ({model_name}):")
    for _, row in rag_helped.head(3).iterrows():
        log(f"    Q: {row['question'][:100]}...")
        log(f"    Domain: {row.get('domain','?')}")
        log()

    # Sample questions where RAG hurt
    log(f"\n  Sample questions where RAG HURT ({model_name}):")
    for _, row in rag_hurt.head(3).iterrows():
        log(f"    Q: {row['question'][:100]}...")
        log(f"    Domain: {row.get('domain','?')}")
        log()

# ── Plot 1: stacked bar — helped/hurt/both per model ──────
fig, ax = plt.subplots(figsize=(10, 6))
model_names = [r["model"] for r in summary_rows]
helped  = [r["rag_helped"]  for r in summary_rows]
hurt    = [r["rag_hurt"]    for r in summary_rows]
b_right = [r["both_right"]  for r in summary_rows]
b_wrong = [r["both_wrong"]  for r in summary_rows]

x = np.arange(len(model_names))
w = 0.5

p1 = ax.bar(x, b_right, w, label="Both correct",  color="#5cb85c", edgecolor="white")
p2 = ax.bar(x, helped,  w, bottom=b_right,         label="RAG helped", color="steelblue", edgecolor="white")
p3 = ax.bar(x, b_wrong, w, bottom=[a+b for a,b in zip(b_right, helped)],
            label="Both wrong", color="lightgray", edgecolor="white")
p4 = ax.bar(x, hurt,    w,
            bottom=[a+b+c for a,b,c in zip(b_right, helped, b_wrong)],
            label="RAG hurt", color="#d9534f", edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=12)
ax.set_ylabel("Number of questions", fontsize=12)
ax.set_title("RAG Effect Breakdown per Model", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.set_ylim(0, 220)
ax.grid(axis="y", alpha=0.3)

# Annotate net effect
for i, r in enumerate(summary_rows):
    net = r["net"]
    ax.text(i, 205, f"net {net:+d}", ha="center", fontsize=10, fontweight="bold",
            color="steelblue" if net >= 0 else "#d9534f")

plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/error_rag_effect_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"\nSaved → {GRAPHS_DIR}/error_rag_effect_breakdown.png")

# ── Plot 2: helped vs hurt by domain (all models combined) ─
all_helped_domains = []
all_hurt_domains   = []

for model_name, (norag_file, rag_file) in pairs.items():
    df_norag = load(norag_file)
    df_rag   = load(rag_file)
    if df_norag is None or df_rag is None:
        continue
    df = df_norag[["id","is_correct"]].merge(
        df_rag[["id","is_correct","domain"]],
        on="id", suffixes=("_norag","_rag")
    )
    helped = df[(df["is_correct_rag"]==True)  & (df["is_correct_norag"]==False)]
    hurt   = df[(df["is_correct_rag"]==False) & (df["is_correct_norag"]==True)]
    if "domain" in df.columns:
        all_helped_domains.extend(helped["domain"].tolist())
        all_hurt_domains.extend(hurt["domain"].tolist())

if all_helped_domains or all_hurt_domains:
    all_domains = sorted(set(all_helped_domains + all_hurt_domains))
    helped_counts = pd.Series(all_helped_domains).value_counts()
    hurt_counts   = pd.Series(all_hurt_domains).value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    x   = np.arange(len(all_domains))
    w   = 0.35
    h_v = [helped_counts.get(d, 0) for d in all_domains]
    hu_v= [hurt_counts.get(d, 0)   for d in all_domains]

    ax.bar(x - w/2, h_v,  w, label="RAG helped", color="steelblue", edgecolor="white")
    ax.bar(x + w/2, hu_v, w, label="RAG hurt",   color="#d9534f",   edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(all_domains, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Number of questions (all models)", fontsize=12)
    ax.set_title("RAG Helps vs Hurts by Domain (All Models Combined)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/error_domain_helped_hurt.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved → {GRAPHS_DIR}/error_domain_helped_hurt.png")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines))
log(f"\nSaved → {SUMMARY_PATH}")
print("Done!")