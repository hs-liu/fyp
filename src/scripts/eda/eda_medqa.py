# eda_and_baseline.py
"""
EDA on MedQA test set (200 samples) + heuristic baselines.
Each plot saved as a separate file in src/graphs/eda/medqa/
"""
import os, random
import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

RESULTS_DIR  = "./results/eda"
GRAPHS_DIR   = "./graphs/eda/medqa"
SUMMARY_PATH = f"{RESULTS_DIR}/eda_medqa_summary.txt"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

N_TEST = 200
SEED   = 42
random.seed(SEED)
np.random.seed(SEED)

print("Loading MedQA...")
dataset   = datasets.load_dataset("bigbio/med_qa", "med_qa_en_source", trust_remote_code=True)
test_ds   = list(dataset["test"])[:N_TEST]
full_test = list(dataset["test"])
train_ds  = list(dataset["train"])
val_ds    = list(dataset["validation"])

lines = []
def log(s=""):
    print(s)
    lines.append(s)

# ── 1. Dataset overview ────────────────────────────────────
log("=" * 60)
log("MEDQA DATASET OVERVIEW")
log("=" * 60)
log(f"Full test set size:       {len(full_test):,}")
log(f"Train set size:           {len(train_ds):,}")
log(f"Validation set size:      {len(val_ds):,}")
log(f"Subset used (N_TEST):     {N_TEST}")
log()

# ── 2. Justify N=200 ──────────────────────────────────────
log("=" * 60)
log("JUSTIFICATION FOR N=200")
log("=" * 60)
log(f"Full test set: {len(full_test)} questions")
log(f"N=200 covers:  {N_TEST/len(full_test)*100:.1f}% of test set")
log()
log("Statistical justification:")
log("  At 95% confidence, margin of error for proportion p=0.5:")
for n in [50, 100, 200, 500, len(full_test)]:
    moe = 1.96 * np.sqrt(0.5 * 0.5 / n)
    log(f"  N={n:5d}: ±{moe:.3f} ({moe*100:.1f}%)")
log()
log("N=200 gives ±6.9% margin of error at 95% confidence.")
log("Practical reason: each model run takes 2-4 hrs on A30.")
log(f"Full test ({len(full_test)}) would take ~{len(full_test)/N_TEST*3:.0f}x longer.")
log()

# ── 3. Answer distribution ────────────────────────────────
log("=" * 60)
log("ANSWER DISTRIBUTION (N=200 subset)")
log("=" * 60)
answer_counts      = Counter(s["answer_idx"] for s in test_ds)
full_answer_counts = Counter(s["answer_idx"] for s in full_test)
options            = sorted(set(s["answer_idx"] for s in full_test))

for k in sorted(answer_counts):
    pct = answer_counts[k] / N_TEST * 100
    log(f"  {k}: {answer_counts[k]:3d} ({pct:.1f}%)")
log()
log("ANSWER DISTRIBUTION (full test set)")
for k in sorted(full_answer_counts):
    pct = full_answer_counts[k] / len(full_test) * 100
    log(f"  {k}: {full_answer_counts[k]:3d} ({pct:.1f}%)")
log()

# ── 4. Question length stats ──────────────────────────────
log("=" * 60)
log("QUESTION LENGTH STATISTICS (N=200)")
log("=" * 60)
q_lengths  = [len(s["question"].split()) for s in test_ds]
opt_counts = [len(s["options"]) for s in test_ds]
log(f"  Mean question length:   {np.mean(q_lengths):.1f} words")
log(f"  Median question length: {np.median(q_lengths):.1f} words")
log(f"  Std question length:    {np.std(q_lengths):.1f} words")
log(f"  Min question length:    {np.min(q_lengths)} words")
log(f"  Max question length:    {np.max(q_lengths)} words")
log(f"  Mean options per q:     {np.mean(opt_counts):.1f}")
log()

# ── 5. Sample questions ───────────────────────────────────
log("=" * 60)
log("SAMPLE QUESTIONS")
log("=" * 60)
for i in [0, 50, 100, 150, 199]:
    s = test_ds[i]
    log(f"  [{i}] Question: {s['question']}")
    log(f"       Options:")
    for opt in s["options"]:
        marker = " ← correct" if opt["key"] == s["answer_idx"] else ""
        log(f"         {opt['key']}. {opt['value']}{marker}")
    log(f"       Answer: {s['answer_idx']}")
    log()

# ── 6. Heuristic baselines ────────────────────────────────
log("=" * 60)
log("HEURISTIC BASELINES")
log("=" * 60)
gt_labels      = [s["answer_idx"] for s in test_ds]
most_common    = answer_counts.most_common(1)[0][0]
n_options      = len(options)
theoretical    = 1 / n_options

random_preds   = [random.choice(options) for _ in gt_labels]
random_acc     = sum(p == g for p, g in zip(random_preds, gt_labels)) / N_TEST
majority_acc   = sum(g == most_common for g in gt_labels) / N_TEST
weights        = [answer_counts[o] / N_TEST for o in options]
weighted_preds = random.choices(options, weights=weights, k=N_TEST)
weighted_acc   = sum(p == g for p, g in zip(weighted_preds, gt_labels)) / N_TEST
always_a_acc   = sum(g == "A" for g in gt_labels) / N_TEST

log(f"Random (uniform):        {random_acc:.2%}")
log(f"Always predict '{most_common}':      {majority_acc:.2%}")
log(f"Weighted random:         {weighted_acc:.2%}")
log(f"Always predict 'A':      {always_a_acc:.2%}")
log(f"Theoretical random:      {theoretical:.2%} (1/{n_options})")
log()
log("Any model >20% has learned something.")
log("Any model >25% beats majority class baseline.")

heuristic_df = pd.DataFrame([
    {"method": "Random uniform",          "accuracy": random_acc},
    {"method": f"Always '{most_common}'", "accuracy": majority_acc},
    {"method": "Weighted random",         "accuracy": weighted_acc},
    {"method": "Always 'A'",              "accuracy": always_a_acc},
    {"method": "Theoretical (1/5)",       "accuracy": theoretical},
])
heuristic_df.to_csv(f"{RESULTS_DIR}/heuristic_baselines.csv", index=False)
log(f"\nSaved → {RESULTS_DIR}/heuristic_baselines.csv")

# ══════════════════════════════════════════════════════════
# PLOTS — one per file
# ══════════════════════════════════════════════════════════

# Plot 1: Answer distribution subset vs full
fig, ax = plt.subplots(figsize=(10, 6))
x  = sorted(answer_counts.keys())
bw = 0.35
subset_vals = [answer_counts[k] / N_TEST * 100 for k in x]
full_vals   = [full_answer_counts[k] / len(full_test) * 100 for k in x]
ax.bar([i - bw/2 for i in range(len(x))], subset_vals, bw,
       label=f"Subset (n={N_TEST})", color="steelblue")
ax.bar([i + bw/2 for i in range(len(x))], full_vals, bw,
       label=f"Full test (n={len(full_test)})", color="orange")
ax.set_xticks(range(len(x)))
ax.set_xticklabels(x, fontsize=12)
ax.set_xlabel("Answer option", fontsize=12)
ax.set_ylabel("Percentage (%)", fontsize=12)
ax.set_title("Answer Distribution: 200 Subset vs Full Test Set", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/01_answer_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/01_answer_distribution.png")

# Plot 2: Question length histogram
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(q_lengths, bins=25, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(np.mean(q_lengths), color="red", linestyle="--",
           linewidth=2, label=f"Mean = {np.mean(q_lengths):.0f} words")
ax.axvline(np.median(q_lengths), color="orange", linestyle="--",
           linewidth=2, label=f"Median = {np.median(q_lengths):.0f} words")
ax.set_xlabel("Question length (words)", fontsize=12)
ax.set_ylabel("Number of questions", fontsize=12)
ax.set_title("Question Length Distribution (N=200)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/02_question_length.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/02_question_length.png")

# Plot 3: Heuristic baselines
fig, ax = plt.subplots(figsize=(10, 6))
methods = heuristic_df["method"].tolist()
accs    = heuristic_df["accuracy"].tolist()
colors  = ["#d9534f" if a < 0.25 else "#5cb85c" for a in accs]
bars    = ax.barh(methods, [a * 100 for a in accs], color=colors, edgecolor="white")
ax.set_xlabel("Accuracy (%)", fontsize=12)
ax.set_title("Heuristic Baseline Accuracies", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.set_xlim(0, 35)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{acc:.1%}", va="center", fontsize=11, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/03_heuristic_baselines.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/03_heuristic_baselines.png")

# Plot 4: Margin of error vs N
fig, ax = plt.subplots(figsize=(10, 6))
ns   = range(10, len(full_test) + 1, 10)
moes = [1.96 * np.sqrt(0.25 / n) * 100 for n in ns]
ax.plot(list(ns), moes, color="steelblue", linewidth=2)
ax.axvline(N_TEST, color="red", linestyle="--", linewidth=2, label=f"N={N_TEST} (chosen)")
ax.axhline(1.96 * np.sqrt(0.25 / N_TEST) * 100, color="orange", linestyle="--",
           linewidth=2, label=f"MoE = ±{1.96*np.sqrt(0.25/N_TEST)*100:.1f}%")
ax.fill_between(list(ns), moes, alpha=0.1, color="steelblue")
ax.set_xlabel("Sample size N", fontsize=12)
ax.set_ylabel("Margin of error (%)", fontsize=12)
ax.set_title("Statistical Power: Margin of Error vs Sample Size", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/04_margin_of_error.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/04_margin_of_error.png")

# Plot 5: Dataset split sizes
fig, ax = plt.subplots(figsize=(10, 6))
splits = ["Train", "Validation", "Test (full)", f"Test (used, n={N_TEST})"]
sizes  = [len(train_ds), len(val_ds), len(full_test), N_TEST]
colors = ["steelblue", "orange", "green", "red"]
bars   = ax.bar(splits, sizes, color=colors, edgecolor="white", width=0.5)
ax.set_ylabel("Number of questions", fontsize=12)
ax.set_title("MedQA Dataset Split Sizes", fontsize=14, fontweight="bold")
for bar, val in zip(bars, sizes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
            f"{val:,}", ha="center", fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(sizes) * 1.15)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/05_dataset_splits.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/05_dataset_splits.png")

# Plot 6: Cumulative answer distribution
fig, ax = plt.subplots(figsize=(10, 6))
for opt in options:
    cumulative = [sum(1 for a in gt_labels[:i+1] if a == opt) / (i+1) * 100
                  for i in range(N_TEST)]
    ax.plot(cumulative, label=f"Option {opt}", linewidth=2, alpha=0.8)
ax.axhline(20, color="black", linestyle="--", linewidth=1.5,
           alpha=0.6, label="Uniform 20%")
ax.set_xlabel("Sample index", fontsize=12)
ax.set_ylabel("Cumulative percentage (%)", fontsize=12)
ax.set_title("Cumulative Answer Distribution Stability", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 45)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/06_cumulative_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/06_cumulative_distribution.png")

# ── Save summary ───────────────────────────────────────────
with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines))
print(f"\nSaved → {SUMMARY_PATH}")
print("Done!")