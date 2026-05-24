# eda_and_baseline.py
"""
EDA on MedQA test set (200 samples) + heuristic baselines.
Outputs:
  - results/eda_summary.txt
  - results/heuristic_baselines.csv
  - graphs/eda_distribution.png
  - graphs/eda_question_length.png
"""
import os, random, string
import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

RESULTS_DIR = "./results/eda"
GRAPHS_DIR  = "./graphs/eda/medqa"
SUMMARY_PATH = f"{RESULTS_DIR}/eda_medqa_summary.txt"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

N_TEST = 200
SEED   = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Load dataset ───────────────────────────────────────────
print("Loading MedQA...")
dataset = datasets.load_dataset(
    "bigbio/med_qa", "med_qa_en_source", trust_remote_code=True
)
test_ds  = list(dataset["test"])[:N_TEST]
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
answer_counts = Counter(s["answer_idx"] for s in test_ds)
for k in sorted(answer_counts):
    pct = answer_counts[k] / N_TEST * 100
    log(f"  {k}: {answer_counts[k]:3d} ({pct:.1f}%)")
log()

# Full test answer distribution
full_answer_counts = Counter(s["answer_idx"] for s in full_test)
log("ANSWER DISTRIBUTION (full test set)")
for k in sorted(full_answer_counts):
    pct = full_answer_counts[k] / len(full_test) * 100
    log(f"  {k}: {full_answer_counts[k]:3d} ({pct:.1f}%)")
log()

# ── 4. Question length stats ──────────────────────────────
log("=" * 60)
log("QUESTION LENGTH STATISTICS (N=200)")
log("=" * 60)
q_lengths   = [len(s["question"].split()) for s in test_ds]
opt_counts  = [len(s["options"]) for s in test_ds]
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
    for opt in s['options']:
        marker = " ← correct" if opt['key'] == s['answer_idx'] else ""
        log(f"         {opt['key']}. {opt['value']}{marker}")
    log(f"       Answer: {s['answer_idx']}")
    log()

# ── 6. Heuristic baselines ────────────────────────────────
log("=" * 60)
log("HEURISTIC BASELINES")
log("=" * 60)

results = []
gt_labels = [s["answer_idx"] for s in test_ds]
options   = sorted(set(s["answer_idx"] for s in full_test))  # A-E

# 6a. Random uniform
random_preds = [random.choice(options) for _ in gt_labels]
random_acc   = sum(p == g for p, g in zip(random_preds, gt_labels)) / N_TEST
log(f"Random (uniform):        {random_acc:.2%}")

# 6b. Always predict most common answer
most_common   = answer_counts.most_common(1)[0][0]
majority_preds = [most_common] * N_TEST
majority_acc   = sum(p == g for p, g in zip(majority_preds, gt_labels)) / N_TEST
log(f"Always predict '{most_common}':   {majority_acc:.2%}")

# 6c. Weighted random (sample proportional to answer freq)
weights = [answer_counts[o] / N_TEST for o in options]
weighted_preds = random.choices(options, weights=weights, k=N_TEST)
weighted_acc   = sum(p == g for p, g in zip(weighted_preds, gt_labels)) / N_TEST
log(f"Weighted random:         {weighted_acc:.2%}")

# 6d. Always predict A
always_a_acc = sum(g == "A" for g in gt_labels) / N_TEST
log(f"Always predict 'A':      {always_a_acc:.2%}")

# 6e. Theoretical random (1/n_options)
n_options = len(options)
theoretical = 1 / n_options
log(f"Theoretical random:      {theoretical:.2%} (1/{n_options})")
log()
log(f"Expected random baseline: ~20% for 5-choice MCQ")
log(f"Any model scoring >20% has learned something.")
log(f"Any model scoring >25% beats majority class baseline.")

# Save heuristic results
heuristic_df = pd.DataFrame([
    {"method": "Random uniform",     "accuracy": random_acc,   "n": N_TEST},
    {"method": f"Always '{most_common}'", "accuracy": majority_acc, "n": N_TEST},
    {"method": "Weighted random",    "accuracy": weighted_acc, "n": N_TEST},
    {"method": "Always 'A'",         "accuracy": always_a_acc, "n": N_TEST},
    {"method": "Theoretical (1/5)",  "accuracy": theoretical,  "n": N_TEST},
])
heuristic_df.to_csv(f"{RESULTS_DIR}/heuristic_baselines.csv", index=False)
log()
log(f"Saved → {RESULTS_DIR}/heuristic_baselines.csv")

# ── 7. Plots ───────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig)

# Plot 1: answer distribution subset vs full
ax1 = fig.add_subplot(gs[0, 0])
x   = sorted(answer_counts.keys())
subset_vals = [answer_counts[k] / N_TEST * 100 for k in x]
full_vals   = [full_answer_counts[k] / len(full_test) * 100 for k in x]
bw  = 0.35
ax1.bar([i - bw/2 for i in range(len(x))], subset_vals, bw, label=f"Subset (n={N_TEST})", color="steelblue")
ax1.bar([i + bw/2 for i in range(len(x))], full_vals,   bw, label=f"Full test (n={len(full_test)})", color="orange")
ax1.axhline(20, color="red", linestyle="--", label="Uniform (20%)")
ax1.set_xticks(range(len(x)))
ax1.set_xticklabels(x)
ax1.set_xlabel("Answer option")
ax1.set_ylabel("Percentage (%)")
ax1.set_title("Answer Distribution")
ax1.legend(fontsize=8)

# Plot 2: question length histogram
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(q_lengths, bins=20, color="steelblue", edgecolor="white")
ax2.axvline(np.mean(q_lengths), color="red", linestyle="--",
            label=f"Mean={np.mean(q_lengths):.0f}")
ax2.axvline(np.median(q_lengths), color="orange", linestyle="--",
            label=f"Median={np.median(q_lengths):.0f}")
ax2.set_xlabel("Question length (words)")
ax2.set_ylabel("Count")
ax2.set_title("Question Length Distribution")
ax2.legend()

# Plot 3: heuristic baselines bar
ax3 = fig.add_subplot(gs[0, 2])
methods = heuristic_df["method"].tolist()
accs    = heuristic_df["accuracy"].tolist()
colors  = ["#d9534f" if a < 0.25 else "#5cb85c" for a in accs]
bars    = ax3.barh(methods, [a * 100 for a in accs], color=colors)
ax3.axvline(20, color="red", linestyle="--", label="Random (20%)")
ax3.set_xlabel("Accuracy (%)")
ax3.set_title("Heuristic Baselines")
ax3.legend(fontsize=8)
for bar, acc in zip(bars, accs):
    ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f"{acc:.1%}", va="center", fontsize=8)

# Plot 4: margin of error vs N
ax4 = fig.add_subplot(gs[1, 0])
ns   = range(10, len(full_test) + 1, 10)
moes = [1.96 * np.sqrt(0.25 / n) * 100 for n in ns]
ax4.plot(list(ns), moes, color="steelblue")
ax4.axvline(N_TEST, color="red", linestyle="--", label=f"N={N_TEST}")
ax4.axhline(1.96 * np.sqrt(0.25 / N_TEST) * 100, color="orange",
            linestyle="--", label=f"MoE={1.96*np.sqrt(0.25/N_TEST)*100:.1f}%")
ax4.set_xlabel("Sample size N")
ax4.set_ylabel("Margin of error (%)")
ax4.set_title("Statistical Power: MoE vs N")
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: split sizes
ax5 = fig.add_subplot(gs[1, 1])
splits = ["Train", "Validation", "Test\n(full)", f"Test\n(used,\nn={N_TEST})"]
sizes  = [len(train_ds), len(val_ds), len(full_test), N_TEST]
ax5.bar(splits, sizes, color=["steelblue", "orange", "green", "red"])
ax5.set_ylabel("Number of questions")
ax5.set_title("Dataset Split Sizes")
for i, (s, v) in enumerate(zip(splits, sizes)):
    ax5.text(i, v + 50, str(v), ha="center", fontsize=9)

# Plot 6: cumulative answer distribution
ax6 = fig.add_subplot(gs[1, 2])
sorted_answers = sorted(gt_labels)
counts_cum = [sorted_answers[:i+1].count(o) / (i+1) * 100
              for i in range(len(sorted_answers))
              for o in [sorted_answers[i]]]
for opt in options:
    cumulative = [sum(1 for a in gt_labels[:i+1] if a == opt) / (i+1) * 100
                  for i in range(N_TEST)]
    ax6.plot(cumulative, label=opt, alpha=0.7)
ax6.axhline(20, color="black", linestyle="--", alpha=0.5, label="Uniform 20%")
ax6.set_xlabel("Sample index")
ax6.set_ylabel("Cumulative % of answers")
ax6.set_title("Cumulative Answer Distribution")
ax6.legend(fontsize=8)

plt.suptitle("MedQA Test Set EDA (N=200)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/eda_medqa_distribution.png", dpi=150, bbox_inches="tight")
print(f"Saved → {GRAPHS_DIR}/eda_medqa_distribution.png")

# ── Save summary ───────────────────────────────────────────
with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines))
print(f"Saved → {SUMMARY_PATH}")
print("\nDone!")