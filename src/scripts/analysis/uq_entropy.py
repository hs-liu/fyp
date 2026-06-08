# analysis_uq_entropy.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

RESULTS_DIR = "./results/UQ/"
GRAPHS_DIR  = "./graphs/analysis/uq_analysis"
os.makedirs(GRAPHS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

CONFIGS = {
    "BioMistral-7B": "results_biomistral_medhireuqrag_0.7_20.csv",
    "Llama-3.1-8B":  "results_llama_medhireuqrag_0.7_10.csv",
    "Qwen2.5-7B":    "results_qwen_medhireuqrag_0.7_10.csv",
}

MODEL_COLORS = {
    "BioMistral-7B": "#2E86C1",
    "Llama-3.1-8B":  "#1E8449",
    "Qwen2.5-7B":    "#7D3C98",
}

CORRECT_COLOR   = "#2E86C1"
INCORRECT_COLOR = "#C0392B"

def get_confidence_label(consistency):
    if consistency >= 0.9:   return "Very High"
    elif consistency >= 0.7: return "High"
    elif consistency >= 0.5: return "Medium"
    elif consistency >= 0.3: return "Low"
    else:                    return "Very Low"

def load(fname):
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    df = pd.read_csv(path)
    df["greedy_correct"]   = df["greedy_correct"].fillna(False).astype(bool)
    df["confidence_label"] = df["uq_consistency"].apply(get_confidence_label)
    return df

all_data = {mn: load(f) for mn, f in CONFIGS.items()}

lines = []
def log(s=""): print(s); lines.append(s)

log("=" * 60)
log("ENTROPY ANALYSIS")
log("=" * 60)

# ══════════════════════════════════════════════════════════
# PLOT 1: Boxplot — entropy by correctness per model
# ══════════════════════════════════════════════════════════
# Plot 1: Boxplot — entropy by correctness per model
fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)
fig.suptitle("Predictive Entropy: Correct vs Incorrect Predictions",
             fontsize=14, fontweight="bold", y=1.02)

CORRECT_COLOR   = "#2E86C1"
INCORRECT_COLOR = "#C0392B"

for ax, model_name in zip(axes, all_data.keys()):
    df = all_data[model_name]
    if df is None:
        continue

    correct   = df[df["greedy_correct"]==True]["uq_entropy"]
    incorrect = df[df["greedy_correct"]==False]["uq_entropy"]

    t_stat, p_val = stats.ttest_ind(correct, incorrect)
    p_str = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"

    bp = ax.boxplot(
        [correct, incorrect],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(linewidth=1.5, color="#444"),
        capprops=dict(linewidth=1.5, color="#444"),
        flierprops=dict(marker="o", markersize=3,
                        alpha=0.3, markeredgewidth=0,
                        markerfacecolor="#888"),
        boxprops=dict(linewidth=1.5),
        showfliers=False,
    )

    bp["boxes"][0].set_facecolor(CORRECT_COLOR)
    bp["boxes"][0].set_alpha(0.85)
    bp["boxes"][1].set_facecolor(INCORRECT_COLOR)
    bp["boxes"][1].set_alpha(0.85)

    # Mean markers
    for xi, (vals, color) in enumerate(
        zip([correct, incorrect], [CORRECT_COLOR, INCORRECT_COLOR]), start=1
    ):
        ax.scatter(xi, vals.mean(), marker="D", s=70, zorder=5,
                   color="white", edgecolors=color, linewidths=2)

    # Mean annotations — placed clearly above box
    y_offset = max(correct.max(), incorrect.max()) * 0.05
    ax.text(1, correct.max() + y_offset,
            f"mean={correct.mean():.2f}",
            ha="center", va="bottom", fontsize=9,
            color=CORRECT_COLOR, fontweight="bold")
    ax.text(2, incorrect.max() + y_offset,
            f"mean={incorrect.mean():.2f}",
            ha="center", va="bottom", fontsize=9,
            color=INCORRECT_COLOR, fontweight="bold")

    # Stats box — bottom right to avoid overlap with annotations
    ax.text(0.97, 0.97,
            f"t={t_stat:.2f},  {p_str}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.95))

    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        [f"Correct\n(n={len(correct)})",
         f"Incorrect\n(n={len(incorrect)})"],
        fontsize=11
    )
    ax.set_ylabel("Predictive entropy" if model_name == list(all_data.keys())[0] else "",
                  fontsize=11)
    ax.set_title(model_name, fontsize=12, fontweight="bold", pad=10)

    # Add some top padding
    ymax = max(correct.max(), incorrect.max())
    ax.set_ylim(-0.02, ymax * 1.25)

    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/06_entropy_correct_vs_incorrect.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {GRAPHS_DIR}/06_entropy_correct_vs_incorrect.png")

# ══════════════════════════════════════════════════════════
# PLOT 2: Entropy vs Consistency scatter with regression
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)
fig.suptitle("Predictive Entropy vs Consistency Score\n",
             fontsize=13, fontweight="bold")

for ax, model_name in zip(axes, all_data.keys()):
    df = all_data[model_name]
    if df is None:
        continue

    correct   = df[df["greedy_correct"]==True]
    incorrect = df[df["greedy_correct"]==False]

    # Scatter — correct vs incorrect
    ax.scatter(correct["uq_consistency"], correct["uq_entropy"],
               alpha=0.4, s=15, color=CORRECT_COLOR,
               label=f"Correct (n={len(correct)})")
    ax.scatter(incorrect["uq_consistency"], incorrect["uq_entropy"],
               alpha=0.4, s=15, color=INCORRECT_COLOR,
               label=f"Incorrect (n={len(incorrect)})")

    # Regression line
    r, p = stats.pearsonr(df["uq_consistency"], df["uq_entropy"])
    m, b = np.polyfit(df["uq_consistency"], df["uq_entropy"], 1)
    x_line = np.linspace(df["uq_consistency"].min(),
                         df["uq_consistency"].max(), 200)
    ax.plot(x_line, m * x_line + b,
            color="black", linewidth=1, linestyle="--",
            alpha=0.8, label="Regression")

    p_str = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    log(f"\n{model_name} entropy-consistency correlation:")
    log(f"  r={r:.3f}, {p_str}")

    ax.text(0.97, 0.97,
            f"r={r:.3f}\n{p_str}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    ax.set_xlabel("Consistency score", fontsize=11)
    ax.set_ylabel("Predictive entropy" if model_name == "BioMistral-7B" else "",
                  fontsize=11)
    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/07_entropy_vs_consistency.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/07_entropy_vs_consistency.png")

# ── Save summary ───────────────────────────────────────────
with open(f"./results/analysis/entropy_summary.txt", "w") as f:
    f.write("\n".join(lines))
print(f"Saved → ./results/analysis/entropy_summary.txt")
print("\nDone!")