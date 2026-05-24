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

# Best config per model (T=0.7, calibration-optimised)
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

CONFIDENCE_LABELS = ["Low", "Medium", "High", "Very High"]

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
    df["greedy_correct"] = df["greedy_correct"].fillna(False).astype(bool)
    df["confidence_label"] = df["uq_consistency"].apply(get_confidence_label)
    return df

all_data = {mn: load(f) for mn, f in CONFIGS.items()}

# Plot 1: Entropy distribution — correct vs incorrect
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
fig.suptitle("Predictive Entropy Distribution: Correct vs Incorrect Predictions",
             fontsize=13, fontweight="bold")

COLORS = {
    "correct":   "#2E86C1",  # professional blue
    "incorrect": "#C0392B",  # professional red
}

for ax, model_name in zip(axes, all_data.keys()):
    df = all_data[model_name]
    if df is None:
        continue

    correct   = df[df["greedy_correct"]==True]["uq_entropy"]
    incorrect = df[df["greedy_correct"]==False]["uq_entropy"]

    # Use KDE instead of histogram to avoid overlap
    from scipy.stats import gaussian_kde

    all_vals = pd.concat([correct, incorrect])
    x_range  = np.linspace(all_vals.min(), all_vals.max(), 300)

    if len(correct) > 3:
        kde_c = gaussian_kde(correct)
        ax.fill_between(x_range, kde_c(x_range), alpha=0.4,
                        color=COLORS["correct"], label=f"Correct (n={len(correct)})")
        ax.plot(x_range, kde_c(x_range), color=COLORS["correct"], linewidth=2)

    if len(incorrect) > 3:
        kde_i = gaussian_kde(incorrect)
        ax.fill_between(x_range, kde_i(x_range), alpha=0.4,
                        color=COLORS["incorrect"], label=f"Incorrect (n={len(incorrect)})")
        ax.plot(x_range, kde_i(x_range), color=COLORS["incorrect"], linewidth=2)

    # Mean lines
    ax.axvline(correct.mean(), color=COLORS["correct"], linestyle="--",
               linewidth=1.5, label=f"Correct mean={correct.mean():.2f}")
    ax.axvline(incorrect.mean(), color=COLORS["incorrect"], linestyle="--",
               linewidth=1.5, label=f"Incorrect mean={incorrect.mean():.2f}")

    # T-test
    t_stat, p_val = stats.ttest_ind(correct, incorrect)
    p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
    ax.text(0.97, 0.97,
            f"t={t_stat:.2f}\n{p_str}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    ax.set_xlabel("Predictive entropy", fontsize=11)
    ax.set_ylabel("Density" if model_name == "BioMistral-7B" else "",
                  fontsize=11)
    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/06_entropy_correct_vs_incorrect.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/06_entropy_correct_vs_incorrect.png")

# ══════════════════════════════════════════════════════════
# PLOT 2: Entropy vs Consistency scatter
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
fig.suptitle("Entropy vs Consistency Score",
             fontsize=13, fontweight="bold")

for ax, model_name in zip(axes, all_data.keys()):
    df = all_data[model_name]
    if df is None:
        continue

    correct   = df[df["greedy_correct"]==True]
    incorrect = df[df["greedy_correct"]==False]

    ax.scatter(correct["uq_consistency"],   correct["uq_entropy"],
               alpha=0.5, s=20, color="steelblue", label="Correct")
    ax.scatter(incorrect["uq_consistency"], incorrect["uq_entropy"],
               alpha=0.5, s=20, color="lightcoral", label="Incorrect")

    # Correlation
    r, p = stats.pearsonr(df["uq_consistency"], df["uq_entropy"])
    ax.text(0.97, 0.97,
            f"r={r:.3f}\np={p:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    ax.set_xlabel("Consistency score", fontsize=11)
    ax.set_ylabel("Predictive entropy" if model_name == "BioMistral-7B" else "",
                  fontsize=11)
    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/07_entropy_vs_consistency.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/07_entropy_vs_consistency.png")

print("\nDone!")