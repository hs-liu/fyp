# analysis_uq_configs.py
"""
UQ Configuration Analysis:
1. Accuracy comparison across all configs per model
2. Greedy vs majority vote per model
3. Calibration curves (consistency vs accuracy)
4. Coverage vs accuracy tradeoff
5. Heatmap — all configs x all models
6. Best config selection
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR  = "./results/UQ"
GRAPHS_DIR   = "./graphs/analysis/uq_analysis"
SUMMARY_PATH = "./results/analysis/uq_analysis_summary.txt"
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

lines = []
def log(s=""): print(s); lines.append(s)

# ── Config — all UQ result files ──────────────────────────
# Each entry: (fname, greedy_col, majority_col)
UQ_CONFIGS = {
    "BioMistral-7B": {
        "T=0.7 N=10": ("results_biomistral_medhireuqrag_0.7_10.csv", "greedy_correct", "uq_correct"),
        "T=0.7 N=20": ("results_biomistral_medhireuqrag_0.7_20.csv", "greedy_correct", "uq_correct"),
        "T=0.3 N=10": ("results_biomistral_medhireuqrag_0.3_10.csv", "greedy_correct", "uq_correct"),
        "T=0.3 N=20": ("results_biomistral_medhireuqrag_0.3_20.csv", "greedy_correct", "uq_correct"),
    },
    "Llama-3.1-8B": {
        "T=0.7 N=10": ("results_llama_medhireuqrag_0.7_10.csv", "greedy_correct", "uq_correct"),
        "T=0.7 N=20": ("results_llama_medhireuqrag_0.7_20.csv", "greedy_correct", "uq_correct"),
        "T=0.3 N=10": ("results_llama_medhireuqrag_0.3_10.csv", "greedy_correct", "uq_correct"),
        "T=0.3 N=20": ("results_llama_medhireuqrag_0.3_20.csv", "greedy_correct", "uq_correct"),
    },
    "Qwen2.5-7B": {
        "T=0.7 N=10": ("results_qwen_medhireuqrag_0.7_10.csv", "greedy_correct", "uq_correct"),
        "T=0.7 N=20": ("results_qwen_medhireuqrag_0.7_20.csv", "greedy_correct", "uq_correct"),
        "T=0.3 N=10": ("results_qwen_medhireuqrag_0.3_10.csv", "greedy_correct", "uq_correct"),
        "T=0.3 N=20": ("results_qwen_medhireuqrag_0.3_20.csv", "greedy_correct", "uq_correct"),
    },
}

CONFIG_COLORS = {
    "T=0.7 N=10": "#E74C3C",
    "T=0.7 N=20": "#E67E22",
    "T=0.3 N=10": "#2E86C1",
    "T=0.3 N=20": "#1E8449",
}
model_names   = list(UQ_CONFIGS.keys())
config_labels = list(list(UQ_CONFIGS.values())[0].keys())
MODEL_COLORS = {
    "BioMistral-7B": "#2E86C1",
    "Llama-3.1-8B":      "#1E8449",
    "Qwen2.5-7B":       "#CBB545",
}

def load_uq(fname):
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    return pd.read_csv(path)

def get_acc(df, col):
    if df is None or col not in df.columns:
        return np.nan
    return df[col].fillna(False).astype(bool).mean() * 100

# ── Load all data ──────────────────────────────────────────
log("=" * 60)
log("UQ CONFIGURATION ANALYSIS")
log("=" * 60)

all_data = {}  # model -> config -> df
all_accs = {}  # model -> config -> {greedy, majority, best}

for model_name, configs in UQ_CONFIGS.items():
    all_data[model_name] = {}
    all_accs[model_name] = {}
    log(f"\n--- {model_name} ---")
    for config_label, (fname, gcol, mcol) in configs.items():
        df = load_uq(fname)
        all_data[model_name][config_label] = df
        greedy_acc   = get_acc(df, gcol)
        majority_acc = get_acc(df, mcol)
        best_acc     = max(greedy_acc, majority_acc)
        best_type    = "greedy" if greedy_acc >= majority_acc else "majority"
        all_accs[model_name][config_label] = {
            "greedy":   greedy_acc,
            "majority": majority_acc,
            "best":     best_acc,
            "best_type": best_type,
        }
        log(f"  {config_label}: greedy={greedy_acc:.1f}%  majority={majority_acc:.1f}%  "
            f"→ best={best_acc:.1f}% ({best_type})")

# ══════════════════════════════════════════════════════════
# PLOT 1: Greedy vs Majority — per model, all configs
# ══════════════════════════════════════════════════════════
""" fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=False)
fig.suptitle("Greedy vs Majority Vote Accuracy",
             fontsize=14, fontweight="bold")

for ax, model_name in zip(axes, UQ_CONFIGS.keys()):
    configs      = list(all_accs[model_name].keys())
    greedy_vals  = [all_accs[model_name][c]["greedy"]   for c in configs]
    majority_vals= [all_accs[model_name][c]["majority"] for c in configs]

    x     = np.arange(len(configs))
    width = 0.35
    b1 = ax.bar(x - width/2, greedy_vals,   width,
                label="Greedy",   color="#7D3C98", edgecolor="white")
    b2 = ax.bar(x + width/2, majority_vals, width,
                label="Majority", color=MODEL_COLORS[model_name], edgecolor="white")

    for bar, val in zip(list(b1)+list(b2), greedy_vals+majority_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right", fontsize=9)
    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)" if model_name == "BioMistral-7B" else "")
    max_v = max(max(greedy_vals), max(majority_vals))
    ax.set_ylim(0, max_v * 1.2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/01_greedy_vs_majority.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"\nSaved → {GRAPHS_DIR}/01_greedy_vs_majority.png") """

# ══════════════════════════════════════════════════════════
# PLOT 2: Best accuracy per config — cross model comparison
# ══════════════════════════════════════════════════════════
""" fig, ax = plt.subplots(figsize=(14, 7))

config_labels = list(list(UQ_CONFIGS.values())[0].keys())
model_names   = list(UQ_CONFIGS.keys())
n_models      = len(model_names)
n_configs     = len(config_labels)
x             = np.arange(n_configs)
width         = 0.22
all_vals      = []

for i, model_name in enumerate(model_names):
    best_vals = [all_accs[model_name][c]["best"] for c in config_labels]
    best_types= [all_accs[model_name][c]["best_type"] for c in config_labels]
    offset    = (i - (n_models-1)/2) * width
    bars      = ax.bar(x + offset, best_vals, width,
                       label=model_name, color=MODEL_COLORS[model_name],
                       edgecolor="white", linewidth=0.5)
    for bar, val, bt in zip(bars, best_vals, best_types):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=7, fontweight="bold")
    all_vals.extend([v for v in best_vals if not np.isnan(v)])

ax.set_xticks(x)
ax.set_xticklabels(config_labels, fontsize=11)
ax.set_ylabel("Best Accuracy (%)", fontsize=11)
ax.set_title("Best UQ Accuracy per Config Cross Model Comparison",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0, max(all_vals) * 1.2)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.legend(fontsize=11, framealpha=0.9, loc="upper right", edgecolor="#cccccc")
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/02_cross_model_best_config.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"Saved → {GRAPHS_DIR}/02_cross_model_best_config.png")
 """
# ══════════════════════════════════════════════════════════
# PLOT 3: Heatmap — all configs x all models (greedy / majority / best)
# ══════════════════════════════════════════════════════════
""" for metric, metric_label in [("greedy", "Greedy"), ("majority", "Majority"), ("best", "Best")]:
    matrix = []
    for config_label in config_labels:
        row = [all_accs[mn][config_label][metric] for mn in model_names]
        matrix.append(row)
    matrix = np.array(matrix, dtype=float)

    vmin = np.nanmin(matrix) - 2
    vmax = np.nanmax(matrix) + 2

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=12)
    ax.set_yticks(range(len(config_labels)))
    ax.set_yticklabels(config_labels, fontsize=11)
    ax.set_title(f"UQ {metric_label} Accuracy Heatmap: All Configs with All Models",
                 fontsize=12, fontweight="bold", pad=12)

    for i in range(len(config_labels)):
        for j in range(len(model_names)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > (vmin + (vmax-vmin)*0.6) else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=text_color)

    # Highlight best per model
    for j in range(len(model_names)):
        col    = matrix[:, j]
        best_i = np.nanargmax(col)
        ax.add_patch(plt.Rectangle((j-0.5, best_i-0.5), 1, 1,
                                   fill=False, edgecolor="gold", linewidth=3))

    plt.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)
    plt.tight_layout()
    out = f"{GRAPHS_DIR}/03_heatmap_{metric}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved → {out}") """

# ══════════════════════════════════════════════════════════
# PLOT 4 (REVISED): Calibration curves — all configs per model
# Uses majority_correct (mcol) instead of greedy_correct
# X-axis: confidence label (Very Low → Very High)
# ══════════════════════════════════════════════════════════

""" def get_confidence_label(score):
    if score >= 0.9:   return "Very High"
    elif score >= 0.7: return "High"
    elif score >= 0.5: return "Medium"
    elif score >= 0.3: return "Low"
    else:              return "Very Low"

# Drop "Very Low" since no model ever reaches it
CONFIDENCE_ORDER = ["Low", "Medium", "High", "Very High"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("UQ Calibration: Confidence Label vs Majority Vote Accuracy\n",
             fontsize=13, fontweight="bold")

for ax, model_name in zip(axes, UQ_CONFIGS.keys()):
    for config_label, (fname, gcol, mcol) in UQ_CONFIGS[model_name].items():
        df = all_data[model_name][config_label]
        if df is None or mcol not in df.columns:
            continue
        df = df.copy()

        # ── Use confidence_label column directly ──────────────────────────
        if "confidence_label" in df.columns:
            df["conf_label"] = df["confidence_label"]
        else:
            df["conf_label"] = df["uq_consistency"].apply(get_confidence_label)

        df["correct"] = df[mcol].fillna(False).astype(bool)

        calib = (
            df.groupby("conf_label", observed=True)["correct"]
            .agg(acc="mean", n="count")
            .reindex(CONFIDENCE_ORDER)
            .dropna(subset=["acc"])
        )

        # Allow n >= 1 so sparse Qwen bins (High/Medium) still appear
        calib = calib[calib["n"] >= 3]

        if calib.empty:
            continue

        x_pos = [CONFIDENCE_ORDER.index(label) for label in calib.index]

        ax.plot(
            x_pos,
            calib["acc"].values * 100,
            marker="o", linewidth=2, markersize=5,
            label=config_label, color=CONFIG_COLORS[config_label]
        )

    # Always show all 4 ticks
    ax.set_xticks(range(len(CONFIDENCE_ORDER)))
    ax.set_xticklabels(CONFIDENCE_ORDER, rotation=15, ha="right", fontsize=9)
    ax.set_xlabel("Confidence Label", fontsize=10)
    ax.set_ylabel("Majority Vote Accuracy (%)" if model_name == "BioMistral-7B" else "",
                  fontsize=10)
    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, framealpha=0.9)

plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/04_calibration_curves_majority.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"Saved → {GRAPHS_DIR}/04_calibration_curves_majority.png")
 """

# ══════════════════════════════════════════════════════════
# PLOT 5: Coverage vs Accuracy — all configs per model
# ══════════════════════════════════════════════════════════
# Plot 5: Coverage vs Accuracy using confidence labels
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("UQ Coverage vs Accuracy",
             fontsize=13, fontweight="bold")

CONFIDENCE_LABELS = ["Very Low", "Low", "Medium", "High", "Very High"]
CONFIDENCE_THRESHOLDS = {
    "Very High": 0.9,
    "High":      0.7,
    "Medium":    0.5,
    "Low":       0.3,
    "Very Low":  0.0,
}
LABEL_COLORS = {
    "T=0.7 N=10": "#E74C3C",
    "T=0.7 N=20": "#E67E22",
    "T=0.3 N=10": "#2E86C1",
    "T=0.3 N=20": "#1E8449",
}

def get_confidence_label(consistency: float) -> str:
    if consistency >= 0.9:   return "Very High"
    elif consistency >= 0.7: return "High"
    elif consistency >= 0.5: return "Medium"
    elif consistency >= 0.3: return "Low"
    else:                    return "Very Low"

for ax, model_name in zip(axes, UQ_CONFIGS.keys()):
    for config_label, (fname, gcol, mcol) in UQ_CONFIGS[model_name].items():
        df = all_data[model_name][config_label]
        if df is None or "uq_consistency" not in df.columns:
            continue
        df = df.copy()
        df["correct"]           = df[gcol].fillna(False).astype(bool)
        df["confidence_label"]  = df["uq_consistency"].apply(get_confidence_label)

        # For each confidence threshold — include questions at or above that level
        coverages, accuracies, label_ticks = [], [], []
        for label in ["Very High", "High", "Medium", "Low", "Very Low"]:
            threshold = CONFIDENCE_THRESHOLDS[label]
            sub = df[df["uq_consistency"] >= threshold]
            if len(sub) < 3:
                continue
            coverages.append(len(sub) / len(df))
            accuracies.append(sub["correct"].mean() * 100)
            label_ticks.append(label)

        if len(coverages) > 1:
            ax.plot(coverages, accuracies,
                    marker="o", linewidth=2, markersize=6,
                    label=config_label,
                    color=LABEL_COLORS.get(config_label, "#888"))

            # Annotate confidence label at each point
            for cov, acc, lbl in zip(coverages, accuracies, label_ticks):
                ax.annotate(lbl,
                            xy=(cov, acc),
                            xytext=(4, 4), textcoords="offset points",
                            fontsize=6, color="gray")

    # Mark full coverage baseline
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5, label="Full coverage")

    ax.set_xlabel("Coverage", fontsize=10)
    ax.set_ylabel("Accuracy (%)" if model_name == list(UQ_CONFIGS.keys())[0] else "",
                  fontsize=10)
    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.invert_xaxis()   # left = selective, right = full coverage
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, framealpha=0.9)

plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/05_coverage_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"Saved → {GRAPHS_DIR}/05_coverage_accuracy.png")

# Add this analysis to analysis_uq_configs.py
for model_name in UQ_CONFIGS:
    for config_label, (fname, gcol, mcol) in UQ_CONFIGS[model_name].items():
        df = all_data[model_name][config_label]
        if df is None or "uq_consistency" not in df.columns:
            continue
        df = df.copy()
        df["correct"] = df[gcol].fillna(False).astype(bool)
        df["confidence_label"] = df["uq_consistency"].apply(get_confidence_label)

        vh = df[df["confidence_label"] == "Very High"]
        h  = df[df["confidence_label"].isin(["Very High","High"])]

        log(f"\n  {model_name} | {config_label}:")
        log(f"    Overall acc:          {df['correct'].mean()*100:.1f}%")
        log(f"    Very High coverage:   {len(vh)/len(df)*100:.1f}% of questions")
        log(f"    Very High accuracy:   {vh['correct'].mean()*100:.1f}%")
        log(f"    High+ coverage:       {len(h)/len(df)*100:.1f}% of questions")
        log(f"    High+ accuracy:       {h['correct'].mean()*100:.1f}%")
        log(f"    Calibration gap:      "
            f"{vh['correct'].mean()*100 - df['correct'].mean()*100:+.1f}%")

# Generate LaTeX table
print("\n% LaTeX confidence distribution table")
print(r"\begin{table}[H]")
print(r"\centering\small")
print(r"\begin{tabular}{llccccc c}")
print(r"\toprule")
print(r"\textbf{Model} & \textbf{Config} & \textbf{VL} & \textbf{L} & \textbf{M} & \textbf{H} & \textbf{VH} & \textbf{Total} \\")
print(r"\midrule")

for model_name in UQ_CONFIGS:
    first = True
    n_configs = len(UQ_CONFIGS[model_name])
    for config_label, (fname, gcol, mcol) in UQ_CONFIGS[model_name].items():
        df = all_data[model_name][config_label]
        if df is None:
            continue
        df = df.copy()
        df["confidence_label"] = df["uq_consistency"].apply(get_confidence_label)
        counts = {l: (df["confidence_label"]==l).sum() for l in CONFIDENCE_LABELS}
        
        model_cell = f"\\multirow{{{n_configs}}}{{*}}{{{model_name}}}" \
                     if first else ""
        first = False
        
        vl = counts["Very Low"]
        l  = counts["Low"]
        m  = counts["Medium"]
        h  = counts["High"]
        vh = counts["Very High"]
        
        # Replace 0 with ---
        def fmt(v): return "---" if v == 0 else str(v)
        
        print(f"    {model_cell} & ${config_label}$ & "
              f"{fmt(vl)} & {fmt(l)} & {fmt(m)} & {fmt(h)} & {fmt(vh)} & 200 \\\\")
    print(r"    \midrule")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\caption{Confidence label distribution across UQ configurations.}")
print(r"\label{tab:conf_dist}")
print(r"\end{table}")

for model_name in UQ_CONFIGS:
    for config_label, (fname, gcol, mcol) in UQ_CONFIGS[model_name].items():
        df = all_data[model_name][config_label]
        if df is None: continue
        df["confidence_label"] = df["uq_consistency"].apply(get_confidence_label)
        for label in ["Low", "Medium", "High", "Very High"]:
            subset = df[df["confidence_label"]==label]
            if len(subset) >= 3:
                acc = subset[mcol].mean() * 100
                print(f"{model_name} {config_label} {label}: {acc:.1f}%")
            else:
                print(f"{model_name} {config_label} {label}: ---")

# ══════════════════════════════════════════════════════════
# SUMMARY: best config per model
# ══════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("BEST CONFIG PER MODEL:")
log("=" * 60)
for model_name in model_names:
    best_config = max(all_accs[model_name].items(),
                      key=lambda x: x[1]["best"])
    config_label, stats = best_config
    log(f"  {model_name:<12} → {config_label}  "
        f"({stats['best_type']}={stats['best']:.1f}%)")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines))
print(f"\nDone! → {SUMMARY_PATH}")