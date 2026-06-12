# scripts/analysis/analysis_reliability.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "./results"
GRAPHS_DIR  = "./graphs/analysis/uq_analysis"
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/analysis", exist_ok=True)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

lines = []
def log(s=""): print(s); lines.append(s)

MEDHIRERAG_FILES = {
    "BioMistral-7B": "medhirerag/results_biomistral.csv",
    "Llama-3.1-8B":  "rerun/results_llama.csv",
    "Qwen2.5-7B":    "medhirerag/results_qwen.csv",
}

UQ_BEST_FILES = {
    "BioMistral-7B": "UQ/results_biomistral_medhireuqrag_0.7_20.csv",
    "Llama-3.1-8B":  "rerun/results_llama_medhireuqrag_0.7_20.csv",
    "Qwen2.5-7B":    "UQ/results_qwen_medhireuqrag_0.7_10.csv",
}

UQ_ALL_CONFIGS = {
    "BioMistral-7B": {
        "T=0.7 N=10": "UQ/results_biomistral_medhireuqrag_0.7_10.csv",
        "T=0.7 N=20": "UQ/results_biomistral_medhireuqrag_0.7_20.csv",
        "T=0.3 N=10": "UQ/results_biomistral_medhireuqrag_0.3_10.csv",
        "T=0.3 N=20": "UQ/results_biomistral_medhireuqrag_0.3_20.csv",
    },
    "Llama-3.1-8B": {
        "T=0.7 N=10": "rerun/results_llama_medhireuqrag_0.7_10.csv",
        "T=0.7 N=20": "rerun/results_llama_medhireuqrag_0.7_20.csv",
        "T=0.3 N=10": "rerun/results_llama_medhireuqrag_0.3_10.csv",
        "T=0.3 N=20": "rerun/results_llama_medhireuqrag_0.3_20.csv",
    },
    "Qwen2.5-7B": {
        "T=0.7 N=10": "UQ/results_qwen_medhireuqrag_0.7_10.csv",
        "T=0.7 N=20": "UQ/results_qwen_medhireuqrag_0.7_20.csv",
        "T=0.3 N=10": "UQ/results_qwen_medhireuqrag_0.3_10.csv",
        "T=0.3 N=20": "UQ/results_qwen_medhireuqrag_0.3_20.csv",
    },
}

MODEL_COLORS = {
    "BioMistral-7B": "#2E86C1",
    "Llama-3.1-8B":  "#1E8449",
    "Qwen2.5-7B":    "#7D3C98",
}

CONFIG_COLORS = {
    "T=0.7 N=10": "#2E86C1",
    "T=0.7 N=20": "#1A5276",
    "T=0.3 N=10": "#C0392B",
    "T=0.3 N=20": "#7B241C",
}

def load_df(fpath, correct_col="is_correct"):
    path = os.path.join(RESULTS_DIR, fpath)
    if not os.path.exists(path):
        log(f"  [MISSING] {path}")
        return None
    df = pd.read_csv(path)
    df[correct_col] = df[correct_col].fillna(False).astype(bool)
    # Also load uq_correct if present
    if "uq_correct" in pd.read_csv(path).columns:
        df["uq_correct"] = pd.read_csv(path)["uq_correct"]\
            .fillna(False).astype(bool)
    return df

def compute_ece(df, confidence_col, correct_col, n_bins=5):
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(df)
    for i in range(n_bins):
        mask = (df[confidence_col] >= bins[i]) & \
               (df[confidence_col] <  bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc  = df[mask][correct_col].mean()
        bin_conf = df[mask][confidence_col].mean()
        ece     += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return ece

# ── Load UQ files with both correct cols ──────────────────
def load_uq_df(fpath):
    """Load UQ results with both greedy_correct and uq_correct."""
    path = os.path.join(RESULTS_DIR, fpath)
    if not os.path.exists(path):
        log(f"  [MISSING] {path}")
        return None
    df = pd.read_csv(path)
    df["greedy_correct"] = df["greedy_correct"].fillna(False).astype(bool)
    df["uq_correct"]     = df["uq_correct"].fillna(False).astype(bool)
    return df

# ══════════════════════════════════════════════════════════
# Compute + log metrics — using uq_correct for MedHireUQRAG
# ══════════════════════════════════════════════════════════
log("=" * 60)
log("RELIABILITY METRICS — MedHireRAG vs MedHireUQRAG")
log("(MedHireUQRAG evaluated on majority-vote predictions)")
log("=" * 60)

model_names   = list(MODEL_COLORS.keys())
config_labels = list(CONFIG_COLORS.keys())

summary_rows = []

for model_name in model_names:
    log(f"\n--- {model_name} ---")

    # MedHireRAG baseline — greedy
    df_rag  = load_df(MEDHIRERAG_FILES[model_name])
    rag_acc = df_rag["is_correct"].mean() * 100 \
              if df_rag is not None else np.nan
    log(f"  MedHireRAG (greedy): {rag_acc:.1f}%")

    # Best UQ config — majority vote (uq_correct)
    df_uq = load_uq_df(UQ_BEST_FILES[model_name])
    if df_uq is not None:
        # ECE and AUC use uq_correct (majority vote)
        ece = compute_ece(df_uq, "uq_consistency", "uq_correct")
        try:
            auc = roc_auc_score(
                df_uq["uq_correct"], df_uq["uq_consistency"])
        except Exception:
            auc = np.nan

        full_acc = df_uq["uq_correct"].mean() * 100
        vh       = df_uq[df_uq["uq_consistency"] >= 0.9]
        vh_acc   = vh["uq_correct"].mean() * 100 \
                   if len(vh) >= 3 else np.nan
        vh_cov   = len(vh) / len(df_uq) * 100
        h_plus   = df_uq[df_uq["uq_consistency"] >= 0.7]
        hp_acc   = h_plus["uq_correct"].mean() * 100 \
                   if len(h_plus) >= 3 else np.nan
        hp_cov   = len(h_plus) / len(df_uq) * 100

        log(f"  MedHireUQRAG (majority, best T=0.7):")
        log(f"    ECE:            {ece:.4f}")
        log(f"    AUC:            {auc:.4f}")
        log(f"    Full acc:       {full_acc:.1f}%")
        log(f"    VH accuracy:    {vh_acc:.1f}% ({vh_cov:.1f}% coverage)")
        log(f"    High+ accuracy: {hp_acc:.1f}% ({hp_cov:.1f}% coverage)")

        summary_rows.append({
            "model":    model_name,
            "rag_acc":  rag_acc,
            "ece":      ece,
            "auc":      auc,
            "full_acc": full_acc,
            "vh_acc":   vh_acc,
            "vh_cov":   vh_cov,
            "hp_acc":   hp_acc,
            "hp_cov":   hp_cov,
        })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(
    f"{RESULTS_DIR}/analysis/reliability_summary.csv", index=False)

# ══════════════════════════════════════════════════════════
# PLOT 1: ECE across all UQ configs — using uq_correct
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
fig.suptitle("Expected Calibration Error (ECE) — Majority Vote",
             fontsize=13, fontweight="bold")

x     = np.arange(len(config_labels))
width = 0.6

for ax, model_name in zip(axes, model_names):
    eces   = []
    colors = []
    for config_label, fname in UQ_ALL_CONFIGS[model_name].items():
        df_c = load_uq_df(fname)
        ece  = compute_ece(
            df_c, "uq_consistency", "uq_correct") \
               if df_c is not None else np.nan
        eces.append(ece)
        colors.append(CONFIG_COLORS[config_label])

    bars = ax.bar(
        x, [v if not np.isnan(v) else 0 for v in eces],
        width, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, eces):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + 0.002, f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(
        config_labels, rotation=20, ha="right", fontsize=9)
    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.set_ylabel(
        "ECE" if model_name == model_names[0] else "", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/10_ece_comparison.png",
            dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════
# PLOT 2: AUC across all UQ configs — using uq_correct
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
fig.suptitle("AUC — Consistency Score as Correctness Predictor "
             "(Majority Vote)\n",
             fontsize=13, fontweight="bold")

for ax, model_name in zip(axes, model_names):
    aucs   = []
    colors = []
    for config_label, fname in UQ_ALL_CONFIGS[model_name].items():
        df_c = load_uq_df(fname)
        if df_c is not None:
            try:
                auc = roc_auc_score(
                    df_c["uq_correct"], df_c["uq_consistency"])
            except Exception:
                auc = np.nan
        else:
            auc = np.nan
        aucs.append(auc)
        colors.append(CONFIG_COLORS[config_label])

    bars = ax.bar(
        x, [v if not np.isnan(v) else 0 for v in aucs],
        width, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, aucs):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + 0.005, f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(
        config_labels, rotation=20, ha="right", fontsize=9)
    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.set_ylabel(
        "AUC" if model_name == model_names[0] else "", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/11_auc_comparison.png",
            dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════
# PLOT 3: MedHireRAG vs MedHireUQRAG selective accuracy
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=False)
fig.suptitle(
    "MedHireRAG vs MedHireUQRAG — Accuracy at Full vs Selective Coverage\n"
    "Hatched = full coverage (100%)   |   Solid = VH confidence only",
    fontsize=12, fontweight="bold"
)

for ax, model_name in zip(axes, model_names):
    labels  = []
    accs    = []
    colors  = []
    hatches = []

    # MedHireRAG — full coverage (greedy)
    df_rag = load_df(MEDHIRERAG_FILES[model_name])
    if df_rag is not None:
        rag_acc = df_rag["is_correct"].mean() * 100
        labels.append("MedHireRAG\n(100%)")
        accs.append(rag_acc)
        colors.append(MODEL_COLORS[model_name])
        hatches.append("//")

    # MedHireUQRAG — full coverage (greedy, for fair comparison)
    df_uq = load_uq_df(UQ_BEST_FILES[model_name])
    if df_uq is not None:
        full_acc = df_uq["uq_correct"].mean() * 100
        labels.append("MedHireUQRAG\n(100%)")
        accs.append(full_acc)
        colors.append("#8E44AD")
        hatches.append("//")

        # MedHireUQRAG — VH only (majority vote)
        vh = df_uq[df_uq["uq_consistency"] >= 0.9]
        if len(vh) >= 3:
            vh_acc = vh["uq_correct"].mean() * 100
            vh_cov = len(vh) / len(df_uq) * 100
            labels.append(
                f"MedHireUQRAG\n(VH: {vh_cov:.0f}%)")
            accs.append(vh_acc)
            colors.append("#8E44AD")
            hatches.append("")

    x_pos = np.arange(len(labels))
    for i, (acc, color, hatch) in enumerate(
            zip(accs, colors, hatches)):
        ax.bar(i, acc, width=0.55, color=color,
               edgecolor="white", linewidth=0.5,
               hatch=hatch, alpha=0.85)
        ax.text(i, acc + 0.5, f"{acc:.1f}%",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9, ha="center")
    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.set_ylabel(
        "Accuracy (%)" if model_name == model_names[0] else "",
        fontsize=11)
    max_val = max(accs) if accs else 100
    ax.set_ylim(0, max_val * 1.2)
    ax.yaxis.set_major_formatter(
        mticker.FormatStrFormatter("%.0f%%"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

plt.subplots_adjust(bottom=0.18)
plt.savefig(f"{GRAPHS_DIR}/12_reliability_medhire_vs_uq.png",
            dpi=150, bbox_inches="tight")
plt.close()

# ── Save summary ───────────────────────────────────────────
with open(f"{RESULTS_DIR}/analysis/reliability_summary.txt", "w") as f:
    f.write("\n".join(lines))
log(f"Saved → {RESULTS_DIR}/analysis/reliability_summary.txt")
print("Done!")