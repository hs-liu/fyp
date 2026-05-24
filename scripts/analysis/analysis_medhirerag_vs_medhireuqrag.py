# analysis_rag_vs_uq.py
"""
Pairwise comparison: My RAG vs My RAG + UQ per domain, per model.
Uses best UQ config per model based on experimental results.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_RAG_DIR  = "./results/medhirerag"
RESULTS_UQ_DIR = "./results/UQ"
RESULTS_DIR = "./results"
GRAPHS_DIR   = "./graphs/analysis/medhirerag_vs_medhireuqrag"
SUMMARY_PATH = "./results/analysis/medhirerag_vs_medhireuqrag.txt"
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs("./results/analysis", exist_ok=True)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

lines = []
def log(s=""):
    print(s)
    lines.append(s)

# Best UQ config per model based on experimental results
configs = {
    "BioMistral": {
        "medhirerag":      "results_biomistral.csv",
        "uq_file":  "results_biomistral_medhireuqrag_0.3_10.csv",
        "uq_col":   "uq_correct",  # majority doesn't help
        "uq_label": "MedHireUQRAG (T=0.3, N=10, majority)",
        "color":    "#C17813",
        "overall_rag": 47.0,
        "overall_uq":  49.5,
    },
    "Llama": {
        "medhirerag":      "results_llama.csv",
        "uq_file":  "results_llama_medhireuqrag_0.3_20.csv",
        "uq_col":   "uq_correct",      # majority T=0.3 N=20 best
        "uq_label": "MedHireUQRAG (T=0.3, N=20, majority)",
        "color":    "#1E8449",
        "overall_rag": 58.5,
        "overall_uq":  59.0,
    },
    "Qwen": {
        "medhirerag":      "results_qwen.csv",
        "uq_file":  "results_qwen_medhireuqrag_0.3_10.csv",
        "uq_col":   "uq_correct",      # majority T=0.3 N=10 best
        "uq_label": "MedHireUQRAG (T=0.3, N=10, majority)",
        "color":    "#7D3C98",
        "overall_rag": 58.5,
        "overall_uq":  59.0,
    },
}

DOMAIN_LABEL_FILES = {
    "BioMistral": "results_biomistral_myrag_v5_6.csv",
    "Llama":      "results_llama_myrag_v4.csv",
    "Qwen":       "results_qwen_myrag.csv",
}

def load_rag(fname):
    path = os.path.join(RESULTS_RAG_DIR, fname)
    print("path is", path)
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    df = pd.read_csv(path)
    df["is_correct"] = df["is_correct"].fillna(False).astype(bool)
    return df

def load_domain(fname):
    path = os.path.join(RESULTS_DIR, fname)
    print("path is", path)
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    df = pd.read_csv(path)
    df["is_correct"] = df["is_correct"].fillna(False).astype(bool)
    return df

def load_uq(fname, uq_col):
    path = os.path.join(RESULTS_UQ_DIR, fname)
    if not os.path.exists(path):
        print(f"  [MISSING UQ] {path}")
        return None
    df = pd.read_csv(path)
    df["is_correct"] = df[uq_col].fillna(False).astype(bool)
    return df

def bar_labels(ax, bars, pad=0.8):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + pad,
                    f"{h:.1f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#333333")

log("=" * 60)
log("PAIRWISE: MedHireRAG vs MedHireUQRAG — Per Domain")
log("=" * 60)

all_stats = {}

for model_name, cfg in configs.items():
    df_medhirerag = load_rag(cfg["medhirerag"])
    df_medhireuqrag  = load_uq(cfg["uq_file"], cfg["uq_col"])
    if df_medhirerag is None or df_medhireuqrag is None:
        continue

    # Load domain labels from old classifier results
    df_domain  = load_domain(DOMAIN_LABEL_FILES[model_name])
    if df_domain is None or "domain" not in df_domain.columns:
        log(f"  [SKIP] {model_name} — no domain labels available")
        continue
    domain_map = df_domain[["id","domain"]].dropna()

    # Merge domain into both files
    if "domain" not in df_medhirerag.columns:
        df_medhirerag = df_medhirerag.merge(domain_map, on="id", how="left")
    if "domain" not in df_medhireuqrag.columns:
        df_medhireuqrag = df_medhireuqrag.merge(domain_map, on="id", how="left")

    df_medhirerag["domain"]   = df_medhirerag["domain"].fillna("Unknown")
    df_medhireuqrag["domain"] = df_medhireuqrag["domain"].fillna("Unknown")

    domains = sorted(set(
        df_medhirerag[df_medhirerag.groupby("domain")["domain"]
                      .transform("count") >= 3]["domain"]
    ) - {"Unknown"})

    medhirerag_acc, medhireuqrag_acc, counts = [], [], []
    for d in domains:
        sr = df_medhirerag[df_medhirerag["domain"]==d]
        su = df_medhireuqrag[df_medhireuqrag["domain"]==d]
        medhirerag_acc.append(sr["is_correct"].mean()*100 if len(sr)>=3 else 0)
        medhireuqrag_acc.append(su["is_correct"].mean()*100  if len(su)>=3 else 0)
        counts.append(len(sr))

    overall_rag = df_medhirerag["is_correct"].mean() * 100
    overall_uq  = df_medhireuqrag["is_correct"].mean()  * 100

    log(f"\n--- {model_name} ---")
    log(f"  Overall MedHireRAG:   {overall_rag:.1f}%")
    log(f"  Overall MedHireUQRAG:   {overall_uq:.1f}%")
    log(f"  Delta:            {overall_uq - overall_rag:+.1f}%")
    log(f"  UQ config:        {cfg['uq_label']}")
    for d, ra, ua, n in zip(domains, medhirerag_acc, medhireuqrag_acc, counts):
        log(f"  {d:<35} MedHireRAG={ra:.1f}%  MedHireUQRAG={ua:.1f}%  Δ={ua-ra:+.1f}%  n={n}")

    all_stats[model_name] = {
        "medhirerag": df_medhirerag, "medhireuqrag": df_medhireuqrag,
        "domains": domains,
        "medhirerag_acc": medhirerag_acc, "medhireuqrag_acc": medhireuqrag_acc,
        "counts": counts,
        "overall_rag": overall_rag, "overall_uq": overall_uq,
        "uq_label": cfg["uq_label"],
        "color": cfg["color"],
    }

# ── Plot 1: per model ──────────────────────────────────────
for model_name, s in all_stats.items():
    domains = s["domains"]
    medhirerag_acc = s["medhirerag_acc"]
    medhireuqrag_acc = s["medhireuqrag_acc"]
    counts  = s["counts"]
    color   = s["color"]
    x       = np.arange(len(domains))
    width   = 0.35
    max_val = max(max(medhirerag_acc or [0]), max(medhireuqrag_acc or [0]))

    fig, ax = plt.subplots(figsize=(max(10, len(domains)*2.5), 7))
    bars1 = ax.bar(x - width/2, medhirerag_acc, width,
                   label="MedHireRAG", color="#C0392B",
                   edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, medhireuqrag_acc,   width,
                   label="MedHireUQRAG", color=color,
                   edgecolor="white", linewidth=0.5)
    bar_labels(ax, bars1)
    bar_labels(ax, bars2)

    # Delta annotations below x-axis
    for i, (na, ra) in enumerate(zip(medhirerag_acc, medhireuqrag_acc)):
        delta = ra - na
        col   = "#1E8449" if delta >= 0 else "#C0392B"
        ax.annotate(f"Δ{delta:+.1f}%",
                    xy=(i, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -28), textcoords="offset points",
                    ha="center", fontsize=9, color=col, fontweight="bold")

    # Sample sizes
    for i, n in enumerate(counts):
        ax.annotate(f"n={n}",
                    xy=(i, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -42), textcoords="offset points",
                    ha="center", fontsize=8, color="gray")

    # Overall summary — use fig.text instead of ax.text with transform
    delta_overall = s["overall_uq"] - s["overall_rag"]

    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, max_val * 1.25)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title(f"Per-Domain Accuracy: MedHireRAG vs MedHireUQRAG — {model_name}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=11, framealpha=0.9,
              loc="upper right", edgecolor="#cccccc")
    plt.subplots_adjust(bottom=0.18)   # fixed bottom margin, no rect
    out = f"{GRAPHS_DIR}/medhirerag_vs_medhireuqrag_{model_name.lower()}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

# ── Plot 2: all models side by side ───────────────────────
domain_sets    = [set(s["domains"]) for s in all_stats.values()]
common_domains = sorted(set.intersection(*domain_sets)) if domain_sets else []

if common_domains:
    model_names  = list(all_stats.keys())
    model_colors = [all_stats[m]["color"] for m in model_names]
    uq_colors    = ["#E67E22", "#27AE60", "#8E44AD"]
    n_models     = len(model_names)
    width        = 0.13
    x            = np.arange(len(common_domains))
    all_accs     = []

    fig, ax = plt.subplots(figsize=(max(13, len(common_domains)*4), 7))

    for i, model_name in enumerate(model_names):
        s        = all_stats[model_name]
        rag_accs = [s["medhirerag_acc"][s["domains"].index(d)]
                    if d in s["domains"] else 0 for d in common_domains]
        uq_accs  = [s["medhireuqrag_acc"][s["domains"].index(d)]
                    if d in s["domains"] else 0 for d in common_domains]

        base = (i - (n_models-1)/2) * (2*width + 0.03)
        b1 = ax.bar(x + base,        rag_accs, width,
                    label=f"{model_name} MedHireRAG",
                    color=model_colors[i], edgecolor="white", linewidth=0.5)
        b2 = ax.bar(x + base + width, uq_accs,  width,
                    label=f"{model_name} MedHireUQRAG",
                    color=uq_colors[i], edgecolor="white", linewidth=0.5)
        bar_labels(ax, b1, pad=0.5)
        bar_labels(ax, b2, pad=0.5)
        all_accs.extend(rag_accs + uq_accs)

    ax.set_xticks(x)
    ax.set_xticklabels(common_domains, fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, max(all_accs) * 1.25)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title("Per-Domain Accuracy: MedHireRAG vs MedHireUQRAG — All Models",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9, framealpha=0.9, ncol=2,
              loc="upper right", edgecolor="#cccccc")
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/medhirerag_vs_medhireuqrag_all_models.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved → {GRAPHS_DIR}/medhirerag_vs_medhireuqrag_all_models.png")

# ── Plot 3: Delta heatmap ─────────────────────────────────
if common_domains:
    delta_matrix = []
    for model_name in model_names:
        s   = all_stats[model_name]
        row = []
        for d in common_domains:
            if d in s["domains"]:
                idx = s["domains"].index(d)
                row.append(s["medhireuqrag_acc"][idx] - s["medhirerag_acc"][idx])
            else:
                row.append(np.nan)
        delta_matrix.append(row)

    delta_matrix = np.array(delta_matrix, dtype=float)
    abs_max      = max(np.nanmax(np.abs(delta_matrix)), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(common_domains)*2), 4))
    im = ax.imshow(delta_matrix, aspect="auto",
                   cmap="RdYlGn", vmin=-abs_max, vmax=abs_max)
    ax.set_xticks(range(len(common_domains)))
    ax.set_xticklabels(common_domains, fontsize=12)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=11)
    ax.set_title("UQ Improvement Heatmap: Δ Accuracy (MedHireUQRAG - MedHireRAG) per Domain",
                 fontsize=12, fontweight="bold", pad=12)

    for i in range(len(model_names)):
        for j in range(len(common_domains)):
            val = delta_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if abs(val) > abs_max*0.6 else "black")

    plt.colorbar(im, ax=ax, label="Δ Accuracy (%)", shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/medhirerag_vs_medhireuqrag_delta_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved → {GRAPHS_DIR}/medhirerag_vs_medhireuqrag_delta_heatmap.png")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines))
log(f"\nSaved → {SUMMARY_PATH}")
print("Done!")