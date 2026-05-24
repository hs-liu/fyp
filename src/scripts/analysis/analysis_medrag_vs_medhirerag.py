# analysis_medrag_vs_myrag.py
"""
Pairwise comparison: MedRAG vs My RAG per domain, per model.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR  = "./results"
GRAPHS_DIR   = "./graphs/analysis/medrag_vs_medhirerag"
SUMMARY_PATH = "./results/analysis/medrag_vs_medhirerag.txt"
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

pairs = {
    "Llama": {
        "medrag": "results_llama_medrag.csv",
        "medhirerag":    "results_llama_myrag_v4.csv",
        "color":  "#1E8449",
    },
    "Qwen": {
        "medrag": "results_qwen_medrag.csv",
        "medhirerag":    "results_qwen_myrag.csv",
        "color":  "#7D3C98",
    },
}

def load(fname):
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    df = pd.read_csv(path)
    df["is_correct"] = df["is_correct"].fillna(False).astype(bool)
    return df

def bar_labels(ax, bars, pad=0.8):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + pad,
                    f"{h:.1f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#333333")

log("=" * 60)
log("PAIRWISE: MedRAG vs MedHireRAG — Per Domain")
log("=" * 60)

all_stats = {}

for model_name, cfg in pairs.items():
    df_medrag = load(cfg["medrag"])
    df_medhirerag    = load(cfg["medhirerag"])
    if df_medrag is None or df_medhirerag is None:
        continue
    if "domain" not in df_medhirerag.columns:
        log(f"  [SKIP] {model_name} — no domain column")
        continue

    domain_map          = df_medhirerag[["id","domain"]].dropna()
    df_medrag           = df_medrag.merge(domain_map, on="id", how="left")
    df_medrag["domain"] = df_medrag["domain"].fillna("Unknown")

    domains = sorted(set(
        df_medhirerag[df_medhirerag.groupby("domain")["domain"]
               .transform("count") >= 3]["domain"]
    ) - {"Unknown"})

    medrag_acc, medhirerag_acc, counts = [], [], []
    for d in domains:
        sm = df_medrag[df_medrag["domain"]==d]
        sr = df_medhirerag[df_medhirerag["domain"]==d]
        medrag_acc.append(sm["is_correct"].mean()*100 if len(sm)>=3 else 0)
        medhirerag_acc.append(sr["is_correct"].mean()*100    if len(sr)>=3 else 0)
        counts.append(len(sr))

    overall_medrag = df_medrag["is_correct"].mean() * 100
    overall_medhirerag    = df_medhirerag["is_correct"].mean()    * 100

    log(f"\n--- {model_name} ---")
    log(f"  Overall MedRAG:  {overall_medrag:.1f}%")
    log(f"  Overall MedHireRAG:  {overall_medhirerag:.1f}%")
    log(f"  Delta:           {overall_medhirerag - overall_medrag:+.1f}%")
    for d, ma, ra, n in zip(domains, medrag_acc, medhirerag_acc, counts):
        log(f"  {d:<35} MedRAG={ma:.1f}%  MedHireRAG={ra:.1f}%  Δ={ra-ma:+.1f}%  n={n}")

    all_stats[model_name] = {
        "medrag": df_medrag, "medhirerag": df_medhirerag,
        "domains": domains,
        "medrag_acc": medrag_acc, "medhirerag_acc": medhirerag_acc,
        "counts": counts,
        "overall_medrag": overall_medrag,
        "overall_medhirerag": overall_medhirerag,
        "color": cfg["color"],
    }

# ── Plot 1: per model ──────────────────────────────────────
for model_name, s in all_stats.items():
    domains    = s["domains"]
    medrag_acc = s["medrag_acc"]
    medhirerag_acc    = s["medhirerag_acc"]
    counts     = s["counts"]
    color      = s["color"]
    x          = np.arange(len(domains))
    width      = 0.35
    max_val    = max(max(medrag_acc or [0]), max(medhirerag_acc or [0]))

    fig, ax = plt.subplots(figsize=(max(10, len(domains)*2.5), 7))
    bars1 = ax.bar(x - width/2, medrag_acc, width,
                   label="MedRAG", color="#E67E22",
                   edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, medhirerag_acc,    width,
                   label="MedHireRAG", color=color,
                   edgecolor="white", linewidth=0.5)
    bar_labels(ax, bars1)
    bar_labels(ax, bars2)

    for i, (ma, ra) in enumerate(zip(medrag_acc, medhirerag_acc)):
        delta = ra - ma
        col   = "#1E8449" if delta >= 0 else "#C0392B"
        ax.annotate(f"Δ{delta:+.1f}%",
                    xy=(i, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -28), textcoords="offset points",
                    ha="center", fontsize=9, color=col, fontweight="bold")

    for i, n in enumerate(counts):
        ax.annotate(f"n={n}",
                    xy=(i, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -42), textcoords="offset points",
                    ha="center", fontsize=8, color="gray")

    delta_overall = s["overall_medhirerag"] - s["overall_medrag"]

    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, max_val * 1.25)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title(f"Per-Domain Accuracy: MedRAG vs MedHireRAG — {model_name}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=11, framealpha=0.9,
              loc="upper right", edgecolor="#cccccc")
    plt.subplots_adjust(bottom=0.18)
    out = f"{GRAPHS_DIR}/medrag_vs_medhirerag_{model_name.lower()}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"\nSaved → {out}")

# ── Plot 2: all models side by side ───────────────────────
domain_sets    = [set(s["domains"]) for s in all_stats.values()]
common_domains = sorted(set.intersection(*domain_sets)) if domain_sets else []

if common_domains:
    model_names  = list(all_stats.keys())
    model_colors = [all_stats[m]["color"] for m in model_names]
    n_models     = len(model_names)
    width        = 0.18
    x            = np.arange(len(common_domains))
    all_accs     = []

    fig, ax = plt.subplots(figsize=(max(12, len(common_domains)*4), 7))

    for i, model_name in enumerate(model_names):
        s           = all_stats[model_name]
        medrag_accs = [s["medrag_acc"][s["domains"].index(d)]
                       if d in s["domains"] else 0 for d in common_domains]
        medhirerag_accs    = [s["medhirerag_acc"][s["domains"].index(d)]
                       if d in s["domains"] else 0 for d in common_domains]

        base = (i - (n_models-1)/2) * (2*width + 0.04)
        b1 = ax.bar(x + base,        medrag_accs, width,
                    label=f"{model_name} MedRAG",
                    color="#E67E22", edgecolor="white",
                    linewidth=0.5, alpha=0.6 + i*0.2)
        b2 = ax.bar(x + base + width, medhirerag_accs,   width,
                    label=f"{model_name} MedHireRAG",
                    color=model_colors[i], edgecolor="white", linewidth=0.5)
        bar_labels(ax, b1, pad=0.5)
        bar_labels(ax, b2, pad=0.5)
        all_accs.extend(medrag_accs + medhirerag_accs)

    ax.set_xticks(x)
    ax.set_xticklabels(common_domains, fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, max(all_accs) * 1.25)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title("Per-Domain Accuracy: MedRAG vs MedHireRAG — All Models",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9, framealpha=0.9, ncol=2,
              loc="upper right", edgecolor="#cccccc")
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/medrag_vs_medhirerag_all_models.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved → {GRAPHS_DIR}/medrag_vs_medhirerag_all_models.png")

# ── Plot 3: Delta heatmap ─────────────────────────────────
if common_domains:
    delta_matrix = []
    for model_name in model_names:
        s   = all_stats[model_name]
        row = []
        for d in common_domains:
            if d in s["domains"]:
                idx = s["domains"].index(d)
                row.append(s["medhirerag_acc"][idx] - s["medrag_acc"][idx])
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
    ax.set_title("MedHireRAG Improvement over MedRAG: Δ Accuracy (MedHireRAG − MedRAG) per Domain",
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
    plt.savefig(f"{GRAPHS_DIR}/medrag_vs_medhirerag_delta_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved → {GRAPHS_DIR}/medrag_vs_medhirerag_delta_heatmap.png")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines))
log(f"\nSaved → {SUMMARY_PATH}")
print("Done!")