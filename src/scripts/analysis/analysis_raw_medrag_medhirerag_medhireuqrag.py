# analysis_all_four.py
"""
Combined comparison: No RAG vs MedRAG vs MedHireRAG vs MedHireUQRAG
All models, all domains.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR  = "./results"
RESULTS_RAG_DIR  = "./results/medhirerag"
RESULTS_UQ_DIR = "./results/UQ"
GRAPHS_DIR   = "./graphs/analysis/overall_comparison"
SUMMARY_PATH = "./results/analysis/all_models.txt"
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

METHOD_COLORS = {
    "Raw Model":         "#C0392B",
    "MedRAG":         "#E67E22",
    "MedHireRAG":     "#2E86C1",
    "MedHireUQRAG":   "#8E44AD",
}

configs = {
    "BioMistral-7B": {
        "norag":    "results_local_biomistral.csv",
        "medrag":   'results_biomistral_medrag.csv',
        "hirerag":  "results_biomistral.csv",
        "uq_file":  "results_biomistral_medhireuqrag_0.7_20.csv",
        "uq_col":   "uq_correct",
        "color":    "#2E86C1",
        "overall":  {
            "Raw Model":       41.5,
            "MedRAG":       42.0,
            "MedHireRAG":   47.0,
            "MedHireUQRAG": 46.5,
        },
    },
    "Llama-3.1-8B": {
        "norag":    "results_llama_local_no_rag.csv",
        "medrag":   "results_llama_medrag.csv",
        "hirerag":  "results_llama.csv",
        "uq_file":  "results_llama_medhireuqrag_0.7_10.csv",
        "uq_col":   "uq_correct",
        "color":    "#1E8449",
        "overall":  {
            "Raw Model":       60.0,
            "MedRAG":       54.0,
            "MedHireRAG":   58.5,
            "MedHireUQRAG": 54.5,
        },
    },
    "Qwen2.5-7B": {
        "norag":    "results_qwen_norag.csv",
        "medrag":   "results_qwen_medrag.csv",
        "hirerag":  "results_qwen.csv",
        "uq_file":  "results_qwen_medhireuqrag_0.7_10.csv",
        "uq_col":   "uq_correct",
        "color":    "#7D3C98",
        "overall":  {
            "Raw Model":       58.5,
            "MedRAG":       56.0,
            "MedHireRAG":   58.5,
            "MedHireUQRAG": 58.5,
        }, 
    },
}

DOMAIN_LABEL_FILES = {
    "BioMistral-7B": "results_biomistral_myrag_v5_6.csv",
    "Llama-3.1-8B":      "results_llama_myrag_v4.csv",
    "Qwen2.5-7B":       "results_qwen_myrag.csv",
}


def load(fname, uq_col=None, hirerag=False):
    if fname is None:
        return None
    if hirerag:
        path = os.path.join(RESULTS_RAG_DIR, fname)
    elif uq_col:
        path = os.path.join(RESULTS_UQ_DIR, fname)
    else:
        path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    df = pd.read_csv(path)
    if uq_col:
        df["is_correct"] = df[uq_col].fillna(False).astype(bool)
    else:
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

def bar_labels(ax, bars, pad=0.5):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + pad,
                    f"{h:.1f}%", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#333333")

log("=" * 60)
log("ALL FOUR: `Raw Model` vs MedRAG vs MedHireRAG vs MedHireUQRAG")
log("=" * 60)

all_stats = {}

for model_name, cfg in configs.items():
    df_norag   = load(cfg["norag"])
    df_medrag  = load(cfg["medrag"])
    df_hirerag = load(cfg["hirerag"], hirerag=True)
    df_uq      = load(cfg["uq_file"], uq_col=cfg["uq_col"])
    df_domain  = load_domain(DOMAIN_LABEL_FILES[model_name])
    if df_domain is None or "domain" not in df_domain.columns:
        log(f"  [SKIP] {model_name} — missing domain column")
        continue

    domain_map = df_domain[["id","domain"]].dropna()

    def merge_domain(df):
        if df is None:
            return None
        if "domain" not in df.columns:
            df = df.merge(domain_map, on="id", how="left")
        df["domain"] = df["domain"].fillna("Unknown")
        return df

    df_norag  = merge_domain(df_norag)
    df_medrag = merge_domain(df_medrag)
    df_hirerag = merge_domain(df_hirerag)
    df_uq     = merge_domain(df_uq)

    domains = sorted(set(
        df_hirerag[df_hirerag.groupby("domain")["domain"]
                   .transform("count") >= 3]["domain"]
    ) - {"Unknown"})

    dfs  = {
        "Raw Model":       df_norag,
        "MedRAG":       df_medrag,
        "MedHireRAG":   df_hirerag,
        "MedHireUQRAG": df_uq,
    }
    accs   = {m: [] for m in dfs}
    counts = []

    for d in domains:
        for method, df in dfs.items():
            if df is not None:
                sub = df[df["domain"]==d]
                accs[method].append(
                    sub["is_correct"].mean()*100 if len(sub)>=3 else np.nan
                )
            else:
                accs[method].append(np.nan)
        counts.append(len(df_hirerag[df_hirerag["domain"]==d]))

    log(f"\n--- {model_name} ---")
    for d, n in zip(domains, counts):
        idx = domains.index(d)
        parts = "  ".join([
            f"{m}={accs[m][idx]:.1f}%" if not np.isnan(accs[m][idx]) else f"{m}=N/A"
            for m in dfs
        ])
        log(f"  {d:<30} {parts}  n={n}")

    all_stats[model_name] = {
        "domains": domains, "accs": accs,
        "counts": counts, "color": cfg["color"],
        "overall": cfg["overall"],
    }

# ── Plot 1: per model — all four methods ──────────────────
for model_name, s in all_stats.items():
    domains = s["domains"]
    accs    = s["accs"]
    counts  = s["counts"]
    overall = s["overall"]

    methods   = ["Raw Model", "MedRAG", "MedHireRAG", "MedHireUQRAG"]
    valid     = [(m) for m in methods
                 if not all(np.isnan(v) if isinstance(v, float)
                            else v is None
                            for v in accs[m])]
    n_methods = len(valid)
    width     = 0.18
    x         = np.arange(len(domains))
    all_vals  = [v for m in valid for v in accs[m]
                 if not (isinstance(v, float) and np.isnan(v))]
    max_val   = max(all_vals) if all_vals else 100

    fig, ax = plt.subplots(figsize=(max(13, len(domains)*4), 10))
    offset_start = -(n_methods - 1) / 2 * width

    for i, method in enumerate(valid):
        vals = [v if not (isinstance(v, float) and np.isnan(v)) else 0
                for v in accs[method]]
        bars = ax.bar(x + offset_start + i*width, vals, width,
                      label=method, color=METHOD_COLORS[method],
                      edgecolor="white", linewidth=0.5)
        bar_labels(ax, bars)

    # Overall summary
    summary_parts = [
        f"{m}: {overall[m]:.1f}%"
        for m in methods if overall.get(m) is not None
    ]

    for i, n in enumerate(counts):
        ax.annotate(f"n={n}",
                    xy=(i, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -28), textcoords="offset points",
                    ha="center", fontsize=8, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, max_val * 1.25)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title(
        f"Per-Domain Accuracy: All Methods — {model_name}",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.legend(fontsize=10, framealpha=0.9,
              loc="upper right", edgecolor="#cccccc")
    plt.subplots_adjust(bottom=0.16)
    out = f"{GRAPHS_DIR}/all_four_{model_name.lower()}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"\nSaved → {out}")

# ── Plot 2: Overall accuracy — all models all methods ─────
fig, ax = plt.subplots(figsize=(16, 10))

model_names = list(all_stats.keys())
methods     = ["Raw Model", "MedRAG", "MedHireRAG", "MedHireUQRAG"]
n_methods   = len(methods)
x           = np.arange(len(model_names))
width       = 0.18
all_vals    = []

for i, method in enumerate(methods):
    vals = [all_stats[m]["overall"].get(method) for m in model_names]
    plot_vals = [v if v is not None else 0 for v in vals]
    bars = ax.bar(x + (i - (n_methods-1)/2) * width, plot_vals, width,
                  label=method, color=METHOD_COLORS[method],
                  edgecolor="white", linewidth=0.5)
    for bar, val, orig in zip(bars, plot_vals, vals):
        if orig is not None and orig > 0:
            ax.text(bar.get_x() + bar.get_width()/2, orig + 0.5,
                    f"{orig:.1f}%", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#333333")
        elif orig is None:
            # Mark N/A on x-axis level
            ax.text(bar.get_x() + bar.get_width()/2, 1.5,
                    "N/A", ha="center", va="bottom",
                    fontsize=7, color="gray", style="italic")
    all_vals.extend([v for v in plot_vals if v > 0])

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=13)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_ylim(0, max(all_vals) * 1.2)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.set_title("Overall Accuracy: All Methods — All Models",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=11, framealpha=0.9,
          loc="upper right", edgecolor="#cccccc")
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/all_four_overall.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"Saved → {GRAPHS_DIR}/all_four_overall.png")

# ── Plot 3: Heatmap — all methods × models × domain ───────
domain_sets    = [set(s["domains"]) for s in all_stats.values()]
common_domains = sorted(set.intersection(*domain_sets)) if domain_sets else []

if common_domains:
    row_labels = []
    matrix     = []

    for model_name in model_names:
        s = all_stats[model_name]
        for method in methods:
            row_labels.append(f"{model_name} — {method}")
            row = []
            for d in common_domains:
                if d in s["domains"]:
                    idx = s["domains"].index(d)
                    val = s["accs"][method][idx]
                    row.append(val if not (isinstance(val, float)
                                          and np.isnan(val)) else np.nan)
                else:
                    row.append(np.nan)
            matrix.append(row)

    matrix = np.array(matrix, dtype=float)
    n_rows = len(row_labels)

    fig, ax = plt.subplots(
        figsize=(max(10, len(common_domains)*2.5), n_rows*0.6 + 1.5)
    )
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=30, vmax=85)

    ax.set_xticks(range(len(common_domains)))
    ax.set_xticklabels(common_domains, fontsize=12)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)

    # Separators between models
    n_m = len(methods)
    for sep in range(n_m, n_rows, n_m):
        ax.axhline(sep - 0.5, color="white", linewidth=2.5)

    for i in range(n_rows):
        for j in range(len(common_domains)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 65 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=text_color)
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=8, color="gray")

    plt.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.6)
    ax.set_title(
        "Accuracy Heatmap: All Methods × All Models × Domain",
        fontsize=12, fontweight="bold", pad=12
    )
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/all_four_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved → {GRAPHS_DIR}/all_four_heatmap.png")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines))
log(f"\nSaved → {SUMMARY_PATH}")
print("Done!")