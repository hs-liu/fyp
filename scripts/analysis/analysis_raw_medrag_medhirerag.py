# analysis_all_methods.py
"""
Combined comparison: No RAG vs MedRAG vs My RAG
All models, all domains.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR  = "./results"
GRAPHS_DIR   = "./graphs/analysis/overall_comparison"
SUMMARY_PATH = "./results/analysis/raw_medrag_medhirerag.txt"
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

configs = {
    "BioMistral": {
        "norag":   "results_biomistral.csv",
        "medrag":  None,
        "medhirerag":     "results_biomistral_myrag_v5_6.csv",
        "color":   "#2E86C1",
        "overall": {"norag": 41.5, "medrag": None, "medhirerag": 48.5},
    },
    "Llama": {
        "norag":   "results_llama_local_no_rag.csv",
        "medrag":  "results_llama_medrag.csv",
        "medhirerag":     "results_llama_myrag_v4.csv",
        "color":   "#1E8449",
        "overall": {"norag": 60.0, "medrag": 54.0, "medhirerag": 57.5},
    },
    "Qwen": {
        "norag":   "results_qwen_norag.csv",
        "medrag":  "results_qwen_medrag.csv",
        "medhirerag":     "results_qwen_myrag.csv",
        "color":   "#7D3C98",
        "overall": {"norag": 58.5, "medrag": 56.0, "medhirerag": 59.0},
    },
}

METHOD_COLORS = {
    "Raw Model":  "#C0392B",
    "MedRAG":  "#E67E22",
    "MedHireRAG":  "#2E86C1",
}

def load(fname):
    if fname is None:
        return None
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
                    fontsize=8, fontweight="bold", color="#333333")

log("=" * 60)
log("METHODS COMPARISON: Raw Model vs MedRAG vs MedHireRAG")
log("=" * 60)

all_stats = {}

for model_name, cfg in configs.items():
    df_norag  = load(cfg["norag"])
    df_medrag = load(cfg["medrag"])
    df_medhirerag = load(cfg["medhirerag"])
    if df_medhirerag is None or "domain" not in df_medhirerag.columns:
        continue

    domain_map = df_medhirerag[["id","domain"]].dropna()

    def merge_domain(df):
        if df is None:
            return None
        if "domain" not in df.columns:
            df = df.merge(domain_map, on="id", how="left")
        df["domain"] = df["domain"].fillna("Unknown")
        return df

    df_norag  = merge_domain(df_norag)
    df_medrag = merge_domain(df_medrag)

    domains = sorted(set(
        df_medhirerag[df_medhirerag.groupby("domain")["domain"]
               .transform("count") >= 3]["domain"]
    ) - {"Unknown"})

    accs   = {"norag": [], "medrag": [], "medhirerag": []}
    dfs    = {"norag": df_norag, "medrag": df_medrag, "medhirerag": df_medhirerag}
    counts = []

    for d in domains:
        for key, df in dfs.items():
            if df is not None:
                sub = df[df["domain"]==d]
                accs[key].append(sub["is_correct"].mean()*100
                                 if len(sub)>=3 else np.nan)
            else:
                accs[key].append(np.nan)
        counts.append(len(df_medhirerag[df_medhirerag["domain"]==d]))

    log(f"\n--- {model_name} ---")
    for d, na, ma, ra, n in zip(
        domains, accs["norag"], accs["medrag"], accs["medhirerag"], counts
    ):
        log(f"  {d:<30} "
            f"Raw Model={na:.1f}%  "
            f"MedRAG={'N/A' if np.isnan(ma) else f'{ma:.1f}%'}  "
            f"MedHireRAG={ra:.1f}%  n={n}")

    all_stats[model_name] = {
        "domains": domains, "accs": accs,
        "counts": counts, "color": cfg["color"],
        "overall": cfg["overall"],
    }

# ── Plot 1: per model ──────────────────────────────────────
for model_name, s in all_stats.items():
    domains = s["domains"]
    accs    = s["accs"]
    counts  = s["counts"]
    overall = s["overall"]

    methods = [("Raw Model", "norag"), ("MedRAG", "medrag"), ("MedHireRAG", "medhirerag")]
    valid   = [(m, k) for m, k in methods
               if not all(np.isnan(v) if isinstance(v, float)
                          else v is None for v in accs[k])]

    n_methods = len(valid)
    width     = 0.22
    x         = np.arange(len(domains))
    all_vals  = [v for _, k in valid for v in accs[k]
                 if not (isinstance(v, float) and np.isnan(v))]
    max_val   = max(all_vals) if all_vals else 100

    fig, ax = plt.subplots(figsize=(max(12, len(domains)*3.5), 7))

    offset_start = -(n_methods - 1) / 2 * width
    for i, (method, key) in enumerate(valid):
        vals = [v if not (isinstance(v, float) and np.isnan(v)) else 0
                for v in accs[key]]
        bars = ax.bar(x + offset_start + i*width, vals, width,
                      label=method,
                      color=METHOD_COLORS[method],
                      edgecolor="white", linewidth=0.5)
        bar_labels(ax, bars, pad=0.5)

    # Overall summary
    summary_parts = []
    label_map = {"norag": "Raw Model", "medrag": "MedRAG", "medhirerag": "MedHireRAG"}
    for key, label in label_map.items():
        val = overall.get(key)
        if val is not None:
            summary_parts.append(f"{label}: {val:.1f}%")

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
    ax.set_title(f"Per-Domain Accuracy: Raw Model vs MedRAG vs MedHireRAG — {model_name}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=11, framealpha=0.9,
              loc="upper right", edgecolor="#cccccc")
    plt.subplots_adjust(bottom=0.16)
    out = f"{GRAPHS_DIR}/three_methods_{model_name.lower()}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"\nSaved → {out}")

# ── Plot 2: Overall accuracy — all models ─────────────────
fig, ax = plt.subplots(figsize=(12, 7))

model_names = list(all_stats.keys())
methods     = [("Raw Model", "norag"), ("MedRAG", "medrag"), ("MedHireRAG", "medhirerag")]
n_methods   = len(methods)
x           = np.arange(len(model_names))
width       = 0.22
all_vals    = []

for i, (method, key) in enumerate(methods):
    vals = [all_stats[m]["overall"].get(key) for m in model_names]
    vals = [v if v is not None else 0 for v in vals]
    bars = ax.bar(x + (i - (n_methods-1)/2) * width, vals, width,
                  label=method, color=METHOD_COLORS[method],
                  edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#333333")
    all_vals.extend([v for v in vals if v > 0])

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=13)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_ylim(0, max(all_vals) * 1.2)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.set_title("Overall Accuracy: Raw Model vs MedRAG vs MedHireRAG — All Models",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=11, framealpha=0.9,
          loc="upper right", edgecolor="#cccccc")
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/three_methods_overall.png",
            dpi=150, bbox_inches="tight")
plt.close()
log(f"Saved → {GRAPHS_DIR}/three_methods_overall.png")

# ── Plot 3: Heatmap — all methods × all models × domain ───
domain_sets    = [set(s["domains"]) for s in all_stats.values()]
common_domains = sorted(set.intersection(*domain_sets)) if domain_sets else []

if common_domains:
    methods_labels = ["Raw Model", "MedRAG", "MedHireRAG"]
    keys_map       = ["norag",  "medrag",  "medhirerag"]
    row_labels     = []
    matrix         = []

    for model_name in model_names:
        s = all_stats[model_name]
        for key, label in zip(keys_map, methods_labels):
            row_labels.append(f"{model_name} — {label}")
            row = []
            for d in common_domains:
                if d in s["domains"]:
                    idx = s["domains"].index(d)
                    val = s["accs"][key][idx]
                    row.append(val if not (isinstance(val, float)
                                          and np.isnan(val)) else np.nan)
                else:
                    row.append(np.nan)
            matrix.append(row)

    matrix = np.array(matrix, dtype=float)
    n_rows = len(row_labels)

    fig, ax = plt.subplots(
        figsize=(max(10, len(common_domains)*2.5), n_rows*0.65 + 1.5)
    )
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=30, vmax=85)

    ax.set_xticks(range(len(common_domains)))
    ax.set_xticklabels(common_domains, fontsize=12)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Separator between models
    n_m = len(methods_labels)
    for sep in range(n_m, n_rows, n_m):
        ax.axhline(sep - 0.5, color="white", linewidth=2.5)

    for i in range(n_rows):
        for j in range(len(common_domains)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 65 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_color)
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=9, color="gray")

    plt.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.6)
    ax.set_title("Accuracy Heatmap: Raw Model vs MedRAG vs MedHireRAG × Model × Domain",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/three_methods_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved → {GRAPHS_DIR}/three_methods_heatmap.png")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines))
log(f"\nSaved → {SUMMARY_PATH}")
print("Done!")