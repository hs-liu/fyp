# analysis_ablation_progressive.py
"""
Ablation study — section by section following report structure.

5.3.1 KG Only vs No RAG vs MedRAG
5.3.2 KG Only vs KG+Textbook
5.3.3 KG Only vs KG+PubMed
5.3.4 KG Only vs KG+Textbook vs KG+PubMed vs KG+Both
5.4.2 KG+Both vs MedHireRAG (+Classifier)
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR  = "./results"
GRAPHS_DIR   = "./graphs/analysis/ablation"
SUMMARY_PATH = "./results/analysis/ablation_progressive.txt"
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

lines = []
def log(s=""): print(s); lines.append(s)

# ── File mapping ───────────────────────────────────────────
FILES = {
    "BioMistral-7B": {
        "Raw Model":     ("results_local_biomistral.csv",                                    None),
        "MedRAG":     ("results_biomistral_medrag.csv",                                                        None),
        "Graph-based RAG":    ("albation/albation_biomistral_kg_only.csv",          None),
        "Graph-based RAG (Textbook)":("albation/albation_biomistral_textbook.csv",         None),
        "Graph-based RAG (PubMed)":  ("albation/albation_biomistral_pubmed.csv",           None),
        "MedHireRAG": ("medhirerag/results_biomistral.csv",                         None),
    },
    "Llama-3.1-8B": {
        "Raw Model":     ("results_llama_local_no_rag.csv",                            None),
        "MedRAG":     ("results_llama_medrag.csv",                                  None),
        "Graph-based RAG":    ("albation/albation_llama_kg_only.csv",               None),
        "Graph-based RAG (Textbook)":("albation/albation_llama_textbook.csv",              None),
        "Graph-based RAG (PubMed)":  ("albation/albation_llama_pubmed.csv",                None),
        "MedHireRAG": ("medhirerag/results_llama.csv",                                None),
    },
    "Qwen2.5-7B": {
        "Raw Model":     ("results_qwen_norag.csv",                                    None),
        "MedRAG":     ("results_qwen_medrag.csv",                                   None),
        "Graph-based RAG":    ("albation/albation_qwen_kg_only.csv",                None),
        "Graph-based RAG (Textbook)":("albation/albation_qwen_textbook.csv",               None),
        "Graph-based RAG (PubMed)":  ("albation/albation_qwen_pubmed.csv",                 None),
        "MedHireRAG": ("medhirerag/results_qwen.csv",                                    None),
    },
}

MODEL_COLORS = {
    "BioMistral-7B": "#2E86C1",
    "Llama-3.1-8B":      "#1E8449",
    "Qwen2.5-7B":       "#7D3C98",
}

METHOD_COLORS = {
    "Raw Model":      "#E67E22",
    "Graph-based RAG":     "#E74C3C",
    "Graph-based RAG (Textbook)": "#3498DB",
    "Graph-based RAG (PubMed)":   "#9B59B6",
    "MedHireRAG":  "#1ABC9C",
}

# ── Utilities ──────────────────────────────────────────────
def load_df(fname, col=None):
    if fname is None:
        return None
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    df = pd.read_csv(path)
    if col:
        df["is_correct"] = df[col].fillna(False).astype(bool)
    else:
        df["is_correct"] = df["is_correct"].fillna(False).astype(bool)
    return df

def get_acc(df):
    if df is None:
        return np.nan
    return df["is_correct"].mean() * 100

def get_domain_accs(df, domain_map, min_n=3):
    if df is None or domain_map is None:
        return {}
    if "domain" not in df.columns:
        df = df.merge(domain_map, on="id", how="left")
    df = df.copy()
    df["domain"] = df["domain"].fillna("Unknown")
    result = {}
    for d, grp in df[df["domain"] != "Unknown"].groupby("domain"):
        if len(grp) >= min_n:
            result[d] = grp["is_correct"].mean() * 100
    return result

def get_helped_hurt(df_a, df_b):
    if df_a is None or df_b is None:
        return None, None, None, None
    merged = df_a[["id","is_correct"]].merge(
        df_b[["id","is_correct"]], on="id", suffixes=("_a","_b")
    )
    helped     = (( merged["is_correct_b"]==True) & (merged["is_correct_a"]==False)).sum()
    hurt       = ((merged["is_correct_b"]==False) & (merged["is_correct_a"]==True)).sum()
    both_right = (( merged["is_correct_b"]==True) & (merged["is_correct_a"]==True)).sum()
    both_wrong = ((merged["is_correct_b"]==False) & (merged["is_correct_a"]==False)).sum()
    return int(helped), int(hurt), int(both_right), int(both_wrong)

def bar_labels(ax, bars, pad=0.5, fontsize=9):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + pad,
                    f"{h:.1f}%", ha="center", va="bottom",
                    fontsize=fontsize, fontweight="bold", color="#333333")

def plot_accuracy_all_models(methods, title, tag, all_dfs, all_accs):
    """Grouped bar — all models, all methods in one plot."""
    model_names = list(FILES.keys())
    n_methods   = len([m for m in methods if any(
        not np.isnan(all_accs[mn].get(m, np.nan)) for mn in model_names)])
    valid_methods = [m for m in methods if any(
        not np.isnan(all_accs[mn].get(m, np.nan)) for mn in model_names)]

    x     = np.arange(len(model_names))
    width = min(0.7 / len(valid_methods), 0.22)
    all_vals = []

    fig, ax = plt.subplots(figsize=(max(12, len(valid_methods)*3), 7))
    for i, method in enumerate(valid_methods):
        vals   = [all_accs[mn].get(method, np.nan) for mn in model_names]
        offset = (i - (len(valid_methods)-1)/2) * width
        bars   = ax.bar(x + offset,
                        [v if not np.isnan(v) else 0 for v in vals],
                        width, label=method,
                        color=METHOD_COLORS.get(method, "#888"),
                        edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if not np.isnan(val) and val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.4,
                        f"{val:.1f}%", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")
        all_vals.extend([v for v in vals if not np.isnan(v)])

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, max(all_vals)*1.2 if all_vals else 100)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, framealpha=0.9,
              loc="upper left", bbox_to_anchor=(1.01,1),
              borderaxespad=0, edgecolor="#cccccc")
    plt.tight_layout()
    out = f"{GRAPHS_DIR}/{tag}_accuracy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved → {out}")

def plot_accuracy_per_model(methods, title, tag, all_dfs, all_accs):
    """One subplot per model."""
    model_names = list(FILES.keys())
    valid_methods = [m for m in methods if any(
        not np.isnan(all_accs[mn].get(m, np.nan)) for mn in model_names)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, model_name in zip(axes, model_names):
        vals   = [all_accs[model_name].get(m, np.nan) for m in valid_methods]
        valid  = [(i, v) for i, v in enumerate(vals) if not np.isnan(v)]
        xi, yi = [v[0] for v in valid], [v[1] for v in valid]
        colors = [METHOD_COLORS.get(valid_methods[i], "#888") for i in xi]

        bars = ax.bar(range(len(xi)), yi, color=colors,
                      edgecolor="white", linewidth=0.5, width=0.65)
        for bar, val in zip(bars, yi):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.4,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

        # Delta between consecutive
        for j in range(1, len(yi)):
            delta = yi[j] - yi[j-1]
            col   = "#1E8449" if delta >= 0 else "#C0392B"
            ax.annotate(f"Δ{delta:+.1f}%",
                        xy=(j, 0), xycoords=("data","axes fraction"),
                        xytext=(0,-28), textcoords="offset points",
                        ha="center", fontsize=9, fontweight="bold", color=col)

        ax.set_xticks(range(len(xi)))
        ax.set_xticklabels([valid_methods[i] for i in xi],
                           rotation=20, ha="right", fontsize=9)
        ax.set_title(model_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Accuracy (%)" if model_name == model_names[0] else "")
        ax.set_ylim(0, max(yi)*1.25 if yi else 100)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    plt.subplots_adjust(bottom=0.18)
    out = f"{GRAPHS_DIR}/{tag}_per_model.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved → {out}")

def plot_domain(methods, title, tag, all_dfs, domain_maps):
    """Domain accuracy — one subplot per model."""
    model_names   = list(FILES.keys())
    valid_methods = [m for m in methods if any(
        all_dfs[mn].get(m) is not None for mn in model_names)]

    # Use union of all domains
    all_domains = set()
    for mn in model_names:
        dm = domain_maps.get(mn)
        for m in valid_methods:
            da = get_domain_accs(all_dfs[mn].get(m), dm)
            all_domains.update(da.keys())
    all_domains = sorted(all_domains - {"Unknown"})
    if not all_domains:
        log(f"  [SKIP] No domain data for {tag}")
        return

    if len(valid_methods) == 2:
        # Side by side bars with delta
        fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=False)
        fig.suptitle(f"{title} — Domain Accuracy", fontsize=12, fontweight="bold")

        for ax, model_name in zip(axes, model_names):
            dm = domain_maps.get(model_name)
            da = get_domain_accs(all_dfs[model_name].get(valid_methods[0]), dm)
            db = get_domain_accs(all_dfs[model_name].get(valid_methods[1]), dm)
            x  = np.arange(len(all_domains))
            w  = 0.35

            va = [da.get(d, np.nan) for d in all_domains]
            vb = [db.get(d, np.nan) for d in all_domains]

            b1 = ax.bar(x-w/2, [v if not np.isnan(v) else 0 for v in va], w,
                        label=valid_methods[0],
                        color=METHOD_COLORS.get(valid_methods[0],"#888"),
                        edgecolor="white", linewidth=0.5)
            b2 = ax.bar(x+w/2, [v if not np.isnan(v) else 0 for v in vb], w,
                        label=valid_methods[1],
                        color=METHOD_COLORS.get(valid_methods[1],"#2E86C1"),
                        edgecolor="white", linewidth=0.5)
            bar_labels(ax, b1, fontsize=8)
            bar_labels(ax, b2, fontsize=8)

            for xi, (a, b) in enumerate(zip(va, vb)):
                if not (np.isnan(a) or np.isnan(b)):
                    delta = b - a
                    col   = "#1E8449" if delta >= 0 else "#C0392B"
                    ax.annotate(f"Δ{delta:+.1f}%",
                                xy=(xi, 0), xycoords=("data","axes fraction"),
                                xytext=(0,-28), textcoords="offset points",
                                ha="center", fontsize=8, fontweight="bold", color=col)

            max_v = max([v for v in va+vb if not np.isnan(v)] or [100])
            ax.set_xticks(x)
            ax.set_xticklabels(all_domains, fontsize=10)
            ax.set_title(model_name, fontsize=12, fontweight="bold")
            ax.set_ylabel("Accuracy (%)" if model_name == model_names[0] else "")
            ax.set_ylim(0, max_v*1.35)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
            ax.yaxis.grid(True, linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
            ax.legend(fontsize=9)

        plt.subplots_adjust(bottom=0.16)

    else:
        # Heatmap for >2 methods
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"{title} — Domain Heatmap", fontsize=12, fontweight="bold")

        for ax, model_name in zip(axes, model_names):
            dm     = domain_maps.get(model_name)
            matrix = []
            for m in valid_methods:
                da = get_domain_accs(all_dfs[model_name].get(m), dm)
                matrix.append([da.get(d, np.nan) for d in all_domains])
            matrix = np.array(matrix, dtype=float)
            vmin   = np.nanmin(matrix)-2 if not np.all(np.isnan(matrix)) else 0
            vmax   = np.nanmax(matrix)+2 if not np.all(np.isnan(matrix)) else 100

            im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(all_domains)))
            ax.set_xticklabels(all_domains, fontsize=10)
            ax.set_yticks(range(len(valid_methods)))
            ax.set_yticklabels(valid_methods, fontsize=9)
            ax.set_title(model_name, fontsize=12, fontweight="bold")

            for i in range(len(valid_methods)):
                for j in range(len(all_domains)):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        tc = "white" if val > (vmin+(vmax-vmin)*0.6) else "black"
                        ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                                fontsize=9, fontweight="bold", color=tc)
                    else:
                        ax.text(j, i, "N/A", ha="center", va="center",
                                fontsize=8, color="gray")

    out = f"{GRAPHS_DIR}/{tag}_domain.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved → {out}")

def plot_helped_hurt(method_a, method_b, title, tag, all_dfs):
    """Helped/hurt stacked bar — all models."""
    model_names = list(FILES.keys())
    results     = []

    for model_name in model_names:
        df_a = all_dfs[model_name].get(method_a)
        df_b = all_dfs[model_name].get(method_b)
        h, hu, br, bw = get_helped_hurt(df_a, df_b)
        if h is None:
            results.append(None)
        else:
            results.append({"helped": h, "hurt": hu,
                            "both_right": br, "both_wrong": bw,
                            "net": h - hu})
            log(f"  {model_name}: helped={h} hurt={hu} net={h-hu:+d}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"{title} — Helped vs Hurt", fontsize=13, fontweight="bold")

    # Left: stacked bar
    ax = axes[0]
    x  = np.arange(len(model_names))
    w  = 0.5
    bottoms = [0] * len(model_names)

    for label, color in [
        ("both_right", "#5cb85c"),
        ("helped",     METHOD_COLORS.get(method_b, "#2E86C1")),
        ("both_wrong", "lightgray"),
        ("hurt",       METHOD_COLORS.get(method_a, "#C0392B")),
    ]:
        vals = [r[label] if r else 0 for r in results]
        ax.bar(x, vals, w, bottom=bottoms, label=label.replace("_"," ").title(),
               color=color, edgecolor="white", linewidth=0.5)
        bottoms = [b+v for b, v in zip(bottoms, vals)]

    for i, r in enumerate(results):
        if r:
            ax.text(i, bottoms[i]+2, f"net {r['net']:+d}",
                    ha="center", fontsize=10, fontweight="bold",
                    color="#1E8449" if r["net"] >= 0 else "#C0392B")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.set_ylabel("Number of questions", fontsize=12)
    ax.set_title("Outcome breakdown", fontsize=11)
    ax.set_ylim(0, max(bottoms)*1.15 if bottoms else 220)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, framealpha=0.9,
              loc="upper left", bbox_to_anchor=(1.01,1),
              borderaxespad=0, edgecolor="#cccccc")

    # Right: helped vs hurt side by side
    ax = axes[1]
    helped_vals = [r["helped"] if r else 0 for r in results]
    hurt_vals   = [r["hurt"]   if r else 0 for r in results]
    b1 = ax.bar(x-0.2, helped_vals, 0.35,
                label=f"{method_b} helped",
                color=METHOD_COLORS.get(method_b, "#2E86C1"),
                edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x+0.2, hurt_vals, 0.35,
                label=f"{method_b} hurt",
                color=METHOD_COLORS.get(method_a, "#C0392B"),
                edgecolor="white", linewidth=0.5)
    for bar, val in zip(list(b1)+list(b2), helped_vals+hurt_vals):
        if val > 0:
            ax.text(bar.get_x()+bar.get_width()/2, val+0.4,
                    str(val), ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.set_ylabel("Number of questions", fontsize=12)
    ax.set_title("Helped vs Hurt", fontsize=11)
    max_v = max(max(helped_vals or [1]), max(hurt_vals or [1]))
    ax.set_ylim(0, max_v*1.3)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, framealpha=0.9, edgecolor="#cccccc")

    plt.tight_layout()
    out = f"{GRAPHS_DIR}/{tag}_helped_hurt.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved → {out}")

def plot_delta_heatmap(methods, title, tag, all_accs, baseline_method):
    """Delta heatmap vs baseline method."""
    model_names   = list(FILES.keys())
    valid_methods = [m for m in methods if m != baseline_method and any(
        not np.isnan(all_accs[mn].get(m, np.nan)) for mn in model_names)]
    if not valid_methods:
        return

    matrix = []
    for m in valid_methods:
        row = []
        for mn in model_names:
            base = all_accs[mn].get(baseline_method, np.nan)
            val  = all_accs[mn].get(m, np.nan)
            row.append(val - base if not (np.isnan(base) or np.isnan(val)) else np.nan)
        matrix.append(row)

    matrix  = np.array(matrix, dtype=float)
    abs_max = max(np.nanmax(np.abs(matrix)), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(model_names)*3), len(valid_methods)*1.2+2))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                   vmin=-abs_max, vmax=abs_max)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=12)
    ax.set_yticks(range(len(valid_methods)))
    ax.set_yticklabels(valid_methods, fontsize=10)
    ax.set_title(f"{title}\nΔ vs {baseline_method}",
                 fontsize=12, fontweight="bold", pad=12)

    for i in range(len(valid_methods)):
        for j in range(len(model_names)):
            val = matrix[i, j]
            if not np.isnan(val):
                tc = "white" if abs(val) > abs_max*0.6 else "black"
                ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=tc)
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=9, color="gray")

    plt.colorbar(im, ax=ax, label=f"Δ vs {baseline_method} (%)", shrink=0.8)
    plt.tight_layout()
    out = f"{GRAPHS_DIR}/{tag}_delta_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved → {out}")

# ── Load everything ────────────────────────────────────────
log("=" * 60)
log("PROGRESSIVE ABLATION STUDY")
log("=" * 60)

all_dfs  = {}
all_accs = {}
domain_maps = {}

for model_name, methods in FILES.items():
    all_dfs[model_name]  = {}
    all_accs[model_name] = {}

    DOMAIN_LABEL_FILES = {
        "BioMistral-7B": "results_biomistral_myrag_v5_6.csv",
        "Llama-3.1-8B":  "results_llama_myrag_v4.csv",
        "Qwen2.5-7B":    "results_qwen_myrag.csv",
    }

    df_domain = load_df(DOMAIN_LABEL_FILES[model_name])
    if df_domain is not None and "domain" in df_domain.columns:
        domain_maps[model_name] = df_domain[["id","domain"]].dropna()

    log(f"\n--- {model_name} ---")
    for method, (fname, col) in methods.items():
        df  = load_df(fname, col)
        acc = get_acc(df)
        all_dfs[model_name][method]  = df
        all_accs[model_name][method] = acc
        log(f"  {method:<30} {f'{acc:.1f}%' if not np.isnan(acc) else 'N/A'}")
# ══════════════════════════════════════════════════════════
# 5.3.1: KG Only vs No RAG vs MedHireRAG
# ══════════════════════════════════════════════════════════
log("\n" + "="*60)
log("5.3.1: Raw Model vs MedRAG vs Graph-based RAG")
log("="*60)

methods = ["Raw Model", "MedRAG", "Graph-based RAG"]
tag     = "raw_medrag_graphbasedrag"

plot_accuracy_all_models(methods, "Raw Model vs MedRAG vs Graph-based RAG",
                         tag, all_dfs, all_accs)
plot_accuracy_per_model(methods, "Raw Model vs MedRAG vs Graph-based RAG — Per Model",
                        tag, all_dfs, all_accs)
plot_domain(methods, "Per-Domain Accuracy: Raw Model vs MedRAG vs Graph-based RAG",
            tag, all_dfs, domain_maps)
plot_delta_heatmap(methods, "Raw Model vs MedRAG vs Graph-based RAG",
                   tag, all_accs, baseline_method="Raw Model")
# Helped/hurt: Raw Model → Graph-based RAG
plot_helped_hurt("Raw Model", "Graph-based RAG",
                 "Question Outcome: Raw Model vs Graph-based RAG", tag+"_raw_vs_graphbasedrag", all_dfs)
# Helped/hurt: MedRAG Model → Graph-based RAG
plot_helped_hurt("MedRAG", "Graph-based RAG",
                 "Question Outcome: MedRAG vs Graph-based RAG", tag+"_medrag_vs_graphbasedrag", all_dfs)

""" # ══════════════════════════════════════════════════════════
# 5.3.2: KG Only vs KG+Textbook
# ══════════════════════════════════════════
log("\n" + "="*60)
log("5.3.2: Graph-based RAG vs Graph-based RAG (Textbook)")
log("="*60)

methods = ["Graph-based RAG", "Graph-based RAG (Textbook)"]
tag     = "s532_graph_vs_graphtextbook"

plot_accuracy_all_models(methods, "Graph-based RAG vs Graph-based RAG (Textbook)",
                         tag, all_dfs, all_accs)
plot_accuracy_per_model(methods, "Graph-based RAG vs Graph-based RAG (Textbook) — Per Model",
                        tag, all_dfs, all_accs)
plot_domain(methods, "Per-Domain Accuracy: Graph-based RAG vs Graph-based RAG (Textbook)",
            tag, all_dfs, domain_maps)
plot_helped_hurt("Graph-based RAG", "Graph-based RAG (Textbook)",
                 "Question Outcome: Graph-based RAG vs Graph-based RAG (Textbook)", tag, all_dfs)
plot_delta_heatmap(methods, "Graph-based RAG vs Graph-based RAG (Textbook)",
                   tag, all_accs, baseline_method="Graph-based RAG")

# ══════════════════════════════════════════════════════════
# 5.3.3: KG Only vs KG+PubMed
# ══════════════════════════════════════════════════════════
log("\n" + "="*60)
log("5.3.3: Graph-based RAG vs Graph-based RAG (PubMed)")
log("="*60)

methods = ["Graph-based RAG", "Graph-based RAG (PubMed)"]
tag     = "s533_graph_vs_graphtextbook"

plot_accuracy_all_models(methods, "Graph-based RAG vs Graph-based RAG (PubMed)",
                         tag, all_dfs, all_accs)
plot_accuracy_per_model(methods, "Graph-based RAG vs Graph-based RAG (PubMed) — Per Model",
                        tag, all_dfs, all_accs)
plot_domain(methods, "Per-Domain Accuracy: Graph-based RAG vs Graph-based RAG (PubMed)",
            tag, all_dfs, domain_maps)
plot_helped_hurt("Graph-based RAG", "Graph-based RAG (PubMed)",
                 "Question Outcome: Graph-based RAG vs Graph-based RAG (PubMed)", tag, all_dfs)
plot_delta_heatmap(methods, "Graph-based RAG vs Graph-based RAG (PubMed)",
                   tag, all_accs, baseline_method="Graph-based RAG")

# ══════════════════════════════════════════════════════════
# 5.3.4: KG Only vs KG+TB vs KG+PM vs MedHireRAG 
# ══════════════════════════════════════════════════════════
log("\n" + "="*60)
log("5.3.4: Graph-based RAG vs Graph-based RAG (Textbook) vs Graph-based RAG (PubMed) vs MedHireRAG")
log("="*60)

methods = ["Graph-based RAG", "Graph-based RAG (Textbook)", "Graph-based RAG (PubMed)", "MedHireRAG"]
tag     = "s534_corpus_ablation"

plot_accuracy_all_models(methods, "Corpus Source Ablation",
                         tag, all_dfs, all_accs)
plot_accuracy_per_model(methods, "Corpus Source Ablation — Per Model",
                        tag, all_dfs, all_accs)
plot_domain(methods, "Per-Domain Accuracy: Corpus Source Ablation",
            tag, all_dfs, domain_maps)
plot_delta_heatmap(methods, "Corpus Source Ablation",
                   tag, all_accs, baseline_method="Graph-based RAG")
# Helped/hurt: KG Only → KG+Both (the final step)
plot_helped_hurt("Graph-based RAG", "MedHireRAG",
                 "Question Outcome: Graph-based RAG vs MedHireRAG", tag+"_kg_vs_medhirerag", all_dfs)
plot_helped_hurt("Graph-based RAG (Textbook)", "MedHireRAG",
                 "Question Outcome: Graph-based RAG (Textbook) vs MedHireRAG", tag+"_kg_vs_medhirerag", all_dfs)
plot_helped_hurt("Graph-based RAG (PubMed)", "MedHireRAG",
                 "Question Outcome: Graph-based RAG (PubMed) vs MedHireRAG", tag+"_kg_vs_medhirerag", all_dfs)
 """
# ══════════════════════════════════════════════════════════
# OVERALL PROGRESSION
# ══════════════════════════════════════════════════════════

log("\n" + "="*60)
log("OVERALL PROGRESSION")
log("="*60)

ALL_METHODS = ["Raw Model", "MedRAG", "Graph-based RAG","Graph-based RAG (Textbook)","Graph-based RAG (PubMed)","MedHireRAG"]
SHORT       = ["Raw", "MedRAG", "Graph-based RAG","Graph-based RAG (Textbook)","Graph-based RAG (PubMed)", "MedHireRAG"]

fig, ax = plt.subplots(figsize=(14, 7))

for model_name in FILES:
    accs    = [all_accs[model_name].get(m, np.nan) for m in ALL_METHODS]
    valid_x = [i for i, a in enumerate(accs) if not np.isnan(a)]
    valid_y = [accs[i] for i in valid_x]
    ax.plot(valid_x, valid_y, marker="o", linewidth=2.5, markersize=8,
            label=model_name, color=MODEL_COLORS[model_name], zorder=3)
    for xi, yi in zip(valid_x, valid_y):
        ax.annotate(f"{yi:.1f}%", xy=(xi, yi),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8, fontweight="bold",
                    color=MODEL_COLORS[model_name])
    log(f"  {model_name}: " + " → ".join(
        f"{SHORT[i]}={accs[i]:.1f}%"
        for i in valid_x))

ax.set_xticks(range(len(ALL_METHODS)))
ax.set_xticklabels(SHORT, fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Overall Accuracy Progression",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=11, framealpha=0.9, edgecolor="#cccccc")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
all_valid = [a for mn in FILES for m in ALL_METHODS
             for a in [all_accs[mn].get(m, np.nan)] if not np.isnan(a)]
ax.set_ylim(min(all_valid)-5, max(all_valid)+10)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/00_overall_progression.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"Saved → {GRAPHS_DIR}/00_overall_progression.png")

# ══════════════════════════════════════════════════════════
# RAW MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════
log("\n" + "="*60)
log("RAW MODEL PERFORMANCE")
log("="*60)

model_names = list(FILES.keys())
raw_accs    = [all_accs[mn].get("Raw Model", np.nan) for mn in model_names]
all_valid   = [v for v in raw_accs if not np.isnan(v)]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(model_names,
              [v if not np.isnan(v) else 0 for v in raw_accs],
              color=[MODEL_COLORS[mn] for mn in model_names],
              edgecolor="white", linewidth=0.5, width=0.5)

for bar, val in zip(bars, raw_accs):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

# Random baseline reference line
""" ax.axhline(20, color="gray", linestyle="--", linewidth=1.5,
           label="Random baseline (20%)")
ax.text(len(model_names)-0.5, 21,
        "Random baseline (20%)",
        ha="right", fontsize=9, color="gray", style="italic") """

ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Raw Model Accuracy",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0, max(all_valid)*1.2 if all_valid else 100)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/raw_model_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()
log(f"Saved → {GRAPHS_DIR}/raw_model_accuracy.png")

# ══════════════════════════════════════════════════════════
# RAW MODEL vs MEDRAG
# ══════════════════════════════════════════════════════════
log("\n" + "="*60)
log("RAW MODEL vs MEDRAG")
log("="*60)

methods = ["Raw Model", "MedRAG"]
tag     = "raw_vs_medrag"

plot_accuracy_all_models(
    methods,
    "Raw Model vs MedRAG — Overall Accuracy",
    tag, all_dfs, all_accs
)
plot_accuracy_per_model(
    methods,
    "Raw Model vs MedRAG — Per Model",
    tag, all_dfs, all_accs
)
plot_domain(
    methods,
    "Raw Model vs MedRAG — åPer-Domain Accuracy",
    tag, all_dfs, domain_maps
)
plot_delta_heatmap(
    methods,
    "MedRAG Improvement over Raw Model",
    tag, all_accs,
    baseline_method="Raw Model"
)
plot_helped_hurt(
    "Raw Model", "MedRAG",
    "Raw Model vs MedRAG — Question Outcome",
    tag, all_dfs
)
# ── Summary table ──────────────────────────────────────────
log("\n" + "="*60)
log("SUMMARY TABLE")
log("="*60)
log(f"\n{'Model':<12} " +
    " ".join(f"{s:<12}" for s in SHORT))
for model_name in FILES:
    accs = [all_accs[model_name].get(m, np.nan) for m in ALL_METHODS]
    row  = " ".join(f"{a:<12.1f}" if not np.isnan(a) else f"{'N/A':<12}" for a in accs)
    log(f"{model_name:<12} {row}")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines))
print(f"\nDone! → {SUMMARY_PATH}")