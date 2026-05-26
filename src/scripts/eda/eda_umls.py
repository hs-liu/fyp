# eda_umls.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

UMLS_DIR     = "/vol/bitbucket/hl2622/umls/2025AB/META"
RESULTS_DIR  = "./src/results/eda"
GRAPHS_DIR   = "./src/graphs/eda/umls"
SUMMARY_PATH = f"{RESULTS_DIR}/eda_umls_summary.txt"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

lines = []
def log(s=""):
    print(s)
    lines.append(s)

log("=" * 60)
log("RAW UMLS EDA")
log("=" * 60)

# ── MRSTY ─────────────────────────────────────────────────
log("\n--- MRSTY.RRF (Semantic Types) ---")
mrsty = pd.read_csv(
    f"{UMLS_DIR}/MRSTY.RRF", sep="|", header=None,
    names=["cui","tui","stn","sty","atui","cvf","_"],
    usecols=["cui","tui","sty"], dtype=str
)
log(f"Total rows:            {len(mrsty):,}")
log(f"Unique CUIs:           {mrsty['cui'].nunique():,}")
log(f"Unique TUIs:           {mrsty['tui'].nunique():,}")
log(f"Unique semantic types: {mrsty['sty'].nunique():,}")
log()
log("Top 20 semantic types:")
for sty, cnt in mrsty["sty"].value_counts().head(20).items():
    log(f"  {sty:<45} {cnt:>8,}")
log()

cuis_per_tui = mrsty.groupby("tui")["cui"].nunique()
log("CUIs per TUI stats:")
log(f"  Mean:   {cuis_per_tui.mean():.1f}")
log(f"  Median: {cuis_per_tui.median():.1f}")
log(f"  Max:    {cuis_per_tui.max():,}")
log()

# ── MRCONSO ───────────────────────────────────────────────
log("--- MRCONSO.RRF (Concepts/Synonyms) ---")
log("Loading English rows...")
mrconso_chunks = []
for chunk in pd.read_csv(
    f"{UMLS_DIR}/MRCONSO.RRF", sep="|", header=None,
    names=["cui","lat","ts","lui","stt","sui","ispref","aui",
           "saui","scui","sdui","sab","tty","code","str",
           "srl","suppress","cvf","_"],
    usecols=["cui","lat","sab","tty","suppress","str"],
    dtype=str, chunksize=500_000
):
    eng = chunk[chunk["lat"] == "ENG"]
    mrconso_chunks.append(eng)
    print(f"  {sum(len(c) for c in mrconso_chunks):,} ENG rows...", end="\r")
mrconso = pd.concat(mrconso_chunks, ignore_index=True)
print()

log(f"Total ENG rows:        {len(mrconso):,}")
log(f"Unique CUIs:           {mrconso['cui'].nunique():,}")
log(f"Unique source SABs:    {mrconso['sab'].nunique():,}")
log()
log("Top 15 source vocabularies (SAB):")
for sab, cnt in mrconso["sab"].value_counts().head(15).items():
    log(f"  {sab:<15} {cnt:>8,}")
log()
log("Term type (TTY) distribution (top 10):")
for tty, cnt in mrconso["tty"].value_counts().head(10).items():
    log(f"  {tty:<15} {cnt:>8,}")
log()

syn_per_cui = mrconso.groupby("cui")["str"].count()
log("Synonyms per concept:")
log(f"  Mean:   {syn_per_cui.mean():.2f}")
log(f"  Median: {syn_per_cui.median():.1f}")
log(f"  Max:    {syn_per_cui.max()}")
log(f"  Min:    {syn_per_cui.min()}")
log()

# ── MRREL ─────────────────────────────────────────────────
log("--- MRREL.RRF (Relations, 5M sample) ---")
mrrel = pd.read_csv(
    f"{UMLS_DIR}/MRREL.RRF", sep="|", header=None,
    names=["cui1","aui1","stype1","rel","cui2","aui2",
           "stype2","rela","rui","srui","sab","sl",
           "rg","dir","suppress","cvf","_"],
    usecols=["cui1","cui2","rel","rela","sab","suppress"],
    dtype=str, nrows=5_000_000
)
log(f"Rows loaded (sample):  {len(mrrel):,}")
log(f"Unique REL types:      {mrrel['rel'].nunique()}")
log(f"Unique RELA types:     {mrrel['rela'].nunique()}")
log()
log("Top 10 relation types (REL):")
for rel, cnt in mrrel["rel"].value_counts().head(10).items():
    log(f"  {rel:<20} {cnt:>8,}")
log()
log("Top 10 relation attributes (RELA):")
for rela, cnt in mrrel["rela"].dropna().value_counts().head(10).items():
    log(f"  {str(rela):<35} {cnt:>8,}")
log()
log("Top 10 source vocabularies in relations:")
for sab, cnt in mrrel["sab"].value_counts().head(10).items():
    log(f"  {sab:<15} {cnt:>8,}")
log()

# ══════════════════════════════════════════════════════════
# PLOTS — one per file, labels inside bars to avoid boundary issues
# ══════════════════════════════════════════════════════════

# Plot 1: Top semantic types (MRSTY)
fig, ax = plt.subplots(figsize=(14, 9))
top_stys = mrsty["sty"].value_counts().head(15)
bars = ax.barh(top_stys.index[::-1], top_stys.values[::-1],
               color="steelblue", edgecolor="white")
ax.set_xlabel("Number of concepts", fontsize=12)
ax.set_title("Top 15 Semantic Types (MRSTY)", fontsize=14, fontweight="bold")
ax.tick_params(axis="y", labelsize=10)
ax.set_xlim(0, top_stys.values.max() * 1.18)  # 18% padding for labels
for bar, val in zip(bars, top_stys.values[::-1]):
    ax.text(bar.get_width() + top_stys.values.max() * 0.01,
            bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", ha="left", fontsize=9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/01_semantic_types.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 2: Top source vocabularies (MRCONSO)
fig, ax = plt.subplots(figsize=(14, 9))
top_sabs = mrconso["sab"].value_counts().head(15)
bars = ax.barh(top_sabs.index[::-1], top_sabs.values[::-1],
               color="orange", edgecolor="white")
ax.set_xlabel("Number of terms", fontsize=12)
ax.set_title("Top 15 Source Vocabularies (MRCONSO)", fontsize=14, fontweight="bold")
ax.tick_params(axis="y", labelsize=10)
ax.set_xlim(0, top_sabs.values.max() * 1.18)
for bar, val in zip(bars, top_sabs.values[::-1]):
    ax.text(bar.get_width() + top_sabs.values.max() * 0.01,
            bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", ha="left", fontsize=9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/02_source_vocabularies.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 4: Top relation types (MRREL) — vertical bars
fig, ax = plt.subplots(figsize=(10, 6))
top_rels = mrrel["rel"].value_counts().head(10)
bars = ax.bar(top_rels.index, top_rels.values,
              color="steelblue", edgecolor="white", width=0.6)
ax.set_xlabel("Relation type", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Top 10 Relation Types (MRREL sample)", fontsize=14, fontweight="bold")
ax.tick_params(axis="x", rotation=30, labelsize=11)
ax.set_ylim(0, top_rels.values.max() * 1.15)
for bar, val in zip(bars, top_rels.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + top_rels.values.max() * 0.01,
            f"{val:,}", ha="center", va="bottom", fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/04_relation_types.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 5: Top relation attributes (MRREL)
fig, ax = plt.subplots(figsize=(14, 9))
top_relas = mrrel["rela"].dropna().value_counts().head(12)
bars = ax.barh([str(r)[:35] for r in top_relas.index[::-1]],
               top_relas.values[::-1], color="orange", edgecolor="white")
ax.set_xlabel("Count", fontsize=12)
ax.set_title("Top 12 Relation Attributes (MRREL sample)", fontsize=14, fontweight="bold")
ax.tick_params(axis="y", labelsize=10)
ax.set_xlim(0, top_relas.values.max() * 1.18)
for bar, val in zip(bars, top_relas.values[::-1]):
    ax.text(bar.get_width() + top_relas.values.max() * 0.01,
            bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", ha="left", fontsize=9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/05_relation_attributes.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 6: Unique CUIs per TUI (top 20)
fig, ax = plt.subplots(figsize=(13, 6))
top_tui = cuis_per_tui.sort_values(ascending=False).head(20)
bars = ax.bar(range(len(top_tui)), top_tui.values,
              color="steelblue", edgecolor="white", width=0.6)
ax.set_xticks(range(len(top_tui)))
ax.set_xticklabels(top_tui.index, rotation=45, ha="right", fontsize=9)
ax.set_xlabel("TUI", fontsize=12)
ax.set_ylabel("Unique CUIs", fontsize=12)
ax.set_title("Unique CUIs per TUI (Top 20)", fontsize=14, fontweight="bold")
ax.set_ylim(0, top_tui.values.max() * 1.15)
for bar, val in zip(bars, top_tui.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + top_tui.values.max() * 0.01,
            f"{val:,}", ha="center", va="bottom", fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/06_cuis_per_tui.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 7: Skewness of semantic type distribution (MRSTY)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Skewness of Semantic Type Distribution (MRSTY)", fontsize=14, fontweight="bold")

sty_counts = mrsty["sty"].value_counts()

# Left: full distribution (log scale) to show skewness
ax = axes[0]
ax.bar(range(len(sty_counts)), sty_counts.values, color="steelblue", edgecolor="none", width=1.0)
ax.set_yscale("log")
ax.set_xlabel("Semantic types (ranked by frequency)", fontsize=12)
ax.set_ylabel("Number of concepts (log scale)", fontsize=12)
ax.set_title("Full Distribution (log scale)", fontsize=12)
ax.axvline(len(sty_counts) * 0.1, color="red", linestyle="--", linewidth=1.5,
           label=f"Top 10% types\nhold {sty_counts.head(int(len(sty_counts)*0.1)).sum()/sty_counts.sum()*100:.0f}% of concepts")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

# Right: CDF to show concentration
ax = axes[1]
cumulative = sty_counts.values.cumsum() / sty_counts.sum() * 100
ax.plot(range(1, len(sty_counts)+1), cumulative, color="steelblue", linewidth=2)
ax.axhline(80, color="red", linestyle="--", linewidth=1.5, label="80% of concepts")
ax.axhline(50, color="orange", linestyle="--", linewidth=1.5, label="50% of concepts")
# Mark where 80% and 50% are reached
idx_80 = next(i for i, v in enumerate(cumulative) if v >= 80)
idx_50 = next(i for i, v in enumerate(cumulative) if v >= 50)
ax.axvline(idx_80+1, color="red", linestyle=":", alpha=0.7,
           label=f"Reached at type #{idx_80+1}")
ax.axvline(idx_50+1, color="orange", linestyle=":", alpha=0.7,
           label=f"Reached at type #{idx_50+1}")
ax.set_xlabel("Number of semantic types (ranked)", fontsize=12)
ax.set_ylabel("Cumulative % of all concepts", fontsize=12)
ax.set_title("Cumulative Distribution (CDF)", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)

# Add skewness annotation
from scipy.stats import skew
sk = skew(sty_counts.values)
axes[0].text(0.97, 0.97, f"Skewness = {sk:.2f}",
             transform=axes[0].transAxes, ha="right", va="top",
             fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
axes[0].text(0.97, 0.85,
             f"Mean   = {sty_counts.mean():,.0f}\nMedian = {sty_counts.median():,.0f}",
             transform=axes[0].transAxes, ha="right", va="top",
             fontsize=10,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/07_semantic_type_skewness.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/07_semantic_type_skewness.png")
# ── Save summary ───────────────────────────────────────────
with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines))
log(f"Saved → {SUMMARY_PATH}")
print("Done!")