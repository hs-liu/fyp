# eda_pubmed.py
"""
EDA on raw MedRAG PubMed corpus (500k sample).
For Chapter 3: Data Description.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

HF_CACHE     = "/vol/bitbucket/hl2622/huggingface_cache"
RESULTS_DIR  = "./results/eda"
GRAPHS_DIR   = "./graphs/eda"
SUMMARY_PATH = f"{RESULTS_DIR}/eda_pubmed_summary.txt"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

lines = []
def log(s=""):
    print(s)
    lines.append(s)

log("=" * 60)
log("RAW PUBMED CORPUS EDA (MedRAG/pubmed, 500k sample)")
log("=" * 60)

N_SAMPLE = 500_000
print(f"Loading {N_SAMPLE:,} PubMed chunks (streaming)...")
pubmed = load_dataset("MedRAG/pubmed", split="train",
                      streaming=True, cache_dir=HF_CACHE)

chunks = []
for i, row in enumerate(pubmed):
    if i >= N_SAMPLE:
        break
    content = row.get("content", "") or row.get("abstract", "") or ""
    chunks.append({
        "pmid":    row.get("PMID", ""),
        "title":   row.get("title", ""),
        "content": content,
        "n_words": len(content.split()),
        "n_chars": len(content),
    })
    if (i+1) % 50_000 == 0:
        print(f"  {i+1:,} loaded...")

pm_df = pd.DataFrame(chunks)
log(f"Chunks loaded:         {len(pm_df):,}")
log(f"Note: Full PubMed corpus has ~23.9M chunks (238GB)")
log(f"We use {N_SAMPLE:,} ({N_SAMPLE/23_900_000*100:.1f}% of full corpus)")
log()
log(f"Unique PMIDs:          {pm_df['pmid'].nunique():,}")
log(f"Empty chunks (0 words): {(pm_df['n_words']==0).sum():,}")
log(f"Total words:           {pm_df['n_words'].sum():,}")
log()
log("Chunk length stats (words):")
log(f"  Mean:   {pm_df['n_words'].mean():.1f}")
log(f"  Median: {pm_df['n_words'].median():.1f}")
log(f"  Std:    {pm_df['n_words'].std():.1f}")
log(f"  Min:    {pm_df['n_words'].min()}")
log(f"  Max:    {pm_df['n_words'].max()}")
log(f"  <50 words:  {(pm_df['n_words']<50).sum():,} ({(pm_df['n_words']<50).mean()*100:.1f}%)")
log(f"  50-150 words: {((pm_df['n_words']>=50)&(pm_df['n_words']<150)).sum():,}")
log(f"  >150 words: {(pm_df['n_words']>=150).sum():,} ({(pm_df['n_words']>=150).mean()*100:.1f}%)")
log()
log("Chunk length stats (chars):")
log(f"  Mean:   {pm_df['n_chars'].mean():.1f}")
log(f"  Median: {pm_df['n_chars'].median():.1f}")
log(f"  Max:    {pm_df['n_chars'].max()}")
log()

# Sample chunks
log("Sample chunks:")
for i in [0, 1000, 10000]:
    row = pm_df.iloc[i]
    log(f"  [{i}] PMID={row['pmid']}: {row['content'][:120]}...")
log()

# ── Plots ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Raw PubMed Corpus EDA (500k sample)", fontsize=14, fontweight="bold")

ax = axes[0, 0]
pm_clipped = pm_df["n_words"].clip(upper=300)
ax.hist(pm_clipped, bins=50, color="orange", edgecolor="white")
ax.axvline(pm_df["n_words"].mean(), color="red", linestyle="--",
           label=f"Mean={pm_df['n_words'].mean():.0f}")
ax.axvline(pm_df["n_words"].median(), color="blue", linestyle="--",
           label=f"Median={pm_df['n_words'].median():.0f}")
ax.set_xlabel("Words per chunk (clipped at 300)")
ax.set_ylabel("Count")
ax.set_title("PubMed Chunk Word Length")
ax.legend()

ax = axes[0, 1]
bins   = [0, 50, 100, 150, 200, 300, float("inf")]
labels = ["<50", "50-100", "100-150", "150-200", "200-300", ">300"]
pm_df["length_bin"] = pd.cut(pm_df["n_words"], bins=bins, labels=labels)
bin_counts = pm_df["length_bin"].value_counts().sort_index()
ax.bar(bin_counts.index, bin_counts.values, color="orange", edgecolor="white")
ax.set_xlabel("Word count range")
ax.set_ylabel("Number of chunks")
ax.set_title("Chunk Length Bins")
for i, (idx, val) in enumerate(bin_counts.items()):
    ax.text(i, val + 500, f"{val:,}", ha="center", fontsize=8)

ax = axes[1, 0]
ax.hist(pm_df["n_chars"].clip(upper=2000), bins=50,
        color="orange", edgecolor="white")
ax.axvline(pm_df["n_chars"].mean(), color="red", linestyle="--",
           label=f"Mean={pm_df['n_chars'].mean():.0f}")
ax.set_xlabel("Characters per chunk (clipped at 2000)")
ax.set_ylabel("Count")
ax.set_title("PubMed Chunk Character Length")
ax.legend()

ax = axes[1, 1]
sources = ["PubMed\n(500k used)", "Textbooks\n(125k)", "PubMed\n(23.9M full)"]
sizes   = [500_000, 125_847, 23_900_000]
colors  = ["orange", "steelblue", "lightgray"]
bars    = ax.bar(sources, sizes, color=colors)
ax.set_ylabel("Number of chunks")
ax.set_title("Corpus Size Comparison (log scale)")
ax.set_yscale("log")
for bar, val in zip(bars, sizes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
            f"{val:,}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/eda_raw_pubmed.png", dpi=150, bbox_inches="tight")
log(f"Saved → {GRAPHS_DIR}/eda_raw_pubmed.png")
plt.close()

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines))
log(f"Saved → {SUMMARY_PATH}")
print("Done!")