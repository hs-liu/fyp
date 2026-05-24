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
GRAPHS_DIR   = "./graphs/eda/pubmed"
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
log(f"Chunks loaded:          {len(pm_df):,}")
log(f"Note: Full PubMed corpus has ~23.9M chunks (238GB)")
log(f"We use {N_SAMPLE:,} ({N_SAMPLE/23_900_000*100:.1f}% of full corpus)")
log()
log(f"Unique PMIDs:           {pm_df['pmid'].nunique():,}")
log(f"Empty chunks (0 words): {(pm_df['n_words']==0).sum():,}")
log(f"Total words:            {pm_df['n_words'].sum():,}")
log()
log("Chunk length stats (words):")
log(f"  Mean:   {pm_df['n_words'].mean():.1f}")
log(f"  Median: {pm_df['n_words'].median():.1f}")
log(f"  Std:    {pm_df['n_words'].std():.1f}")
log(f"  Min:    {pm_df['n_words'].min()}")
log(f"  Max:    {pm_df['n_words'].max()}")
log(f"  <50 words:    {(pm_df['n_words']<50).sum():,} ({(pm_df['n_words']<50).mean()*100:.1f}%)")
log(f"  50-150 words: {((pm_df['n_words']>=50)&(pm_df['n_words']<150)).sum():,}")
log(f"  >150 words:   {(pm_df['n_words']>=150).sum():,} ({(pm_df['n_words']>=150).mean()*100:.1f}%)")
log()
log("Chunk length stats (chars):")
log(f"  Mean:   {pm_df['n_chars'].mean():.1f}")
log(f"  Median: {pm_df['n_chars'].median():.1f}")
log(f"  Max:    {pm_df['n_chars'].max()}")
log()
log("Sample chunks:")
for i in [0, 1000, 10000]:
    row = pm_df.iloc[i]
    log(f"  [{i}] PMID={row['pmid']}: {row['content'][:120]}...")
log()

# ══════════════════════════════════════════════════════════
# PLOTS — one per file
# ══════════════════════════════════════════════════════════

# Plot 1: Chunk word length distribution
fig, ax = plt.subplots(figsize=(10, 6))
pm_clipped = pm_df["n_words"].clip(upper=300)
ax.hist(pm_clipped, bins=50, color="orange", edgecolor="white", alpha=0.85)
ax.axvline(pm_df["n_words"].mean(), color="red", linestyle="--", linewidth=2,
           label=f"Mean = {pm_df['n_words'].mean():.0f} words")
ax.axvline(pm_df["n_words"].median(), color="blue", linestyle="--", linewidth=2,
           label=f"Median = {pm_df['n_words'].median():.0f} words")
ax.set_xlabel("Words per chunk (clipped at 300)", fontsize=12)
ax.set_ylabel("Number of chunks", fontsize=12)
ax.set_title("PubMed Chunk Word Length Distribution (500k sample)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/01_chunk_word_length.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/01_chunk_word_length.png")

# Plot 2: Chunk length bins
fig, ax = plt.subplots(figsize=(10, 6))
bins   = [0, 50, 100, 150, 200, 300, float("inf")]
labels = ["<50", "50-100", "100-150", "150-200", "200-300", ">300"]
pm_df["length_bin"] = pd.cut(pm_df["n_words"], bins=bins, labels=labels)
bin_counts = pm_df["length_bin"].value_counts().sort_index()
bars = ax.bar(bin_counts.index, bin_counts.values, color="orange",
              edgecolor="white", width=0.6)
ax.set_xlabel("Word count range", fontsize=12)
ax.set_ylabel("Number of chunks", fontsize=12)
ax.set_title("PubMed Chunk Length Bins", fontsize=14, fontweight="bold")
for bar, val in zip(bars, bin_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f"{val:,}", ha="center", fontsize=10, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/02_chunk_length_bins.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/02_chunk_length_bins.png")

# Plot 3: Chunk character length distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(pm_df["n_chars"].clip(upper=2000), bins=50,
        color="orange", edgecolor="white", alpha=0.85)
ax.axvline(pm_df["n_chars"].mean(), color="red", linestyle="--", linewidth=2,
           label=f"Mean = {pm_df['n_chars'].mean():.0f} chars")
ax.axvline(pm_df["n_chars"].median(), color="blue", linestyle="--", linewidth=2,
           label=f"Median = {pm_df['n_chars'].median():.0f} chars")
ax.set_xlabel("Characters per chunk (clipped at 2000)", fontsize=12)
ax.set_ylabel("Number of chunks", fontsize=12)
ax.set_title("PubMed Chunk Character Length Distribution", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/03_chunk_char_length.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/03_chunk_char_length.png")

# Plot 4: Corpus size comparison
fig, ax = plt.subplots(figsize=(10, 6))
sources = ["Textbooks\n(125k)", "PubMed\n(500k used)", "PubMed\n(23.9M full)"]
sizes   = [125_847, 500_000, 23_900_000]
colors  = ["steelblue", "orange", "lightcoral"]
bars    = ax.bar(sources, sizes, color=colors, edgecolor="white", width=0.5)
ax.set_ylabel("Number of chunks (log scale)", fontsize=12)
ax.set_title("Corpus Size Comparison", fontsize=14, fontweight="bold")
ax.set_yscale("log")
ax.set_ylim(min(sizes) * 0.5, max(sizes) * 3)  # padding for labels on log scale
for bar, val in zip(bars, sizes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.4,
            f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/04_corpus_size_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/04_corpus_size_comparison.png")

# Plot 5: Word length CDF
fig, ax = plt.subplots(figsize=(10, 6))
sorted_words = np.sort(pm_df["n_words"].clip(upper=300).values)
cdf = np.arange(1, len(sorted_words)+1) / len(sorted_words)
ax.plot(sorted_words, cdf, color="orange", linewidth=2)
ax.axvline(pm_df["n_words"].median(), color="blue", linestyle="--", linewidth=2,
           label=f"Median = {pm_df['n_words'].median():.0f}")
ax.axvline(pm_df["n_words"].mean(), color="red", linestyle="--", linewidth=2,
           label=f"Mean = {pm_df['n_words'].mean():.0f}")
ax.axhline(0.5, color="gray", linestyle=":", alpha=0.7)
ax.axhline(0.9, color="gray", linestyle=":", alpha=0.7)
ax.set_xlabel("Words per chunk (clipped at 300)", fontsize=12)
ax.set_ylabel("Cumulative proportion", fontsize=12)
ax.set_title("PubMed Chunk Word Length CDF", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/05_word_length_cdf.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/05_word_length_cdf.png")

# ── Save summary ───────────────────────────────────────────
with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines))
log(f"Saved → {SUMMARY_PATH}")
print("Done!")