# eda_textbook.py
"""
EDA on raw MedRAG textbook corpus.
For Chapter 3: Data Description.
"""
import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CORPUS_DIR   = "/vol/bitbucket/hl2622/fyp/corpus/textbooks/chunk"
RESULTS_DIR  = "./results/eda"
GRAPHS_DIR   = "./graphs/eda"
SUMMARY_PATH = f"{RESULTS_DIR}/eda_textbook_summary.txt"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

lines = []
def log(s=""):
    print(s)
    lines.append(s)

log("=" * 60)
log("RAW TEXTBOOK CORPUS EDA")
log("=" * 60)

textbook_files = sorted([f for f in os.listdir(CORPUS_DIR) if f.endswith(".jsonl")])
log(f"Number of textbooks: {len(textbook_files)}")
log()

all_chunks = []
per_book   = {}
for fname in textbook_files:
    chunks = []
    with open(os.path.join(CORPUS_DIR, fname)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            chunks.append({
                "title":   row.get("title", fname.replace(".jsonl", "")),
                "content": row.get("content", ""),
                "id":      row.get("id", ""),
            })
    per_book[fname.replace(".jsonl", "")] = len(chunks)
    all_chunks.extend(chunks)

tb_df = pd.DataFrame(all_chunks)
tb_df["n_words"] = tb_df["content"].str.split().str.len()
tb_df["n_chars"] = tb_df["content"].str.len()

log(f"Total chunks:        {len(tb_df):,}")
log(f"Total words:         {tb_df['n_words'].sum():,}")
log(f"Total chars:         {tb_df['n_chars'].sum():,}")
log()
log("Per-textbook chunk counts:")
for book, cnt in sorted(per_book.items(), key=lambda x: -x[1]):
    pct = cnt / len(tb_df) * 100
    log(f"  {book:<40} {cnt:>6,} ({pct:.1f}%)")
log()
log("Chunk length stats (words):")
log(f"  Mean:   {tb_df['n_words'].mean():.1f}")
log(f"  Median: {tb_df['n_words'].median():.1f}")
log(f"  Std:    {tb_df['n_words'].std():.1f}")
log(f"  Min:    {tb_df['n_words'].min()}")
log(f"  Max:    {tb_df['n_words'].max()}")
log()
log("Chunk length stats (chars):")
log(f"  Mean:   {tb_df['n_chars'].mean():.1f}")
log(f"  Median: {tb_df['n_chars'].median():.1f}")
log(f"  Std:    {tb_df['n_chars'].std():.1f}")
log(f"  Min:    {tb_df['n_chars'].min()}")
log(f"  Max:    {tb_df['n_chars'].max()}")
log()

# Sample chunks
log("Sample chunks:")
for i in [0, 500, 1000]:
    row = tb_df.iloc[i]
    log(f"  [{i}] {row['title']}: {row['content'][:120]}...")
log()

# ── Plots ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Raw Textbook Corpus EDA", fontsize=14, fontweight="bold")

ax = axes[0, 0]
books = list(per_book.keys())
cnts  = [per_book[b] for b in books]
ax.barh([b[:30] for b in books], cnts, color="steelblue")
ax.set_xlabel("Number of chunks")
ax.set_title("Chunks per Textbook")
ax.tick_params(axis="y", labelsize=8)

ax = axes[0, 1]
ax.hist(tb_df["n_words"], bins=40, color="steelblue", edgecolor="white")
ax.axvline(tb_df["n_words"].mean(), color="red", linestyle="--",
           label=f"Mean={tb_df['n_words'].mean():.0f}")
ax.axvline(tb_df["n_words"].median(), color="orange", linestyle="--",
           label=f"Median={tb_df['n_words'].median():.0f}")
ax.set_xlabel("Words per chunk")
ax.set_ylabel("Count")
ax.set_title("Chunk Word Length Distribution")
ax.legend()

ax = axes[1, 0]
ax.hist(tb_df["n_chars"], bins=40, color="orange", edgecolor="white")
ax.axvline(tb_df["n_chars"].mean(), color="red", linestyle="--",
           label=f"Mean={tb_df['n_chars'].mean():.0f}")
ax.set_xlabel("Characters per chunk")
ax.set_ylabel("Count")
ax.set_title("Chunk Character Length Distribution")
ax.legend()

ax = axes[1, 1]
per_book_words = tb_df.groupby("title")["n_words"].sum().sort_values(ascending=False)
ax.barh([t[:30] for t in per_book_words.index[::-1]],
        per_book_words.values[::-1], color="steelblue")
ax.set_xlabel("Total words")
ax.set_title("Total Words per Textbook")
ax.tick_params(axis="y", labelsize=8)

plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/eda_raw_textbook.png", dpi=150, bbox_inches="tight")
log(f"Saved → {GRAPHS_DIR}/eda_raw_textbook.png")
plt.close()

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines))
log(f"Saved → {SUMMARY_PATH}")
print("Done!")