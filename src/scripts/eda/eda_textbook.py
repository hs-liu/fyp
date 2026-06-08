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
GRAPHS_DIR   = "./graphs/eda/textbook"
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
log("Sample chunks:")
for i in [0, 500, 1000]:
    row = tb_df.iloc[i]
    log(f"  [{i}] {row['title']}: {row['content'][:120]}...")
log()

# ══════════════════════════════════════════════════════════
# PLOTS — one per file
# ══════════════════════════════════════════════════════════

# Plot 1: Chunks per textbook
fig, ax = plt.subplots(figsize=(12, 8))
books = list(per_book.keys())
cnts  = [per_book[b] for b in books]
sorted_pairs = sorted(zip(cnts, books), reverse=True)
cnts_s, books_s = zip(*sorted_pairs)
bars = ax.barh([b[:35] for b in books_s], cnts_s,
               color="steelblue", edgecolor="white")
ax.set_xlabel("Number of chunks", fontsize=12)
ax.set_title("Chunks per Textbook", fontsize=14, fontweight="bold")
ax.tick_params(axis="y", labelsize=10)
for bar, val in zip(bars, cnts_s):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/01_chunks_per_textbook.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/01_chunks_per_textbook.png")

# Plot 2: Chunk word length distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(tb_df["n_words"], bins=40, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(tb_df["n_words"].mean(), color="red", linestyle="--", linewidth=2,
           label=f"Mean = {tb_df['n_words'].mean():.0f} words")
ax.axvline(tb_df["n_words"].median(), color="orange", linestyle="--", linewidth=2,
           label=f"Median = {tb_df['n_words'].median():.0f} words")
ax.set_xlabel("Words per chunk", fontsize=12)
ax.set_ylabel("Number of chunks", fontsize=12)
ax.set_title("Chunk Word Length Distribution", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/02_chunk_word_length.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/02_chunk_word_length.png")

# Plot 3: Chunk character length distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(tb_df["n_chars"], bins=40, color="orange", edgecolor="white", alpha=0.85)
ax.axvline(tb_df["n_chars"].mean(), color="red", linestyle="--", linewidth=2,
           label=f"Mean = {tb_df['n_chars'].mean():.0f} chars")
ax.axvline(tb_df["n_chars"].median(), color="blue", linestyle="--", linewidth=2,
           label=f"Median = {tb_df['n_chars'].median():.0f} chars")
ax.set_xlabel("Characters per chunk", fontsize=12)
ax.set_ylabel("Number of chunks", fontsize=12)
ax.set_title("Chunk Character Length Distribution", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/03_chunk_char_length.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/03_chunk_char_length.png")

# Plot 4: Total words per textbook
fig, ax = plt.subplots(figsize=(12, 8))
per_book_words = tb_df.groupby("title")["n_words"].sum().sort_values(ascending=True)
bars = ax.barh([t[:35] for t in per_book_words.index],
               per_book_words.values, color="steelblue", edgecolor="white")
ax.set_xlabel("Total words", fontsize=12)
ax.set_title("Total Words per Textbook", fontsize=14, fontweight="bold")
ax.tick_params(axis="y", labelsize=10)
ax.set_xlim(0, per_book_words.values.max() * 1.18)
for bar, val in zip(bars, per_book_words.values):
    ax.text(bar.get_width() + per_book_words.values.max() * 0.01,
            bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", ha="left", fontsize=9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/04_total_words_per_textbook.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/04_total_words_per_textbook.png")

# Plot 5: Mean chunk length per textbook with std error bars
fig, ax = plt.subplots(figsize=(12, 9))

stats_per_book = tb_df.groupby("title")["n_words"].agg(
    mean="mean", std="std"
).round(1).sort_values("mean", ascending=True)

# Shorten long textbook names
short_names = [t[:35] + "..." if len(t) > 35 else t
               for t in stats_per_book.index]

bars = ax.barh(range(len(stats_per_book)), stats_per_book["mean"], capsize=4,
               color="#2E86C1", edgecolor="white",)

# Annotate mean value
# Get the max x value for consistent label placement
max_x = (stats_per_book["mean"] + stats_per_book["std"]).max()

for i, (mean, std) in enumerate(zip(stats_per_book["mean"],
                                     stats_per_book["std"])):
    ax.text(max_x + 3, i, f"{mean:.0f}",
            va="center", fontsize=8, color="#333")

# Extend x limit to fit labels
ax.set_xlim(0, max_x + 30)

ax.set_yticks(range(len(stats_per_book)))
ax.set_yticklabels(short_names, fontsize=9)
ax.set_xlabel("Mean chunk length (words)", fontsize=11)
ax.set_title("Mean Chunk Word Length per Textbook",
             fontsize=13, fontweight="bold", pad=12)
ax.axvline(tb_df["n_words"].mean(), color="red", linestyle="--",
           linewidth=1.5, label=f"Corpus mean ({tb_df['n_words'].mean():.0f})")
ax.legend(fontsize=10)
ax.xaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/05_mean_chunk_length_per_textbook.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/05_mean_chunk_length_per_textbook.png")
# ── Save summary ───────────────────────────────────────────
with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines))
log(f"Saved → {SUMMARY_PATH}")
print("Done!")