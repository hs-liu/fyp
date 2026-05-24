# eda_umls.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

UMLS_DIR     = "/vol/bitbucket/hl2622/umls/2025AB/META"
RESULTS_DIR  = "./results/eda"
GRAPHS_DIR   = "./graphs/eda"
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

# CUIs per TUI
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

# ── Plots ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Raw UMLS EDA (MRCONSO + MRREL + MRSTY)", fontsize=14, fontweight="bold")

ax = axes[0, 0]
top_stys = mrsty["sty"].value_counts().head(15)
ax.barh(top_stys.index[::-1], top_stys.values[::-1], color="steelblue")
ax.set_xlabel("Count")
ax.set_title("Top 15 Semantic Types (MRSTY)")
ax.tick_params(axis="y", labelsize=7)

ax = axes[0, 1]
top_sabs = mrconso["sab"].value_counts().head(15)
ax.barh(top_sabs.index[::-1], top_sabs.values[::-1], color="orange")
ax.set_xlabel("Count")
ax.set_title("Top 15 Source Vocabularies (MRCONSO)")
ax.tick_params(axis="y", labelsize=7)

ax = axes[0, 2]
syn_clipped = syn_per_cui.clip(upper=50)
ax.hist(syn_clipped, bins=40, color="steelblue", edgecolor="white")
ax.axvline(syn_per_cui.mean(), color="red", linestyle="--",
           label=f"Mean={syn_per_cui.mean():.1f}")
ax.set_xlabel("Synonyms per concept (clipped at 50)")
ax.set_ylabel("Count")
ax.set_title("Synonyms per Concept")
ax.legend()

ax = axes[1, 0]
top_rels = mrrel["rel"].value_counts().head(10)
ax.bar(top_rels.index, top_rels.values, color="steelblue")
ax.set_xlabel("Relation type")
ax.set_ylabel("Count")
ax.set_title("Top 10 Relation Types (MRREL)")
ax.tick_params(axis="x", rotation=45)

ax = axes[1, 1]
top_relas = mrrel["rela"].dropna().value_counts().head(12)
ax.barh([str(r)[:30] for r in top_relas.index[::-1]],
        top_relas.values[::-1], color="orange")
ax.set_xlabel("Count")
ax.set_title("Top 12 Relation Attributes (MRREL)")
ax.tick_params(axis="y", labelsize=7)

ax = axes[1, 2]
top_tui = cuis_per_tui.sort_values(ascending=False).head(20)
ax.bar(top_tui.index, top_tui.values, color="steelblue")
ax.set_xlabel("TUI")
ax.set_ylabel("Unique CUIs")
ax.set_title("Unique CUIs per TUI (Top 20)")
ax.tick_params(axis="x", rotation=90, labelsize=7)

plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/eda_raw_umls.png", dpi=150, bbox_inches="tight")
log(f"Saved → {GRAPHS_DIR}/eda_raw_umls.png")
plt.close()

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines))
log(f"Saved → {SUMMARY_PATH}")
print("Done!")