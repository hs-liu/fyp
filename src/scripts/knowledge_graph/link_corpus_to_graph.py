# build_corpus_index.py
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR   = "/vol/bitbucket/hl2622/fyp/src/data"
CORPUS_DIR = "/vol/bitbucket/hl2622/fyp/corpus/textbooks/chunk"
HF_CACHE   = "/vol/bitbucket/hl2622/huggingface_cache"
N_PUBMED   = 500_000

# ── Step 1: Load graph ────────────────────────────────────
print("Loading graph...")
G = pickle.load(open(f"{DATA_DIR}/umls_graph_filtered_new.pkl", "rb"))
print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

name_to_cui = {
    G.nodes[n].get("name","").lower(): n
    for n in G.nodes()
    if G.nodes[n].get("name","") and len(G.nodes[n].get("name","")) > 3
}
print(f"  {len(name_to_cui):,} concept names indexed")

# ── Step 2: Load textbook chunks ──────────────────────────
print("\nLoading textbook chunks...")
textbook_chunks = []
for fpath in sorted(Path(CORPUS_DIR).glob("*.jsonl")):
    book = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line:
                book.append(json.loads(line))
    print(f"  {fpath.name:<45} {len(book):>6,} chunks")
    textbook_chunks.extend(book)
print(f"  Total textbook chunks: {len(textbook_chunks):,}")

# ── Step 3: Load PubMed chunks (streaming) ────────────────
print(f"\nLoading {N_PUBMED:,} PubMed chunks (streaming)...")
pubmed_chunks = []
pubmed_ds = load_dataset(
    "MedRAG/pubmed",
    split="train",
    streaming=True,
    cache_dir=HF_CACHE,
)
for i, row in enumerate(pubmed_ds):
    if i >= N_PUBMED:
        break
    pubmed_chunks.append({
        "id"      : row.get("id", f"pubmed_{i}"),
        "title"   : row.get("title", ""),
        "content" : row.get("content", "") or row.get("abstract", ""),
        "source"  : "pubmed",
    })
    if (i+1) % 50_000 == 0:
        print(f"  loaded {i+1:,} PubMed chunks")
        sys.stdout.flush()
print(f"  Total PubMed chunks: {len(pubmed_chunks):,}")

# ── Step 4: Tag source on textbook chunks ─────────────────
for c in textbook_chunks:
    c["source"] = "textbook"

all_chunks = textbook_chunks + pubmed_chunks
print(f"\nTotal chunks: {len(all_chunks):,}")

# ── Step 5: Entity linking ────────────────────────────────
from collections import defaultdict
import re

print("Building fast concept index...")
first_word_index = defaultdict(list)
for name, cui in name_to_cui.items():
    words = name.split()
    if words:
        first_word_index[words[0]].append((name, cui))
print(f"  First-word index: {len(first_word_index):,} keys")
sys.stdout.flush()

def link_chunk_fast(text, first_word_idx, max_concepts=15):
    text_lower = text.lower()
    found = {}
    words_in_text = set(re.findall(r'\b\w+\b', text_lower))
    for word in words_in_text:
        if word not in first_word_idx:
            continue
        for name, cui in first_word_idx[word]:
            if cui not in found and name in text_lower:
                found[cui] = name
        if len(found) >= max_concepts:
            break
    return list(found.keys())

print("\nLinking chunks to UMLS concepts...")
# add checkpointing every 10k chunks in case of interruption
checkpoint_path = f"{DATA_DIR}/corpus_linking_umls_checkpoint.csv"
checkpoint_interval = 10_000
checkpoint_results, done_indices = [], set()
if os.path.exists(checkpoint_path):
    done_df = pd.read_csv(checkpoint_path)
    done_indices = set(done_df["chunk_id"].tolist())
    checkpoint_results = done_df.to_dict("records")
    print(f"Resuming linking — {len(done_indices):,}/{len(all_chunks):,} already done.")

remaining = [(i, c) for i, c in enumerate(all_chunks) if i not in done_indices]
print(f"Chunks left to link: {len(remaining):,}")

linked = list(checkpoint_results)
# linked = []
for i, chunk in remaining:
    if i in done_indices:
        continue
    text = f"{chunk.get('title','')} {chunk.get('content','')}"
    cuis = link_chunk_fast(text, first_word_index)
    linked.append({
        "chunk_id"  : chunk.get("id", i),
        "title"     : chunk.get("title",""),
        "content"   : chunk.get("content",""),
        "source"    : chunk.get("source",""),
        "text"      : text,
        "cuis"      : ",".join(cuis),
        "n_concepts": len(cuis),
    })
    if (i+1) % 10_000 == 0:
        print(f"  linked {i+1:,}/{len(all_chunks):,}")
        sys.stdout.flush()

df = pd.DataFrame(linked)
print(f"\n  Chunks with ≥1 concept : {(df['n_concepts']>0).sum():,} "
      f"({(df['n_concepts']>0).mean():.1%})")
print(f"  Avg concepts per chunk : {df['n_concepts'].mean():.2f}")
print(f"  Textbook chunks        : {(df['source']=='textbook').sum():,}")
print(f"  PubMed chunks          : {(df['source']=='pubmed').sum():,}")

df.to_parquet(f"{DATA_DIR}/corpus_linked.parquet", index=False)
print(f"  Saved → corpus_linked.parquet")
sys.stdout.flush()

# ── Step 6: Build embeddings ──────────────────────────────
print("\nBuilding embeddings...")
model = SentenceTransformer(
    "pritamdeka/S-PubMedBert-MS-MARCO",
    cache_folder=HF_CACHE,
)

texts = df["text"].tolist()
print(f"  Encoding {len(texts):,} chunks...")

# Encode in two batches — textbooks then pubmed — to allow partial recovery
tb_end = len(textbook_chunks)

print("  Encoding textbooks...")
tb_embs = model.encode(
    texts[:tb_end],
    batch_size=256,
    show_progress_bar=True,
    normalize_embeddings=True,
)
np.save(f"{DATA_DIR}/embeddings_textbooks.npy", tb_embs)
print(f"  Textbook embeddings: {tb_embs.shape} — saved")
sys.stdout.flush()

print("  Encoding PubMed...")
pm_embs = model.encode(
    texts[tb_end:],
    batch_size=256,
    show_progress_bar=True,
    normalize_embeddings=True,
)
np.save(f"{DATA_DIR}/embeddings_pubmed.npy", pm_embs)
print(f"  PubMed embeddings: {pm_embs.shape} — saved")
sys.stdout.flush()

# Combine
all_embs = np.vstack([tb_embs, pm_embs])
np.save(f"{DATA_DIR}/corpus_embeddings.npy", all_embs)
print(f"  Combined embeddings: {all_embs.shape} — saved")

# ── Summary ───────────────────────────────────────────────
print("\n" + "="*50)
print("  CORPUS INDEX SUMMARY")
print("="*50)
print(f"  Total chunks      : {len(df):,}")
print(f"  Textbook chunks   : {(df['source']=='textbook').sum():,}")
print(f"  PubMed chunks     : {(df['source']=='pubmed').sum():,}")
print(f"  Linked chunks     : {(df['n_concepts']>0).sum():,} ({(df['n_concepts']>0).mean():.1%})")
print(f"  Embedding shape   : {all_embs.shape}")
print(f"  Embedding size    : {all_embs.nbytes/1e9:.2f} GB")
print("="*50)
print("\nDone!")