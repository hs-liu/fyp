# scripts/retrieval_pipeline.py
import pickle
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_DIR  = "/vol/bitbucket/hl2622/fyp/data"
MODEL_DIR = "/vol/bitbucket/hl2622/fyp/models/domain_classifier"

import sys
sys.stdout.reconfigure(line_buffering=True)

# ── Load retrieval index ───────────────────────────────────
print("Loading retrieval index...")
G          = pickle.load(open(f"{DATA_DIR}/umls_graph_filtered_new.pkl", "rb"))
corpus_df  = pd.read_parquet(f"{DATA_DIR}/corpus_linked.parquet")
embeddings = np.load(f"{DATA_DIR}/corpus_embeddings.npy")

print("Building CUI index...")
cui_to_rows = {}
for i, cuis_str in enumerate(corpus_df["cuis"]):
    if not cuis_str:
        continue
    for cui in str(cuis_str).split(","):
        cui = cui.strip()
        if cui:
            cui_to_rows.setdefault(cui, []).append(i)

print("Building name lookup...")
name_to_cui = {}
for cui, attrs in G.nodes(data=True):
    name = attrs.get("name", "")
    if name and len(name) > 3:
        name_to_cui[name.lower()] = cui

print(f"Ready. {len(name_to_cui):,} concepts, {len(cui_to_rows):,} indexed CUIs")

# ── Load domain classifier ─────────────────────────────────
print("Loading domain classifier...")
_clf_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
_clf_model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
_clf_device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_clf_model     = _clf_model.to(_clf_device).eval()

_label_map = json.load(open(f"{MODEL_DIR}/label_map.json"))
_id2label  = {int(k): v for k, v in _label_map["id2label"].items()}

# Load temperature if calibrated
try:
    _temperature = json.load(open(f"{MODEL_DIR}/temperature.json"))["temperature"]
    print(f"  Classifier temperature: {_temperature:.4f}")
except FileNotFoundError:
    _temperature = 1.0
    print("  No temperature file, using T=1.0")

# Domain → source routing
# These groups route to textbook (guidelines), rest to pubmed (evidence)
TEXTBOOK_GROUPS = {
    "Disorders", "Physiology", "Anatomy", "Procedures", "Chemicals & Drugs"
}

def classify_domain(query: str) -> tuple[str, str, float]:
    """
    Returns (domain_label, source_filter, confidence)
    source_filter: 'textbook', 'pubmed', or None (both)
    """
    enc = _clf_tokenizer(
        query,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(_clf_device)

    with torch.no_grad():
        logits = _clf_model(**enc).logits / _temperature
        probs  = torch.softmax(logits, dim=-1)[0]

    top_idx   = probs.argmax().item()
    top_label = _id2label[top_idx]
    top_conf  = probs[top_idx].item()

    # Route based on domain
    if top_conf < 0.4:
        source = None  # low confidence → search both
    elif top_label in TEXTBOOK_GROUPS:
        source = "textbook"
    else:
        source = "pubmed"

    return top_label, source, top_conf


# ── Entity extraction ──────────────────────────────────────
def extract_cuis(text: str, max_cuis: int = 10) -> list:
    text_lower = text.lower()
    found = []
    for name, cui in name_to_cui.items():
        if name in text_lower:
            found.append(cui)
        if len(found) >= max_cuis:
            break
    return found


# ── Graph expansion ────────────────────────────────────────
def expand_cuis(seed_cuis: list, hops: int = 1) -> set:
    expanded = set(seed_cuis)
    for cui in seed_cuis:
        if cui in G:
            neighbours = set(G.neighbors(cui))
            expanded.update(neighbours)
            if hops == 2:
                for nb in neighbours:
                    expanded.update(G.neighbors(nb))
    return expanded


# ── Core retrieval ─────────────────────────────────────────
def retrieve(query, query_embedding, top_k=5, source_filter=None, hops=1):
    seed_cuis = extract_cuis(query)
    if seed_cuis:
        expanded = expand_cuis(seed_cuis, hops=hops)
        candidate_rows = set()
        for cui in expanded:
            candidate_rows.update(cui_to_rows.get(cui, []))
        candidate_rows = list(candidate_rows)
    else:
        candidate_rows = list(range(len(corpus_df)))

    if source_filter:
        mask = corpus_df.iloc[candidate_rows]["source"] == source_filter
        candidate_rows = [r for r, m in zip(candidate_rows, mask) if m]

    if not candidate_rows:
        candidate_rows = list(range(len(corpus_df)))

    candidate_embs = embeddings[candidate_rows]
    scores = candidate_embs @ query_embedding
    top_local  = np.argsort(scores)[::-1][:top_k]
    top_global = [candidate_rows[i] for i in top_local]

    results = corpus_df.iloc[top_global].copy()
    results["score"] = scores[top_local]
    return results[["chunk_id", "title", "content", "source", "score"]]


# ── Full hierarchical pipeline ─────────────────────────────
def hierarchical_retrieve(query: str, encoder, top_k: int = 5) -> dict:
    """
    L1: Domain classifier → routes query to textbook / pubmed / both
    L2: Textbook retrieval (guideline-level knowledge)
    L3: PubMed retrieval (evidence-level knowledge)
    """
    # L1 — classify domain
    domain, source_route, confidence = classify_domain(query)

    # Encode query
    q_emb = encoder.encode([query], normalize_embeddings=True)[0]

    # L2/L3 — retrieve based on routing
    if source_route == "textbook":
        l2 = retrieve(query, q_emb, top_k=top_k, source_filter="textbook", hops=1)
        l3 = retrieve(query, q_emb, top_k=2,     source_filter="pubmed",   hops=1)
    elif source_route == "pubmed":
        l2 = retrieve(query, q_emb, top_k=2,     source_filter="textbook", hops=1)
        l3 = retrieve(query, q_emb, top_k=top_k, source_filter="pubmed",   hops=1)
    else:
        # Low confidence — search both equally
        l2 = retrieve(query, q_emb, top_k=3, source_filter="textbook", hops=1)
        l3 = retrieve(query, q_emb, top_k=3, source_filter="pubmed",   hops=1)

    combined = pd.concat([l2, l3]).drop_duplicates("chunk_id")
    context  = "\n\n".join(combined["content"].tolist())

    return {
        "query":       query,
        "domain":      domain,
        "confidence":  confidence,
        "source_route": source_route,
        "context":     context,
        "l2_chunks":   l2,
        "l3_chunks":   l3,
    }