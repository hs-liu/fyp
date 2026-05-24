# rag_utils.py
"""
Shared RAG inference utilities used across all experiment scripts.
Handles context retrieval, prompt building, and UQ computation.
"""
import numpy as np
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
import scripts.rag.retrieval_pipeline as R


# ── Context retrieval ──────────────────────────────────────
# the pubmed score can be ignored 
def get_context(sample, encoder, max_chunk_chars=400,
                pubmed_score_threshold=0.85) -> tuple:
    """
    L2: top-2 textbook chunks (coarse)
    L3: top-1 pubmed chunk if score > threshold (fine, conditioned on L2)
    """
    result = R.hierarchical_retrieve(sample["question"], encoder)
    l2     = result["l2_chunks"]
    l3     = result["l3_chunks"]

    parts = []

    # L2 — coarse clinical knowledge
    for _, row in l2.head(2).iterrows():
        parts.append(f"[Textbook] {row['content'][:max_chunk_chars]}")

    # L3 — fine evidence, only if relevant enough

    parts.append(f"[Evidence] {l3.iloc[0]['content'][:300]}")

    return "\n\n".join(parts)

# ── Prompt building ────────────────────────────────────────
def build_rag_prompt(sample, context: str, format_question_fn) -> str:
    """Build RAG prompt with context injected."""
    base = format_question_fn(sample)
    if context:
        return (
            "You are a medical expert. Use the reference passages below to answer "
            "the question. Reply with ONLY the single letter of the correct answer.\n\n"
            f"### Reference passages\n{context}\n\n"
            f"### Question\n{base}"
        )
    return base


def build_norag_prompt(sample, format_question_fn) -> str:
    """Build plain prompt without context."""
    return format_question_fn(sample)


# ── Checkpoint utilities ───────────────────────────────────
def load_checkpoint(path: str) -> tuple:
    """Load checkpoint CSV. Returns (results_list, done_indices_set)."""
    import os
    import pandas as pd
    if os.path.exists(path):
        done_df = pd.read_csv(path)
        done_df["is_correct"] = done_df["is_correct"].fillna(False).astype(bool)
        done_indices = set(done_df["id"].tolist())
        results = done_df.to_dict("records")
        print(f"Resuming — {len(done_indices)} already done.")
        return results, done_indices
    return [], set()


def save_checkpoint(results: list, path: str, step: int = None, total: int = None):
    """Save results to CSV and print accuracy."""
    import pandas as pd
    pd.DataFrame(results).to_csv(path, index=False)
    acc = sum(r.get("is_correct", r.get("greedy_correct", False)) for r in results) / len(results)
    if step and total:
        print(f"  [checkpoint] {len(results)} saved, acc: {acc:.2%}")


def save_summary(path: str, line: str):
    """Append a line to the summary file."""
    with open(path, "a") as f:
        f.write(line + "\n")