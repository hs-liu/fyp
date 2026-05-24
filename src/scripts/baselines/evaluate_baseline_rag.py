import os, re, time, json, signal, pickle
import datasets
import pandas as pd
import numpy as np
import faiss
from groq import Groq

from scripts.baselines.baseline_utils import format_question, parse_answer

# config 
CORPUS_BASE  = "/vol/bitbucket/hl2622/fyp/corpus/textbooks"
CHUNK_DIR    = f"{CORPUS_BASE}/chunk"
INDEX_PATH   = f"{CORPUS_BASE}/index/ncbi/MedCPT-Article-Encoder/faiss.index"
META_PATH    = f"{CORPUS_BASE}/index/ncbi/MedCPT-Article-Encoder/metadatas.jsonl"
EMBED_DIR    = f"{CORPUS_BASE}/index/ncbi/MedCPT-Article-Encoder/embedding"

N_TEST       = 200
TOP_K        = 3
MAX_CTX_CHARS= 1800
RESULTS_DIR  = "./results"
CHECKPOINT   = f"{RESULTS_DIR}/results_groq_rag.csv"
GROQ_MODEL   = "llama-3.3-70b-versatile"
DELAY        = 60.0 / 28

os.makedirs(RESULTS_DIR, exist_ok=True)
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# load faiss  
print("Loading FAISS index …")
index = faiss.read_index(INDEX_PATH)
print(f"  Index loaded: {index.ntotal:,} vectors, dim={index.d}")

# load metadata 
print("Loading metadata …")
metadata = []
with open(META_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            metadata.append(json.loads(line))
print(f"  Metadata rows: {len(metadata):,}")

#load texts 
print("Loading chunk texts …")
chunk_lookup = {}  
for fname in os.listdir(CHUNK_DIR):
    if not fname.endswith(".jsonl"):
        continue
    fpath = os.path.join(CHUNK_DIR, fname)
    with open(fpath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = str(row.get("id", row.get("pmid", "")))
            chunk_lookup[cid] = {
                "title":   row.get("title", fname.replace(".jsonl", "")),
                "content": row.get("content", row.get("abstract", "")),
            }
print(f"  Chunks loaded: {len(chunk_lookup):,}")

# load query encoder for retrieval 
print("Loading MedCPT query encoder …")
from sentence_transformers import SentenceTransformer
query_encoder = SentenceTransformer("ncbi/MedCPT-Query-Encoder")

#retrieve 
def retrieve(query: str, k: int = TOP_K) -> str:
    q_emb = query_encoder.encode([query], show_progress_bar=True).astype("float32")
    # MedCPT uses inner-product (embeddings are already normalised)
    _, ids = index.search(q_emb, k)

    snippets = []
    for vec_id in ids[0]:
        if vec_id < 0 or vec_id >= len(metadata):
            continue
        meta = metadata[vec_id]
        cid  = str(meta.get("id", meta.get("pmid", "")))
        chunk = chunk_lookup.get(cid)
        if chunk:
            snippets.append(f"[{chunk['title']}]\n{chunk['content'][:600]}")
        else:
            # fall back to whatever metadata has
            snippets.append(meta.get("title", "") + "\n" + meta.get("abstract", "")[:600])

    return "\n\n".join(snippets)

# build prompts
def build_prompt(sample, context: str) -> str:
    base = format_question(sample)
    return (
        "You are a medical expert. Use the reference passages below to answer "
        "the question. Reply with ONLY the single letter of the correct answer.\n\n"
        f"### Reference passages\n{context[:MAX_CTX_CHARS]}\n\n"
        f"### Question\n{base}"
    )

# timeout 
class _Timeout(Exception): pass
def _alarm(sig, frame): raise _Timeout()
def with_timeout(fn, seconds=25):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:    return fn()
    finally: signal.alarm(0)

# groq
def infer(sample, max_retries=4) -> str:
    context = retrieve(sample["question"])
    prompt  = build_prompt(sample, context)

    for attempt in range(max_retries):
        try:
            t0  = time.time()
            raw = with_timeout(lambda: client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                timeout=25,
            ).choices[0].message.content or "", seconds=28)

            gap = DELAY - (time.time() - t0)
            if gap > 0:
                time.sleep(gap)
            return raw.strip()

        except _Timeout:
            wait = 2 ** attempt
            print(f"  [TIMEOUT] attempt {attempt+1}, retry in {wait}s")
            time.sleep(wait)

        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                match = re.search(r'retry.?after[: ]+(\d+\.?\d*)', err, re.I)
                wait  = float(match.group(1)) + 2 if match else 30 * (attempt + 1)
                print(f"  [RATE LIMIT] attempt {attempt+1}, waiting {wait:.0f}s")
                time.sleep(wait)
            else:
                print(f"  [ERROR] {e}")
                return ""
    return ""


# dataset 
print("Loading MedQA …")
med_ds  = datasets.load_dataset("bigbio/med_qa", "med_qa_en_source", trust_remote_code=True)
test_ds = list(med_ds["test"])[:N_TEST]

checkpoint_results, done_indices = [], set()
if os.path.exists(CHECKPOINT):
    done_df = pd.read_csv(CHECKPOINT)
    done_df["is_correct"] = done_df["is_correct"].fillna(False).astype(bool)
    done_indices       = set(done_df["id"].tolist())
    checkpoint_results = done_df.to_dict("records")
    print(f"Resuming — {len(done_indices)}/{N_TEST} already done.")

remaining = [(i, s) for i, s in enumerate(test_ds) if i not in done_indices]
print(f"Samples left: {len(remaining)}")

# main loop results = list(checkpoint_results)
results = list(checkpoint_results)
for step, (i, sample) in enumerate(remaining, 1):
    raw    = infer(sample)
    parsed = parse_answer(raw)
    gt     = sample["answer_idx"]
    ok     = parsed == gt

    results.append({
        "id":           i,
        "question":     sample["question"],
        "ground_truth": gt,
        "raw_answer":   raw,
        "model_answer": parsed,
        "is_correct":   bool(ok),
    })
    print(f"  [{step:>3}/{len(remaining)}]  raw={raw!r:8}  parsed={parsed}  gt={gt}  {'✓' if ok else '✗'}")

    if step % 5 == 0:
        pd.DataFrame(results).to_csv(CHECKPOINT, index=False)
        acc = sum(r["is_correct"] for r in results) / len(results)
        print(f"  [checkpoint] {len(results)} saved — running acc: {acc:.2%}")

# outputs 
pd.DataFrame(results).to_csv(CHECKPOINT, index=False)
n_correct = sum(r["is_correct"] for r in results)
print(f"\nFinal accuracy: {n_correct/len(results):.2%}  ({n_correct}/{len(results)})")
print(f"Results → {CHECKPOINT}")