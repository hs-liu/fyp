# baseline_groq_medrag.py
import os, re, time, json, signal, faiss, datasets
import pandas as pd
from groq import Groq
from sentence_transformers import SentenceTransformer
from src.scripts.baselines.baseline_utils import format_question, parse_answer

N_TEST        = 200
TOP_K         = 3
MAX_CTX_CHARS = 1800
RESULTS_DIR   = "./results"
CHECKPOINT_PATH = f"{RESULTS_DIR}/results_groq_medrag.csv"
os.makedirs(RESULTS_DIR, exist_ok=True)

CORPUS_BASE = "/vol/bitbucket/hl2622/fyp/corpus/textbooks"
CHUNK_DIR   = f"{CORPUS_BASE}/chunk"
INDEX_PATH  = f"{CORPUS_BASE}/index/ncbi/MedCPT-Article-Encoder/faiss.index"
META_PATH   = f"{CORPUS_BASE}/index/ncbi/MedCPT-Article-Encoder/metadatas.jsonl"

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL  = "llama-3.3-70b-versatile"
DELAY  = 60.0 / 28

# load FAISS
print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

# load metadata
print("Loading metadata...")
metadata = []
with open(META_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            metadata.append(json.loads(line))

# load chunks
print("Loading chunks...")
chunk_lookup = {}
for fname in os.listdir(CHUNK_DIR):
    if not fname.endswith(".jsonl"):
        continue
    with open(os.path.join(CHUNK_DIR, fname)) as f:
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

print("Loading query encoder...")
query_encoder = SentenceTransformer("ncbi/MedCPT-Query-Encoder")

def retrieve(query: str) -> str:
    q_emb = query_encoder.encode([query]).astype("float32")
    _, ids = index.search(q_emb, TOP_K)
    snippets = []
    for vec_id in ids[0]:
        if vec_id < 0 or vec_id >= len(metadata):
            continue
        meta  = metadata[vec_id]
        cid   = str(meta.get("id", meta.get("pmid", "")))
        chunk = chunk_lookup.get(cid)
        if chunk:
            snippets.append(f"[{chunk['title']}]\n{chunk['content'][:600]}")
        else:
            snippets.append(meta.get("title", "") + "\n" + meta.get("abstract", "")[:600])
    return "\n\n".join(snippets)

def build_prompt(sample, context: str) -> str:
    base = format_question(sample)
    return (
        "You are a medical expert. Use the reference passages below to answer "
        "the question. Reply with ONLY the single letter of the correct answer.\n\n"
        f"### Reference passages\n{context[:MAX_CTX_CHARS]}\n\n"
        f"### Question\n{base}"
    )

# dataset
dataset = datasets.load_dataset("bigbio/med_qa", "med_qa_en_source", trust_remote_code=True)
test_ds = list(dataset["test"])[:N_TEST]

# checkpoint
checkpoint_results, done_indices = [], set()
if os.path.exists(CHECKPOINT_PATH):
    done_df = pd.read_csv(CHECKPOINT_PATH)
    done_df["is_correct"] = done_df["is_correct"].fillna(False).astype(bool)
    done_indices = set(done_df["id"].tolist())
    checkpoint_results = done_df.to_dict("records")
    print(f"Resuming — {len(done_indices)}/{N_TEST} already done.")

class TimeoutError(Exception): pass
def _handler(sig, frame): raise TimeoutError()
def run_with_timeout(fn, seconds=25):
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:    return fn()
    finally: signal.alarm(0)

def call_api(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        timeout=30,
    )
    return (resp.choices[0].message.content or "").strip()

def infer(sample, max_retries=4) -> str:
    context = retrieve(sample["question"])
    prompt  = build_prompt(sample, context)
    for attempt in range(max_retries):
        try:
            t0  = time.time()
            raw = run_with_timeout(lambda: call_api(prompt), seconds=25)
            gap = DELAY - (time.time() - t0)
            if gap > 0:
                time.sleep(gap)
            return raw
        except TimeoutError:
            time.sleep(2 ** attempt)
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                import re
                match = re.search(r'retry.?after[: ]+(\d+\.?\d*)', err, re.I)
                wait  = float(match.group(1)) + 2 if match else 30 * (attempt + 1)
                print(f"  [RATE LIMIT] waiting {wait:.0f}s")
                time.sleep(wait)
            else:
                print(f"  [ERROR] {e}")
                return ""
    return ""

# main
remaining = [(i, s) for i, s in enumerate(test_ds) if i not in done_indices]
print(f"Samples left: {len(remaining)}")
results = list(checkpoint_results)

for step, (i, sample) in enumerate(remaining, 1):
    raw    = infer(sample)
    parsed = parse_answer(raw)
    gt     = sample["answer_idx"]
    ok     = parsed == gt

    results.append({
        "id": i, "question": sample["question"],
        "ground_truth": gt, "raw_answer": raw,
        "model_answer": parsed, "is_correct": bool(ok),
    })
    print(f"  [{step:>3}/{len(remaining)}] raw={raw!r:10} parsed={parsed} gt={gt} {'✓' if ok else '✗'}")

    if step % 5 == 0:
        pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
        acc = sum(r["is_correct"] for r in results) / len(results)
        print(f"  [checkpoint] acc: {acc:.2%}")

pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
n_correct = sum(r["is_correct"] for r in results)
print(f"\nFinal accuracy: {n_correct/len(results):.2%} ({n_correct}/{len(results)})")
with open(f"{RESULTS_DIR}/local_model_summary.txt", "a") as f:
    f.write(f"Groq 70B (MedRAG) Accuracy: {n_correct/len(results):.2%} ({n_correct}/{len(results)})\n")