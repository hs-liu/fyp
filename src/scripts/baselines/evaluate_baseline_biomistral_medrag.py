# baseline_biomistral_medrag.py
import os, json, faiss, torch, datasets
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.scripts.baselines.baseline_utils import format_question, parse_answer

N_TEST          = 200
TOP_K           = 3
MAX_CTX_CHARS   = 1800
RESULTS_DIR     = "./results/appendix"
CHECKPOINT_PATH = f"{RESULTS_DIR}/results_biomistral_medrag.csv"
os.makedirs(RESULTS_DIR, exist_ok=True)

CORPUS_BASE  = "/vol/bitbucket/hl2622/fyp/corpus/textbooks"
CHUNK_DIR    = f"{CORPUS_BASE}/chunk"
INDEX_PATH   = f"{CORPUS_BASE}/index/ncbi/MedCPT-Article-Encoder/faiss.index"
META_PATH    = f"{CORPUS_BASE}/index/ncbi/MedCPT-Article-Encoder/metadatas.jsonl"
MODEL_PATH   = "/vol/bitbucket/hl2622/fyp/src/models/biomistral-7b"

# ── Load FAISS ─────────────────────────────────────────────
print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)
print(f"  {index.ntotal:,} vectors, dim={index.d}")

# ── Load metadata ──────────────────────────────────────────
print("Loading metadata...")
metadata = []
with open(META_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            metadata.append(json.loads(line))

# ── Load chunks ────────────────────────────────────────────
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
print(f"  {len(chunk_lookup):,} chunks")

# ── Load query encoder ─────────────────────────────────────
print("Loading MedCPT query encoder...")
query_encoder = SentenceTransformer("ncbi/MedCPT-Query-Encoder")

def retrieve(query: str, k: int = TOP_K) -> str:
    q_emb = query_encoder.encode([query]).astype("float32")
    _, ids = index.search(q_emb, k)
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
            snippets.append(
                meta.get("title", "") + "\n" + meta.get("abstract", "")[:600]
            )
    return "\n\n".join(snippets)

# ── Load BioMistral ────────────────────────────────────────
print("Loading BioMistral...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="cuda:0",
    dtype=torch.float16,
)
model.eval()
print("Model loaded.")

# ── Dataset ────────────────────────────────────────────────
dataset = datasets.load_dataset(
    "bigbio/med_qa", "med_qa_en_source", trust_remote_code=True
)
test_ds = list(dataset["test"])[:N_TEST]

# ── Checkpoint ─────────────────────────────────────────────
checkpoint_results, done_indices = [], set()
if os.path.exists(CHECKPOINT_PATH):
    done_df = pd.read_csv(CHECKPOINT_PATH)
    done_df["is_correct"] = done_df["is_correct"].fillna(False).astype(bool)
    done_indices = set(done_df["id"].tolist())
    checkpoint_results = done_df.to_dict("records")
    print(f"Resuming — {len(done_indices)}/{N_TEST} already done.")

def build_prompt(sample, context: str) -> str:
    base = format_question(sample)
    return (
        "You are a medical expert. Use the reference passages below to answer "
        "the question. Reply with ONLY the single letter of the correct answer.\n\n"
        f"### Reference passages\n{context[:MAX_CTX_CHARS]}\n\n"
        f"### Question\n{base}"
    )

def call_local(prompt: str) -> str:
    # BioMistral uses standard tokenizer — no chat template
    inputs    = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)
    input_ids = inputs["input_ids"]
    if input_ids[0, -1] == tokenizer.eos_token_id:
        input_ids = input_ids[:, :-1]
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=5,
            min_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def infer(sample) -> str:
    try:
        context = retrieve(sample["question"])
        prompt  = build_prompt(sample, context)
        return call_local(prompt)
    except Exception as e:
        print(f"  [ERROR] {e}")
        return ""

# ── Main loop ──────────────────────────────────────────────
remaining = [(i, s) for i, s in enumerate(test_ds) if i not in done_indices]
print(f"Samples left: {len(remaining)}")
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
    print(f"  [{step:>3}/{len(remaining)}] raw={raw!r:10} parsed={parsed} gt={gt} {'✓' if ok else '✗'}")

    if step % 5 == 0:
        pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
        acc = sum(r["is_correct"] for r in results) / len(results)
        print(f"  [checkpoint] acc: {acc:.2%}")

pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
n_correct = sum(r["is_correct"] for r in results)
print(f"\nFinal accuracy: {n_correct/len(results):.2%} ({n_correct}/{len(results)})")
with open(f"{RESULTS_DIR}/more_test_summary.txt", "a") as f:
    f.write(
        f"BioMistral-7B (MedRAG) Accuracy: "
        f"{n_correct/len(results):.2%} ({n_correct}/{len(results)})\n"
    )