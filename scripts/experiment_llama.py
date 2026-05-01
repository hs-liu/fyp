# baseline_llama_myrag.py
import os, torch, datasets
import pandas as pd
import sys
sys.path.insert(0, "/vol/bitbucket/hl2622/fyp")
sys.path.insert(0, "/vol/bitbucket/hl2622/fyp/MedRAG/src")
from baseline_utils import format_question, parse_answer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import retrieval_pipeline as R

N_TEST          = 200
MAX_CHUNK_CHARS = 400
RESULTS_DIR     = "./results"
CHECKPOINT_PATH = f"{RESULTS_DIR}/results_llama_myrag_v3_no_rerank.csv"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = "/vol/bitbucket/hl2622/fyp/models/llama-3.1-8b"

print("Loading Llama...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="cuda:0",
)
model.eval()

print("Loading encoder...")
encoder = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
print("All models loaded.")

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

""" def rerank_chunks(chunks_df, query: str, top_k: int = 1):
    query_words = set(query.lower().split())
    scores = []
    for _, row in chunks_df.iterrows():
        content_words = set(row["content"].lower().split())
        overlap = len(query_words & content_words) / (len(query_words) + 1)
        scores.append(row["score"] * 0.7 + overlap * 0.3)
    chunks_df = chunks_df.copy()
    chunks_df["rerank_score"] = scores
    return chunks_df.nlargest(top_k, "rerank_score") """

def call_local(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def infer(sample) -> tuple:
    try:
        result     = R.hierarchical_retrieve(sample["question"], encoder)
        confidence = result["confidence"]
        domain     = result["domain"]
        route      = result["source_route"]

        if confidence > 0.55:
            parts = []
            # top-2 textbook, no reranking
            for _, row in result["l2_chunks"].head(2).iterrows():
                parts.append(f"[Textbook] {row['content'][:400]}")
            # top-1 pubmed only if score high enough
            l3 = result["l3_chunks"]
            if len(l3) > 0 and l3.iloc[0]["score"] > 0.80:
                parts.append(f"[Evidence] {l3.iloc[0]['content'][:300]}")
            context = "\n\n".join(parts)
        else:
            context = ""

    except Exception as e:
        print(f"  [ERROR] {e}")
        context, domain, route = "", "", ""

    base_prompt = format_question(sample)
    if context:
        prompt = (
            "You are a medical expert. Use the reference passages below to answer "
            "the question. Reply with ONLY the single letter of the correct answer.\n\n"
            f"### Reference passages\n{context}\n\n"
            f"### Question\n{base_prompt}"
        )
    else:
        prompt = base_prompt

    try:
        return call_local(prompt), domain, route
    except Exception as e:
        print(f"  [ERROR] {e}")
        return "", domain, route

# main
remaining = [(i, s) for i, s in enumerate(test_ds) if i not in done_indices]
print(f"Samples left: {len(remaining)}")
results = list(checkpoint_results)

for step, (i, sample) in enumerate(remaining, 1):
    raw, domain, route = infer(sample)
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
        "domain":       domain,
        "source_route": route,
    })
    print(f"  [{step:>3}/{len(remaining)}] domain={domain} route={route} "
          f"parsed={parsed} gt={gt} {'✓' if ok else '✗'}")

    if step % 5 == 0:
        pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
        acc = sum(r["is_correct"] for r in results) / len(results)
        print(f"  [checkpoint] acc: {acc:.2%}")

pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
n_correct = sum(r["is_correct"] for r in results)
print(f"\nFinal accuracy: {n_correct/len(results):.2%} ({n_correct}/{len(results)})")
with open(f"{RESULTS_DIR}/local_model_summary.txt", "a") as f:
    f.write(f"Llama-3.1-8B (My RAG v3, no reranking) Accuracy: {n_correct/len(results):.2%} ({n_correct}/{len(results)})\n")