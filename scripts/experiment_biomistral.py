# baseline_biomistral_myrag.py
import os, torch, datasets
import pandas as pd
import sys
sys.path.insert(0, "/vol/bitbucket/hl2622/fyp")
sys.path.insert(0, "/vol/bitbucket/hl2622/fyp/MedRAG/src")
from baseline_utils import format_question, parse_answer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import retrieval_pipeline as R

load_dotenv()

N_TEST          = 200
MAX_CHUNK_CHARS = 400
RESULTS_DIR     = "./results"
CHECKPOINT_PATH = f"{RESULTS_DIR}/results_biomistral_myrag_v5_6.csv"
SUMMARY_PATH    = f"{RESULTS_DIR}/local_model_summary.txt"
os.makedirs(RESULTS_DIR, exist_ok=True)

BIOMISTRAL_PATH = "/vol/bitbucket/hl2622/fyp/models/biomistral-7b"

print("Loading BioMistral...")
bm_tokenizer = AutoTokenizer.from_pretrained(BIOMISTRAL_PATH)
bm_model = AutoModelForCausalLM.from_pretrained(
    BIOMISTRAL_PATH,
    device_map="cuda:0",
    dtype=torch.float16,
)
bm_model.eval()

print("Loading encoder...")
encoder = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
print("All models loaded.")

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

def biomistral_rag_fn(sample):
    try:
        result     = R.hierarchical_retrieve(sample["question"], encoder)
        confidence = result["confidence"]
        domain     = result["domain"]
        route      = result["source_route"]

        """ if confidence > 0.60:
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
            context = "" """
        parts = []
        # top-2 textbook, no reranking
        for _, row in result["l2_chunks"].head(2).iterrows():
            parts.append(f"[Textbook] {row['content'][:400]}")
        # top-1 pubmed only if score high enough
        l3 = result["l3_chunks"]
        parts.append(f"[Evidence] {l3.iloc[0]['content'][:300]}")
        context = "\n\n".join(parts)

    except Exception as e:
        print(f"  [RETRIEVAL ERROR] {e}")
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

    inputs = bm_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(bm_model.device)

    input_ids = inputs["input_ids"]
    if input_ids[0, -1] == bm_tokenizer.eos_token_id:
        input_ids = input_ids[:, :-1]
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = bm_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=5,
            min_new_tokens=1,
            do_sample=False,
            pad_token_id=bm_tokenizer.eos_token_id,
        )

    new_token_ids = output_ids[0][input_ids.shape[1]:]
    raw = bm_tokenizer.decode(new_token_ids, skip_special_tokens=True)
    return raw, domain, route

# main
remaining = [(i, s) for i, s in enumerate(test_ds) if i not in done_indices]
print(f"Samples left: {len(remaining)}")
results = list(checkpoint_results)

for step, (i, sample) in enumerate(remaining, 1):
    raw, domain, route = biomistral_rag_fn(sample)
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
        print(f"  [checkpoint] acc so far: {acc:.2%}")

pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
n_correct = sum(r["is_correct"] for r in results)
accuracy  = n_correct / len(results) if results else 0
print(f"\nFinal accuracy: {accuracy:.2%} ({n_correct}/{len(results)})")
with open(SUMMARY_PATH, "a") as f:
    f.write(f"BioMistral-7B (My RAG v5, remove confidence) Accuracy: {accuracy:.2%} ({n_correct}/{len(results)})\n")
print(f"Results saved → {CHECKPOINT_PATH}")