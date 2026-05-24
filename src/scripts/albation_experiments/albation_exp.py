# ablation_experiment.py
"""
Runs ablation study across all retrieval modes for a given model.
Usage: python3 ablation_experiment.py --model biomistral
       python3 ablation_experiment.py --model llama
       python3 ablation_experiment.py --model qwen
"""
import os, torch, datasets, argparse
import pandas as pd
import sys
sys.path.insert(0, "/vol/bitbucket/hl2622/fyp")
sys.path.insert(0, "/vol/bitbucket/hl2622/fyp/MedRAG/src")
from scripts.baselines.baseline_utils import format_question, parse_answer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from scripts.rag.retrieval_utils import build_rag_prompt, load_checkpoint, save_summary
import scripts.rag.retrieval_pipeline as R

# ── Args ───────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["biomistral", "llama", "qwen"], required=True)
parser.add_argument("--mode", choices=["kg_only", "textbook", "pubmed", "both", "no_classifier"], required=True)
args = parser.parse_args()

MODEL_MAP = {
    "biomistral": "/vol/bitbucket/hl2622/fyp/models/biomistral-7b",
    "llama":      "/vol/bitbucket/hl2622/fyp/models/llama-3.1-8b",
    "qwen":       "/vol/bitbucket/hl2622/fyp/models/qwen2.5-7b",
}

N_TEST      = 200
RESULTS_DIR = "./results/ablation"
SUMMARY_PATH = "./results/ablation_summary.txt"
os.makedirs(RESULTS_DIR, exist_ok=True)

CHECKPOINT_PATH = f"{RESULTS_DIR}/ablation_{args.model}_{args.mode}.csv"
MODEL_PATH      = MODEL_MAP[args.model]

# ── Load model ─────────────────────────────────────────────
print(f"Loading {args.model}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.float16, device_map="cuda:0",
)
model.eval()

print("Loading encoder...")
encoder = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
print("All loaded.")

# ── Dataset ────────────────────────────────────────────────
dataset = datasets.load_dataset("bigbio/med_qa", "med_qa_en_source", trust_remote_code=True)
test_ds = list(dataset["test"])[:N_TEST]

# ── Checkpoint ─────────────────────────────────────────────
checkpoint_results, done_indices = [], set()
if os.path.exists(CHECKPOINT_PATH):
    done_df = pd.read_csv(CHECKPOINT_PATH)
    done_df["is_correct"] = done_df["is_correct"].fillna(False).astype(bool)
    done_indices = set(done_df["id"].tolist())
    checkpoint_results = done_df.to_dict("records")
    print(f"Resuming — {len(done_indices)} done.")

# ── Inference ──────────────────────────────────────────────
def call_model(prompt: str) -> str:
    is_llama_style = args.model in ["llama", "qwen"]
    if is_llama_style:
        messages  = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
    else:
        input_ids = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=2048,
        ).input_ids.to(model.device)
        if input_ids[0, -1] == tokenizer.eos_token_id:
            input_ids = input_ids[:, :-1]

    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=5, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def infer(sample) -> tuple:
    try:
        if args.mode == "no_classifier":
            result  = R.hierarchical_retrieve_no_classifier(
                sample["question"], encoder
            )
            parts = []
            for _, row in result["l2_chunks"].head(2).iterrows():
                parts.append(f"[Textbook] {row['content'][:400]}")
            l3 = result["l3_chunks"]
            if len(l3) > 0 and l3.iloc[0]["score"] > 0.85:
                parts.append(f"[Evidence] {l3.iloc[0]['content'][:300]}")
            context = "\n\n".join(parts)
            route   = "both_equal"
        else:
            result  = R.hierarchical_retrieve_ablation(
                sample["question"], encoder, mode=args.mode
            )
            context = result["context"]
            route   = result["source_route"]

        base = format_question(sample)
        if context:
            prompt = (
                "You are a medical expert. Use the reference passages below to answer "
                "the question. Reply with ONLY the single letter of the correct answer.\n\n"
                f"### Reference passages\n{context}\n\n"
                f"### Question\n{base}"
            )
        else:
            prompt = base

        return call_model(prompt), route

    except Exception as e:
        print(f"  [ERROR] {e}")
        return "", ""

# ── Main loop ──────────────────────────────────────────────
remaining = [(i, s) for i, s in enumerate(test_ds) if i not in done_indices]
print(f"\nModel: {args.model} | Mode: {args.mode} | Samples left: {len(remaining)}")
results = list(checkpoint_results)

for step, (i, sample) in enumerate(remaining, 1):
    raw, route = infer(sample)
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
        "mode":         args.mode,
        "route":        route,
    })

    print(f"  [{step:>3}/{len(remaining)}] parsed={parsed} gt={gt} {'✓' if ok else '✗'}")

    if step % 5 == 0:
        pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
        acc = sum(r["is_correct"] for r in results) / len(results)
        print(f"  [checkpoint] acc: {acc:.2%}")

pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
n_correct = sum(r["is_correct"] for r in results)
acc = n_correct / len(results)
print(f"\nFinal accuracy: {acc:.2%} ({n_correct}/{len(results)})")
save_summary(SUMMARY_PATH,
    f"{args.model:12} | {args.mode:10} | {acc:.2%} ({n_correct}/{len(results)})")