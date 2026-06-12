# src/scripts/ablation/ablation_llama.py
import os, torch, datasets
import pandas as pd
import sys
sys.path.insert(0, "/vol/bitbucket/hl2622/fyp/src")
sys.path.insert(0, "/vol/bitbucket/hl2622/fyp")
from src.scripts.baselines.baseline_utils import format_question, parse_answer
from src.scripts.rag.retrieval_utils import load_checkpoint, save_checkpoint, save_summary
from src.scripts.rag.retrieval_pipeline import hierarchical_retrieve_ablation
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

N_TEST      = 200
RESULTS_DIR = "./results/rerun/ablation"
MODEL_PATH  = "/vol/bitbucket/hl2622/fyp/src/models/llama-3.1-8b"
MODEL_NAME  = "Llama-3.1-8B"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("./logs/ablation", exist_ok=True)

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cuda:0",
)
model.eval()

print("Loading encoder...")
encoder = SentenceTransformer(
    'pritamdeka/S-PubMedBert-MS-MARCO',
    cache_folder="/vol/bitbucket/hl2622/huggingface_cache"
)
print("All models loaded.")

dataset = datasets.load_dataset(
    "bigbio/med_qa", "med_qa_en_source",
    trust_remote_code=True,
    cache_dir="/vol/bitbucket/hl2622/huggingface_cache/datasets"
)
test_ds = list(dataset["test"])[:N_TEST]

def call_local(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt"
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
    return tokenizer.decode(
        new_tokens, skip_special_tokens=True).strip()

def build_ablation_prompt(sample, context):
    # Use build_rag_prompt from retrieval_utils — same as main pipeline
    from src.scripts.rag.retrieval_utils import build_rag_prompt
    if context and context.strip():
        return build_rag_prompt(sample, context, format_question)
    else:
        from src.scripts.rag.retrieval_utils import build_norag_prompt
        return build_norag_prompt(sample, format_question)

def run_ablation(mode: str):
    checkpoint_path = f"{RESULTS_DIR}/results_llama_{mode}.csv"
    checkpoint_results, done_indices = load_checkpoint(checkpoint_path)

    remaining = [
        (i, s) for i, s in enumerate(test_ds)
        if i not in done_indices
    ]
    print(f"\n{'='*60}")
    print(f"Mode: {mode} | Samples left: {len(remaining)}")
    print(f"{'='*60}")

    results = list(checkpoint_results)

    for step, (i, sample) in enumerate(remaining, 1):
        # Retrieval
        try:
            retrieval = hierarchical_retrieve_ablation(
                sample["question"], encoder, mode=mode
            )
            context = retrieval.get("context", "")
        except Exception as e:
            print(f"  [RETRIEVAL ERROR] {e}")
            context = ""

        # Inference
        try:
            prompt = build_ablation_prompt(sample, context)
            raw    = call_local(prompt)
        except Exception as e:
            print(f"  [INFERENCE ERROR] {e}")
            raw = ""

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
            "mode":         mode,
        })

        print(f"  [{step:>3}/{len(remaining)}] "
              f"mode={mode} parsed={parsed} gt={gt} "
              f"{'✓' if ok else '✗'}")

        if step % 5 == 0:
            save_checkpoint(
                results, checkpoint_path,
                step, len(remaining))

    save_checkpoint(results, checkpoint_path)
    n_correct = sum(r["is_correct"] for r in results)
    acc = n_correct / len(results)
    print(f"\n[{MODEL_NAME}][{mode}] Accuracy: "
          f"{acc:.2%} ({n_correct}/{len(results)})")
    save_summary(
        f"{RESULTS_DIR}/summary_llama_{mode}.txt",
        f"{MODEL_NAME} [{mode}] Accuracy: "
        f"{acc:.2%} ({n_correct}/{len(results)})"
    )
    return acc

# ── Run all ablation modes ─────────────────────────────────
MODES = ["kg_only", "textbook", "pubmed"]

print(f"\nStarting ablation — {MODEL_NAME}")
summary = {}
for mode in MODES:
    acc = run_ablation(mode)
    summary[mode] = acc

print(f"\n{'='*60}")
print(f"ABLATION SUMMARY — {MODEL_NAME}")
print(f"{'='*60}")
for mode, acc in summary.items():
    print(f"  {mode:<12}: {acc:.2%}")