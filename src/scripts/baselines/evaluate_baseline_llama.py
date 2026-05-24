# baseline_llama_local.py
import os, re, time, datasets
import pandas as pd
from scripts.baselines.baseline_utils import format_question, parse_answer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# basic setup
N_TEST = 500
RESULTS_DIR = "./results/appendix"
CHECKPOINT_PATH = f"{RESULTS_DIR}/results_llama_raw.csv"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = "/vol/bitbucket/hl2622/fyp/models/llama-3.1-8b"

# load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="auto",
)
model.eval()
print("Model loaded.")

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

def call_local(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=5,
            min_new_tokens=1,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def infer(sample) -> str:
    prompt = format_question(sample)
    try:
        return call_local(prompt)
    except Exception as e:
        print(f"  [ERROR] {e}")
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
        print(f"  [checkpoint] {len(results)} saved, acc: {acc:.2%}")

pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
n_correct = sum(r["is_correct"] for r in results)
print(f"\nFinal accuracy: {n_correct/len(results):.2%}  ({n_correct}/{len(results)})")
with open(f"{RESULTS_DIR}/more_test_summary.txt", "a") as f:
    f.write(f"LLama-3.1-8B (Raw) Accuracy: {n_correct/len(results):.2%} ({n_correct}/{len(results)})\n")