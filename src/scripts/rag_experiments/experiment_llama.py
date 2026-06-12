# baseline_llama_myrag.py
import os, torch, datasets
import pandas as pd
import sys
sys.path.insert(0, "/vol/bitbucket/hl2622/fyp/src")
sys.path.insert(0, "/vol/bitbucket/hl2622/fyp/MedRAG/src")
from src.scripts.baselines.baseline_utils import format_question, parse_answer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from src.scripts.rag.retrieval_utils import get_context, build_rag_prompt, load_checkpoint, save_checkpoint, save_summary

N_TEST          = 200
MAX_CHUNK_CHARS = 400
RESULTS_DIR     = "./results/rerun"
CHECKPOINT_PATH = f"{RESULTS_DIR}/results_llama.csv"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = "/vol/bitbucket/hl2622/fyp/src/models/llama-3.1-8b"

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
checkpoint_results, done_indices = load_checkpoint(CHECKPOINT_PATH)

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

def infer(sample) -> str:
    try:
        context = get_context(sample, encoder)
    except Exception as e:
        print(f"  [RETRIEVAL ERROR] {e}")
        context = ""

    try:
        prompt = build_rag_prompt(sample, context, format_question)
        return call_local(prompt)
    except Exception as e:
        print(f"  [INFERENCE ERROR] {e}")
        return ""

# main
remaining = [(i, s) for i, s in enumerate(test_ds) if i not in done_indices]
print(f"Samples left: {len(remaining)}")
results = list(checkpoint_results)

for step, (i, sample) in enumerate(remaining, 1):
    raw = infer(sample)
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
    print(f"  [{step:>3}/{len(remaining)}] parsed={parsed} gt={gt} {'✓' if ok else '✗'}")

    if step % 5 == 0:
        save_checkpoint(results, CHECKPOINT_PATH, step, len(remaining))

save_checkpoint(results, CHECKPOINT_PATH)
n_correct = sum(r["is_correct"] for r in results)
print(f"\nFinal accuracy: {n_correct/len(results):.2%} ({n_correct}/{len(results)})")
save_summary(f"{RESULTS_DIR}/more_test_summary.txt", f"Llama-3.1-8B Accuracy: {n_correct/len(results):.2%} ({n_correct}/{len(results)})")