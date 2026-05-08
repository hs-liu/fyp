import os, torch, datasets
import pandas as pd
import sys
sys.path.insert(0, "/vol/bitbucket/hl2622/fyp")
sys.path.insert(0, "/vol/bitbucket/hl2622/fyp/MedRAG/src")
from scripts.baselines.baseline_utils import format_question, parse_answer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from scripts.rag.retrieval_utils import get_context, build_rag_prompt, load_checkpoint, save_checkpoint, save_summary
from uq_utils import compute_uq, print_uq_summary, plot_uq

N_TEST          = 200
N_SAMPLES       = 10
TEMPERATURE     = 0.3
RESULTS_DIR     = "./results"
GRAPHS_DIR      = "./graphs"
CHECKPOINT_PATH = f"{RESULTS_DIR}/results_qwen_myrag_uq_0_3_10.csv"
SUMMARY_PATH    = f"{RESULTS_DIR}/local_model_summary.txt"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

MODEL_PATH = "/vol/bitbucket/hl2622/fyp/models/qwen2.5-7b"

print("Loading Qwen...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.float16, device_map="cuda:0",
)
model.eval()

print("Loading encoder...")
encoder = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
print("All models loaded.")

dataset = datasets.load_dataset("bigbio/med_qa", "med_qa_en_source", trust_remote_code=True)
test_ds = list(dataset["test"])[:N_TEST]

checkpoint_results, done_indices = load_checkpoint(CHECKPOINT_PATH)

def inference_fn(prompt: str, do_sample: bool = False, temperature: float = 1.0) -> str:
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
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def infer_with_uq(sample) -> dict:
    try:
        context, domain, route = get_context(sample, encoder)
        prompt = build_rag_prompt(sample, context, format_question)

        greedy_raw    = inference_fn(prompt, do_sample=False)
        greedy_parsed = parse_answer(greedy_raw)
        uq = compute_uq(prompt, inference_fn, n_samples=N_SAMPLES, temperature=TEMPERATURE)

        return {"domain": domain, "route": route,
                "greedy_answer": greedy_parsed, "greedy_raw": greedy_raw, **uq}

    except Exception as e:
        print(f"  [ERROR] {e}")
        return {"domain": "", "route": "",
                "greedy_answer": "", "greedy_raw": "",
                "uq_answer": "UNKNOWN", "uq_consistency": 0.0,
                "uq_entropy": 1.0, "uq_samples": "[]", "n_valid": 0}

# main
remaining = [(i, s) for i, s in enumerate(test_ds) if i not in done_indices]
print(f"Samples left: {len(remaining)}")
results = list(checkpoint_results)

for step, (i, sample) in enumerate(remaining, 1):
    r  = infer_with_uq(sample)
    gt = sample["answer_idx"]
    greedy_ok = r["greedy_answer"] == gt
    uq_ok     = r["uq_answer"] == gt

    results.append({
        "id": i, "question": sample["question"],
        "ground_truth": gt,
        "greedy_answer": r["greedy_answer"], "greedy_raw": r["greedy_raw"],
        "greedy_correct": bool(greedy_ok),
        "uq_answer": r["uq_answer"], "uq_consistency": r["uq_consistency"],
        "uq_entropy": r["uq_entropy"], "uq_correct": bool(uq_ok),
        "uq_samples": r["uq_samples"], "n_valid": r["n_valid"],
        "domain": r["domain"], "source_route": r["route"],
        "is_correct": bool(greedy_ok),
    })

    print(f"  [{step:>3}/{len(remaining)}] "
          f"greedy={r['greedy_answer']} uq={r['uq_answer']} "
          f"consistency={r['uq_consistency']:.2f} gt={gt} "
          f"greedy={'✓' if greedy_ok else '✗'} uq={'✓' if uq_ok else '✗'}")

    if step % 5 == 0:
        save_checkpoint(results, CHECKPOINT_PATH, step, len(remaining))
        u_acc = sum(r["uq_correct"] for r in results) / len(results)
        print(f"  uq_acc: {u_acc:.2%}")

pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
n_greedy = sum(r["greedy_correct"] for r in results)
n_uq     = sum(r["uq_correct"] for r in results)
greedy_acc = n_greedy / len(results)
uq_acc     = n_uq / len(results)

print(f"\nGreedy accuracy:      {greedy_acc:.2%} ({n_greedy}/{len(results)})")
print(f"UQ majority accuracy: {uq_acc:.2%} ({n_uq}/{len(results)})")
print(f"UQ improvement:       {uq_acc - greedy_acc:+.2%}")

save_summary(SUMMARY_PATH, f"Qwen2.5-7B (My RAG + UQ greedy, 0.3, 10):   {greedy_acc:.2%} ({n_greedy}/{len(results)})")
save_summary(SUMMARY_PATH, f"Qwen2.5-7B (My RAG + UQ majority, 0.3, 10):  {uq_acc:.2%} ({n_uq}/{len(results)})")

df = pd.read_csv(CHECKPOINT_PATH)
print_uq_summary(df)
plot_uq(df, save_path=f"{GRAPHS_DIR}/uq_qwen_0.3_10.png", model_name="Qwen2.5-7B + My RAG, 0.3, 10)")