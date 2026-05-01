# baseline_groq_myrag.py
import os, re, time, signal, datasets
import pandas as pd
from groq import Groq
from sentence_transformers import SentenceTransformer
from baseline_utils import format_question, parse_answer
import retrieval_pipeline as R

N_TEST          = 200
MAX_CTX_CHARS   = 1800
RESULTS_DIR     = "./results"
CHECKPOINT_PATH = f"{RESULTS_DIR}/results_groq_myrag.csv"
os.makedirs(RESULTS_DIR, exist_ok=True)

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL  = "llama-3.3-70b-versatile"
DELAY  = 60.0 / 28

print("Loading encoder...")
encoder = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')

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

def build_prompt(sample, context: str) -> str:
    base = format_question(sample)
    return (
        "You are a medical expert. Use the reference passages below to answer "
        "the question. Reply with ONLY the single letter of the correct answer.\n\n"
        f"### Reference passages\n{context[:MAX_CTX_CHARS]}\n\n"
        f"### Question\n{base}"
    )

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

def infer(sample, max_retries=4) -> tuple:
    try:
        result  = R.hierarchical_retrieve(sample["question"], encoder)
        context = "\n\n".join(result["l2_chunks"].head(2)["content"].tolist())
        domain  = result["domain"]
        route   = result["source_route"]
    except Exception as e:
        print(f"  [RETRIEVAL ERROR] {e}")
        context, domain, route = "", "", ""

    prompt = build_prompt(sample, context)

    for attempt in range(max_retries):
        try:
            t0  = time.time()
            raw = run_with_timeout(lambda: call_api(prompt), seconds=25)
            gap = DELAY - (time.time() - t0)
            if gap > 0:
                time.sleep(gap)
            return raw, domain, route
        except TimeoutError:
            time.sleep(2 ** attempt)
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                match = re.search(r'retry.?after[: ]+(\d+\.?\d*)', err, re.I)
                wait  = float(match.group(1)) + 2 if match else 30 * (attempt + 1)
                print(f"  [RATE LIMIT] waiting {wait:.0f}s")
                time.sleep(wait)
            else:
                print(f"  [ERROR] {e}")
                return "", "", ""
    return "", "", ""

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
        "id": i, "question": sample["question"],
        "ground_truth": gt, "raw_answer": raw,
        "model_answer": parsed, "is_correct": bool(ok),
        "domain": domain, "source_route": route,
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
    f.write(f"Groq 70B (My RAG) Accuracy: {n_correct/len(results):.2%} ({n_correct}/{len(results)})\n")