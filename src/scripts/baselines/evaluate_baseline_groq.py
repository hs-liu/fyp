import os, re, time, datasets, pandas as pd
from scripts.baselines.baseline_utils import format_question, parse_answer

# basic setup 
N_TEST = 200
RESULTS_DIR = "./results"
CHECKPOINT_PATH = f"{RESULTS_DIR}/groq_results_no_rag.csv"
os.makedirs(RESULTS_DIR, exist_ok=True)

#backend models setting 
from groq import Groq
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL  = "llama-3.3-70b-versatile"
DELAY  = 60.0 / 28

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

#inference 
def call_api(prompt: str) -> str:
    """Single-attempt call; raises on error so retry wrapper handles it."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        timeout=30,          # explicit 30 s socket timeout
    )
    return (resp.choices[0].message.content or "").strip()

# handle timeout 
import signal

class TimeoutError(Exception): pass

def _handler(sig, frame): raise TimeoutError()

def run_with_timeout(fn, seconds=25):
    """POSIX only (Linux/Mac). On Windows, remove this and rely on socket timeout."""
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)


def infer(sample, max_retries=4) -> str:
    prompt = format_question(sample)
    for attempt in range(max_retries):
        try:
            t0  = time.time()
            raw = run_with_timeout(lambda: call_api(prompt), seconds=25)
            # pace to stay within rate limit
            gap = DELAY - (time.time() - t0)
            if gap > 0:
                time.sleep(gap)
            return raw

        except TimeoutError:
            wait = 2 ** attempt        # 1, 2, 4, 8 s
            print(f"  [TIMEOUT] attempt {attempt+1}, retrying in {wait}s")
            time.sleep(wait)

        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                match = re.search(r'retry.?after[: ]+(\d+\.?\d*)', err, re.I)
                wait  = float(match.group(1)) + 2 if match else 30 * (attempt + 1)
                print(f"  [RATE LIMIT] attempt {attempt+1}, waiting {wait:.0f}s")
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
        "id":           i,
        "question":     sample["question"],
        "ground_truth": gt,
        "raw_answer":   raw,
        "model_answer": parsed,
        "is_correct":   bool(ok),
    })
    print(f"  [{step:>3}/{len(remaining)}] raw={raw!r:10} parsed={parsed} gt={gt} {'✓' if ok else '✗'}")

    # save every 5 samples (not 20) so you lose very little on crash
    if step % 5 == 0:
        pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
        acc_so_far = sum(r["is_correct"] for r in results) / len(results)
        print(f"  [checkpoint] {len(results)} saved, acc so far: {acc_so_far:.2%}")


# end 
pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
n_correct = sum(r["is_correct"] for r in results)
print(f"\nFinal accuracy: {n_correct/len(results):.2%}  ({n_correct}/{len(results)})")
with open(f"{RESULTS_DIR}/local_model_summary.txt", "a") as f:
    f.write(f"Groq Accuracy: {n_correct/len(results):.2%} ({n_correct}/{len(results)})\n")