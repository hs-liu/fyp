# scripts/analysis/robustness_experiment.py
"""
Robustness analysis — how stable are MedHireUQRAG answers
under small query perturbations, across all three models.

Perturbation types:
1. Paraphrase  — reword question, same meaning
2. Typo        — random character errors
3. Synonym     — replace medical terms with synonyms
4. Shuffle     — shuffle non-critical words
5. Truncation  — remove last sentence of question
"""
import os
import re
import random
import torch
import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys
sys.path.insert(0, "/vol/bitbucket/hl2622/fyp/src")
sys.path.insert(0, "/vol/bitbucket/hl2622/fyp/MedRAG/src")

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from src.scripts.baselines.baseline_utils import format_question, parse_answer
from src.scripts.rag.retrieval_utils import get_context, build_rag_prompt
from src.scripts.uq_experiments.uq_utils import compute_uq

# ── Config ─────────────────────────────────────────────────
N_TEST          = 50    # subset — robustness is expensive
N_PERTURBATIONS = 5     # perturbations per question per type
RESULTS_DIR     = "./results/robustness"
GRAPHS_DIR      = "./graphs/analysis/robustness"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR,  exist_ok=True)

random.seed(42)
np.random.seed(42)

# ── Model configs ──────────────────────────────────────────
# ── Best UQ config per model (calibration-optimised T=0.7) ─
MODEL_CONFIGS = {
    "BioMistral-7B": {
        "path":        "/vol/bitbucket/hl2622/fyp/src/models/biomistral-7b",
        "use_chat":    False,
        "color":       "#2E86C1",
        "temperature": 0.7,   # calibration-optimised
        "n_samples":   20,    # best calibration gap +17.6%
    },
    "Llama-3.1-8B": {
        "path":        "/vol/bitbucket/hl2622/fyp/src/models/llama-3.1-8b",
        "use_chat":    True,
        "color":       "#1E8449",
        "temperature": 0.7,   # calibration-optimised
        "n_samples":   10,    # best calibration gap +12.5%
    },
    "Qwen2.5-7B": {
        "path":        "/vol/bitbucket/hl2622/fyp/src/models/qwen2.5-7b",
        "use_chat":    True,
        "color":       "#7D3C98",
        "temperature": 0.7,   # calibration-optimised
        "n_samples":   10,    # marginal signal, T=0.7 still preferred
    },
}

# ── Medical synonyms ───────────────────────────────────────
SYNONYMS = {
    "patient":              ["individual", "person", "subject"],
    "presents":             ["comes in", "reports", "is seen"],
    "history":              ["background", "past", "record"],
    "diagnosis":            ["condition", "finding", "assessment"],
    "treatment":            ["therapy", "management", "intervention"],
    "symptoms":             ["signs", "complaints", "manifestations"],
    "chest pain":           ["thoracic pain", "chest discomfort"],
    "fever":                ["pyrexia", "elevated temperature"],
    "shortness of breath":  ["dyspnoea", "breathlessness"],
    "hypertension":         ["high blood pressure", "elevated BP"],
    "diabetes":             ["diabetes mellitus", "hyperglycaemia"],
}

# ── Perturbation functions ─────────────────────────────────
def inject_typos(text: str, rate: float = 0.02) -> str:
    chars = list(text)
    n_errors = max(1, int(len(chars) * rate))
    for _ in range(n_errors):
        idx = random.randint(0, len(chars)-1)
        action = random.choice(["swap", "delete", "duplicate"])
        if action == "swap" and idx < len(chars)-1:
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
        elif action == "delete":
            chars.pop(idx)
        elif action == "duplicate":
            chars.insert(idx, chars[idx])
    return "".join(chars)

def swap_synonyms(text: str) -> str:
    result = text
    for term, syns in SYNONYMS.items():
        if term.lower() in result.lower():
            replacement = random.choice(syns)
            result = re.sub(re.escape(term), replacement,
                            result, flags=re.IGNORECASE, count=1)
    return result

def shuffle_words(text: str) -> str:
    sentences = text.split(". ")
    if len(sentences) <= 2:
        return text
    middle = sentences[1:-1]
    words  = " ".join(middle).split()
    random.shuffle(words)
    sentences[1:-1] = [" ".join(words)]
    return ". ".join(sentences)

def truncate(text: str) -> str:
    sentences = text.split(". ")
    if len(sentences) <= 1:
        return text
    return ". ".join(sentences[:-1]) + "."

def paraphrase_simple(text: str) -> str:
    text = swap_synonyms(text)
    text = re.sub(r"(\d+)-year-old",
                  lambda m: f"{m.group(1)} years old", text)
    return text

PERTURBATION_FNS = {
    "typo":       lambda q: inject_typos(q, rate=0.02),
    "synonym":    swap_synonyms,
    "shuffle":    shuffle_words,
    "truncate":   truncate,
    "paraphrase": paraphrase_simple,
}

# ── Dataset ────────────────────────────────────────────────
print("Loading dataset...")
dataset = datasets.load_dataset(
    "bigbio/med_qa", "med_qa_en_source", trust_remote_code=True
)
test_ds = list(dataset["test"])[:N_TEST]

print("Loading encoder...")
encoder = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")

# ══════════════════════════════════════════════════════════
# Run experiment per model
# ══════════════════════════════════════════════════════════
all_results = []

for model_name, cfg in MODEL_CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    # Load model
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        dtype=torch.float16,
        device_map="cuda:0"
    )
    model.eval()



    # ── Inference functions ────────────────────────────────
    def call_local(prompt: str, do_sample: bool = False, temperature: float = 1.0) -> str:
        if cfg["use_chat"]:
            messages  = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
        else:
            inputs    = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=2048,
            ).to(model.device)
            input_ids = inputs["input_ids"]
            if input_ids[0, -1] == tokenizer.eos_token_id:
                input_ids = input_ids[:, :-1]

        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=5,
                min_new_tokens=1,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0][input_ids.shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


    def infer_with_uq(sample) -> dict:
        try:
            context = get_context(sample, encoder)
            prompt  = build_rag_prompt(sample, context, format_question)
            greedy_raw    = call_local(prompt)
            greedy_answer = parse_answer(greedy_raw)
            uq = compute_uq(
                prompt, call_local,
                n_samples=cfg["n_samples"],      # ← per model
                temperature=cfg["temperature"],   # ← per model
            )
            return {
                "greedy_answer":  greedy_answer,
                "uq_answer":      uq["uq_answer"],
                "uq_consistency": uq["uq_consistency"],
            }
        except Exception as e:
            print(f"  [ERROR] {e}")
            return {
                "greedy_answer":  "",
                "uq_answer":      "",
                "uq_consistency": 0.0,
            }

    # ── Run per question ───────────────────────────────────
    for i, sample in enumerate(test_ds):
        original_q = sample["question"]
        gt         = sample["answer_idx"]

        # Original — MedHireUQRAG
        orig = infer_with_uq(sample)
        orig_greedy  = orig["greedy_answer"]
        orig_uq      = orig["uq_answer"]
        orig_correct = orig_greedy == gt
        orig_consist = orig["uq_consistency"]

        print(f"\n[{i+1}/{N_TEST}] {model_name} | GT={gt} "
              f"greedy={orig_greedy} uq={orig_uq} "
              f"consistency={orig_consist:.2f} "
              f"{'✓' if orig_correct else '✗'}")

        for perturb_type, perturb_fn in PERTURBATION_FNS.items():
            greedy_answers  = []
            uq_answers      = []
            consistencies   = []

            for _ in range(N_PERTURBATIONS):
                perturbed_sample           = dict(sample)
                perturbed_sample["question"] = perturb_fn(original_q)

                out = infer_with_uq(perturbed_sample)
                greedy_answers.append(out["greedy_answer"])
                uq_answers.append(out["uq_answer"])
                consistencies.append(out["uq_consistency"])

            # Metrics — using greedy answer for consistency
            greedy_same  = [a == orig_greedy for a in greedy_answers]
            uq_same      = [a == orig_uq     for a in uq_answers]
            greedy_correct_perturb = [a == gt for a in greedy_answers]
            uq_correct_perturb     = [a == gt for a in uq_answers]

            greedy_consistency = sum(greedy_same) / len(greedy_same)
            uq_consistency_rate= sum(uq_same)     / len(uq_same)
            greedy_accuracy    = sum(greedy_correct_perturb) / len(greedy_correct_perturb)
            uq_accuracy        = sum(uq_correct_perturb)     / len(uq_correct_perturb)
            mean_uq_score      = np.mean(consistencies)
            flip_rate          = 1 - greedy_consistency

            all_results.append({
                "model":                model_name,
                "id":                   i,
                "ground_truth":         gt,
                "original_greedy":      orig_greedy,
                "original_uq":          orig_uq,
                "original_correct":     orig_correct,
                "original_consistency": orig_consist,
                "perturbation_type":    perturb_type,
                "greedy_consistency":   greedy_consistency,
                "uq_consistency_rate":  uq_consistency_rate,
                "greedy_accuracy":      greedy_accuracy,
                "uq_accuracy":          uq_accuracy,
                "mean_uq_score":        mean_uq_score,
                "flip_rate":            flip_rate,
            })

            print(f"  {perturb_type:<12} "
                  f"greedy_consist={greedy_consistency:.2f} "
                  f"uq_consist={uq_consistency_rate:.2f} "
                  f"flip={flip_rate:.2f}")

    # Free GPU memory before loading next model
    del model
    torch.cuda.empty_cache()

# ── Save results ───────────────────────────────────────────
df = pd.DataFrame(all_results)
df.to_csv(f"{RESULTS_DIR}/robustness_results.csv", index=False)
print(f"\nSaved → {RESULTS_DIR}/robustness_results.csv")

# ══════════════════════════════════════════════════════════
# ANALYSIS + PLOTS
# ══════════════════════════════════════════════════════════
model_names   = list(MODEL_CONFIGS.keys())
model_colors  = {m: cfg["color"] for m, cfg in MODEL_CONFIGS.items()}
perturb_types = list(PERTURBATION_FNS.keys())

PERTURB_COLORS = {
    "typo":       "#E74C3C",
    "synonym":    "#E67E22",
    "shuffle":    "#2E86C1",
    "truncate":   "#1E8449",
    "paraphrase": "#7D3C98",
}

# ── Plot 1: Greedy consistency per perturbation — all models
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.suptitle("Answer Consistency Under Query Perturbations — All Models\n"
             "(MedHireUQRAG greedy, higher = more robust)",
             fontsize=13, fontweight="bold")

for ax, model_name in zip(axes, model_names):
    sub    = df[df["model"] == model_name]
    means  = [sub[sub["perturbation_type"]==p]["greedy_consistency"].mean()
               for p in perturb_types]
    stds   = [sub[sub["perturbation_type"]==p]["greedy_consistency"].std()
               for p in perturb_types]
    colors = [PERTURB_COLORS[p] for p in perturb_types]

    bars = ax.bar(perturb_types, means, color=colors,
                  edgecolor="white", linewidth=0.5, width=0.6,
                  yerr=stds, capsize=4,
                  error_kw={"linewidth": 1.5})
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.axhline(1.0, color="gray", linestyle="--",
               linewidth=1.5, label="Perfect consistency")
    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Perturbation type", fontsize=10)
    ax.set_ylabel("Consistency rate" if model_name == model_names[0] else "",
                  fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.set_xticklabels(perturb_types, rotation=20, ha="right", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if model_name == model_names[0]:
        ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/01_consistency_all_models.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/01_consistency_all_models.png")

# ── Plot 2: Greedy vs UQ consistency rate ─────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
fig.suptitle("Greedy vs UQ Majority Consistency Under Perturbations\n"
             "UQ majority should be more stable than greedy",
             fontsize=13, fontweight="bold")

x     = np.arange(len(perturb_types))
width = 0.35

for ax, model_name in zip(axes, model_names):
    sub    = df[df["model"] == model_name]
    greedy = [sub[sub["perturbation_type"]==p]["greedy_consistency"].mean()
               for p in perturb_types]
    uq     = [sub[sub["perturbation_type"]==p]["uq_consistency_rate"].mean()
               for p in perturb_types]

    b1 = ax.bar(x - width/2, greedy, width, label="Greedy",
                color=model_colors[model_name], edgecolor="white",
                linewidth=0.5, alpha=0.6)
    b2 = ax.bar(x + width/2, uq,     width, label="UQ majority",
                color=model_colors[model_name], edgecolor="white",
                linewidth=0.5)
    for bar, val in zip(list(b1)+list(b2), greedy+uq):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(perturb_types, rotation=20, ha="right", fontsize=9)
    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.set_ylabel("Consistency rate" if model_name == model_names[0] else "",
                  fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/02_greedy_vs_uq_consistency.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/02_greedy_vs_uq_consistency.png")

# ── Plot 3: Accuracy under perturbation vs original ────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
fig.suptitle("Accuracy Under Perturbations vs Original Accuracy\n"
             "MedHireUQRAG — greedy and UQ majority",
             fontsize=13, fontweight="bold")

for ax, model_name in zip(axes, model_names):
    sub      = df[df["model"] == model_name]
    orig_acc = sub["original_correct"].mean() * 100

    greedy_accs = [sub[sub["perturbation_type"]==p]["greedy_accuracy"].mean()*100
                   for p in perturb_types]
    uq_accs     = [sub[sub["perturbation_type"]==p]["uq_accuracy"].mean()*100
                   for p in perturb_types]

    ax.axhline(orig_acc, color="black", linestyle="--",
               linewidth=2, label=f"Original ({orig_acc:.1f}%)", zorder=3)

    ax.plot(perturb_types, greedy_accs, marker="o", linewidth=2,
            markersize=7, color=model_colors[model_name],
            alpha=0.6, label="Greedy")
    ax.plot(perturb_types, uq_accs, marker="s", linewidth=2,
            markersize=7, color=model_colors[model_name],
            label="UQ majority")

    for i, (ga, ua) in enumerate(zip(greedy_accs, uq_accs)):
        ax.annotate(f"{ga:.1f}%", (i, ga),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=7,
                    color=model_colors[model_name], alpha=0.8)
        ax.annotate(f"{ua:.1f}%", (i, ua),
                    xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=7,
                    color=model_colors[model_name])

    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)" if model_name == model_names[0] else "",
                  fontsize=11)
    ax.set_xticklabels(perturb_types, rotation=20, ha="right", fontsize=9)
    max_val = max(max(greedy_accs), max(uq_accs), orig_acc)
    ax.set_ylim(0, max_val * 1.2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=9, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/03_accuracy_under_perturbation.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/03_accuracy_under_perturbation.png")

# ── Plot 4: UQ score under perturbation ───────────────────
# Does UQ consistency drop when question is perturbed?
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
fig.suptitle("UQ Consistency Score Under Perturbations\n"
             "Does perturbation make the model less confident?",
             fontsize=13, fontweight="bold")

for ax, model_name in zip(axes, model_names):
    sub      = df[df["model"] == model_name]
    orig_uq  = sub["original_consistency"].mean()

    uq_scores = [sub[sub["perturbation_type"]==p]["mean_uq_score"].mean()
                  for p in perturb_types]

    ax.axhline(orig_uq, color="black", linestyle="--",
               linewidth=2, label=f"Original ({orig_uq:.2f})", zorder=3)
    bars = ax.bar(perturb_types, uq_scores,
                  color=[PERTURB_COLORS[p] for p in perturb_types],
                  edgecolor="white", linewidth=0.5, width=0.6)
    for bar, val in zip(bars, uq_scores):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean UQ consistency score" if model_name == model_names[0] else "",
                  fontsize=11)
    ax.set_xticklabels(perturb_types, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/04_uq_score_under_perturbation.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/04_uq_score_under_perturbation.png")

# ── Plot 5: Heatmap — flip rate per model × perturbation ──
fig, ax = plt.subplots(figsize=(12, 6))

matrix = np.array([
    [df[(df["model"]==m) & (df["perturbation_type"]==p)]["flip_rate"].mean()
     for p in perturb_types]
    for m in model_names
])

im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
ax.set_xticks(range(len(perturb_types)))
ax.set_xticklabels(perturb_types, fontsize=12)
ax.set_yticks(range(len(model_names)))
ax.set_yticklabels(model_names, fontsize=11)
ax.set_title("Answer Flip Rate Heatmap — All Models × Perturbations\n"
             "Red = fragile, Green = robust",
             fontsize=13, fontweight="bold", pad=12)

for i in range(len(model_names)):
    for j in range(len(perturb_types)):
        val = matrix[i, j]
        tc  = "white" if val > 0.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=12, fontweight="bold", color=tc)

plt.colorbar(im, ax=ax, label="Flip rate", shrink=0.8)
plt.tight_layout()
plt.savefig(f"{GRAPHS_DIR}/05_flip_heatmap_all_models.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {GRAPHS_DIR}/05_flip_heatmap_all_models.png")

# ── Summary ────────────────────────────────────────────────
print("\n" + "="*60)
print("ROBUSTNESS SUMMARY")
print("="*60)
summary = df.groupby(["model","perturbation_type"]).agg(
    greedy_consistency=("greedy_consistency","mean"),
    uq_consistency_rate=("uq_consistency_rate","mean"),
    flip_rate=("flip_rate","mean"),
    greedy_accuracy=("greedy_accuracy","mean"),
    uq_accuracy=("uq_accuracy","mean"),
).round(3)
print(summary.to_string())
summary.to_csv(f"{RESULTS_DIR}/robustness_summary.csv")
print(f"\nSaved → {RESULTS_DIR}/robustness_summary.csv")
print("Done!")