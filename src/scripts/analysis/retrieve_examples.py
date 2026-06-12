# scripts/analysis/retrieve_examples.py
import sys
import os
sys.path.insert(0, "/vol/bitbucket/hl2622/fyp/src")
sys.path.insert(0, "/vol/bitbucket/hl2622/fyp")

from src.scripts.rag.retrieval_pipeline import (
    hierarchical_retrieve, extract_cuis, expand_cuis,
    cui_to_rows, G
)
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = "./results/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

encoder = SentenceTransformer(
    'pritamdeka/S-PubMedBert-MS-MARCO',
    cache_folder="/vol/bitbucket/hl2622/huggingface_cache"
)

lines = []
def log(s=""): print(s); lines.append(s)

# ── Cases to analyse ──────────────────────────────────────
CASES = {
    "HELPED_4_conduction_aphasia": {
        "question": """A 67-year-old male is seen by neurology after he was noticed to be speaking strangely by his family. After acute treatment with tissue plasminogen activator (tPA), the patient is able to recover most of his speech. Subsequent neurologic exam finds that the patient is fluent while speaking and is able to comprehend both one and two step instructions. Noticeably the patient remains unable to complete tasks involving verbal repetition. Residual damage to which of the following structures is most likely responsible for this patient's residual deficit?""",
        "ground_truth": "A",
        "raw_answer":        "E",
        "medhirerag_answer": "A",
        "medrag_answer":     "E",
        "note": "HELPED vs Raw + KG ADV 5 vs MedRAG",
    },
    "HELPED_10_gynecomastia": {
        "question": """A 13-year-old boy is brought to his pediatrician due to a left breast lump under his nipple. He noticed it last month and felt that it has increased slightly in size. It is tender to touch but has no overlying skin changes. There is no breast discharge. The patient has cryptorchidism as an infant and underwent a successful orchiopexy. In addition, he was recently diagnosed with ADHD and is currently on methylphenidate with improvement in his symptoms. He has a family history of type I diabetes in his maternal grandfather. Which of the following is the most likely diagnosis?""",
        "ground_truth": "B",
        "raw_answer":        "E",
        "medhirerag_answer": "B",
        "medrag_answer":     "C",
        "note": "HELPED vs Raw + KG ADV 15 vs MedRAG",
    },
    "HURT_6_parkinsons_amantadine": {
        "question": """A 65-year old man presents with gradually worsening rigidity of his arms and legs and slowness in performing tasks. He says he has also noticed hand tremors, which increase at rest and decrease with focused movements. On examination, the patient does not swing his arms while walking and has a shortened, shuffling gait. An antiviral drug is prescribed which alleviates the patient's symptoms. Which of the following drugs was most likely prescribed to this patient?""",
        "ground_truth": "A",
        "raw_answer":        "A",
        "medhirerag_answer": "D",
        "medrag_answer":     "A",
        "note": "HURT vs Raw — symptom CUIs only",
    },
    "HURT_11_post_thyroidectomy": {
        "question": """A 43-year-old man comes to the emergency department with nausea, abdominal discomfort, diarrhea, and progressive perioral numbness for the past 24 hours. 3 days ago, he underwent a total thyroidectomy for treatment of papillary thyroid cancer. His only medication is a multivitamin supplement. He appears fatigued. While measuring the patient's blood pressure, the nurse observes a spasm in the patient's hand. Physical examination shows a well-healing surgical wound on the neck. Which of the following laboratory findings would most likely be seen in this patient?""",
        "ground_truth": "B",
        "raw_answer":        "B",
        "medhirerag_answer": "C",
        "medrag_answer":     "B",
        "note": "HURT vs Raw — L2→L3 conditioning failure",
    },
    "BM_KG_ADV_5_typhoid": {
        "question": """A 14-year-old girl is brought to the physician by her father because of fever, chills, abdominal pain, and profuse non-bloody diarrhea. Her symptoms began one week ago, when she had several days of low-grade fever and constipation. She returned from Indonesia 2 weeks ago, where she spent the summer with her grandparents. Her temperature is 39.3°C (102.8°F). Examination shows diffuse abdominal tenderness and mild hepatosplenomegaly. There is a faint salmon-colored maculopapular rash on her trunk.""",
        "ground_truth": "D",
        "raw_answer":        "C",
        "medhirerag_answer": "D",
        "medrag_answer":     "C",
        "note": "BioMistral cross-model KG advantage",
    },
}

# ── Run retrieval for each case ────────────────────────────
log("=" * 70)
log("RETRIEVAL CONTEXT ANALYSIS — updated pipeline")
log("=" * 70)

for case_name, case in CASES.items():
    log(f"\n{'─'*70}")
    log(f"CASE: {case_name}")
    log(f"NOTE: {case['note']}")
    log(f"GT: {case['ground_truth']} | "
        f"Raw: {case['raw_answer']} | "
        f"MedHireRAG: {case['medhirerag_answer']} | "
        f"MedRAG: {case['medrag_answer']}")
    log(f"{'─'*70}")

    question = case["question"]

    # ── L1 — extracted CUIs ───────────────────────────────
    seed_cuis = extract_cuis(question)
    log(f"\n[L1] Extracted seed CUIs ({len(seed_cuis)}):")
    for cui in seed_cuis:
        name = G.nodes[cui].get("name", cui) if cui in G else cui
        sty  = G.nodes[cui].get("sty_name", "") if cui in G else ""
        log(f"  {cui}: {name} [{sty}]")

    # ── L1 — candidate pool ───────────────────────────────
    if seed_cuis:
        expanded = expand_cuis(seed_cuis, hops=1)
        candidate_rows = set()
        for cui in expanded:
            candidate_rows.update(cui_to_rows.get(cui, []))
        log(f"\n[L1] Candidate pool size: {len(candidate_rows)} chunks")
    else:
        log(f"\n[L1] No CUIs extracted — full corpus fallback")
        candidate_rows = set()

    # ── L2 + L3 — full retrieval ──────────────────────────
    try:
        result = hierarchical_retrieve(question, encoder)

        l2 = result["l2_chunks"]
        log(f"\n[L2] Top-2 textbook chunks:")
        for i, (_, row) in enumerate(l2.head(2).iterrows(), 1):
            log(f"\n  Chunk {i} (score={row['score']:.3f}):")
            log(f"  Source: {row['title'][:80]}")
            log(f"  Content: {row['content'][:300]}")

        l3 = result["l3_chunks"]
        log(f"\n[L3] Top-1 PubMed chunk:")
        if len(l3) > 0:
            row = l3.iloc[0]
            log(f"  Score: {row['score']:.3f}")
            log(f"  Source: {row['title'][:80]}")
            log(f"  Content: {row['content'][:300]}")
        else:
            log("  No PubMed chunks retrieved")

        # Assembled context
        l2_text = "\n\n".join(
            f"[Textbook] {r['content'][:400]}"
            for _, r in l2.head(2).iterrows()
        )
        l3_text = ""
        if len(l3) > 0:
            l3_text = f"\n\n[Evidence] {l3.iloc[0]['content'][:300]}"
        context = l2_text + l3_text
        log(f"\n[ASSEMBLED CONTEXT]:")
        log(context[:800])

    except Exception as e:
        log(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()

    log(f"\n{'─'*70}")

# ── Save ───────────────────────────────────────────────────
with open(f"{OUTPUT_DIR}/retrieval_context_examples_rerun.txt", "w") as f:
    f.write("\n".join(lines))
print(f"\nSaved → {OUTPUT_DIR}/retrieval_context_examples_rerun.txt")
print("Done!")