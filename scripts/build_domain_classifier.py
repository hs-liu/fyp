import pickle
import pandas as pd
import datasets
import re
from collections import defaultdict, Counter
import sys

sys.stdout.reconfigure(line_buffering=True)

UMLS_DIR = "/vol/bitbucket/hl2622/umls/2025AB/META"
DATA_DIR  = "/vol/bitbucket/hl2622/fyp/data"
HF_CACHE  = "/vol/bitbucket/hl2622/huggingface_cache"

# ── Load graph ────────────────────────────────────────────
print("Loading graph...")
G = pickle.load(open(f"{DATA_DIR}/umls_graph_filtered_new.pkl", "rb"))
print(f"  {G.number_of_nodes():,} nodes")
graph_nodes = set(G.nodes())

# ── Load semantic type → group mapping ────────────────────
print("Loading semantic type groups...")
sty_df = pd.read_csv(f"{DATA_DIR}/semantic_types_full.csv")
sty_to_group = dict(zip(sty_df["name"], sty_df["group_name"]))

# Clinical groups only — derived from CSV, no hardcoding
EXCLUDE_GROUPS = {
    "Concepts & Ideas", "Living Beings", "Objects",
    "Geographic Areas", "Occupations", "Organizations",
}
GROUP_LABELS = sorted([
    g for g in sty_df["group_name"].dropna().unique().tolist()
    if g not in EXCLUDE_GROUPS
])
label2id = {g: i for i, g in enumerate(GROUP_LABELS)}
id2label = {i: g for g, i in label2id.items()}
print(f"  {len(GROUP_LABELS)} clinical groups: {GROUP_LABELS}")

# ── Load ALL English synonyms from MRCONSO ────────────────
print("\nLoading all English synonyms from MRCONSO...")
cui_to_names = defaultdict(list)
for chunk in pd.read_csv(
    f"{UMLS_DIR}/MRCONSO.RRF", sep="|", header=None,
    names=["cui","lat","ts","lui","stt","sui","ispref","aui",
           "saui","scui","sdui","sab","tty","code","str",
           "srl","suppress","cvf","_"],
    usecols=["cui","lat","suppress","str"],
    dtype=str, chunksize=500_000
):
    chunk = chunk[
        (chunk["lat"]      == "ENG") &
        (chunk["suppress"] != "Y")
    ][["cui","str"]]
    for row in chunk.itertuples():
        if row.cui in graph_nodes and isinstance(row.str, str):
            cui_to_names[row.cui].append(row.str.lower())

print(f"  Loaded synonyms for {len(cui_to_names):,} concepts in graph")

# ── Build name → group from all synonyms ──────────────────
print("Building name → group lookup...")
name_to_group = {}
for cui, names in cui_to_names.items():
    sty   = G.nodes[cui].get("sty_name","")
    group = sty_to_group.get(sty, "")
    if not group or group in EXCLUDE_GROUPS:
        continue
    for name in names:
        if len(name) > 3 and name not in name_to_group:
            name_to_group[name] = group

print(f"  {len(name_to_group):,} name → group mappings")

# Verify key terms
print("\n  Verification:")
for t in ["hypertension","stroke","cancer","surgery","heart attack",
          "warfarin","diabetes mellitus","biopsy","atrial fibrillation"]:
    print(f"    '{t}' → '{name_to_group.get(t, 'NOT FOUND')}'")

# ── Build first-word index for fast lookup ────────────────
print("\nBuilding first-word index...")
first_word_idx = defaultdict(list)
for name, group in name_to_group.items():
    w = name.split()[0] if name.split() else ""
    if w:
        first_word_idx[w].append((name, group))
print(f"  {len(first_word_idx):,} first-word keys")

# ── Domain labelling function ─────────────────────────────
def get_dominant_group(text):
    text_lower = text.lower()
    words      = set(re.findall(r'\b\w+\b', text_lower))
    group_counts = Counter()
    for word in words:
        for name, group in first_word_idx.get(word, []):
            if name in text_lower and group in label2id:
                group_counts[group] += 1
    if not group_counts:
        return None, 0.0
    top_group, top_count = group_counts.most_common(1)[0]
    confidence = top_count / sum(group_counts.values())
    return top_group, confidence

# Sanity check
print("\nSanity check:")
tests = [
    "A patient with diabetes mellitus presents with chest pain",
    "Which drug is used to treat hypertension?",
    "The patient has atrial fibrillation and needs anticoagulation",
    "Biopsy of the lung showed adenocarcinoma",
    "Surgery is indicated for this patient with bowel obstruction",
    "The gene mutation causes familial hypercholesterolaemia",
]
for q in tests:
    group, conf = get_dominant_group(q)
    print(f"  {q[:58]:<58} → {str(group):<25} ({conf:.2f})")

# ── Load MedQA and label ──────────────────────────────────
print("\nLoading MedQA...")
dataset = datasets.load_dataset(
    "bigbio/med_qa",
    "med_qa_en_source",
    trust_remote_code=True,
    cache_dir=HF_CACHE,
)

# Merge tiny classes before labelling
MERGE_MAP = {
    "Genes & Molecular Sequences": "Physiology",
    "Devices"                    : "Procedures",
    "Phenomena"                  : "Disorders",
}

records = []
skipped = 0
for split in ["train", "validation", "test"]:
    for sample in dataset[split]:
        text  = sample["question"]
        group, conf = get_dominant_group(text)
        if not group or conf < 0.3:
            skipped += 1
            continue
        # Merge tiny classes
        group = MERGE_MAP.get(group, group)
        records.append({
            "text"      : text,
            "label"     : group,
            "label_id"  : label2id.get(group, -1),
            "confidence": round(conf, 3),
            "split"     : split,
        })

# Rebuild label2id after merging (some classes may be gone)
final_groups = sorted(set(r["label"] for r in records))
label2id_final = {g: i for i, g in enumerate(final_groups)}
for r in records:
    r["label_id"] = label2id_final[r["label"]]

df = pd.DataFrame(records)
total = sum(len(dataset[s]) for s in ["train","validation","test"])

print(f"\nTotal MedQA questions : {total:,}")
print(f"Labelled              : {len(df):,} ({len(df)/total:.1%})")
print(f"Skipped (low conf)    : {skipped:,} ({skipped/total:.1%})")
print(f"\nLabel distribution:")
print(df["label"].value_counts().to_string())
print(f"\nPer-split counts:")
print(df.groupby(["split","label"]).size().unstack(fill_value=0).to_string())

# Save
df.to_csv(f"{DATA_DIR}/classifier_training_data.csv", index=False)

# Save label map for use in training
import json
with open(f"{DATA_DIR}/classifier_label_map.json", "w") as f:
    json.dump({
        "label2id": label2id_final,
        "id2label": {str(i): g for g, i in label2id_final.items()},
        "groups"  : final_groups,
    }, f, indent=2)

print(f"\nSaved → classifier_training_data.csv")
print(f"Saved → classifier_label_map.json")
print("Done!")