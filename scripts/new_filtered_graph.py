# filter_graph_v4.py — semantic group filter only, no degree filter
import pickle
import pandas as pd
import networkx as nx
import sys

sys.stdout.reconfigure(line_buffering=True)
DATA_DIR = "/vol/bitbucket/hl2622/fyp/data"

print("Loading original graph...")
G = pickle.load(open(f"{DATA_DIR}/umls_graph.pkl", "rb"))
print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# Load semantic type → group mapping
sty_groups = pd.read_csv(f"{DATA_DIR}/semantic_types_full.csv", dtype=str)
sty_to_group = dict(zip(sty_groups["name"], sty_groups["group_name"]))

# Exclude only non-clinical groups
EXCLUDE_GROUPS = {"Living Beings", "Objects"}

keep_nodes = {
    n for n in G.nodes()
    if sty_to_group.get(G.nodes[n].get("sty_name",""), "") not in EXCLUDE_GROUPS
}
print(f"  Keeping {len(keep_nodes):,} nodes after group filter")

# Verify canonical concepts are kept
canonical = {
    "C0004238": "Atrial Fibrillation",
    "C0043031": "Warfarin",
    "C0038454": "Stroke",
    "C0020538": "Hypertension",
    "C0011849": "Diabetes Mellitus",
    "C0027051": "Myocardial Infarction",
    "C0004057": "Aspirin",
    "C0018801": "Heart Failure",
    "C0025598": "Metformin",
    "C0032285": "Pneumonia",
}
print("\n  Canonical concept check:")
for cui, name in canonical.items():
    status = "✅ KEPT" if cui in keep_nodes else "❌ REMOVED"
    print(f"    {status} | {cui} | {name}")

# Build subgraph — no degree filter, no isolate removal
G4 = G.subgraph(keep_nodes).copy()
print(f"\n  Final: {G4.number_of_nodes():,} nodes, {G4.number_of_edges():,} edges")

# Save
print("Saving...")
with open(f"{DATA_DIR}/umls_graph_filtered_new.pkl", "wb") as f:
    pickle.dump(G4, f)

pd.DataFrame([
    {"cui": n, "str": G4.nodes[n].get("name",""),
     "tui": G4.nodes[n].get("tui",""),
     "sty": G4.nodes[n].get("sty_name","")}
    for n in G4.nodes()
]).to_csv(f"{DATA_DIR}/graph_nodes_filtered_new.csv", index=False)

import csv
with open(f"{DATA_DIR}/graph_edges_filtered_new.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["cui1","cui2","rel"])
    for u, v, d in G4.edges(data=True):
        writer.writerow([u, v, d.get("relation","")])

print("Done!")