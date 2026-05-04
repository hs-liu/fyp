# Replace your ENTIRE concept + relation fetching with this
import sys

import pandas as pd
import networkx as nx
import pickle

UMLS_DIR = "/vol/bitbucket/hl2622/umls/2025AB/META"  # adjust to where your UMLS download is
DATA_DIR  = "/vol/bitbucket/hl2622/fyp/data"

# ── Step 1: Load MRSTY (CUI → TUI mapping) ────────────────
print("Loading MRSTY...")
mrsty = pd.read_csv(
    f"{UMLS_DIR}/MRSTY.RRF", sep="|", header=None,
    names=["cui","tui","stn","sty","atui","cvf","_"],
    usecols=["cui","tui","sty"], dtype=str
)
print(f"  {len(mrsty):,} rows")

# ── Step 2: Load MRCONSO (preferred English names) ────────
print("Loading MRCONSO...")
mrconso = pd.read_csv(
    f"{UMLS_DIR}/MRCONSO.RRF", sep="|", header=None,
    names=["cui","lat","ts","lui","stt","sui","ispref","aui",
           "saui","scui","sdui","sab","tty","code","str","srl","suppress","cvf","_"],
    usecols=["cui","lat","ts","ispref","str"], dtype=str
)
# English preferred terms only
preferred = mrconso[
    (mrconso["lat"]    == "ENG") &
    (mrconso["ts"]     == "P") &
    (mrconso["ispref"] == "Y")
][["cui","str"]].drop_duplicates("cui")
print(f"  {len(preferred):,} preferred terms")

# ── Step 3: Filter to your TUIs, merge names ──────────────
TUIS = [
    "T001","T002","T004","T005","T007","T008","T010","T011","T012","T013",
    "T014","T015","T016","T017","T018","T019","T020","T021","T022","T023",
    "T024","T025","T026","T028","T029","T030","T031","T032","T033","T034",
    "T037","T038","T039","T040","T041","T042","T043","T044","T045","T046",
    "T047","T048","T049","T050","T051","T052","T053","T054","T055","T056",
    "T057","T058","T059","T060","T061","T062","T063","T064","T065","T066",
    "T067","T068","T069","T070","T071","T072","T073","T074","T075","T077",
    "T078","T079","T080","T081","T082","T083","T085","T086","T087","T088",
    "T089","T090","T091","T092","T093","T094","T095","T096","T097","T098",
    "T099","T100","T101","T102","T103","T104","T109","T114","T116","T120",
    "T121","T122","T123","T125","T126","T127","T129","T130","T131","T167",
    "T168","T169","T170","T171","T184","T185","T190","T191","T192","T194",
    "T195","T196","T197","T200","T201","T203","T204"
]
tui_set = set(TUIS)
filtered = mrsty[mrsty["tui"].isin(tui_set)].merge(preferred, on="cui", how="left")
filtered = filtered.dropna(subset=["str"])
print(f"  {len(filtered):,} concepts after filtering")

# ── Step 4: Load MRREL (relations) ────────────────────────
# ── Step 4: Load MRREL in chunks (too big for RAM) ────────
print("Loading MRREL in chunks...")
cui_set_graph = set(filtered["cui"].unique())

edge_list = []
chunk_size = 500_000

for chunk in pd.read_csv(
    f"{UMLS_DIR}/MRREL.RRF", sep="|", header=None,
    names=["cui1","aui1","stype1","rel","cui2","aui2","stype2",
           "rela","rui","srui","sab","sl","rg","dir","suppress","cvf","_"],
    usecols=["cui1","rel","cui2","rela","suppress"],
    dtype=str,
    chunksize=chunk_size
):
    # Filter chunk immediately before accumulating
    chunk = chunk[chunk["suppress"] != "Y"]
    chunk = chunk[chunk["cui1"] != chunk["cui2"]]
    chunk = chunk[
        chunk["cui1"].isin(cui_set_graph) &
        chunk["cui2"].isin(cui_set_graph)
    ]
    edge_list.append(chunk)
    print(f"  chunk done, kept {len(chunk)} edges so far...")
    sys.stdout.flush()

edges = pd.concat(edge_list, ignore_index=True)
print(f"  {len(edges):,} total edges")

# ── Step 5: Build graph ────────────────────────────────────
print("Building graph...")
cui_set_graph = set(filtered["cui"].unique())

G = nx.DiGraph()

# Add nodes
for _, row in filtered.iterrows():
    G.add_node(row["cui"],
               name=row["str"],
               tui=row["tui"],
               sty_name=row["sty"])

# Add edges — only between nodes in our set
for _, row in edges.iterrows():
    label = row["rela"] if pd.notna(row["rela"]) and row["rela"] else row["rel"]
    G.add_edge(row["cui1"], row["cui2"], relation=label)

print(f"  Nodes: {G.number_of_nodes():,} | Edges: {G.number_of_edges():,}")

# ── Save ──────────────────────────────────────────────────
with open(f"{DATA_DIR}/umls_graph.pkl", "wb") as f:
    pickle.dump(G, f)
filtered.to_csv(f"{DATA_DIR}/graph_nodes.csv", index=False)
print("Done!")