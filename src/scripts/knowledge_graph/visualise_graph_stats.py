import pickle
import pandas as pd
import networkx as nx
from pyvis.network import Network
import numpy as np
import sys
import math

sys.stdout.reconfigure(line_buffering=True)
DATA_DIR = "/vol/bitbucket/hl2622/fyp/src/data"

print("Loading graph...")
G = pickle.load(open(f"{DATA_DIR}/umls_graph.pkl", "rb"))
print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
density   = nx.density(G)
print(f"\n{'='*50}")
print(f"  GRAPH STATISTICS")
print(f"{'='*50}")
print(f"  Nodes  : {num_nodes:,}")
print(f"  Edges  : {num_edges:,}")
print(f"  Density: {density:.8f}")

print("\n  Semantic type distribution:")
stys = pd.Series([G.nodes[n].get("sty_name","") for n in G.nodes()]).value_counts()
for sty, count in stys.head(20).items():
    print(f"    {sty:<45} {count:>8,}")

print("\n  Top 15 most connected concepts:")
degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:15]
for cui, deg in degrees:
    name = G.nodes[cui].get("name", cui)
    sty  = G.nodes[cui].get("sty_name", "")
    print(f"    {name:<45} {sty:<35} degree: {deg:,}")

deg_vals = np.array([d for _, d in G.degree()])
print(f"\n  Degree stats:")
print(f"    Mean   : {deg_vals.mean():.2f}")
print(f"    Median : {np.median(deg_vals):.2f}")
print(f"    Max    : {deg_vals.max():,}")
print(f"    Min    : {deg_vals.min():,}")

print("\n  Top 20 relation types:")
rels = pd.Series([
    d.get("relation","") for _,_,d in list(G.edges(data=True))[:500_000]
]).value_counts()
for rel, count in rels.head(20).items():
    print(f"    {rel:<45} {count:>8,}")
print(f"{'='*50}\n")

# ── Seeds pinned in a circle ───────────────────────────────
print("Setting up seeds...")
SEED_CUIS = {
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

CANVAS = 1400
CX, CY = CANVAS // 2, CANVAS // 2
RADIUS = 520
n_seeds = len(SEED_CUIS)
seed_positions = {}
for i, (cui, label) in enumerate(SEED_CUIS.items()):
    angle = (2 * math.pi * i / n_seeds) - math.pi / 2
    x = CX + RADIUS * math.cos(angle)
    y = CY + RADIUS * math.sin(angle)
    seed_positions[cui] = (int(x), int(y))
    print(f"  {label} → ({int(x)}, {int(y)})")

seed_cuis = set(SEED_CUIS.keys()) & set(G.nodes())

# ── Junk filter ───────────────────────────────────────────
JUNK_STYS = {
    "Temporal Concept", "Spatial Concept", "Quantitative Concept",
    "Qualitative Concept", "Functional Concept", "Intellectual Product",
    "Conceptual Entity", "Idea or Concept", "Classification",
    "Health Care Related Organization", "Language", "Geographic Area",
    "Body Substance", "Body Space or Junction",
}

def is_useful(n):
    return G.nodes[n].get("sty_name","") not in JUNK_STYS

# ── MAX 5 neighbours per seed ─────────────────────────────
print("\nExtracting neighbours...")
neighbour_seed_count = {}
# Store which seed each neighbour belongs to for positioning
neighbour_primary_seed = {}
all_neighbours = set()

for cui in seed_cuis:
    nbrs = [
        n for n in (set(G.successors(cui)) | set(G.predecessors(cui)))
        if n not in seed_cuis and is_useful(n)
    ]
    nbrs_ranked = sorted(nbrs, key=lambda n: G.degree(n), reverse=True)[:5]
    for nbr in nbrs_ranked:
        count = neighbour_seed_count.get(nbr, 0)
        neighbour_seed_count[nbr] = count + 1
        if count == 0:
            neighbour_primary_seed[nbr] = cui
        all_neighbours.add(nbr)

shared = {n for n, c in neighbour_seed_count.items() if c >= 2}
single = {n for n, c in neighbour_seed_count.items() if c == 1}
print(f"  Shared neighbours: {len(shared)}")
print(f"  Single neighbours: {len(single)}")
print(f"  Total nodes: {len(seed_cuis) + len(all_neighbours)}")

print("\n  Top shared neighbours:")
for n in sorted(shared, key=lambda n: neighbour_seed_count[n], reverse=True)[:10]:
    print(f"    {G.nodes[n].get('name',''):<45} seeds: {neighbour_seed_count[n]}")

# ── Pre-compute neighbour positions near their primary seed ─
# Place neighbours in a small cluster around their seed
neighbour_positions = {}
seed_neighbour_lists = {cui: [] for cui in seed_cuis}
for nbr in all_neighbours:
    primary = neighbour_primary_seed.get(nbr)
    if primary:
        seed_neighbour_lists[primary].append(nbr)

for cui, nbrs in seed_neighbour_lists.items():
    sx, sy = seed_positions[cui]
    # Direction from centre to seed
    dx = sx - CX
    dy = sy - CY
    dist = math.sqrt(dx*dx + dy*dy) or 1
    # Outward unit vector
    ux, uy = dx/dist, dy/dist
    # Perpendicular
    px, py = -uy, ux
    spread = 120
    for j, nbr in enumerate(nbrs):
        offset = (j - len(nbrs)/2) * (spread / max(len(nbrs), 1))
        nx_ = sx + ux * 180 + px * offset
        ny_ = sy + uy * 180 + py * offset
        neighbour_positions[nbr] = (int(nx_), int(ny_))

# ── STY colour palette ─────────────────────────────────────
STY_PALETTE = {
    "Disease or Syndrome"                  : "#E57373",
    "Neoplastic Process"                   : "#EF9A9A",
    "Injury or Poisoning"                  : "#FFAB91",
    "Mental or Behavioral Dysfunction"     : "#F48FB1",
    "Pharmacologic Substance"              : "#64B5F6",
    "Clinical Drug"                        : "#4FC3F7",
    "Antibiotic"                           : "#80DEEA",
    "Organic Chemical"                     : "#B39DDB",
    "Biologically Active Substance"        : "#CE93D8",
    "Enzyme"                               : "#A5D6A7",
    "Body Part, Organ, or Organ Component" : "#81C784",
    "Tissue"                               : "#C5E1A5",
    "Finding"                              : "#FFD54F",
    "Diagnostic Procedure"                 : "#FFE082",
    "Therapeutic or Preventive Procedure"  : "#FFF176",
    "Laboratory Procedure"                 : "#E6EE9C",
    "Gene or Genome"                       : "#80CBC4",
    "Immunologic Factor"                   : "#4DB6AC",
}
def get_color(sty):
    return STY_PALETTE.get(sty, "#B0BEC5")

# ── Build pyvis ────────────────────────────────────────────
print("\nBuilding visualisation...")
nodes_to_include = seed_cuis | all_neighbours
subG = G.subgraph(nodes_to_include).copy()

net = Network(height=f"{CANVAS}px", width="100%", directed=True, notebook=False)

# Seed nodes — pinned
for cui, label in SEED_CUIS.items():
    if cui not in G:
        continue
    x, y = seed_positions[cui]
    sty  = G.nodes[cui].get("sty_name","")
    net.add_node(
        cui,
        label=label,
        title=f"<b>{label}</b><br>CUI: {cui}<br>Type: {sty}<br>Degree: {G.degree(cui):,}",
        color={"background":"#FF5722","border":"#BF360C",
               "highlight":{"background":"#FF7043","border":"#BF360C"}},
        size=50,
        shape="dot",
        font={"size":15,"color":"#1a1a1a","bold":True},
        x=x, y=y,
        physics=False,
        fixed=True,
    )

# Neighbour nodes — pre-positioned but free
for node in all_neighbours:
    attrs      = subG.nodes[node]
    name       = attrs.get("name", node)
    sty        = attrs.get("sty_name","")
    is_shared  = node in shared
    seed_count = neighbour_seed_count.get(node, 0)
    color      = "#FFD700" if is_shared else get_color(sty)
    nx_, ny_   = neighbour_positions.get(node, (CX, CY))

    net.add_node(
        node,
        label=name[:22],
        title=(
            f"<b>{name}</b><br>"
            f"CUI: {node}<br>"
            f"Type: {sty}<br>"
            f"Degree: {G.degree(node):,}<br>"
            f"Seeds connected: {seed_count}"
        ),
        color={"background":color,"border":color,
               "highlight":{"background":color,"border":"#333"}},
        size=22 if is_shared else 14,
        shape="dot",
        font={"size":10,"color":"#1a1a1a"},
        x=nx_, y=ny_,
        physics=True,
    )

# Edges — ONLY seed-to-neighbour and seed-to-seed
# Skip neighbour-to-neighbour entirely
edge_count = 0
for u, v, d in subG.edges(data=True):
    u_seed = u in seed_cuis
    v_seed = v in seed_cuis
    if not u_seed and not v_seed:
        continue   # skip neighbour-neighbour edges

    rel        = d.get("relation","")
    both_seeds = u_seed and v_seed

    net.add_edge(
        u, v,
        title=rel,
        label=rel[:18] if rel else "",
        arrows="to",
        color={"color":"#5C6BC0" if both_seeds else "#FF5722",
               "opacity": 0.9},
        width=3 if both_seeds else 2,
        font={"size":9,"color":"#333","strokeWidth":2,"strokeColor":"#fff"},
        smooth={"type":"curvedCW","roundness":0.25},
        dashes=both_seeds,
    )
    edge_count += 1

print(f"  Edges drawn: {edge_count}")

net.set_options("""
{
  "physics": {
    "enabled": true,
    "barnesHut": {
      "gravitationalConstant": -4000,
      "centralGravity": 0.0,
      "springLength": 120,
      "springConstant": 0.08,
      "damping": 0.3,
      "avoidOverlap": 1.0
    },
    "stabilization": {
      "enabled": true,
      "iterations": 300,
      "updateInterval": 10
    }
  },
  "edges": {
    "font": {"size": 9, "align": "middle", "strokeWidth": 2, "strokeColor": "#ffffff"},
    "arrows": {"to": {"enabled": true, "scaleFactor": 0.6}},
    "smooth": {"type": "curvedCW", "roundness": 0.25}
  },
  "nodes": {
    "borderWidth": 1.5,
    "borderWidthSelected": 3,
    "shadow": false
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 80,
    "navigationButtons": true,
    "keyboard": true,
    "multiselect": true,
    "zoomView": true
  }
}
""")

out = f"{DATA_DIR}/graph_viz.html"
net.save_graph(out)

# ── Inject legend ─────────────────────────────────────────
legend_items = ""
for sty in sorted(STY_PALETTE):
    color = STY_PALETTE[sty]
    legend_items += (
        f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0">'
        f'<div style="width:10px;height:10px;border-radius:50%;background:{color};'
        f'flex-shrink:0;border:1px solid rgba(0,0,0,0.15)"></div>'
        f'<span style="font-size:11px;color:#444">{sty}</span></div>'
    )

legend_html = f"""
<div id="legend" style="position:absolute;top:12px;left:12px;
  background:rgba(255,255,255,0.96);border:1px solid #ddd;border-radius:10px;
  padding:14px;width:235px;font-family:-apple-system,BlinkMacSystemFont,sans-serif;
  z-index:999;max-height:92vh;overflow-y:auto;
  box-shadow:0 2px 10px rgba(0,0,0,0.1)">
  <div style="font-weight:700;font-size:14px;margin-bottom:4px;color:#111">
    UMLS Knowledge Graph
  </div>
  <div style="font-size:11px;color:#888;margin-bottom:10px">
    {len(seed_cuis)} seeds · {len(all_neighbours)} neighbours · {edge_count} edges
  </div>
  <div style="margin-bottom:10px">
    <div style="display:flex;align-items:center;gap:7px;margin:4px 0">
      <div style="width:16px;height:16px;border-radius:50%;background:#FF5722;flex-shrink:0"></div>
      <span style="font-size:11px;font-weight:600;color:#222">Seed concept (pinned)</span>
    </div>
    <div style="display:flex;align-items:center;gap:7px;margin:4px 0">
      <div style="width:12px;height:12px;border-radius:50%;background:#FFD700;flex-shrink:0"></div>
      <span style="font-size:11px;font-weight:600;color:#222">Shared connector</span>
    </div>
    <div style="display:flex;align-items:center;gap:7px;margin:4px 0">
      <div style="width:30px;height:2px;background:#5C6BC0;border-top:2px dashed #5C6BC0;flex-shrink:0"></div>
      <span style="font-size:11px;color:#555">Seed–seed relation</span>
    </div>
    <div style="display:flex;align-items:center;gap:7px;margin:4px 0">
      <div style="width:30px;height:2px;background:#FF5722;flex-shrink:0"></div>
      <span style="font-size:11px;color:#555">Seed–neighbour relation</span>
    </div>
  </div>
  <div style="font-weight:600;font-size:11px;margin-bottom:6px;
    color:#333;border-top:1px solid #eee;padding-top:8px">
    Semantic types
  </div>
  {legend_items}
  <div style="margin-top:10px;padding-top:8px;border-top:1px solid #eee;
    font-size:10px;color:#aaa;line-height:1.5">
    Hover for details · Scroll to zoom · Drag to pan
  </div>
</div>
"""

with open(out, "r") as f:
    html = f.read()
html = html.replace("<body>", f"<body>{legend_html}")
with open(out, "w") as f:
    f.write(html)

print(f"  Saved → {out}")
print("\nDone!")