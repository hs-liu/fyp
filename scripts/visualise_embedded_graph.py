# scripts/visualise_graph.py
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import random

DATA_DIR = "/vol/bitbucket/hl2622/fyp/data"

# Load graph
print("Loading graph...")
G = pickle.load(open(f"{DATA_DIR}/umls_graph_filtered_new.pkl", "rb"))

# Pick a seed and extract neighbourhood
SEED_CUI  = "C0004238"   # atrial fibrillation
N_HOPS    = 2
MAX_NODES = 200

print(f"Extracting {N_HOPS}-hop neighbourhood of {SEED_CUI}...")
subgraph_nodes = {SEED_CUI}
frontier = {SEED_CUI}
for _ in range(N_HOPS):
    next_frontier = set()
    for node in frontier:
        if node in G:
            next_frontier.update(G.neighbors(node))
    subgraph_nodes.update(next_frontier)
    frontier = next_frontier
    if len(subgraph_nodes) > MAX_NODES:
        break

# Subsample if too large
if len(subgraph_nodes) > MAX_NODES:
    subgraph_nodes = {SEED_CUI} | set(random.sample(list(subgraph_nodes - {SEED_CUI}), MAX_NODES - 1))

SG = G.subgraph(subgraph_nodes)
print(f"Subgraph: {SG.number_of_nodes()} nodes, {SG.number_of_edges()} edges")

# Get node labels and semantic types
labels    = {n: G.nodes[n].get("name", n)[:20] for n in SG.nodes()}
sty_names = [G.nodes[n].get("sty_name", "Unknown") for n in SG.nodes()]
unique_stys = list(set(sty_names))
color_map   = {s: i for i, s in enumerate(unique_stys)}
colors      = [color_map[s] for s in sty_names]

# Layout
print("Computing layout...")
pos = nx.spring_layout(SG, seed=42, k=2)

# Plot
fig, ax = plt.subplots(figsize=(18, 14))
nx.draw_networkx_edges(SG, pos, alpha=0.2, edge_color="gray", ax=ax)
sc = nx.draw_networkx_nodes(SG, pos, node_color=colors, 
                             cmap=plt.cm.tab20, node_size=300, ax=ax)
nx.draw_networkx_labels(SG, pos, labels=labels, font_size=6, ax=ax)

# Legend
for sty, idx in color_map.items():
    ax.scatter([], [], c=[plt.cm.tab20(idx/len(unique_stys))], label=sty[:30], s=50)
ax.legend(loc="upper left", fontsize=6, ncol=2)
ax.set_title(f"UMLS Graph: 2-hop neighbourhood of '{G.nodes[SEED_CUI].get('name')}' ({MAX_NODES} nodes)")
ax.axis("off")

plt.tight_layout()
plt.savefig("results/embedded_graph_visualisation.png", dpi=150, bbox_inches="tight")
print("Saved → results/embedded_graph_visualisation.png")