import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
os.makedirs("./graphs/method", exist_ok=True)

G = pickle.load(open("/vol/bitbucket/hl2622/fyp/src/data/umls_graph_filtered_new.pkl", "rb"))

# Seed node — Atrial Fibrillation
seed_cui = "C0004238"

# 1-hop subgraph only, limit to 30 neighbours
neighbours = list(G.neighbors(seed_cui))[:25]
nodes = [seed_cui] + neighbours
subgraph = G.subgraph(nodes)

# Build labels
labels = {n: G.nodes[n].get("name", n)[:20] for n in subgraph.nodes()}
print(dict(G.nodes["C0004238"]))
# Colour by semantic type
sty_map = {n: G.nodes[n].get("sty_name", "Unknown") for n in subgraph.nodes()}
unique_stys = list(set(sty_map.values()))
color_map = plt.cm.Set2.colors
node_colors = [color_map[unique_stys.index(sty_map[n]) % len(color_map)]
               for n in subgraph.nodes()]

fig, ax = plt.subplots(figsize=(14, 10))
pos = nx.spring_layout(subgraph, seed=42, k=2.5)

nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors,
                       node_size=800, alpha=0.9, ax=ax)
nx.draw_networkx_labels(subgraph, pos, labels=labels,
                        font_size=7, ax=ax)
nx.draw_networkx_edges(subgraph, pos, alpha=0.3,
                       arrows=True, arrowsize=10,
                       edge_color="#888", ax=ax)

# Highlight seed node
nx.draw_networkx_nodes(subgraph, pos,
                       nodelist=[seed_cui],
                       node_color="red",
                       node_size=1200, ax=ax)

ax.set_title("One-hop Neighbourhood: Atrial Fibrillation (C0004238)",
             fontsize=13, fontweight="bold")
ax.axis("off")

# Legend
from matplotlib.patches import Patch
legend = [Patch(color=color_map[i % len(color_map)], label=sty)
          for i, sty in enumerate(unique_stys[:6])]
ax.legend(handles=legend, loc="lower left", fontsize=8, framealpha=0.9)

plt.tight_layout()
plt.savefig("./graphs/method/kg_visualisation_clean.png",
            dpi=150, bbox_inches="tight")
plt.close()