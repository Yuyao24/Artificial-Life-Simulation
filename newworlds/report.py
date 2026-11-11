"""End-of-run PNG report: population history and a phylogenetic tree of species."""
from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402


def build_tree(tracker) -> nx.DiGraph:
    g = nx.DiGraph()
    for sp in tracker.species.values():
        g.add_node(sp.id, name=sp.name, born=sp.born, extinct=sp.extinct,
                   color=tuple(c / 255 for c in sp.color), peak=sp.peak)
    for sp in tracker.species.values():
        if sp.parent is not None:
            g.add_edge(sp.parent, sp.id)
    return g


def _layout(g: nx.DiGraph) -> dict[int, float]:
    """y position per node: leaves get consecutive rows, parents sit between their children."""
    ys: dict[int, float] = {}
    row = [0]

    def visit(n):
        kids = sorted(g.successors(n), key=lambda k: g.nodes[k]["born"])
        # parent's own continuing lineage gets its own row first
        ys[n] = row[0]
        row[0] += 1
        for k in kids:
            visit(k)

    roots = sorted([n for n in g.nodes if g.in_degree(n) == 0], key=lambda n: g.nodes[n]["born"])
    for r in roots:
        visit(r)
    return ys


def save_report(sim, path: str | None = None) -> str:
    if path is None:
        os.makedirs("reports", exist_ok=True)
        path = os.path.join("reports", f"new_worlds_seed{sim.seed}_{time.strftime('%Y%m%d_%H%M%S')}.png")
    tracker = sim.tracker
    now = sim.t
    fig = plt.figure(figsize=(16, 10), facecolor="#12141c")
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.25], hspace=0.28)
    plt.rcParams.update({"text.color": "#dde2ee", "axes.labelcolor": "#dde2ee",
                         "xtick.color": "#9aa2b5", "ytick.color": "#9aa2b5"})

    # --- population graph ---------------------------------------------
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor("#181c28")
    t = np.array(sim.history_t)
    ids = sorted(tracker.species, key=lambda k: tracker.species[k].born)
    mat = np.array([[h.get(k, 0) for h in sim.history_species] for k in ids], float)
    cols = [tuple(c / 255 for c in tracker.species[k].color) for k in ids]
    if len(t) > 1 and len(ids):
        ax.stackplot(t, mat, colors=cols, linewidth=0)
    ax.plot(t, sim.history_total, color="white", lw=1.2, label="total")
    ax.set_xlim(0, max(now, 1))
    ax.set_ylabel("creatures")
    ax.set_title(f"New Worlds - seed {sim.seed} - {now} steps - "
                 f"{len(tracker.species)} species ever, {len(tracker.living())} alive",
                 color="#78dcaa", fontsize=14, loc="left")
    ax2 = ax.twinx()
    ax2.plot(t, np.array(sim.history_temp) * 60 - 15, color="#ff9f5a", lw=1, alpha=0.8)
    ax2.set_ylabel("temperature (°C)", color="#ff9f5a")
    ax2.tick_params(colors="#ff9f5a")
    for et, msg in sim.event_log:
        ax.axvline(et, color="#ffffff", alpha=0.35, ls="--", lw=0.8)
        ax.text(et, ax.get_ylim()[1] * 0.97, " " + msg.split(" (")[0], rotation=90, va="top",
                fontsize=8, color="#cccccc")
    for a in (ax, ax2):
        for s in a.spines.values():
            s.set_color("#3a4054")

    # --- phylogenetic tree ------------------------------------------
    axt = fig.add_subplot(gs[1])
    axt.set_facecolor("#181c28")
    g = build_tree(tracker)
    ys = _layout(g)
    pos = {n: (g.nodes[n]["born"], ys[n]) for n in g.nodes}
    peak = max([g.nodes[n]["peak"] for n in g.nodes] + [1])
    for n in g.nodes:
        d = g.nodes[n]
        end = d["extinct"] if d["extinct"] is not None else now
        lw = 1.5 + 7 * d["peak"] / peak
        axt.plot([d["born"], end], [ys[n], ys[n]], color=d["color"], lw=lw, solid_capstyle="butt")
        label = d["name"] + ("" if d["extinct"] is None else "  †")
        axt.text(end + now * 0.005, ys[n], label, va="center", fontsize=8,
                 color="#dde2ee" if d["extinct"] is None else "#8a90a0")
    for u, v in g.edges:
        bx = g.nodes[v]["born"]
        axt.plot([bx, bx], [ys[u], ys[v]], color="#9aa2b5", lw=0.9)
    nx.draw_networkx_nodes(g, pos, ax=axt, node_size=28,
                           node_color=[g.nodes[n]["color"] for n in g.nodes],
                           edgecolors="white", linewidths=0.6)
    axt.set_xlim(-now * 0.01, now * 1.18 + 1)
    axt.set_ylim(len(ys) - 0.5, -0.5)
    axt.set_yticks([])
    axt.set_xlabel("simulation step")
    axt.set_title("Phylogenetic tree (bar width = peak population, † = extinct)",
                  color="#78dcaa", fontsize=12, loc="left")
    axt.tick_params(left=False, labelleft=False, bottom=True, labelbottom=True)
    for s in axt.spines.values():
        s.set_color("#3a4054")

    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path
