"""High-dimensional comparison figure: |terminal utility| (log10) and max
drawdown vs number of assets, worst-case measure."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- palette (reference instance, light mode, slots 1-4 in order) ----
BLUE = "#2a78d6"     # neural policy (ours)
ORANGE = "#eb6834"   # certainty-equivalent Merton
AQUA = "#1baf7a"     # fixed-radius robust
YELLOW = "#eda100"   # homothetic robust
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

RES = "/Users/hahaha/Desktop/文章/solver/results"
NS = [30, 50, 100]
SERIES = [("nn", "Neural HJBI solver (ours)", BLUE, "o"),
          ("merton", "Certainty-equivalent Merton", ORANGE, "s"),
          ("fixed_radius", "Fixed-radius robust", AQUA, "^"),
          ("homothetic", "Homothetic robust", YELLOW, "D")]

data = {}
for n in NS:
    d = json.load(open(os.path.join(RES, f"highdim_n{n}.json")))
    b = json.load(open(os.path.join(RES, f"highdim_n{n}_bench.json")))
    wc = d["worst_case"]
    wc.update({k: v for k, v in b["worst_case"].items() if k != "nn"})
    data[n] = wc

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=200)
fig.patch.set_facecolor(SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)

for key, label, color, marker in SERIES:
    U = np.array([abs(data[n][key]["mean_terminal_utility"]) for n in NS])
    DD = np.array([data[n][key]["max_drawdown"] for n in NS])
    axes[0].plot(NS, np.log10(U), color=color, lw=2, marker=marker, ms=6,
                 label=label, zorder=3)
    axes[1].plot(NS, DD, color=color, lw=2, marker=marker, ms=6, zorder=3)

axes[0].set_yscale("linear")
axes[0].set_ylabel(r"$|\,\text{terminal utility}\,|$  ($\log_{10}$)",
                   color=INK, fontsize=11)
axes[1].set_ylabel("Max drawdown", color=INK, fontsize=11)
for ax in axes:
    ax.set_xlabel("$n$ (number of risky assets)", color=INK, fontsize=11)
    ax.set_xticks(NS)
    ax.tick_params(colors=MUTED, labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)

# direct labels on the neural series (the headline curve)
axes[0].text(NS[-1] + 1.5, np.log10(abs(data[100]["nn"]["mean_terminal_utility"])) + 0.2,
             "neural", color=INK, fontsize=10)
axes[1].text(NS[-1] + 1.5, data[100]["nn"]["max_drawdown"] + 0.02,
             "neural", color=INK, fontsize=10)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
           fontsize=9.5, bbox_to_anchor=(0.5, 1.02), labelcolor=INK)
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = "/Users/hahaha/Desktop/文章/overleaf-paper/figures/highdim_comparison.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("saved:", out)
