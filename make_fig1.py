"""Regenerate Fig. 1 (fig1.pdf, fig1.png) from results/*.json.

Run verify_pathsum.py, verify_theta.py and theta_consistency.py first (run_all.py does this).
Every plotted number is read from those JSON files; nothing is recomputed here.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda name: json.load(open(os.path.join(HERE, "results", name + ".json")))
# categorical slots 1-4 of the validated default palette; identity also carried by marker
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MK = ["o", "s", "^", "D"]
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
                     "legend.frameon": False, "lines.linewidth": 1.4})
fig = plt.figure(figsize=(6.4, 5.8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9])
ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]

# (a) Theorem 1: error vs dt, with the rigorous bound
ps = R("verify_pathsum")["convergence"]
pick = [("S=1/2 triangle", "tetrahedral", r"$S=1/2$ triangle (frustrated)"),
        ("S=1/2 4-ring", "tetrahedral", r"$S=1/2$ 4-site ring"),
        ("S=1 triangle", "octahedral", r"$S=1$ triangle"),
        ("S=3/2 triangle", "icosahedral", r"$S=3/2$ triangle")]
for i, (lab, fr, txt) in enumerate(pick):
    rec = next(r for r in ps if r["label"] == lab and r["frame"] == fr)
    dt = [x["dt"] for x in rec["rows"]]
    ax[0].loglog(dt, [x["err"] for x in rec["rows"]], marker=MK[i], ms=4, color=COL[i],
                 label=f"{txt} [{fr[:4]}., M={rec['M']}]")
    ax[0].loglog(dt, [x["bound"] for x in rec["rows"]], ":", color=COL[i], lw=1)
ax[0].set_xlabel(r"time step $\delta t$")
ax[0].set_ylabel(r"$\|\mathcal{P}^{(n)}_\delta-e^{-iTH}\|$")
ax[0].legend(fontsize=7, loc="upper left")
ax[0].text(0.98, 0.04, "dotted: Theorem 1 bound", transform=ax[0].transAxes, ha="right",
           fontsize=7, color=MUTED)
ax[0].set_title("(a)", loc="left", fontsize=9)

# (b) Theorem 3: aliasing remainder vs L
th = R("verify_theta")["fig1b"]
for i, seed in enumerate((0, 1, 2)):
    for n, ls in ((256, "-"), (16, ":")):
        rows = th[f"seed{seed}_n{n}"]
        L = [r["L"] for r in rows]
        e = np.maximum([abs(r["eps"]) for r in rows], 1e-16)
        Q = round(rows[0]["Q"])
        ax[1].semilogy(L, e, ls, marker=MK[i] if n == 256 else None, ms=3.5, color=COL[i],
                       label=f"history {seed + 1} ($Q={Q}$)" if n == 256 else None)
ax[1].set_xlabel(r"chain length $L$")
ax[1].set_ylabel(r"remainder $|\varepsilon|$")
ax[1].legend(fontsize=7, loc="lower left")
ax[1].text(0.98, 0.94, r"solid: $n=256$, dotted: $n=16$ time slices", transform=ax[1].transAxes,
           ha="right", va="top", fontsize=7, color=MUTED)
ax[1].set_title("(b)", loc="left", fontsize=9)

# (c) Theorem 4: spectral radius of the theta-deformed transfer operator K_E
tt = R("theta_transfer")["cases"]
show = [("tetrahedral", 0.5, 4, "tetrahedral, $S=1/2$, $L=4$"),
        ("tetrahedral", 1, 4, "tetrahedral, $S=1$, $L=4$"),
        ("tetrahedral", 0.5, 6, "tetrahedral, $S=1/2$, $L=6$"),
        ("two-tetrahedral", 0.5, 4, "two-tetrahedral ($M=8$), $S=1/2$, $L=4$")]
for i, (fr, S, L, lab) in enumerate(show):
    rec = next(r for r in tt if r["frame"] == fr and r["S"] == S and r["L"] == L)
    x = np.array(rec["dtheta"]) / np.pi
    ax[2].plot(x, rec["spectral_radius"], color=COL[i], lw=1.4, marker=MK[i], ms=3,
               markevery=8, label=lab)
ax[2].axhline(1, color=MUTED, lw=0.6, ls=":")
ax[2].set_xlabel(r"$\Delta\theta/\pi=(\theta-2\pi S)/\pi$")
ax[2].set_ylabel(r"spectral radius $\rho(K_E)$")
ax[2].legend(fontsize=7, loc="upper right")
ax[2].text(0.02, 0.95, r"$K_E$ is an orthogonal projector at $\Delta\theta\in2\pi\mathbb{Z}$",
           transform=ax[2].transAxes, va="top", ha="left", fontsize=7, color=MUTED)
ax[2].set_title("(c)", loc="left", fontsize=9)

for a in ax:
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig1.pdf"))
fig.savefig(os.path.join(HERE, "fig1.png"), dpi=200)
print("wrote fig1.pdf, fig1.png")
