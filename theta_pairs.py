"""Lemma 3: explicit overlapping pairs with non-integer single-step charge, for every frame
of strength t >= 2 in the library, following the construction in the proof (p in F,
s in F \\ {+-p}, r in F off the great circle through p and s; L = 4; frame on m_j)."""
import numpy as np

from lmcore import Checker, FRAMES, Om, coh, design_strength, dim, TOL_IDENTITY

C = Checker("theta_pairs")
out = []


def q_step(ma, mb):
    L = len(ma)
    s = 0.0
    for j in range(L):
        k = (j + 1) % L
        s += Om(ma[j], mb[j], mb[k]) + Om(ma[j], mb[k], ma[k])
    return s / (4 * np.pi)


def overlap(S, ma, mb):
    v = 1.0 + 0j
    for j, (x, y) in enumerate(zip(ma, mb)):
        sg = (-1) ** j
        v *= coh(S, sg * y).conj() @ coh(S, sg * x)
    return v


for fname, f in FRAMES.items():
    F = f()
    t = design_strength(F)
    p = F[0]
    s = next(x for x in F if abs(abs(x @ p) - 1) > 1e-9)
    nrm = np.cross(p, s)
    r = next(x for x in F if abs(x @ nrm) > 1e-9)
    for S in (0.5, 1, 1.5):
        if t < 2 * S:
            continue
        ma = np.array([r, p, p, p])
        mb = np.array([r, s, p, p])
        q = q_step(ma, mb)
        ov = abs(overlap(S, ma, mb))
        q_pred = Om(r, s, p) / (4 * np.pi)
        ov_pred = ((1 + p @ s) / 2) ** S
        rec = {"frame": fname, "M": len(F), "t": t, "S": S, "q": q, "|<b|a>|": ov,
               "min_|dtheta|_with_E=1": 2 * np.pi / abs(q) / np.pi}
        out.append(rec)
        print(f"  {fname:16s} S={S}: q={q:+.6f} |<b|a>|={ov:.4f}  smallest dtheta with E=1:"
              f" {2 / abs(q):.2f} pi")
        C.check(f"{fname} S={S}: q = Omega(r,s,p)/4pi", q - q_pred, TOL_IDENTITY)
        C.check(f"{fname} S={S}: 0 < |q| < 1/2", 0, 0, ok=0 < abs(q) < 0.5)
        C.check(f"{fname} S={S}: |<b|a>| = ((1+p.s)/2)^S > 0", ov - ov_pred, TOL_IDENTITY)

C.finish({"pairs": out})
