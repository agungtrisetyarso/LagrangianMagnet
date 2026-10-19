"""Independent re-implementation of the headline numbers (shares no code with lmcore.py).

Differences from the main scripts:
  * coherent states from the Wigner rotation e^{-i phi Jz} e^{-i theta Jy} e^{i phi Jz} built by
    scipy.linalg.expm from freshly constructed spin matrices (not the closed form);
  * solid angles from L'Huilier's spherical-excess formula with the sign of the triple
    product, not from atan2;
  * the frame path sum is formed as an explicit sum over frame points of rank-one projectors.
It then compares with results/*.json written by the main scripts and fails on disagreement.
"""
import json
import os
import sys

import numpy as np
from scipy.linalg import expm, eigh

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda n: json.load(open(os.path.join(HERE, "results", n + ".json")))
fails = []


def report(label, mine, ref, tol):
    ok = abs(mine - ref) <= tol
    print(f"  [{'OK' if ok else 'MISMATCH'}] {label}: independent {mine:.6g} vs stored {ref:.6g}")
    if not ok:
        fails.append(label)


def spin_ops(S):
    m = np.arange(S, -S - 1, -1)
    d = len(m)
    Sp = np.diag(np.sqrt(S * (S + 1) - m[1:] * (m[1:] + 1)), 1)
    return (Sp + Sp.T) / 2, (Sp - Sp.T) / 2j, np.diag(m)


def cstate(S, v):
    Sx, Sy, Sz = spin_ops(S)
    th, ph = np.arccos(np.clip(v[2], -1, 1)), np.arctan2(v[1], v[0])
    e = np.zeros(len(Sz))
    e[0] = 1
    return expm(-1j * ph * Sz) @ expm(-1j * th * Sy) @ expm(1j * ph * Sz) @ e


def excess(a, b, c):
    """Signed spherical excess by L'Huilier (valid away from antipodes / great circles)."""
    A = np.arccos(np.clip(b @ c, -1, 1))
    B = np.arccos(np.clip(c @ a, -1, 1))
    Cc = np.arccos(np.clip(a @ b, -1, 1))
    s = (A + B + Cc) / 2
    t = np.tan(s / 2) * np.tan((s - A) / 2) * np.tan((s - B) / 2) * np.tan((s - Cc) / 2)
    E = 4 * np.arctan(np.sqrt(max(t, 0.0)))
    return np.sign(a @ np.cross(b, c)) * E


TET = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
OCT = np.vstack([np.eye(3), -np.eye(3)])

print("== Theorem 1: S=1/2 triangle (tetrahedral) and S=1 triangle (octahedral), T=1 ==")
ps = R("verify_pathsum")["convergence"]
for S, F, fname, lab in ((0.5, TET, "tetrahedral", "S=1/2 triangle"),
                         (1, OCT, "octahedral", "S=1 triangle")):
    d, M, N = int(2 * S + 1), len(F), 3
    Sx, Sy, Sz = spin_ops(S)
    I = np.eye(d)
    op = lambda o, i: np.kron(np.kron(np.eye(d ** i), o), np.eye(d ** (N - i - 1)))
    H = sum(op(o, i) @ op(o, (i + 1) % N) for i in range(N) for o in (Sx, Sy, Sz))
    w, U = eigh(H)
    cs = [cstate(S, v) for v in F]
    ref = next(r for r in ps if r["label"] == lab and r["frame"] == fname)
    for row in (ref["rows"][0], ref["rows"][-1]):
        dt, n = row["dt"], row["n"]
        A = np.zeros((d ** N, d ** N), complex)
        for a0 in range(M):
            for a1 in range(M):
                for a2 in range(M):
                    v = np.kron(np.kron(cs[a0], cs[a1]), cs[a2])
                    hv = (S + 1) ** 2 * (F[a0] @ F[a1] + F[a1] @ F[a2] + F[a2] @ F[a0])
                    A += np.exp(-1j * dt * hv) * np.outer(v, v.conj())
        A *= (d / M) ** N
        err = np.linalg.norm(np.linalg.matrix_power(A, n) - (U * np.exp(-1j * w)) @ U.conj().T, 2)
        report(f"{lab} [{fname}] error at dt={dt:.4g}", err, row["err"], 1e-9)

print("== Lemma 2 phase with L'Huilier solid angles ==")
rng = np.random.default_rng(99)
z = np.array([0, 0, 1.0])
worst = 0.0
for S in (0.5, 1, 1.5, 2):
    for _ in range(300):
        a, b = rng.normal(size=(2, 3))
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        ov = cstate(S, b).conj() @ cstate(S, a)
        pred = ((1 + a @ b) / 2) ** S * np.exp(-1j * S * excess(z, a, b))
        worst = max(worst, abs(ov - pred))
report("max |overlap - formula| (S<=2, 1200 pairs)", worst, 0.0, 1e-9)

print("== Theorem 3: Q and eps for Fig. 1(b) history 3 (seed 2), n=64, L=64 and 256 ==")
th = R("verify_theta")["fig1b"]["seed2_n64"]


def trig(L, n, seed):
    x, t = np.arange(L) / L, np.arange(n) / n
    X, T = np.meshgrid(x, t, indexing="ij")
    v = np.zeros((L, n, 3))
    g = np.random.default_rng(seed)
    for i in range(3):
        for kx in range(-2, 3):
            for kt in range(-2, 3):
                c = g.normal() / (1 + kx * kx + kt * kt)
                ph = g.uniform(0, 2 * np.pi)
                v[..., i] += c * np.cos(2 * np.pi * (kx * X + kt * T) + ph)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


for L in (64, 256):
    m = trig(L, 64, 2)
    Rj = np.zeros(L)
    for j in range(L):
        for k in range(64):
            a, ap = m[j, k], m[j, (k + 1) % 64]
            b, bp = m[(j + 1) % L, k], m[(j + 1) % L, (k + 1) % 64]
            Rj[j] += excess(a, ap, bp) + excess(a, bp, b)
    Q = Rj.sum() / (4 * np.pi)
    eps = ((-1.0) ** np.arange(L) * Rj).sum()
    ref = next(r for r in th if r["L"] == L)
    report(f"Q at L={L}", Q, ref["Q"], 1e-8)
    report(f"eps at L={L}", eps, ref["eps"], 1e-8 + 1e-6 * abs(ref["eps"]))

print("== Theorem 4: spectral radius of K_E, tetrahedral S=1/2 L=4, dtheta=pi ==")
L, S = 4, 0.5
M, d = 4, 2
confs = np.array(np.meshgrid(*[np.arange(M)] * L, indexing="ij")).reshape(L, -1).T
vecs = []
for c in confs:
    v = np.array([1.0 + 0j])
    for j, a in enumerate(c):
        v = np.kron(v, cstate(S, (-1) ** j * TET[a]))
    vecs.append(v)
vecs = np.array(vecs)
G = (d / M) ** L * vecs.conj() @ vecs.T
q = np.zeros((len(confs), len(confs)))
for ib, cb in enumerate(confs):
    for ia, ca in enumerate(confs):
        s = 0.0
        for j in range(L):
            k = (j + 1) % L
            A_, B_, C_, D_ = TET[ca[j]], TET[cb[j]], TET[cb[k]], TET[ca[k]]
            s += excess(A_, B_, C_) + excess(A_, C_, D_)
        q[ib, ia] = s / (4 * np.pi)
K = G * np.exp(-1j * np.pi * q)
rho = np.abs(np.linalg.eigvals(K)).max()
tt = next(c for c in R("theta_transfer")["cases"] if c["frame"] == "tetrahedral" and c["S"] == 0.5
          and c["L"] == 4)
report("rho(K_E) at dtheta=pi", rho, tt["max_radius"][1], 1e-8)
K2 = G * np.exp(-2j * np.pi * q)
report("idempotency of K_E at dtheta=2pi", np.abs(K2 @ K2 - K2).max(), 0.0, 1e-10)

print("== Appendix C: Lambda/||H|| for the S=1/2 bond ==")
Sx, Sy, Sz = spin_ops(0.5)
Hb = sum(np.kron(o, o) for o in (Sx, Sy, Sz))
hv = np.array([2.25 * (TET[a] @ TET[b]) for a in range(4) for b in range(4)])
report("Lambda/||H|| (S=1/2 bond)", 4 * np.abs(hv).max() / np.abs(np.linalg.eigvalsh(Hb)).max(),
       R("verify_overhead")["bond"][0]["Lambda/||H||"], 1e-9)

print("== independent_check: %d mismatches ==" % len(fails))
sys.exit(1 if fails else 0)
