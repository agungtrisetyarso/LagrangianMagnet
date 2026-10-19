"""Lemma 1 (finite resolution), Lemma 2 (Berry phase / projector), Proposition 1(iv)
(first-order symbols) and Theorem 2(a) (reference independence <=> 2*kappa in Z).

Every check compares the finite-frame object with an independent matrix computation:
  * Lemma 1: frame sums vs exact cubature (Gauss product rule) and vs the analytic operator;
  * Lemma 2: the closed-form phase vs the overlap computed from the state vectors;
  * the projector property of the kernel G built from Omega (not from overlaps).
"""
import numpy as np

from lmcore import (Checker, FRAMES, Om, ZHAT, coh, design_strength, dim, gauss_sphere,
                    generic, product_states, spin, unit, configs, TOL_IDENTITY, TOL_INTEGER)

C = Checker("verify_all")
rng = np.random.default_rng(20261019)
SPINS = [0.5, 1, 1.5, 2, 2.5, 3]


def T1(S, pts, w, h):
    """Single-site Toeplitz map with weights w (sum w = 1): d * sum w h(n) |n><n|."""
    d = dim(S)
    out = np.zeros((d, d), complex)
    for n, wi in zip(pts, w):
        v = coh(S, n)
        out += wi * h(n) * np.outer(v, v.conj())
    return d * out


out = {"lemma1": [], "lemma2": {}, "projector": [], "theorem2a": {}}

print("== Lemma 1: frame sum vs exact cubature ==")
Q, wQ = gauss_sphere(14)          # exact up to degree 14 (>= 2S+2 for S <= 6)
for fname, f in FRAMES.items():
    F = f()
    t = design_strength(F)
    wF = np.full(len(F), 1 / len(F))
    for S in SPINS:
        row = {"frame": fname, "M": len(F), "t": t, "S": S}
        for label, q, h in (("identity", 0, lambda n: 1.0),
                            ("n_x", 1, lambda n: n[0]),
                            ("n_z^2", 2, lambda n: n[2] ** 2),
                            ("n_x n_y", 2, lambda n: n[0] * n[1])):
            dev = np.abs(T1(S, F, wF, h) - T1(S, Q, wQ, h)).max()
            row[label] = dev
            if t >= 2 * S + q:
                C.check(f"L1 {fname} (t={t}) S={S} {label}", dev, TOL_IDENTITY)
        out["lemma1"].append(row)
# negative controls: hypothesis t >= 2S+q violated
neg = [r for r in out["lemma1"] if r["t"] < 2 * r["S"]]
worst = min(r["identity"] for r in neg)
C.check("L1 negative control: min deviation when t < 2S (must exceed 1e-3)", worst, 0,
        ok=worst > 1e-3)

print("== Proposition 1(iv), first order: T[n_a] = J_a/(S+1) ==")
for S in SPINS:
    J = spin(S)
    dev = max(np.abs(T1(S, Q, wQ, lambda n, a=a: n[a]) - J[a] / (S + 1)).max() for a in range(3))
    C.check(f"P1(iv) S={S}", dev, TOL_IDENTITY)

print("== Lemma 2: overlap = rho * exp(-i S Omega(z,a,b)) ==")
for S in SPINS:
    err = 0.0
    for _ in range(2000):
        a, b = unit(rng.normal(size=(2, 3)))
        ov = coh(S, b).conj() @ coh(S, a)
        pred = ((1 + a @ b) / 2) ** S * np.exp(-1j * S * Om(ZHAT, a, b))
        err = max(err, abs(ov - pred))
    out["lemma2"][str(S)] = err
    C.check(f"L2 S={S} (2000 random pairs)", err, TOL_IDENTITY)
# negative control: the formula is gauge-singular at n = -z
S = 1
a, b = -ZHAT, unit([0.3, 0.2, 0.9])
dev = abs(coh(S, b).conj() @ coh(S, a) - ((1 + a @ b) / 2) ** S * np.exp(-1j * S * Om(ZHAT, a, b)))
out["lemma2"]["at_minus_z"] = dev
C.check("L2 negative control at n_a = -z (must fail)", dev, 0, ok=dev > 1e-3)

print("== Lemma 2: kernel G built from Omega is a rank-d^N orthogonal projector ==")
for fname, S, N in (("tetrahedral", 0.5, 3), ("two-tetrahedral", 0.5, 2),
                    ("octahedral", 1, 2), ("cube", 1, 2), ("icosahedral", 1.5, 2)):
    F = generic(FRAMES[fname]())        # generic orientation: -z is not a frame point
    d, M = dim(S), len(F)
    A = configs(M, N)
    n = F[A]                              # (C, N, 3)
    na, nb = n[None, :, :, :], n[:, None, :, :]
    rho = np.prod(np.clip((1 + np.einsum("...i,...i", na, nb)) / 2, 0, None) ** S, axis=-1)
    L0 = -S * Om(ZHAT, na, nb).sum(-1)
    G = (d / M) ** N * np.where(rho > 0, rho * np.exp(1j * np.nan_to_num(L0)), 0)
    P = product_states(S, F, N)
    dev_vv = np.abs(G - (d / M) ** N * P.conj() @ P.T).max()
    dev_idem = np.abs(G @ G - G).max()
    dev_herm = np.abs(G - G.conj().T).max()
    rank = np.linalg.matrix_rank(G, tol=1e-8)
    out["projector"].append({"frame": fname, "S": S, "N": N, "VVdag": dev_vv,
                             "idempotent": dev_idem, "hermitian": dev_herm, "rank": int(rank),
                             "d^N": d ** N})
    C.check(f"L2 G=VV^dag {fname} S={S} N={N}", dev_vv, TOL_IDENTITY)
    C.check(f"L2 G^2=G {fname} S={S} N={N}", dev_idem, TOL_IDENTITY)
    C.check(f"L2 rank G = d^N = {d ** N} ({fname})", rank - d ** N, 0)

print("== Theorem 2(a): reference change is 4*pi*Z on closed loops ==")
resid, ms = 0.0, []
for _ in range(500):
    K = rng.integers(3, 13)
    loop = unit(rng.normal(size=(K, 3)))
    r, rp = unit(rng.normal(size=(2, 3)))
    D = sum(Om(r, loop[k], loop[(k + 1) % K]) - Om(rp, loop[k], loop[(k + 1) % K])
            for k in range(K)) / (4 * np.pi)
    resid = max(resid, abs(D - round(D)))
    ms.append(int(round(D)))
vals, counts = np.unique(ms, return_counts=True)
out["theorem2a"] = {"max_residual": resid, "m_histogram": dict(zip(map(int, vals), map(int, counts)))}
C.check("T2(a) max |Delta/4pi - nearest integer| over 500 loops", resid, TOL_INTEGER)
C.check("T2(a) m = +-1 occurs (needed for 'only if')", 0, 0, ok=(1 in ms or -1 in ms))
# branch conventions: Omega is undefined exactly when two vertices are antipodal, because
#   num^2 + den^2 = 2 (1+a.b)(1+b.c)(1+c.a);
# on great-circle (num = 0) triangles it takes the value +-2pi, which is harmless mod 4pi.
def cocycle(a, b, c, d):
    return (Om(b, c, d) - Om(a, c, d) + Om(a, b, d) - Om(a, b, c)) / (4 * np.pi)


idw = 0.0
for _ in range(20000):
    a_, b_, c_ = unit(rng.normal(size=(3, 3)))
    num = a_ @ np.cross(b_, c_)
    den = 1 + a_ @ b_ + b_ @ c_ + c_ @ a_
    idw = max(idw, abs(num ** 2 + den ** 2 - 2 * (1 + a_ @ b_) * (1 + b_ @ c_) * (1 + c_ @ a_)))
C.check("identity num^2+den^2 = 2(1+a.b)(1+b.c)(1+c.a)", idw, TOL_IDENTITY)
deg = 0.0
for _ in range(20000):
    u = unit(rng.normal(size=3))
    w = unit(np.cross(u, rng.normal(size=3)))
    k3 = rng.integers(3, 5)                       # 3 or 4 of the points on one great circle
    th = rng.uniform(0, 2 * np.pi, 4)
    pts = [np.cos(t) * u + np.sin(t) * w for t in th]
    if k3 == 3:
        pts[3] = unit(rng.normal(size=3))
    x = cocycle(*pts)
    deg = max(deg, abs(x - round(x)))
out["theorem2a"]["great_circle_cocycle_max_residual"] = deg
C.check("cocycle in 4piZ on 20000 great-circle (degenerate) quadruples", deg, TOL_INTEGER)
reg_worst = 0.0
for _ in range(20000):
    x = cocycle(*unit(rng.normal(size=(4, 3))))
    reg_worst = max(reg_worst, abs(x - round(x)))
out["theorem2a"]["generic_cocycle_max_residual"] = reg_worst
C.check("cocycle in 4piZ on 20000 generic quadruples", reg_worst, TOL_INTEGER)
kappa = 0.3
C.check("T2(a) kappa=0.3 gives a reference-dependent phase",
        abs(np.exp(4j * np.pi * kappa) - 1), 0, ok=abs(np.exp(4j * np.pi * kappa) - 1) > 0.1)

C.finish(out)
