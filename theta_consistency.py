"""Theorem 4 (fixed-frame obstruction to generic theta): the deformed one-step operator at
dt = 0,
    A0(dtheta) = (d/M)^{2L} sum_{a,b} exp(-i dtheta q(b,a)) <b|a> |b><a|,
on a periodic chain of L sites, with the frame placed on the staggered field m_j and
n_j = (-1)^j m_j.  Consistency (Theorem 4(i)) requires A0 = 1.

Two independent constructions are used and cross-checked:
  * brute force over all M^{2L} pairs (a, b)  [L = 4, M = 4];
  * a periodic matrix-product-operator contraction (bond index (a_j, b_j)), which reaches
    L = 6, 8 and the M = 8 two-tetrahedron frame.
Reported for each case: ||A0||_2, ||A0 - 1||_2 and the spectral radius over
dtheta in [0, 4 pi] (161 points), the peak located by golden-section refinement, and the
values at dtheta = 0 and 8 pi.
"""
import itertools

import numpy as np

from lmcore import Checker, FRAMES, Om, coh, configs, dim, product_states, TOL_IDENTITY

C = Checker("theta_consistency")
out = {"cases": []}


def local_data(S, F, L):
    M, d = len(F), dim(S)
    sg = [(-1) ** j for j in range(L)]
    states = [np.array([coh(S, sg[j] * F[a]) for a in range(M)]) for j in range(L)]
    # O_j[a, b] = |b><b|a><a|   (d x d)
    O = []
    for j in range(L):
        cs = states[j]
        ov = cs.conj() @ cs.T                      # ov[b, a] = <b|a>
        Oj = np.einsum("bi,ba,aj->abij", cs, ov, cs.conj())
        O.append(Oj.reshape(M * M, d, d))
    # plaquette charge between neighbouring sites: q(aj, bj, ak, bk)
    ma = F[:, None, None, None, :]
    mb = F[None, :, None, None, :]
    mak = F[None, None, :, None, :]
    mbk = F[None, None, None, :, :]
    q = (Om(ma, mb, mbk) + Om(ma, mbk, mak)) / (4 * np.pi)   # [aj, bj, ak, bk]
    q = q.reshape(M * M, M * M)
    return O, q


def A0_mpo(S, F, L, dth, O=None, q=None):
    M, d = len(F), dim(S)
    if O is None:
        O, q = local_data(S, F, L)
    B = np.exp(-1j * dth * q)
    D = M * M
    T = np.zeros((D, D, d, d), complex)
    for x in range(D):
        T[x, x] = O[0][x]
    for k in range(1, L):
        Tn = np.einsum("uxij,xy->uyij", T, B)
        T = np.einsum("uyij,ykl->uyikjl", Tn, O[k]).reshape(D, D, d ** (k + 1), d ** (k + 1))
    A = np.einsum("uxij,xu->ij", T, B)
    return (d / M) ** (2 * L) * A


def A0_brute(S, F, L, dth):
    M, d = len(F), dim(S)
    cf = configs(M, L)
    sg = np.array([(-1) ** j for j in range(L)], float)
    P = product_states(S, F, L, signs=sg)        # rows |a>
    G = P.conj() @ P.T                          # G[b, a] = <b|a>
    m = F[cf]
    a = m[None]
    b = m[:, None]
    ap, bp = np.roll(a, -1, axis=2), np.roll(b, -1, axis=2)
    qq = (Om(a, b, bp) + Om(a, bp, ap)).sum(-1) / (4 * np.pi)     # q[b, a]
    E = np.exp(-1j * dth * qq)
    return (d / M) ** (2 * L) * np.einsum("bi,ba,aj->ij", P, G * E, P.conj()), qq, G


def metrics(A):
    I = np.eye(len(A))
    return (np.linalg.norm(A, 2), np.linalg.norm(A - I, 2), np.abs(np.linalg.eigvals(A)).max())


def golden_max(f, lo, hi, it=40):
    g = (5 ** 0.5 - 1) / 2
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = f(x1), f(x2)
    for _ in range(it):
        if f1 > f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - g * (hi - lo)
            f1 = f(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + g * (hi - lo)
            f2 = f(x2)
    return (x1, f1) if f1 > f2 else (x2, f2)


print("== cross-check: MPO contraction vs brute force (tetrahedral, L=4) ==")
grid_check = [0.0, 0.7, np.pi, 2 * np.pi, 3.3]
for S in (0.5, 1):
    F = FRAMES["tetrahedral"]()
    O, q = local_data(S, F, 4)
    dev = 0.0
    for t in grid_check:
        Ab, qq, G = A0_brute(S, F, 4, t)
        dev = max(dev, np.abs(Ab - A0_mpo(S, F, 4, t, O, q)).max())
    C.check(f"MPO = brute force, S={S}", dev, TOL_IDENTITY)
    overl = np.abs(G) > 1e-12
    qvals = np.unique(np.round(4 * qq[overl], 9))
    frac = np.abs(4 * qq[overl] - np.round(4 * qq[overl])).max()
    C.check(f"tetrahedral S={S}: 4q integer on every overlapping pair", frac, TOL_IDENTITY)
    out[f"tetra_S{S}_4q_values"] = qvals.tolist()

GRID = np.linspace(0, 4 * np.pi, 161)
CASES = [("tetrahedral", 0.5, 4), ("tetrahedral", 1, 4), ("tetrahedral", 0.5, 6),
         ("tetrahedral", 0.5, 8), ("two-tetrahedral", 0.5, 4)]
for fname, S, L in CASES:
    F = FRAMES[fname]()
    O, q = local_data(S, F, L)
    f = lambda t: A0_mpo(S, F, L, t, O, q)
    rows = np.array([metrics(f(t)) for t in GRID])
    nrm, dev, rad = rows.T
    i_n, i_r = nrm.argmax(), rad.argmax()
    t_n, v_n = golden_max(lambda t: metrics(f(t))[0], GRID[max(i_n - 1, 0)], GRID[min(i_n + 1, 160)])
    t_r, v_r = golden_max(lambda t: metrics(f(t))[2], GRID[max(i_r - 1, 0)], GRID[min(i_r + 1, 160)])
    at0 = np.abs(f(0.0) - np.eye(dim(S) ** L)).max()
    rec = {"frame": fname, "M": len(F), "S": S, "L": L, "dtheta": GRID, "norm": nrm,
           "dev": dev, "spectral_radius": rad, "norm_peak": [t_n / np.pi, v_n],
           "radius_peak": [t_r / np.pi, v_r], "A(0)-1": at0,
           "min_dev_on_(0,4pi]": dev[1:].min()}
    if fname == "tetrahedral":
        rec["A(8pi)-1"] = np.abs(f(8 * np.pi) - np.eye(dim(S) ** L)).max()
        C.check(f"{fname} S={S} L={L}: A0(8 pi) = 1", rec["A(8pi)-1"], TOL_IDENTITY)
    out["cases"].append(rec)
    print(f"  {fname} S={S} L={L}: max||A0|| = {v_n:.4f} at {t_n/np.pi:.4f} pi;"
          f" max rho = {v_r:.4f} at {t_r/np.pi:.4f} pi; min ||A0-1|| on grid (0,4pi] ="
          f" {dev[1:].min():.3e}")
    C.check(f"{fname} S={S} L={L}: A0(0) = 1", at0, TOL_IDENTITY)
    C.check(f"{fname} S={S} L={L}: A0 != 1 at every grid point in (0,4pi]",
            dev[1:].min(), 0, ok=dev[1:].min() > 1e-3)
    C.check(f"{fname} S={S} L={L}: spectral radius > 1 somewhere", v_r - 1, 0, ok=v_r > 1)

C.finish(out)
