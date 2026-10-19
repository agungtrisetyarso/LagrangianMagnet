"""Theorem 4 (revised) and Lemma 3 (revised): theta deformation of the fixed-frame path sum.

The theta-deformed path sum inserts E_ba = exp(-i dtheta q(b,a)) into every step of Eq. (6):
    P_E^(n) = V^dag D (K_E D)^(n-1) V,      K_E = G o E   (entrywise product),
so the one-step (transfer) operator is K_E on l2(F^N), not a Hilbert-space compression.
Checked here, on regular frames (no antipodal pair, no three points on a great circle):
  (a) gauge decomposition  q(b,a) = [Phi(b) - Phi(a)]/4pi + k_ba,  k_ba in Z,
      Phi(a) = sum_j Omega(z, m_j^a, m_{j+1}^a)  (spatial Berry loop of configuration a);
  (b) Lemma 3: the two-step history a -> b -> a obtained by changing two adjacent sites to the
      remaining two vertices of a tetrahedron around the origin has charge +-1 and <b|a> != 0;
  (c) K_E is an orthogonal projector for dtheta in 2piZ and is not idempotent otherwise;
      K_E = U (G o E~) U^dag with E~ = exp(-i dtheta k), so everything is 2pi-periodic;
  (d) numerical illustration: spectral radius of K_E and growth of the free (h = 0) path sum
      ||V^dag K_E^(n-1) V|| with the number of slices n.
"""
import itertools

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

from lmcore import (Checker, FRAMES, Om, ZHAT, coh, configs, dim, generic, product_states,
                    TOL_IDENTITY, TOL_INTEGER)

C = Checker("theta_transfer")
out = {"frames": [], "lemma3": [], "cases": []}


def regular(F, tol=1e-9):
    """No antipodal pair and no three points on a great circle."""
    G = F @ F.T
    anti = np.any(np.isclose(G[~np.eye(len(F), dtype=bool)], -1))
    flat = min(abs(F[i] @ np.cross(F[j], F[k])) for i, j, k in
               itertools.combinations(range(len(F)), 3))
    return (not anti) and flat > tol, flat


def tables(F):
    """Local plaquette charge qloc[aj, bj, ak, bk] and edge loop phi[a, b] = Omega(z, a, b)."""
    ma, mb = F[:, None, None, None], F[None, :, None, None]
    mak, mbk = F[None, None, :, None], F[None, None, None, :]
    qloc = (Om(ma, mb, mbk) + Om(ma, mbk, mak)) / (4 * np.pi)
    phi = Om(ZHAT, F[:, None], F[None, :])
    return qloc, phi


def build(S, F, L):
    M, d = len(F), dim(S)
    A = configs(M, L)
    sg = np.array([(-1) ** j for j in range(L)], float)
    P = product_states(S, F, L, signs=sg)
    V = np.sqrt((d / M) ** L) * P.conj()                      # V[a, psi] = (d/M)^{L/2}<n_a|psi>
    G = V @ V.conj().T                                         # G[b, a]
    qloc, phi = tables(F)
    q = np.zeros((len(A), len(A)))
    for j in range(L):
        k = (j + 1) % L
        q += qloc[A[None, :, j], A[:, None, j], A[None, :, k], A[:, None, k]]
    Phi = sum(phi[A[:, j], A[:, (j + 1) % L]] for j in range(L))
    return V, G, q, Phi


def rho(K):
    if len(K) <= 512:
        return np.abs(np.linalg.eigvals(K)).max()
    op = LinearOperator(K.shape, matvec=lambda x: K @ x, dtype=complex)
    return np.abs(eigs(op, k=1, which="LM", return_eigenvectors=False, tol=1e-10)).max()


print("== frame regularity ==")
for name in ("tetrahedral", "two-tetrahedral", "octahedral", "cube", "icosahedral"):
    ok, flat = regular(FRAMES[name]())
    out["frames"].append({"frame": name, "regular": ok, "min_triple_product": flat})
    print(f"  {name}: regular = {ok} (min |a.(b x c)| over triples = {flat:.3e})")
C.check("tetrahedral frame is regular", 0, 0, ok=out["frames"][0]["regular"])
C.check("two-tetrahedral frame is regular", 0, 0, ok=out["frames"][1]["regular"])


def q_step(ma, mb):
    L = len(ma)
    return sum(Om(ma[j], mb[j], mb[(j + 1) % L]) + Om(ma[j], mb[(j + 1) % L], ma[(j + 1) % L])
               for j in range(L)) / (4 * np.pi)


print("== Lemma 3: two-step closed history with charge +-1 ==")
for name in ("tetrahedral", "two-tetrahedral"):
    F = FRAMES[name]()
    p, s, t, r = F[:4]                      # vertices of a tetrahedron containing the origin
    for S in (0.5, 1):
        for L in (4, 6, 8):
            ma = np.array([p, p, r] + [r] * (L - 3))
            mb = np.array([p, s, t] + [r] * (L - 3))
            Qab = q_step(ma, mb) + q_step(mb, ma)
            ov = np.prod([abs(coh(S, (-1) ** j * mb[j]).conj() @ coh(S, (-1) ** j * ma[j]))
                          for j in range(L)])
            out["lemma3"].append({"frame": name, "S": S, "L": L, "Q": Qab, "|<b|a>|": ov})
            C.check(f"{name} S={S} L={L}: Q(a->b->a) = +-1", abs(Qab) - 1, TOL_INTEGER)
            C.check(f"{name} S={S} L={L}: overlap = ((1+p.s)(1+r.t)/4)^S > 0",
                    ov - ((1 + p @ s) * (1 + r @ t) / 4) ** S, TOL_IDENTITY)

GRID = np.linspace(0, 2 * np.pi, 81)
for name, S, L in (("tetrahedral", 0.5, 4), ("tetrahedral", 1, 4), ("tetrahedral", 0.5, 6),
                   ("two-tetrahedral", 0.5, 4)):
    F = FRAMES[name]()
    print(f"== {name} S={S} L={L} ==")
    V, G, q, Phi = build(S, F, L)
    k = q - (Phi[:, None] - Phi[None, :]) / (4 * np.pi)
    kres = np.abs(k - np.round(k)).max()
    C.check(f"{name} S={S} L={L}: q - dPhi/4pi integer (all {q.size} pairs)", kres, TOL_INTEGER)
    kk = np.round(k)
    ovl = np.abs(G) > 1e-14
    kvals, kcnt = np.unique(kk[ovl], return_counts=True)
    rec = {"frame": name, "S": S, "L": L, "dim_l2": len(G), "k_hist": dict(zip(kvals.tolist(),
           kcnt.tolist())), "k_residue": kres}
    proj = {}
    for dth in (0.0, 2 * np.pi, 4 * np.pi, np.pi / 2, np.pi, 0.3):
        K = G * np.exp(-1j * dth * q)
        Ug = np.exp(-1j * dth * Phi / (4 * np.pi))
        gdev = np.abs(K - Ug[:, None] * (G * np.exp(-1j * dth * kk)) * Ug.conj()[None, :]).max()
        idem = np.abs(K @ K - K).max()
        proj[f"{dth / np.pi:.3f}pi"] = {"idempotency": idem, "gauge_form_dev": gdev,
                                        "hermiticity": np.abs(K - K.conj().T).max()}
        C.check(f"  K_E = U (G o E~) U^dag at dtheta={dth / np.pi:.2f}pi", gdev, TOL_IDENTITY)
        if np.isclose(dth % (2 * np.pi), 0):
            C.check(f"  K_E idempotent at dtheta={dth / np.pi:.0f}pi", idem, TOL_IDENTITY)
        else:
            C.check(f"  K_E not idempotent at dtheta={dth / np.pi:.2f}pi (>1e-8, roundoff ~1e-15)",
                    idem, 0, ok=idem > 1e-8)
    rec["projector_tests"] = proj
    rads = np.array([rho(G * np.exp(-1j * t * q)) for t in GRID])
    rec["dtheta"] = GRID
    rec["spectral_radius"] = rads
    imax = rads.argmax()
    rec["max_radius"] = [GRID[imax] / np.pi, rads[imax]]
    inner = rads[1:-1]
    C.check(f"  rho(K_E) > 1 at every grid point of (0, 2pi)", inner.min() - 1, 0, ok=inner.min() > 1)
    C.check(f"  rho(K_E) = 1 at dtheta = 0, 2pi", max(abs(rads[0] - 1), abs(rads[-1] - 1)),
            1e-8)
    # free path sum growth with slices
    grow = {}
    for dth in (np.pi / 2, np.pi, 2 * np.pi):
        K = G * np.exp(-1j * dth * q)
        W = V.copy()
        norms = []
        for n in range(1, 65):
            if n in (2, 4, 8, 16, 32, 64):
                norms.append(np.linalg.norm(V.conj().T @ W, 2))
            W = K @ W
        grow[f"{dth / np.pi:.1f}pi"] = norms
    rec["free_pathsum_norms_n=2,4,8,16,32,64"] = grow
    out["cases"].append(rec)
    print(f"  k on overlapping pairs: {rec['k_hist']}")
    print(f"  max rho(K_E) = {rads[imax]:.4f} at {GRID[imax] / np.pi:.3f} pi;"
          f" min on (0,2pi) = {inner.min():.4f}")
    for key, v in grow.items():
        print(f"  ||V^dag K^(n-1) V||, dtheta={key}: " + " ".join(f"{x:.3g}" for x in v))

C.finish(out)
