"""Theorem 1 (exact finite path sum) and Corollary 1 inside the dynamics.

Part A  -- the finite sum of Eq. (6) is enumerated history by history, with the Berry phase
           taken from Omega (Lemma 2), and compared with the matrix product (V^dag D V)^n.
Part B  -- (V^dag D V)^n is compared with exp(-i T H) from dense exact diagonalization, for
           several clusters, spins, frames (sizes M = 4..20) and time steps (dt = 0.2 .. 1.6e-3).
           Reported: the error, the rigorous bound T dt ||h||_inf^2, the observed order of
           convergence, and the asymptotic error constant err/dt.
"""
import numpy as np
from scipy.linalg import eigh

from lmcore import (Checker, FRAMES, Om, ZHAT, configs, design_strength, dim, embed,
                    generic, heisenberg, opnorm, product_states, spin, toeplitz_frame,
                    TOL_IDENTITY)

C = Checker("verify_pathsum")
out = {"enumeration": [], "convergence": []}


def native_heis(F, A, bonds, S):
    n = F[A]
    return (S + 1) ** 2 * sum(np.einsum("ci,ci->c", n[:, i], n[:, j]) for i, j in bonds)


def native_aniso(F, A, bonds, S, J, D, N):
    n = F[A]
    h = J * (S + 1) ** 2 * sum(np.einsum("ci,ci->c", n[:, i], n[:, j]) for i, j in bonds)
    h += D * (S + 1) * (2 * S + 3) / 2 * (n[:, :, 2] ** 2).sum(1) - N * D * (S + 1) / 2
    return h


def expm_herm(H, T):
    w, U = eigh(H)
    return (U * np.exp(-1j * T * w)) @ U.conj().T


# ------------------------------------------------------------------ Part A
print("== Part A: Eq. (6) enumerated history by history vs (V^dag D V)^n ==")
for fname, S, N, n in (("tetrahedral", 0.5, 2, 3), ("two-tetrahedral", 0.5, 2, 2),
                       ("cube", 1, 2, 3), ("icosahedral", 1, 2, 2)):
    F = generic(FRAMES[fname]())
    d, M = dim(S), len(F)
    bonds = [(0, 1)]
    dt = 0.17
    A = configs(M, N)
    h = native_heis(F, A, bonds, S)
    P = product_states(S, F, N)
    nvec = F[A]
    # pairwise data for one step b <- a (Lemma 2 form, no state overlaps used)
    na, nb = nvec[None], nvec[:, None]
    rho = np.prod(np.clip((1 + np.einsum("...i,...i", na, nb)) / 2, 0, None) ** S, axis=-1)
    Ldt = -S * Om(ZHAT, na, nb).sum(-1) - dt * h[None, :]       # L_delta(b, a)
    step = rho * np.exp(1j * Ldt)                                  # [b, a]
    # enumerate all histories (a_1, ..., a_n)
    Cn = len(A)
    hist = np.array(np.meshgrid(*[np.arange(Cn)] * n, indexing="ij")).reshape(n, -1).T
    amp = np.exp(-1j * dt * h[hist[:, -1]])
    for k in range(n - 1):
        amp = amp * step[hist[:, k + 1], hist[:, k]]
    w = (d / M) ** (N * n)
    Psum = w * np.einsum("h,hi,hj->ij", amp, P[hist[:, -1]], P[hist[:, 0]].conj())
    Amat = (d / M) ** N * (P.T * np.exp(-1j * dt * h)) @ P.conj()
    dev = np.abs(Psum - np.linalg.matrix_power(Amat, n)).max()
    out["enumeration"].append({"frame": fname, "M": M, "S": S, "N": N, "n": n,
                               "histories": int(len(hist)), "max_abs_dev": dev})
    C.check(f"A {fname} S={S} N={N} n={n}: {len(hist)} histories", dev, TOL_IDENTITY)

# ------------------------------------------------------------------ Part B
print("== Part B: convergence to exp(-iTH) (exact diagonalization) ==")
ring = lambda N: [(i, (i + 1) % N) for i in range(N)]
chain = lambda N: [(i, i + 1) for i in range(N - 1)]
CASES = [
    # label, S, N, bonds, frame, model
    ("S=1/2 triangle", 0.5, 3, ring(3), "tetrahedral", "heis"),
    ("S=1/2 triangle", 0.5, 3, ring(3), "octahedral", "heis"),
    ("S=1/2 triangle", 0.5, 3, ring(3), "icosahedral", "heis"),
    ("S=1/2 4-ring", 0.5, 4, ring(4), "tetrahedral", "heis"),
    ("S=1/2 4-ring", 0.5, 4, ring(4), "cube", "heis"),
    ("S=1/2 6-ring", 0.5, 6, ring(6), "tetrahedral", "heis"),
    ("S=1 triangle", 1, 3, ring(3), "octahedral", "heis"),
    ("S=1 triangle", 1, 3, ring(3), "cube", "heis"),
    ("S=1 triangle", 1, 3, ring(3), "icosahedral", "heis"),
    ("S=1 4-ring", 1, 4, ring(4), "octahedral", "heis"),
    ("S=3/2 dimer", 1.5, 2, chain(2), "icosahedral", "heis"),
    ("S=3/2 triangle", 1.5, 3, ring(3), "icosahedral", "heis"),
    ("S=1 chain J=1 D=0.5", 1, 3, chain(3), "icosahedral", "aniso"),
    ("S=1 chain J=1 D=0.5", 1, 3, chain(3), "dodecahedral", "aniso"),
]
T = 1.0
NSTEPS = [5, 10, 20, 40, 80, 160, 320, 640]
for label, S, N, bonds, fname, model in CASES:
    F = generic(FRAMES[fname]())
    t = design_strength(F)
    q = 1 if model == "heis" else 2
    assert t >= 2 * S + q, (label, fname)
    d, M = dim(S), len(F)
    A = configs(M, N)
    P = product_states(S, F, N)
    if model == "heis":
        H = heisenberg(S, N, bonds)
        h = native_heis(F, A, bonds, S)
    else:
        Jz = spin(S)[2]
        H = heisenberg(S, N, bonds) + 0.5 * sum(embed(Jz @ Jz, i, N, d) for i in range(N))
        h = native_aniso(F, A, bonds, S, 1.0, 0.5, N)
    dev_TH = np.abs(toeplitz_frame(S, F, N, h, P=P) - H).max()
    C.check(f"T[h]=H  {label} [{fname}]", dev_TH, TOL_IDENTITY)
    U = expm_herm(H, T)
    hinf = np.abs(h).max()
    rows = []
    for n in NSTEPS:
        dt = T / n
        Amat = (d / M) ** N * (P.T * np.exp(-1j * dt * h)) @ P.conj()
        err = opnorm(np.linalg.matrix_power(Amat, n) - U)
        rows.append({"n": n, "dt": dt, "err": err, "bound": T * dt * hinf ** 2})
    errs = np.array([r["err"] for r in rows])
    dts = np.array([r["dt"] for r in rows])
    p_local = np.log(errs[-2] / errs[-1]) / np.log(dts[-2] / dts[-1])
    p_fit = np.polyfit(np.log(dts[-4:]), np.log(errs[-4:]), 1)[0]
    within = bool(np.all(errs <= np.array([r["bound"] for r in rows])))
    rec = {"label": label, "S": S, "N": N, "frame": fname, "M": M, "t": t, "dim": d ** N,
           "histories_per_step": M ** N, "h_inf": hinf, "T[h]=H": dev_TH, "rows": rows,
           "order_local": p_local, "order_fit_last4": p_fit, "err_const": errs[-1] / dts[-1],
           "below_bound": within}
    out["convergence"].append(rec)
    C.check(f"bound holds at all dt  {label} [{fname}]", 0, 0, ok=within)
    C.check(f"order -> 1 (|p-1|<0.05, p={p_local:.4f})  {label} [{fname}]", p_local - 1, 0.05)

C.finish(out)
