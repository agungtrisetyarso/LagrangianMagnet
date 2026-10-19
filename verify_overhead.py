"""Appendix C: block-encoding normalisation Lambda / ||H|| for frame-based (Lagrangian-native)
simulation of S=1/2 and S=1 Heisenberg bonds and S=1/2 J1-J2 rings (sparse exact
diagonalization; ||H|| = max(|E_min|, |E_max|), Lanczos tolerance 1e-12)."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from lmcore import Checker, FRAMES, dim, heisenberg, spin, TOL_IDENTITY

C = Checker("verify_overhead")
out = {"bond": [], "rings": []}

print("== single bond ==")
for S, fname in ((0.5, "tetrahedral"), (1, "octahedral")):
    F = FRAMES[fname]()
    M, d = len(F), dim(S)
    hv = (S + 1) ** 2 * (F @ F.T)            # native symbol of J1.J2 on F x F
    b = np.abs(hv).max()
    Lam = d ** 2 * b
    Lam1 = (d / M) ** 2 * np.abs(hv).sum()
    nH = np.abs(np.linalg.eigvalsh(heisenberg(S, 2, [(0, 1)]))).max()
    out["bond"].append({"S": S, "frame": fname, "Lambda": Lam, "Lambda_1": Lam1,
                        "||H||": nH, "Lambda/||H||": Lam / nH, "Lambda_1/||H||": Lam1 / nH})
    print(f"  S={S} [{fname}]: Lambda={Lam:.4f}  Lambda_1={Lam1:.4f}  ||H||={nH:.4f}"
          f"  ratio={Lam / nH:.4f}  ratio_1={Lam1 / nH:.4f}")
C.check("S=1/2 bond Lambda/||H|| = 12", out["bond"][0]["Lambda/||H||"] - 12, TOL_IDENTITY)
C.check("S=1 bond Lambda/||H|| = 18", out["bond"][1]["Lambda/||H||"] - 18, TOL_IDENTITY)
C.check("S=1/2 bond Lambda_1 = 4.5", out["bond"][0]["Lambda_1"] - 4.5, TOL_IDENTITY)

print("== S=1/2 J1-J2 rings ==")
sx = sp.csr_matrix([[0, .5], [.5, 0]])
sy = sp.csr_matrix([[0, -.5j], [.5j, 0]])
sz = sp.csr_matrix([[.5, 0], [0, -.5]])
op = lambda o, i, N: sp.kron(sp.kron(sp.identity(2 ** i), o), sp.identity(2 ** (N - i - 1)),
                             format="csr")
F = FRAMES["tetrahedral"]()
hv = 2.25 * (F @ F.T)
b, l1 = np.abs(hv).max(), (2 / 4) ** 2 * np.abs(hv).sum()
for J2 in (0.0, 0.2, 0.5):
    for N in (8, 10, 12, 14, 16):
        H = 0
        for i in range(N):
            for r, J in ((1, 1.0), (2, J2)):
                if J:
                    H = H + J * sum(op(o, i, N) @ op(o, (i + r) % N, N) for o in (sx, sy, sz))
        emin = sla.eigsh(H, k=1, which="SA", tol=1e-12)[0][0]
        emax = sla.eigsh(H, k=1, which="LA", tol=1e-12)[0][0]
        nH = max(abs(emin), abs(emax))
        wsum = N * (1 + J2)
        rec = {"J2": J2, "N": N, "||H||": nH, "Lambda/||H||": 4 * b * wsum / nH,
               "Lambda_1/||H||": l1 * wsum / nH}
        out["rings"].append(rec)
        print(f"  J2={J2} N={N}: ||H||={nH:.6f}  Lambda/||H||={rec['Lambda/||H||']:.3f}"
              f"  Lambda_1/||H||={rec['Lambda_1/||H||']:.3f}")
C.finish(out)
