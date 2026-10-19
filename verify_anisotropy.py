"""Proposition 1(iv) (second order) and Corollary 1 (renormalized single-ion anisotropy).

1. Symbolic derivation (sympy, exact rational arithmetic) of T[n_z^2] = alpha Jz^2 + beta for
   general S, from the two trace identities used in the analytic proof:
       Tr T[h]      = d <h>                       (resolution of the identity)
       Tr(T[h] B)   = d < h(n) <n|B|n> >          (definition of T), with
       <n|Jz^2|n>   = S(S-1/2) n_z^2 + S/2.
2. Numerical check against exact cubature for S = 1 ... 4 (independent of step 1).
3. The rhombic term: T[n_x^2 - n_y^2] = alpha (Jx^2 - Jy^2).
4. The many-body statement T[h] = H for Eq. (9) is checked in verify_pathsum.py (Part B).
"""
import numpy as np
import sympy as sp

from lmcore import Checker, coh, dim, gauss_sphere, spin, TOL_IDENTITY

C = Checker("verify_anisotropy")
out = {}

print("== symbolic derivation ==")
S = sp.symbols("S", positive=True)
a, b = sp.symbols("alpha beta")
d = 2 * S + 1
trJz2 = S * (S + 1) * (2 * S + 1) / 3
trJz4 = S * (S + 1) * (2 * S + 1) * (3 * S ** 2 + 3 * S - 1) / 15
avg_nz2, avg_nz4 = sp.Rational(1, 3), sp.Rational(1, 5)
eqs = [sp.Eq(a * trJz2 + b * d, d * avg_nz2),
       sp.Eq(a * trJz4 + b * trJz2, d * (S * (S - sp.Rational(1, 2)) * avg_nz4 + S / 2 * avg_nz2))]
sol = sp.solve(eqs, [a, b], dict=True)[0]
alpha, beta = sp.factor(sol[a]), sp.factor(sol[b])
print("alpha =", alpha, "  beta =", beta)
ok_a = sp.simplify(alpha - 2 / ((S + 1) * (2 * S + 3))) == 0
ok_b = sp.simplify(beta - 1 / (2 * S + 3)) == 0
C.check("symbolic alpha = 2/((S+1)(2S+3))", 0, 0, ok=ok_a)
C.check("symbolic beta = 1/(2S+3)", 0, 0, ok=ok_b)
ratio = sp.simplify(((S + 1) * (2 * S + 3) / 2) / (S + 1) ** 2)
C.check("symbolic D/J factor = (2S+3)/(2(S+1))",
        0, 0, ok=sp.simplify(ratio - (2 * S + 3) / (2 * (S + 1))) == 0)
out["symbolic"] = {"alpha": str(alpha), "beta": str(beta), "DJ_factor": str(ratio)}
out["factor_table"] = {str(s): str(sp.nsimplify(ratio.subs(S, sp.Rational(s)))) for s in
                       ("1", "3/2", "2", "5/2", "3")}
print("D/J factors:", out["factor_table"])

print("== numerical check against exact cubature ==")
pts, w = gauss_sphere(16)
devs = {}
for s in (1, 1.5, 2, 2.5, 3, 3.5, 4):
    dd = dim(s)
    Jx, Jy, Jz = spin(s)
    Tz = np.zeros((dd, dd), complex)
    Tr = np.zeros((dd, dd), complex)
    for n, wi in zip(pts, w):
        v = coh(s, n)
        Pn = np.outer(v, v.conj())
        Tz += wi * n[2] ** 2 * Pn
        Tr += wi * (n[0] ** 2 - n[1] ** 2) * Pn
    Tz, Tr = dd * Tz, dd * Tr
    al, be = 2 / ((s + 1) * (2 * s + 3)), 1 / (2 * s + 3)
    dz = np.abs(Tz - (al * Jz @ Jz + be * np.eye(dd))).max()
    dr = np.abs(Tr - al * (Jx @ Jx - Jy @ Jy)).max()
    # Corollary 1 single-site form: Jz^2 = T[ (S+1)(2S+3)/2 n_z^2 - (S+1)/2 ]
    inv = (s + 1) * (2 * s + 3) / 2 * Tz - (s + 1) / 2 * np.eye(dd)
    dinv = np.abs(inv - Jz @ Jz).max()
    devs[str(s)] = {"nz2": dz, "rhombic": dr, "Jz2_inverse": dinv}
    C.check(f"T[n_z^2] S={s}", dz, TOL_IDENTITY)
    C.check(f"T[n_x^2-n_y^2] S={s}", dr, TOL_IDENTITY)
    C.check(f"Jz^2 = T[native symbol] S={s}", dinv, TOL_IDENTITY)
out["numerical"] = devs
out["max_deviation"] = max(max(v.values()) for v in devs.values())
C.finish(out)
