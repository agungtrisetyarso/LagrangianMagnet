"""Theorem 2(b) (Q is an integer for every closed history) and Theorem 3 (exact lattice
Haldane identity, and decay of the aliasing remainder eps).

Data for Fig. 1(b): trig_history seeds 0, 1, 2 with n in {16, 64, 256} time slices and
L in {16, 24, 32, 48, 64, 96, 128, 192, 256}.
"""
import numpy as np

from lmcore import Checker, unit, TOL_INTEGER
from theta_check import analyse, chern_history, finite_smooth_history, trig_history

C = Checker("verify_theta")
rng = np.random.default_rng(7)
LS = [16, 24, 32, 48, 64, 96, 128, 192, 256]
NS = [16, 64, 256]
out = {"fig1b": {}, "random": [], "chern": [], "finite_smooth": []}

print("== Theorem 2(b): Q integer for arbitrary (non-smooth, random) closed histories ==")
worst = 0.0
for _ in range(200):
    L = 2 * rng.integers(2, 9)
    n = rng.integers(3, 12)
    m = unit(rng.normal(size=(L, n, 3)))
    for S in (0.5, 1, 1.5):
        r = analyse(m, S)
        worst = max(worst, r["Q_int_dev"], r["id1"], r["id2"], r["id3"], r["berry"])
    out["random"].append({"L": int(L), "n": int(n), "Q": round(r["Q"])})
C.check("T2(b)+T3(i) random histories: max residue (Q, id1-3, Eq.12)", worst, TOL_INTEGER)

print("== Theorem 3: analytic histories of Fig. 1(b) ==")
for seed in (0, 1, 2):
    for n in NS:
        rows = []
        for L in LS:
            m, vmin = trig_history(L, n, seed)
            r = analyse(m)
            rows.append({"L": L, **r, "min_abs_v": vmin})
        key = f"seed{seed}_n{n}"
        out["fig1b"][key] = rows
        res = max(max(r["Q_int_dev"], r["id1"], r["id2"], r["id3"], r["berry"]) for r in rows)
        Qs = {round(r["Q"]) for r in rows}
        C.check(f"{key}: identities and integrality (Q={sorted(Qs)})", res, TOL_INTEGER)
        C.check(f"{key}: same Q at every L", 0, 0, ok=len(Qs) == 1)
        adm = [r["L"] for r in rows if r["admissible"]]
        eps_last = abs(rows[-1]["eps"])
        print(f"   admissible for L >= {min(adm) if adm else None}; |eps(L=256)| = {eps_last:.2e}")
    # uniformity in n: eps at L=256 for the three n
    e = [abs(out["fig1b"][f"seed{seed}_n{n}"][-1]["eps"]) for n in NS]
    out["fig1b"][f"seed{seed}_eps256_vs_n"] = e

print("== designed winding numbers (two-band maps), L up to 256 ==")
for kx, m0, Qexp in ((1, 1.0, 1), (2, 1.0, 2), (1, 3.0, 0)):
    rows = []
    for L in LS:
        m, vmin = chern_history(L, 64, kx, m0)
        r = analyse(m)
        rows.append({"L": L, **r})
    Qs = {round(r["Q"]) for r in rows}
    out["chern"].append({"kx": kx, "m0": m0, "rows": rows})
    C.check(f"two-band map kx={kx} m0={m0}: |Q| = {Qexp}", 0, 0,
            ok=Qs == {Qexp} or Qs == {-Qexp})
    res = max(max(r["Q_int_dev"], r["id1"], r["id2"], r["id3"], r["berry"]) for r in rows)
    C.check(f"two-band map kx={kx} m0={m0}: identities", res, TOL_INTEGER)

print("== finitely smooth history: algebraic (not exponential) decay ==")
LS2 = [32, 64, 128, 256, 512, 1024]
rows = []
for L in LS2:
    m, vmin = finite_smooth_history(L, 64, p=3)
    r = analyse(m)
    rows.append({"L": L, **r})
eps = np.array([abs(r["eps"]) for r in rows])
slope = np.polyfit(np.log(LS2[2:]), np.log(eps[2:]), 1)[0]
out["finite_smooth"] = {"rows": rows, "fitted_exponent": slope,
                        "note": "history is C^2 but not C^3 in x; Theorem 3(ii) guarantees p>=2"}
print(f"   |eps| = {eps}, fitted exponent {slope:.2f}")
C.check(f"C^2 history: fitted decay exponent {slope:.2f} <= -2 (Theorem 3(ii), p=2)",
        0, 0, ok=slope <= -2 + 0.05)

C.finish(out)
