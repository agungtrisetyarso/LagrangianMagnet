"""Run every verification script, regenerate Fig. 1 and write results/summary.md.

Usage:  python run_all.py
Exit status is non-zero if any check in any script fails.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["verify_all.py", "verify_pathsum.py", "verify_anisotropy.py", "verify_theta.py",
           "theta_pairs.py", "theta_consistency.py", "verify_overhead.py"]

status = {}
for s in SCRIPTS:
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, s)], cwd=HERE,
                       capture_output=True, text=True)
    status[s] = (r.returncode, time.time() - t0)
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "", f"[{status[s][1]:.1f}s]")
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
subprocess.run([sys.executable, os.path.join(HERE, "make_fig1.py")], cwd=HERE, check=True)

R = lambda n: json.load(open(os.path.join(HERE, "results", n + ".json")))
va, ps, an, th, tp, tc, ov = (R(n[:-3]) for n in SCRIPTS)


def case(frame, S, L):
    return next(c for c in tc["cases"] if c["frame"] == frame and c["S"] == S and c["L"] == L)


lines = ["# Summary of reproduced numbers", "",
         f"Environment: {va['environment']}", "",
         "| Manuscript item | Reproduced value | Script |", "|---|---|---|"]
add = lambda a, b, c: lines.append(f"| {a} | {b} | `{c}` |")
add("Lemma 2 phase formula, max deviation (S<=3, 12000 pairs)",
    f"{max(v for k, v in va['lemma2'].items() if k != 'at_minus_z'):.1e}", "verify_all.py")
add("Lemma 2 kernel G: idempotency, max deviation",
    f"{max(p['idempotent'] for p in va['projector']):.1e}", "verify_all.py")
add("Eq. (6) enumerated vs (V^dag D V)^n, max deviation",
    f"{max(e['max_abs_dev'] for e in ps['enumeration']):.1e}", "verify_pathsum.py")
add("Theorem 1: observed order at smallest dt (14 cases)",
    f"{min(c['order_local'] for c in ps['convergence']):.3f} - "
    f"{max(c['order_local'] for c in ps['convergence']):.3f}", "verify_pathsum.py")
add("Theorem 1: error below bound in all cases", str(all(c["below_bound"] for c in ps["convergence"])),
    "verify_pathsum.py")
add("Eq. (9) / Prop. 1(iv), max deviation (S=1..4)", f"{an['max_deviation']:.1e}",
    "verify_anisotropy.py")
fb = th["fig1b"]
add("Theorem 3: max residue of Q-integrality and identities (Fig. 1b)",
    f"{max(max(r['Q_int_dev'], r['id1'], r['id2'], r['id3'], r['berry']) for k, v in fb.items() if k.endswith(('n16', 'n64', 'n256')) for r in v):.1e}",
    "verify_theta.py")
for s in (0, 1, 2):
    add(f"Theorem 3: history {s + 1}, Q and |eps| at L=16 -> 256 (n=256)",
        f"Q={round(fb[f'seed{s}_n256'][0]['Q'])}, {abs(fb[f'seed{s}_n256'][0]['eps']):.1e} -> "
        f"{abs(fb[f'seed{s}_n256'][-1]['eps']):.1e}", "verify_theta.py")
add("Theorem 3(ii): C^2 history, fitted decay exponent of |eps|",
    f"{th['finite_smooth']['fitted_exponent']:.2f}", "verify_theta.py")
for fr, S, L in (("tetrahedral", 0.5, 4), ("tetrahedral", 1, 4), ("tetrahedral", 0.5, 6),
                 ("tetrahedral", 0.5, 8), ("two-tetrahedral", 0.5, 4)):
    c = case(fr, S, L)
    add(f"Theorem 4(iii): {fr}, S={S}, L={L}: max norm / max spectral radius",
        f"{c['norm_peak'][1]:.4f} at {c['norm_peak'][0]:.3f}pi / "
        f"{c['radius_peak'][1]:.4f} at {c['radius_peak'][0]:.3f}pi", "theta_consistency.py")
add("Lemma 3: q on tetra / octa / cube / icosa",
    ", ".join(f"{p['q']:+.4f}" for p in tp["pairs"] if p["S"] == 1 and p["frame"] in
              ("tetrahedral", "octahedral", "cube", "icosahedral")), "theta_pairs.py")
add("App. C: Lambda/||H|| bond S=1/2, S=1",
    f"{ov['bond'][0]['Lambda/||H||']:.2f}, {ov['bond'][1]['Lambda/||H||']:.2f}", "verify_overhead.py")
add("App. C: Lambda/||H|| J1-J2 rings N=16, J2=0,0.2,0.5",
    ", ".join(f"{r['Lambda/||H||']:.2f}" for r in ov["rings"] if r["N"] == 16), "verify_overhead.py")
lines += ["", "| Script | exit code | runtime (s) |", "|---|---|---|"]
lines += [f"| {s} | {c} | {t:.1f} |" for s, (c, t) in status.items()]
open(os.path.join(HERE, "results", "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(int(any(c for c, _ in status.values())))
