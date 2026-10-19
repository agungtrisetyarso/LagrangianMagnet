# LagrangianMagnet — verification code

Numerical checks and figure for

> A. Trisetyarso, *Haldane's weight and renormalized single-ion anisotropy from exact
> coherent-state path sums on spherical designs* (submitted to Int. J. Mod. Phys. B).

The numerical checks support, but are not part of, the proofs: every theorem in the
paper is proved analytically (proofs in Appendix B). The scripts check the proved
statements independently and produce Fig. 1 and every number quoted in the paper.

## Reproduce everything (about 7 minutes on one CPU core)

```bash
python -m pip install -r requirements.txt     # numpy, scipy, sympy, matplotlib
python run_all.py
```

`run_all.py` runs the six verification scripts below and `independent_check.py`, regenerates `fig1.pdf` / `fig1.png`, and writes

* `results/<script>.json` — all raw numbers, every check with its value and tolerance, and the
  software environment;
* `results/summary.md` — a table of every number quoted in the manuscript, with its script.

The exit status is non-zero if any check fails. Each script can also be run on its own.

| Script | Manuscript item | What it compares |
|---|---|---|
| `verify_all.py` | Lemma 1, Lemma 2, Prop. 1(iv) (first order), Thm 2(a) | frame sums vs exact cubature; phase formula vs state overlaps; kernel `G` built from Ω vs `VV†`, `G²=G`, rank |
| `verify_pathsum.py` | Theorem 1, Corollary 1 in the dynamics, Fig. 1(a) | Eq. (6) enumerated history by history vs `(V†DV)^n`; `(V†DV)^n` vs `exp(-iTH)` from exact diagonalization |
| `verify_anisotropy.py` | Prop. 1(iv) (second order), Corollary 1, Table 1 | symbolic derivation (sympy) and exact cubature for S = 1…4 |
| `verify_theta.py` (uses `theta_check.py`) | Theorem 2(b), Theorem 3, Fig. 1(b) | integrality of Q and the exact identities for random, analytic, designed-winding and finitely smooth histories; decay of ε |
| `theta_transfer.py` | Lemma 3, Theorem 4, Fig. 1(c) | gauge decomposition q = ΔΦ/4π + k (k ∈ Z) on all pairs; the charge-one two-step history; K_E = G∘E idempotent iff Δθ ∈ 2πZ; spectral radius of K_E and growth of the free path sum |
| `verify_overhead.py` | Appendix C | block-encoding normalisation vs sparse exact diagonalization |
| `make_fig1.py` | Fig. 1 | plots only; reads `results/*.json` |
| `independent_check.py` | headline numbers | re-implementation sharing no code with `lmcore.py` (Wigner-matrix coherent states, L'Huilier solid angles); fails if it disagrees with `results/*.json` |

## Coverage

* **Frames:** tetrahedral (M=4, t=2), two-tetrahedral (M=8, t=2), octahedral (M=6, t=3),
  cube (M=8, t=3), icosahedral (M=12, t=5), dodecahedral (M=20, t=5). The design strength t
  is computed, not assumed (`lmcore.design_strength`).
* **Spins:** S = 1/2 … 3 (Lemmas 1–2), S = 1 … 4 (anisotropy), S = 1/2, 1, 3/2 (path sums).
* **Clusters for Theorem 1:** S=1/2 triangle, 4-ring and 6-ring; S=1 triangle and 4-ring;
  S=3/2 dimer and triangle; S=1 chain with single-ion anisotropy D=0.5. 14 (cluster, frame)
  cases in all.
* **Time steps:** T = 1 with n = 5, 10, …, 640 (δt = 0.2 … 1.6×10⁻³).
* **Theorem 4:** regular frames only (no antipodal pair, no three points on a great circle):
  tetrahedral frame with S=1/2 (L = 4, 6) and S=1 (L = 4); two-tetrahedral frame with S=1/2 (L = 4).

## Precision, tolerances and convergence criteria

All arithmetic is IEEE-754 double precision (`float64` / `complex128`). Matrix exponentials
come from Hermitian eigendecomposition (`scipy.linalg.eigh`). Operator norms are spectral norms
(largest singular value). Extremal eigenvalues in `verify_overhead.py` use ARPACK (`eigsh`) with
`tol=1e-12`.

| Quantity | Criterion |
|---|---|
| exact algebraic identities (Lemmas 1–2, Prop. 1, Eq. (6), Eq. (9), T[h]=H, A0(0)=1) | max entrywise deviation ≤ 1e-10 |
| integrality of Q, of Δ/4π (Thm 2a) and of 4q; mod-4π identities of Theorem 3 | deviation from nearest integer / residue ≤ 1e-9 |
| design strength | (1/M²) Σ_ab P_l(n_a·n_b) ≤ 1e-12 for l = 1…t |
| Theorem 1 convergence | error ≤ T δt ‖h‖∞² at every δt, and local order p = log(e₇/e₈)/log(δt₇/δt₈) with \|p−1\| < 0.05 |
| Theorem 3(ii) algebraic regime | fitted exponent of \|ε\| vs L on L = 128…1024 is ≤ −2 (proof gives p = 2 for a C² history) |
| Theorem 4 | k = q − ΔΦ/4π integer to 1e-9 on every pair; K_E idempotent to 1e-10 at Δθ ∈ {0, 2π, 4π} and not idempotent (> 1e-8) elsewhere; spectral radius on 81 points of [0, 2π] (dense for dim ≤ 512, Arnoldi with tol 1e-10 otherwise) |
| negative controls | must *fail* by more than 1e-3: Lemma 1 with t < 2S; Lemma 2 at n = −z |

The observed deviations are 10⁻¹⁷–10⁻¹² (see `results/summary.md`), at least two orders of
magnitude inside every tolerance.

## Conventions that matter numerically

* **Gauge.** Coherent states are in the north-pole gauge, which is singular at n = −z.
  Lemma 2's phase formula therefore needs n_a, n_b ≠ −z, and the scripts place every frame in
  a fixed generic orientation (`lmcore.generic`) whenever the Ω form of the kernel is used.
  `verify_all.py` includes the n = −z case as a negative control.
* **Staggered frames.** On chains the frame is placed on the staggered field m_j, and
  n_j = (−1)^j m_j. For antipodally symmetric frames this is the same as placing it on n_j.
  For the tetrahedral frame it is the convention that makes Lemma 3's pairs frame points.
* **Branch conventions.** Ω(a,b,c) = 2 atan2(a·(b×c), 1+a·b+b·c+c·a) is undefined exactly when two
  vertices are antipodal (num² + den² = 2(1+a·b)(1+b·c)(1+c·a)). For three vertices on a great circle
  enclosing the origin it equals ±2π, which is irrelevant modulo 4π. `verify_all.py` checks the
  cocycle on generic and on great-circle quadruples.
* **Theorem 4 is about regular frames.** Centrally symmetric designs (octahedral, cube, icosahedral,
  dodecahedral) contain antipodal pairs, so some plaquette triangles between overlapping
  configurations are undefined; Theorem 4 and `theta_transfer.py` use only the tetrahedral and
  two-tetrahedral frames.

## Changes from the first public version

* `verify_anisotropy.py` (cited in the manuscript but missing) and a figure script were added.
* `theta_consistency.py` and `theta_pairs.py` were removed. They analysed A = V†(G∘E)V, whose powers
  are not the θ-deformed path sum; the correct one-step object is the transfer operator
  K_E = G∘E on ℓ²(F^N) (`theta_transfer.py`). The earlier statement that even Δθ = 2π fails was an
  artefact of that choice: for Δθ ∈ 2πZ the deformation is a pure gauge.
