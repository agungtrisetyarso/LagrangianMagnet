"""Shared numerical core for the LagrangianMagnet verification scripts.

Conventions (identical to the manuscript):
  * spin-S basis |S,m>, ordered m = S, S-1, ..., -S;
  * north-pole-gauge coherent state |n> = e^{-i phi Jz} e^{-i theta Jy} e^{+i phi Jz} |S,S>;
  * signed geodesic solid angle Omega(a,b,c) = 2 atan2(a.(b x c), 1+a.b+b.c+c.a) in (-2pi, 2pi];
  * a coherent frame is a spherical t-design F = {n_1..n_M};
  * on a staggered chain the frame is placed on the staggered field m_j, and the physical
    spin direction is n_j = (-1)^j m_j (for antipodally symmetric frames this is the same as
    placing the frame on n_j).

All arithmetic is IEEE double precision (numpy float64 / complex128).
"""
import itertools
import json
import os
import platform
import sys

import numpy as np
from math import comb

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ZHAT = np.array([0.0, 0.0, 1.0])

# Tolerances used by every script (see README, "Tolerances").
TOL_IDENTITY = 1e-10   # exact algebraic identities evaluated in float64 (dims <= 4096)
TOL_INTEGER = 1e-9     # integrality of Q, q-sums and 4pi-Z residues
TOL_DESIGN = 1e-12     # design-strength test sum_ab P_l(n_a.n_b) / M^2


# --------------------------------------------------------------------- spins
def spin(S):
    """Return (Jx, Jy, Jz) for spin S in the |S,m> basis, m = S..-S."""
    d = int(round(2 * S + 1))
    m = S - np.arange(d)
    Jz = np.diag(m).astype(complex)
    Jp = np.zeros((d, d), complex)
    for k in range(1, d):
        Jp[k - 1, k] = np.sqrt(S * (S + 1) - m[k] * (m[k] + 1))
    return (Jp + Jp.conj().T) / 2, (Jp - Jp.conj().T) / 2j, Jz


def dim(S):
    return int(round(2 * S + 1))


def coh(S, n):
    """North-pole-gauge spin coherent state (closed form).

    <S,m|n> = sqrt(C(2S,S-m)) cos(th/2)^(S+m) sin(th/2)^(S-m) e^{-i(m-S)phi}.
    Ill-defined (gauge singular) only at n = -z.
    """
    n = np.asarray(n, float)
    th = np.arccos(np.clip(n[2], -1.0, 1.0))
    ph = np.arctan2(n[1], n[0])
    d = dim(S)
    twoS = d - 1
    out = np.empty(d, complex)
    c, s = np.cos(th / 2), np.sin(th / 2)
    for k in range(d):
        m = S - k
        out[k] = (np.sqrt(comb(twoS, k)) * c ** (twoS - k) * s ** k
                  * np.exp(-1j * (m - S) * ph))
    return out


def coh_expm(S, n):
    """Reference construction by matrix exponentials (used only to test coh)."""
    from scipy.linalg import expm
    Jx, Jy, Jz = spin(S)
    th = np.arccos(np.clip(n[2], -1, 1))
    ph = np.arctan2(n[1], n[0])
    top = np.zeros(dim(S), complex)
    top[0] = 1
    return expm(-1j * ph * Jz) @ expm(-1j * th * Jy) @ expm(1j * ph * Jz) @ top


# ------------------------------------------------------------------ geometry
def Om(a, b, c):
    """Signed solid angle of the geodesic triangle (a,b,c); broadcasts over leading axes."""
    num = np.einsum("...i,...i", a, np.cross(b, c))
    den = (1 + np.einsum("...i,...i", a, b) + np.einsum("...i,...i", b, c)
           + np.einsum("...i,...i", c, a))
    return 2 * np.arctan2(num, den)


def wrap(x, p=4 * np.pi):
    """Representative of x modulo p in [-p/2, p/2)."""
    return (np.asarray(x) + p / 2) % p - p / 2


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def rot(axis, angle):
    axis = unit(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


# A fixed, generic rotation. Used (i) to build the two-tetrahedron frame and (ii) to place
# frames in "generic orientation", which guarantees -z is not a frame point (Lemma 2 needs
# n != -z, because the north-pole gauge is singular there).
R_GENERIC = rot([0.3, -0.7, 0.55], 0.9137)


# -------------------------------------------------------------------- frames
def tetra():
    return unit(np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float))


def octa():
    return np.vstack([np.eye(3), -np.eye(3)])


def cube():
    return unit(np.array(list(itertools.product([1, -1], repeat=3)), float))


def icosa():
    p = (1 + 5 ** 0.5) / 2
    v = []
    for a in (1, -1):
        for b in (p, -p):
            v += [[0, a, b], [a, b, 0], [b, 0, a]]
    return unit(np.array(v, float))


def dodeca():
    p = (1 + 5 ** 0.5) / 2
    v = [list(x) for x in itertools.product([1, -1], repeat=3)]
    for a in (1, -1):
        for b in (1, -1):
            v += [[0, a / p, b * p], [a / p, b * p, 0], [b * p, 0, a / p]]
    return unit(np.array(v, float))


def two_tetra():
    """Union of the tetrahedron and a generically rotated copy: a 2-design with M=8 and
    no antipodal pairs."""
    T = tetra()
    return np.vstack([T, T @ R_GENERIC.T])


FRAMES = {
    "tetrahedral": tetra,
    "two-tetrahedral": two_tetra,
    "octahedral": octa,
    "cube": cube,
    "icosahedral": icosa,
    "dodecahedral": dodeca,
}


def generic(F):
    """Rotate a frame into generic orientation (still a design of the same strength)."""
    return F @ R_GENERIC.T


def legendre(l, x):
    from numpy.polynomial import legendre as L
    c = np.zeros(l + 1)
    c[l] = 1
    return L.legval(x, c)


def design_strength(F, lmax=12):
    """Largest t such that sum_{a,b} P_l(n_a.n_b) = 0 for l = 1..t (addition theorem)."""
    G = np.clip(F @ F.T, -1, 1)
    t = 0
    for l in range(1, lmax + 1):
        if abs(legendre(l, G).sum()) / len(F) ** 2 > TOL_DESIGN:
            break
        t = l
    return t


def gauss_sphere(deg):
    """Product cubature (Gauss-Legendre in cos(theta) x uniform phi), exact for all
    polynomials of total degree <= deg on S^2. Returns (points, weights) with sum w = 1."""
    K = deg // 2 + 1
    P = deg + 1
    x, wx = np.polynomial.legendre.leggauss(K)
    ph = 2 * np.pi * np.arange(P) / P
    pts, w = [], []
    for xi, wi in zip(x, wx):
        s = np.sqrt(1 - xi ** 2)
        for p in ph:
            pts.append([s * np.cos(p), s * np.sin(p), xi])
            w.append(wi / 2 / P)
    return np.array(pts), np.array(w)


# ----------------------------------------------------------- many-body tools
def embed(op, i, N, d):
    return np.kron(np.kron(np.eye(d ** i), op), np.eye(d ** (N - i - 1)))


def heisenberg(S, N, bonds, J=1.0):
    d = dim(S)
    Js = spin(S)
    return J * sum(embed(Js[a], i, N, d) @ embed(Js[a], j, N, d)
                   for (i, j) in bonds for a in range(3))


def product_states(S, F, N, signs=None):
    """Rows are |n_{a_1}> x ... x |n_{a_N}> for all a in F^N (row index = mixed radix).
    signs[j] = +-1 puts the frame on the staggered field (n_j = signs[j] m_j)."""
    M = len(F)
    if signs is None:
        signs = np.ones(N)
    cs = [np.array([coh(S, signs[j] * F[a]) for a in range(M)]) for j in range(N)]
    P = cs[0]
    for j in range(1, N):
        P = np.einsum("ai,bj->abij", P, cs[j]).reshape(P.shape[0] * M, -1)
    return P


def configs(M, N):
    return np.array(list(itertools.product(range(M), repeat=N)))


def toeplitz_frame(S, F, N, hvals, signs=None, P=None):
    """Frame quantization (d/M)^N sum_a h(a) |a><a| for a symbol tabulated on F^N."""
    if P is None:
        P = product_states(S, F, N, signs)
    w = (dim(S) / len(F)) ** N
    return w * (P.T * hvals) @ P.conj()


def opnorm(A):
    return np.linalg.norm(A, 2)


# ------------------------------------------------------------------- output
def environment():
    import scipy
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(),
            "dtype": "float64/complex128"}


def save(name, payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = dict(payload)
    payload["environment"] = environment()

    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, complex):
            return [o.real, o.imag]
        raise TypeError(type(o))

    with open(os.path.join(RESULTS_DIR, name + ".json"), "w") as f:
        json.dump(payload, f, indent=1, default=conv)


class Checker:
    """Collects pass/fail checks; exits non-zero at the end if any check failed."""

    def __init__(self, script):
        self.script = script
        self.rows = []

    def check(self, label, value, tol, ok=None):
        passed = (abs(value) <= tol) if ok is None else bool(ok)
        self.rows.append({"check": label, "value": float(value), "tol": tol, "pass": passed})
        flag = "PASS" if passed else "FAIL"
        print(f"  [{flag}] {label}: {value:.3e} (tol {tol:.0e})")
        return passed

    def finish(self, payload):
        payload = dict(payload)
        payload["checks"] = self.rows
        save(self.script, payload)
        nfail = sum(not r["pass"] for r in self.rows)
        print(f"== {self.script}: {len(self.rows) - nfail}/{len(self.rows)} checks passed ==")
        if nfail:
            sys.exit(1)
