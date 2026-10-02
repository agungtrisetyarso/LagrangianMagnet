"""Explicit overlapping pairs with non-integer single-step charge q (Theorem 4), and spectral radii."""
import numpy as np, itertools, sys, io
from theta_check import Om
_o=sys.stdout; sys.stdout=io.StringIO(); from verify_all import coh, tetra, octa; sys.stdout=_o
def q_step(ma,mb):
    L=len(ma); s=0.0
    for j in range(L):
        k=(j+1)%L; s+=Om(ma[j],mb[j],mb[k])+Om(ma[j],mb[k],ma[k])
    return s/(4*np.pi)
def overlap(S,ma,mb):
    v=1.0+0j
    for j,(x,y) in enumerate(zip(ma,mb)):
        sg=(-1)**j; v*=coh(S,sg*y).conj()@coh(S,sg*x)
    return v
for name,F,S in (("tetrahedral",tetra(),0.5),("tetrahedral",tetra(),1.0),("octahedral",octa(),1.0),("octahedral",octa(),1.5)):
    p,qv,r=F[0],F[1],F[2]                      # three pairwise non-antipodal points off a great circle
    ma=np.array([r,p,p,p]); mb=np.array([r,qv,p,p])  # staggered field, L=4: only site 1 changes
    q=q_step(ma,mb); ov=overlap(S,ma,mb)
    print(f"{name:11s} S={S}: points {np.round(p,3)},{np.round(qv,3)},{np.round(r,3)}  q={q:+.6f}  Omega(r,q,p)/4pi={Om(r,qv,p)/(4*np.pi):+.6f}  |<b|a>|={abs(ov):.4f}")
