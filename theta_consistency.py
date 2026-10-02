"""Finite-frame consistency of a per-step theta deformation (reference-free plaquette phase)."""
import numpy as np, itertools, sys, io
from theta_check import Om
_o=sys.stdout; sys.stdout=io.StringIO(); from verify_all import coh, tetra; sys.stdout=_o
def A0(S,F,L,dth):
    M=len(F); d=int(2*S+1); confs=np.array(list(itertools.product(range(M),repeat=L))); m=F[confs]; C=len(m)
    P=[]
    for c in confs:
        v=np.array([1.0+0j])
        for j,a in enumerate(c): v=np.kron(v,coh(S,F[a]*(-1)**j))
        P.append(v)
    P=np.array(P); G=P.conj()@P.T          # G[b,a]=<b|a>
    a=np.broadcast_to(m[None],(C,C,L,3)); b=np.broadcast_to(m[:,None],(C,C,L,3))
    ap=np.roll(a,-1,axis=2); bp=np.roll(b,-1,axis=2)
    q=(Om(a,b,bp)+Om(a,bp,ap)).sum(-1)/(4*np.pi)    # reference-free per-step charge q(b,a)
    w=(d/M)**L
    out={}
    for t in dth:
        E=np.exp(-1j*t*q); A=w*w*np.einsum('bi,ba,aj->ij',P,G*E,P.conj())
        ev=np.linalg.eigvalsh((A+A.conj().T)/2)
        out[t]=(ev, np.abs(A-A.conj().T).max(), np.abs(q+q.T).max())
    return out
if __name__=="__main__":
    for S in (0.5,1.0):
        r=A0(S,tetra(),4,[0,np.pi/2,np.pi,2*np.pi])
        for t,(ev,herm,anti) in r.items():
            print(f"S={S} dtheta={t:.3f}: eig(A0) in [{ev.min():.4f},{ev.max():.4f}], #eig>0.999={np.sum(ev>0.999)}/{len(ev)}, herm-dev={herm:.1e}, q-antisym={anti:.1e}")

def scan(S,F,L=4,K=64):
    ts=np.linspace(4*np.pi/K,4*np.pi,K)
    r=A0(S,F,L,ts)
    dev=[]; nrm=[]
    # recompute norms of A0-1 and A0 (not only hermitian part)
    return ts,r

if __name__=="__main__":
    # scan used for Fig. 1(c) and Theorem 4(ii)
    for S in (0.5,1.0):
        ts=np.linspace(0,4*np.pi,161); r=A0(S,tetra(),4,ts)
        print(f"S={S}: min eig spread over (0,4pi]:", min(max(abs(v[0]-1),abs(v[-1]-1)) for t,(v,_,_) in r.items() if t>0))
