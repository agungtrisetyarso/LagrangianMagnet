"""Exact lattice decomposition of staggered Berry phases (Theorem: Haldane term on the lattice)."""
import numpy as np
def Om(a,b,c):
    num=np.einsum('...i,...i',a,np.cross(b,c)); den=1+np.einsum('...i,...i',a,b)+np.einsum('...i,...i',b,c)+np.einsum('...i,...i',c,a)
    return 2*np.arctan2(num,den)
def wrap(x,p=4*np.pi): return (x+p/2)%p-p/2
def field(L,n,deg=1,ell=1.0,r0=0.4,shift=(0.,0.)):
    x=(np.arange(L)+shift[0])*ell/L; t=(np.arange(n)+shift[1])*ell/n
    X,Tt=np.meshgrid(x,t,indexing='ij'); u=(X-ell/2)+1j*(Tt-ell/2); rho=np.abs(u)
    m=np.zeros((L,n,3)); m[...,2]=1.0
    ins=rho<r0*ell; r=rho[ins]/(r0*ell)
    kap=np.exp(-r**2/(1-r**2)); Z=u[ins]/(r0*ell)/kap
    if deg==-1: Z=np.conj(Z)
    if deg==0: Z=np.abs(Z)*np.exp(0.3j)   # contractible (degree 0) bump
    den=np.abs(Z)**2+1
    m[ins]=np.stack([2*Z.real/den,2*Z.imag/den,(np.abs(Z)**2-1)/den],-1)
    # generic global twist so nothing is axis-aligned
    a=0.7; R=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    return m@R.T
def analyse(m,S=0.5):
    L,n,_=m.shape; z=np.array([0,0,1.])
    sgn=(-1)**np.arange(L)
    nphys=m*sgn[:,None,None]
    W=lambda f: Om(z,f,np.roll(f,-1,axis=1)).sum(axis=1)          # per-site WZ sums
    Wphys=W(nphys); Wm=W(m)
    a,ap=m,np.roll(m,-1,axis=1); b,bp=np.roll(m,-1,axis=0),np.roll(np.roll(m,-1,axis=0),-1,axis=1)
    Rj=(Om(a,ap,bp)+Om(a,bp,b)).sum(axis=1)                       # ribbon between site j and j+1
    Q=Rj.sum()/(4*np.pi)
    X=(sgn*Wm).sum(); eps=(sgn*Rj).sum()
    chk1=wrap(Wphys.sum()-X); chk2=wrap(X-(2*np.pi*Q+eps/2))
    chk3=wrap(Wm-np.roll(Wm,-1)-Rj)
    return Q,eps,chk1,chk2,np.abs(chk3).max()
if __name__=="__main__":
    for deg in (1,-1,0):
        print(f"--- degree {deg} smooth bump ---")
        for L in (16,24,32,48,64,96,128):
            Q,eps,c1,c2,c3=analyse(field(L,L,deg))
            print(f"L={L:4d}  Q={Q:+.12f}  eps={eps:+.3e}  [mod-4pi checks {c1:.1e} {c2:.1e} {c3:.1e}]")
    print("--- generic random smooth history (Fourier modes), degree from data ---")
    rng=np.random.default_rng(3)
    def rnd(L,n):
        x=np.arange(L)/L; t=np.arange(n)/n; X,T=np.meshgrid(x,t,indexing='ij'); v=np.zeros((L,n,3))
        for i in range(3):
            for kx in range(-2,3):
                for kt in range(-2,3):
                    c=np.random.default_rng(100*i+5*kx+kt+50).normal()/(1+kx*kx+kt*kt)
                    v[...,i]+=c*np.cos(2*np.pi*(kx*X+kt*T)+i)
        v+=np.array([0,0,0.0]); return v/np.linalg.norm(v,axis=-1,keepdims=True)
    for L in (16,32,64,128,256):
        Q,eps,c1,c2,c3=analyse(rnd(L,L))
        print(f"L={L:4d}  Q={Q:+.12f}  eps={eps:+.3e}  [checks {c1:.1e} {c2:.1e} {c3:.1e}]")
