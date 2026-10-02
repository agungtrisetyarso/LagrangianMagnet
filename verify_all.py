"""Numerical verification for 'Lagrangian-native quantum simulation of quantum magnets'."""
import numpy as np, itertools
from scipy.linalg import expm
rng=np.random.default_rng(7)
def spin(S):
    d=int(round(2*S+1)); m=S-np.arange(d)
    Jz=np.diag(m); Jp=np.zeros((d,d))
    for k in range(1,d): Jp[k-1,k]=np.sqrt(S*(S+1)-m[k]*(m[k]+1))
    return (Jp+Jp.T)/2,(Jp-Jp.T)/(2j),Jz
def coh(S,n):
    Jx,Jy,Jz=spin(S); th=np.arccos(np.clip(n[2],-1,1)); ph=np.arctan2(n[1],n[0])
    top=np.zeros(len(Jz)); top[0]=1
    # north-pole gauge R = e^{-i ph Jz} e^{-i th Jy} e^{+i ph Jz}
    return expm(-1j*ph*Jz)@expm(-1j*th*Jy)@expm(1j*ph*Jz)@top
def tetra(): v=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])/np.sqrt(3); return v
def octa(): return np.vstack([np.eye(3),-np.eye(3)])
def icosa():
    p=(1+5**.5)/2; v=[]
    for a in (1,-1):
        for b in (p,-p): v+= [[0,a,b],[a,b,0],[b,0,a]]
    v=np.array(v,float); return v/np.linalg.norm(v,axis=1)[:,None]
def T(S,F,h):  # frame (Toeplitz) quantization of single-site symbol
    d=int(2*S+1); return d/len(F)*sum(h(n)*np.outer(coh(S,n),coh(S,n).conj()) for n in F)
ok=lambda A,B:np.allclose(A,B,atol=1e-10)
print("== L1 frame identity ==")
for S,F,name in [(.5,tetra(),'tetra'),(1,octa(),'octa'),(1,icosa(),'icosa'),(2,icosa(),'icosa'),(2.5,icosa(),'icosa')]:
    print(S,name,ok(T(S,F,lambda n:1),np.eye(int(2*S+1))))
print("== L3 T[n^a]=J^a/(S+1) ==")
for S,F in [(.5,tetra()),(1,octa()),(2,icosa())]:
    J=spin(S); print(S,all(ok(T(S,F,lambda n,a=a:n[a]),J[a]/(S+1)) for a in range(3)))
print("== l=2 symbol: T[n_z^2] = alpha Jz^2 + beta ==")
for S in [1,1.5,2]:
    J=spin(S); A=T(S,icosa(),lambda n:n[2]**2); d=int(2*S+1)
    X=np.array([np.diag(J[2]@J[2]).real,np.ones(d)]).T; c=np.linalg.lstsq(X,np.diag(A).real,rcond=None)[0]
    print(S,c, 'pred',2/((S+1)*(2*S+3)), 1/(2*S+3))
print("spin-1/2: T[n_z^2] =",np.round(T(.5,octa(),lambda n:n[2]**2),12).real.tolist(),"(octa is 3-design)")
print("== L2 Berry overlap phase ==")
def Om(a,b,c): return 2*np.arctan2(a@np.cross(b,c),1+a@b+b@c+c@a)
z=np.array([0,0,1.]); maxerr=0
for S in [.5,1,1.5,3]:
    for _ in range(200):
        a,b=rng.normal(size=(2,3)); a/=np.linalg.norm(a); b/=np.linalg.norm(b)
        ov=coh(S,b).conj()@coh(S,a)
        pred=((1+a@b)/2)**(S)*np.exp(-1j*S*Om(z,a,b))
        maxerr=max(maxerr,abs(ov-pred))
print("max |<b|a> - ((1+a.b)/2)^S e^{-iS Omega(z,a,b)}| =",maxerr)
print("== L4 gauge quantization of closed loops ==")
for _ in range(3):
    loop=rng.normal(size=(7,3)); loop/=np.linalg.norm(loop,axis=1)[:,None]
    r,rp=rng.normal(size=(2,3)); r/=np.linalg.norm(r); rp/=np.linalg.norm(rp)
    D=sum(Om(r,loop[k],loop[(k+1)%7])-Om(rp,loop[k],loop[(k+1)%7]) for k in range(7))
    print("Delta/(4pi) =",round(D/(4*np.pi),10))
print("== T1 frame path-sum convergence (Heisenberg) ==")
def manybody(S,F,N,bonds,Tt,dts):
    d=int(2*S+1); M=len(F); J=spin(S)
    cs=[coh(S,n) for n in F]
    def emb(op,i): return np.kron(np.kron(np.eye(d**i),op),np.eye(d**(N-i-1)))
    H=sum(emb(J[a],i)@emb(J[a],j) for (i,j) in bonds for a in range(3))
    hmax=(S+1)**2*len(bonds)
    # product frame
    P=[];hv=[]
    for idx in itertools.product(range(M),repeat=N):
        v=cs[idx[0]]
        for k in idx[1:]: v=np.kron(v,cs[k])
        P.append(v); hv.append((S+1)**2*sum(F[idx[i]]@F[idx[j]] for (i,j) in bonds))
    P=np.array(P); hv=np.array(hv); w=(d/M)**N
    Ex=expm(-1j*Tt*H); out=[]
    assert ok(w*(P.T*hv)@P.conj(),H), "T[h] != H"
    for dt in dts:
        n=int(round(Tt/dt)); A=w*(P.T*np.exp(-1j*dt*hv))@P.conj()
        err=np.linalg.norm(np.linalg.matrix_power(A,n)-Ex,2)
        out.append((dt,err,Tt*dt*np.max(np.abs(hv))**2))
    return out
res={}
res['tri12']=manybody(.5,tetra(),3,[(0,1),(1,2),(2,0)],1.0,[0.2,0.1,0.05,0.025,0.0125,0.00625])
res['ring12']=manybody(.5,tetra(),4,[(0,1),(1,2),(2,3),(3,0)],1.0,[0.2,0.1,0.05,0.025,0.0125,0.00625])
res['tri1']=manybody(1,octa(),3,[(0,1),(1,2),(2,0)],1.0,[0.1,0.05,0.025,0.0125,0.00625])
for k,v in res.items():
    print(k); [print(f"  dt={a:.5f} err={b:.3e} bound={c:.3e}") for a,b,c in v]
    sl=np.polyfit(np.log([x[0] for x in v]),np.log([x[1] for x in v]),1)[0]; print("  slope",round(sl,3))
np.save('conv.npy',res,allow_pickle=True)
print("== Block-encoding normalisation, S=1/2 Heisenberg bond ==")
F=tetra(); hv=np.array([(9/4)*F[a]@F[b] for a in range(4) for b in range(4)])
print("l1 lambda =",(1/4)*np.abs(hv).sum()," uniform Lambda =",4*np.abs(hv).max()," ||S.S|| = 0.75")
print("== P4 frame estimator ==")
S=1;F=icosa();d=3; psi=rng.normal(size=3)+1j*rng.normal(size=3); psi/=np.linalg.norm(psi)
g=lambda n: n[0]-2*n[1]*1+0.3*n[2]
Q=np.array([abs(coh(S,n).conj()@psi)**2 for n in F]); p=d/len(F)*Q
print("sum p =",p.sum()," E[g]=",p@np.array([g(n) for n in F])," <T[g]>=",(psi.conj()@T(S,F,g)@psi).real)
print("== Berezin-Lieb: min h <= spec T[h] ==")
S=.5;N=3;bonds=[(0,1),(1,2),(2,0)]; J=spin(S)
emb=lambda op,i: np.kron(np.kron(np.eye(2**i),op),np.eye(2**(N-i-1)))
H=sum(emb(J[a],i)@emb(J[a],j) for (i,j) in bonds for a in range(3))
print("E0 =",np.linalg.eigvalsh(H)[0]," lower bound (S+1)^2*min_cl =",(1.5**2)*(-1.5)," upper bound S^2*min_cl =",0.25*(-1.5))
