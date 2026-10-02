import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
sx=sp.csr_matrix([[0,.5],[.5,0]]); sy=sp.csr_matrix([[0,-.5j],[.5j,0]]); sz=sp.csr_matrix([[.5,0],[0,-.5]])
def op(o,i,N): return sp.kron(sp.kron(sp.identity(2**i),o),sp.identity(2**(N-i-1)),format='csr')
def H(N,J2):
    h=0
    for i in range(N):
        for r,J in ((1,1.0),(2,J2)):
            if J==0: continue
            j=(i+r)%N; h=h+J*sum(op(o,i,N)@op(o,j,N) for o in (sx,sy,sz))
    return h
tet=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])/np.sqrt(3)
dots=tet@tet.T; b=(1.5**2)*np.abs(dots).max(); l1=(0.5**2)*(1.5**2)*np.abs(dots).sum()
print("per-bond: uniform 4b =",4*b," l1 =",l1)
for J2 in (0.0,0.2,0.5):
    for N in (8,12,16):
        Hm=H(N,J2); emin=sla.eigsh(Hm,k=1,which='SA')[0][0]; emax=sla.eigsh(Hm,k=1,which='LA')[0][0]
        nrm=max(abs(emin),abs(emax)); bonds=N*(1+ (J2>0)); wsum=N*(1+J2)
        print(f"J2={J2} N={N}: ||H||={nrm:.4f}  Lambda_unif/||H||={4*b*wsum/nrm:.2f}  Lambda_l1/||H||={l1*wsum/nrm:.2f}")
