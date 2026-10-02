import numpy as np
from theta_check import analyse
def rnd(L,n,seed=0):
    x=np.arange(L)/L; t=np.arange(n)/n; X,T=np.meshgrid(x,t,indexing='ij'); v=np.zeros((L,n,3))
    rng=np.random.default_rng(seed)
    for i in range(3):
        for kx in range(-2,3):
            for kt in range(-2,3):
                c=rng.normal()/(1+kx*kx+kt*kt); ph=rng.uniform(0,2*np.pi)
                v[...,i]+=c*np.cos(2*np.pi*(kx*X+kt*T)+ph)
    return v/np.linalg.norm(v,axis=-1,keepdims=True)
out={}
for seed in (0,1,2):
  for n in (16,64,256):
    rows=[]
    for L in (16,24,32,48,64,96,128,192,256):
        Q,eps,c1,c2,c3=analyse(rnd(L,n,seed))
        rows.append((L,Q,abs(eps),max(abs(c1),abs(c2),c3)))
    out[(seed,n)]=rows
    print(f"seed {seed} n={n}: Q={[round(r[1],9) for r in rows][:3]}..., |eps|=",' '.join(f"{r[2]:.1e}" for r in rows)," maxcheck=%.1e"%max(r[3] for r in rows))
np.save('theta_scan.npy',out,allow_pickle=True)
