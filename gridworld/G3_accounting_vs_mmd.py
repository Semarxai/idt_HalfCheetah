"""G3: MMD and the accounting decomposition on the same windows.
MMD^2 (unbiased) with a delta kernel on (s,a,s') symbols and with an RBF kernel on one-hot
(s,a,s') coordinates, between each window and a healthy reference sample; reported as d
(change / healthy SD), next to the accounting readings, for each G2 case and construction."""
import numpy as np, pandas as pd, gw_core as g, estimators as e
import importlib.util, sys
spec=importlib.util.spec_from_file_location("G2mod","G2_occupancy_vs_mechanism.py")
src=open("G2_occupancy_vs_mechanism.py").read().split("R=pd.concat")[0]
G2=type(sys)("G2mod"); exec(src,G2.__dict__)   # functions only, no rerun
rng=np.random.default_rng(20261110); G2.rng=rng
def mmd2_delta(x,ref):
    # unbiased MMD^2 with k(u,v)=1[u==v]
    def sumsq(z):
        _,c=np.unique(z,return_counts=True); return (c*(c-1)).sum()
    n,m=len(x),len(ref); u,ix=np.unique(np.concatenate([x,ref]),return_inverse=True)
    cx=np.bincount(ix[:n],minlength=len(u)); cr=np.bincount(ix[n:],minlength=len(u))
    return sumsq(x)/(n*(n-1))+sumsq(ref)/(m*(m-1))-2*(cx*cr).sum()/(n*m)
def feats(S,A,Sn):
    xy=np.array(g.cells,float)/(g.N-1)
    return np.hstack([xy[S],np.eye(g.nA)[A]*0.5,xy[Sn]])
def mmd2_rbf(X,Y,bw):
    def k(U,V): d=((U[:,None,:]-V[None,:,:])**2).sum(-1); return np.exp(-d/(2*bw**2))
    n,m=len(X),len(Y); Kxx=k(X,X); Kyy=k(Y,Y); Kxy=k(X,Y)
    return (Kxx.sum()-np.trace(Kxx))/(n*(n-1))+(Kyy.sum()-np.trace(Kyy))/(m*(m-1))-2*Kxy.mean()
def run(N,eps,W,nwin,tag):
    g.setup(N); pi=g.policy(eps); T0=g.kernel(g.slip_map()); st0=g.start_dist(); p0,_=g.population(T0,pi,st0); d0=p0.sum(2)
    cases={"A visitation only":(T0,g.start_dist([(x,0) for x in range(g.WALL_X)])),
           "D action noise 10%":(0.9*T0+0.1*T0.mean(1,keepdims=True),st0),
           "C slip one-sided":(g.kernel(g.slip_map(),side=1.0),st0)}
    sym=lambda S,A,Sn:(S*g.nA+A)*g.nS+Sn
    rows=[]
    for constr in ("i rollout","ii fixed d0"):
        gen=(lambda T,st,n: G2.roll(T,pi,st,n+2000)) if constr=="i rollout" else (lambda T,st,n: G2.onestep(d0,T,n))
        def wins(T,st,n):
            S,A,Sn=gen(T,st,n)
            if constr=="i rollout": S,A,Sn=S[2000:],A[2000:],Sn[2000:]
            return S,A,Sn
        rS,rA,rSn=wins(T0,st0,5000); ref=sym(rS,rA,rSn)
        Fr=feats(rS,rA,rSn)[rng.choice(5000,800,replace=False)]
        D=((Fr[:300,None,:]-Fr[None,:300,:])**2).sum(-1); bw=np.sqrt(np.median(D[D>0])/2)
        S,A,Sn=wins(T0,st0,nwin*W)
        def table(S,A,Sn):
            out=[]
            for k in range(nwin):
                sl=slice(k*W,(k+1)*W); q=e.window_quantities(S[sl],A[sl],Sn[sl],g.nS,g.nA,"plugin")
                q["MMD"]=mmd2_delta(sym(S[sl],A[sl],Sn[sl]),ref)
                F=feats(S[sl],A[sl],Sn[sl]); F=F[rng.choice(len(F),min(len(F),300),replace=False)]
                q["MMD_rbf"]=mmd2_rbf(F,Fr,bw); out.append(q)
            return pd.DataFrame(out)
        H0=table(S,A,Sn)
        for name,(T,st) in cases.items():
            X=table(*wins(T,st,nwin*W))
            for s in ["MMD","MMD_rbf","P","Hf","Hb","I_SSn","I_ASn_S","HS"]:
                rows.append(dict(world=tag,construction=constr,case=name,stat=s,d=(X[s].mean()-H0[s].mean())/H0[s].std()))
    return pd.DataFrame(rows)
R=pd.concat([run(7,0.2,500,150,"7x7 W=500"),run(27,0.4,200,200,"27x27 W=200")]); R.to_csv("G3_results.csv",index=False)
for w in R.world.unique():
    print(f"\n#### {w}")
    print(R[R.world==w].pivot_table(index=["construction","case"],columns="stat",values="d",sort=False)[["MMD","MMD_rbf","P","Hf","Hb","I_SSn","I_ASn_S","HS"]].round(2).to_string())
