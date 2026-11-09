"""G2: occupancy vs mechanism. Two constructions per case:
 (i)  operational rollouts - occupancy free to change;
 (ii) reference-weighted one-step sampling - healthy d0(s,a) held fixed, s' from healthy or perturbed kernel.
For each statistic: exact population change, mean estimated change (plug-in, W windows), the
difference between them (change in estimator bias), and the standardised shift d = change / healthy SD.
Reward = 1 when the goal is reached (windowed reward = goal rate); case F changes reward only."""
import numpy as np, pandas as pd, gw_core as g, estimators as e
rng=np.random.default_rng(20261109)
STATS=["P","Hf","Hb","dH","I_SSn","I_ASn_S","HS"]
def roll(T,pi,start,n):
    cpi=np.cumsum(pi,1); cT=np.cumsum(T,2); cst=np.cumsum(start); u=rng.random((n,3))
    S=np.empty(n,int);A=np.empty(n,int);Sn=np.empty(n,int); s=int(np.searchsorted(cst,rng.random()))
    for t in range(n):
        a=int(np.searchsorted(cpi[s],u[t,0])); sn=int(np.searchsorted(cT[s,a],u[t,1])); S[t],A[t],Sn[t]=s,a,sn
        s=int(np.searchsorted(cst,u[t,2])) if sn==g.G else sn
    return S,A,Sn
def onestep(d0,T,n):
    flat=d0.ravel()/d0.sum(); k=rng.choice(flat.size,size=n,p=flat); S,A=k//g.nA,k%g.nA
    cT=np.cumsum(T,2); u=rng.random(n); Sn=np.array([int(np.searchsorted(cT[s,a],x)) for s,a,x in zip(S,A,u)])
    return S,A,Sn
def windows(S,A,Sn,W,rw_offset=0.0):
    out=[]
    for k in range(len(S)//W):
        sl=slice(k*W,(k+1)*W); q=e.window_quantities(S[sl],A[sl],Sn[sl],g.nS,g.nA,"plugin")
        q["reward"]=(Sn[sl]==g.G).mean()+rw_offset; out.append(q)
    return pd.DataFrame(out)
def run(N,eps,W,nwin,tag):
    g.setup(N); pi=g.policy(eps); T0=g.kernel(g.slip_map()); st0=g.start_dist()
    p0,_=g.population(T0,pi,st0); d0=p0.sum(2); truth0=g.quantities(p0); truth0["reward"]=p0[:,:,g.G].sum()
    zone=g.slip_map()>0.1
    cases={
      "A visitation (start moved), same mechanism": dict(T=T0,start=g.start_dist([(x,0) for x in range(g.WALL_X)])),
      "B env-side stochastic: zone slip 0.30->0.60": dict(T=g.kernel(g.slip_map(zone=0.60)),start=st0),
      "C env-side systematic: zone slip all one side": dict(T=g.kernel(g.slip_map(),side=1.0),start=st0),
      "D agent-side stochastic: 10% executed action random": dict(T=None,start=st0),
      "F reward only (step penalty), loop unchanged": dict(T=T0,start=st0,rw=-0.01),
    }
    cases["D agent-side stochastic: 10% executed action random"]["T"]=0.9*T0+0.1*T0.mean(1,keepdims=True)
    n=nwin*W+2000; S,A,Sn=roll(T0,pi,st0,n); H0=windows(S[2000:],A[2000:],Sn[2000:],W)
    S,A,Sn=onestep(d0,T0,nwin*W); H0b=windows(S,A,Sn,W)
    rows=[]
    for name,c in cases.items():
        for constr in ("i rollout","ii fixed d0"):
            if constr=="i rollout":
                p1,_=g.population(c["T"],pi,c["start"]); S,A,Sn=roll(c["T"],pi,c["start"],n); X=windows(S[2000:],A[2000:],Sn[2000:],W,c.get("rw",0)); base=H0
            else:
                p1=d0[:,:,None]*c["T"]; S,A,Sn=onestep(d0,c["T"],nwin*W); X=windows(S,A,Sn,W,c.get("rw",0)); base=H0b
            t1=g.quantities(p1); t1["reward"]=p1[:,:,g.G].sum()+c.get("rw",0)
            for s in STATS+["reward"]:
                true_d=t1[s]-truth0[s]; est_d=X[s].mean()-base[s].mean()
                rows.append(dict(world=tag,case=name,construction=constr,stat=s,true_change=true_d,est_change=est_d,
                                 bias_change=est_d-true_d,d=est_d/base[s].std()))
    return pd.DataFrame(rows)
R=pd.concat([run(7,0.2,500,200,"7x7 W=500"),run(27,0.4,200,300,"27x27 W=200 (HalfCheetah-like coverage)")])
R.to_csv("G2_results.csv",index=False)
pd.set_option("display.width",250)
for w in R.world.unique():
    print(f"\n######## {w}")
    for constr in ("i rollout","ii fixed d0"):
        sub=R[(R.world==w)&(R.construction==constr)]
        print(f"\n--- construction {constr}: standardised shift d (estimated change / healthy SD)")
        print(sub.pivot_table(index="case",columns="stat",values="d",sort=False)[STATS+["reward"]].round(2).to_string())
        print(f"--- construction {constr}: true change | estimated change  (P, Hf, Hb, I(S;S'), I(A;S'|S))")
        t=sub[sub.stat.isin(["P","Hf","Hb","I_SSn","I_ASn_S"])]
        print(t.pivot_table(index="case",columns="stat",values=["true_change","est_change"],sort=False).round(3).to_string())
