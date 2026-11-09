"""E5 bootstrap (cloud): nested leave-one-agent-out (H5 select, ALPHA 0.25) on E5_trial_scores.csv, per-condition
held-out detection, and agent-level bootstrap (1,000; selection inside) for telemetry - KL differences."""
import os
import numpy as np, pandas as pd
T=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"out","E5_trial_scores.csv"))
ZG=np.round(np.arange(1.5,30.0,0.05),2)
M={"three-of-four":"cs3","P":"P_cs","KL joint":"kl_joint","KL S":"kl_S","reward":"reward","MMD":"MMD"}
def select(inner,c):
    C=inner[inner.ctrl][c].to_numpy(); P=inner[~inner.ctrl][c].to_numpy()
    fa=(C[None,:]>ZG[:,None]).mean(1); det=(P[None,:]>ZG[:,None]).mean(1); ok=np.where(fa<=0.25)[0]
    return ZG[ok[np.flatnonzero(det[ok]==det[ok].max())[-1]]]
def nested(agents,per=False):
    out={m:[] for m in M}; pc=[]; uniq,mult=np.unique(agents,return_counts=True)
    for a,n in zip(uniq,mult):
        inner=pd.concat([T[T.UID==b] for b in agents if b!=a]); held=T[(T.UID==a)&(~T.ctrl)]
        for m,c in M.items():
            z=select(inner,c); hits=list(held[c]>z); out[m]+=hits*n
            if per: pc+=[dict(method=m,cond=cd,hit=h) for cd,h in zip(held.cond,hits)]
    return {m:100*np.mean(v) for m,v in out.items()},pc
AG=np.array(sorted(T.UID.unique())); pt,pc=nested(AG,True); print({m:round(v,1) for m,v in pt.items()})
PC=pd.DataFrame(pc).pivot_table(index="cond",columns="method",values="hit",aggfunc="mean")*100; print(PC.round(1).to_string()); PC.to_csv("E5_by_condition.csv")
rng=np.random.default_rng(20261111); B=[]
for b in range(1000):
    ags=rng.choice(AG,len(AG),replace=True)
    if len(np.unique(ags))<3: continue
    B.append(nested(ags)[0])
B=pd.DataFrame(B); B.to_csv("E5_boot.csv",index=False)
for m in M: print(f"{m:14s} [{B[m].quantile(.025):.1f}, {B[m].quantile(.975):.1f}]")
for a,b in [("three-of-four","KL joint"),("P","KL joint"),("three-of-four","KL S"),("P","KL S"),("KL joint","reward"),("KL joint","MMD")]:
    d=B[a]-B[b]; print(f"{a} - {b}: [{d.quantile(.025):+.1f}, {d.quantile(.975):+.1f}]")
