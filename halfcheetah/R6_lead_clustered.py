"""R6 (review point 6): agent-clustered inference for early detection (replaces pooled sign tests).
From out/H10a_first_alarm.csv (nested operating points, ALPHA 0.25). Agent-level bootstrap (10,000 resamples
of the 21 agents, seed 20260917+600): (a) difference in share of perturbation trials detected within 1 and 5
evaluation episodes; (b) on trials both methods detect, net share earlier = (earlier - later)/both and the
median lead (episodes); (c) agent-level sign test: agents whose median lead is positive vs negative."""
import os, numpy as np, pandas as pd
from math import comb
O=os.path.join(os.path.dirname(os.path.abspath(__file__)),"out")
D=pd.read_csv(f"{O}/H10a_first_alarm.csv"); P=D[D.cond!="no_noise (baseline)"]
W=P.pivot_table(index=["UID","cond"],columns="method",values="first_window",aggfunc="first")
AG=np.array(sorted(P.UID.unique())); rng=np.random.default_rng(20260917+600); B=10000
def stats(a,b,X):
    xa,xb=X[a].to_numpy(),X[b].to_numpy()
    by1=np.mean(xa<20)-np.mean(xb<20); by5=np.mean(xa<100)-np.mean(xb<100)
    both=np.isfinite(xa)&np.isfinite(xb); d=(xb[both]-xa[both])
    net=((d>0).sum()-(d<0).sum())/max(both.sum(),1); med=np.median(d)*50/1000 if both.any() else np.nan
    return by1,by5,net,med
idx={a:W.index.get_level_values(0)==a for a in AG}
out=[]
for a,b in [("three-of-four Chao-Shen","reward"),("P Chao-Shen","reward"),("consensus Chao-Shen","reward"),
            ("three-of-four Chao-Shen","MMD"),("P Chao-Shen","MMD"),("consensus Chao-Shen","MMD"),
            ("P Chao-Shen","H(S)"),("consensus plug-in","reward")]:
    pt=stats(a,b,W); bs=[]
    for _ in range(B):
        ags=rng.choice(AG,len(AG),replace=True); X=pd.concat([W[idx[g]] for g in ags]); bs.append(stats(a,b,X))
    bs=np.array(bs); q=lambda j:np.nanpercentile(bs[:,j],[2.5,97.5])
    # agent-level sign test on per-agent median lead over jointly detected trials
    pos=neg=0
    for g in AG:
        x=W[idx[g]]; both=np.isfinite(x[a])&np.isfinite(x[b])
        if both.sum()==0: continue
        m=np.median((x[b]-x[a])[both])
        pos+=m>0; neg+=m<0
    n=pos+neg; p=min(1.0,2*sum(comb(n,i) for i in range(0,min(pos,neg)+1))/2**n) if n else 1.0
    out.append(dict(comparison=f"{a} vs {b}",
        by_ep1_diff=f"{100*pt[0]:+.1f} [{100*q(0)[0]:+.1f}, {100*q(0)[1]:+.1f}]",
        by_ep5_diff=f"{100*pt[1]:+.1f} [{100*q(1)[0]:+.1f}, {100*q(1)[1]:+.1f}]",
        net_earlier_share=f"{pt[2]:+.2f} [{q(2)[0]:+.2f}, {q(2)[1]:+.2f}]",
        median_lead_ep=f"{pt[3]:+.2f} [{q(3)[0]:+.2f}, {q(3)[1]:+.2f}]",
        agents_earlier_vs_later=f"{pos} v {neg}, p={p:.3f}"))
R=pd.DataFrame(out); R.to_csv(f"{O}/R6_lead_clustered.csv",index=False)
pd.set_option("display.width",250); print(R.to_string(index=False))
