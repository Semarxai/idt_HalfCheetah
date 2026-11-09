"""G4a: estimation stability of the normalisations against exact truth.
Statistics: P = I/C (C = H(S)+H(A)+H(S')), I/min{H(S,A),H(S')}, I/H(S,A,S'), and unnormalised I(S,A;S').
Worlds: primary 7x7 (eps 0.2) and stress 27x27 (eps 0.4, HalfCheetah-like coverage at W 200).
Estimators: plug-in, Miller-Madow, Chao-Shen, each from one coherent set of entropies. 10 seeds (101-110).
Per (world, W, estimator, statistic): relative bias (mean over windows - truth)/truth, relative SD of the window
estimates, and the change of relative bias across the coverage range (stability of the bias)."""
import numpy as np, pandas as pd, gw_core as g, estimators as e
from G1_truth_vs_estimate import fast_rollout
import G1_truth_vs_estimate as G1
def stats_from(q): return {"P":q["P"],"I_min":q["I"]/min(q["HSA"],q["HSn"]),"I_joint":q["I"]/q["HT"],"I_raw":q["I"]}
rows=[]
for N,eps,WS in [(7,0.2,[50,100,200,500,1000,2000]),(27,0.4,[100,200,350,500,1000,2000])]:
    g.setup(N); pi=g.policy(eps); T=g.kernel(g.slip_map()); p,start=g.population(T,pi); tq=g.quantities(p); truth=stats_from(tq)
    for seed in range(101,111):
        G1.rng=np.random.default_rng(seed)
        for W in WS:
            nwin=max(10,min(200,200000//W)); S,A,Sn=fast_rollout(T,pi,start,nwin*W+2000); S,A,Sn=S[2000:],A[2000:],Sn[2000:]
            cov=[]
            for est in ("plugin","mm","cs"):
                vals=[]
                for k in range(nwin):
                    sl=slice(k*W,(k+1)*W); vals.append(stats_from(e.window_quantities(S[sl],A[sl],Sn[sl],g.nS,g.nA,est)))
                    if est=="plugin": cov.append(e.coverage((S[sl]*g.nA+A[sl])*g.nS+Sn[sl])[1])
                V=pd.DataFrame(vals)
                for s in truth:
                    rows.append(dict(world=f"{N}x{N}",W=W,seed=seed,est=est,stat=s,truth=truth[s],rel_bias=(V[s].mean()-truth[s])/truth[s],
                                     rel_sd=V[s].std()/truth[s],cov=np.mean(cov)))
import os
R=pd.DataFrame(rows); R.to_csv("G4a_results.csv",index=False)
pd.set_option("display.width",220)
agg=R.groupby(["world","W","est","stat"]).agg(cov=("cov","mean"),rel_bias=("rel_bias","mean"),rel_bias_sd=("rel_bias","std"),rel_sd=("rel_sd","mean")).reset_index()
agg.to_csv("G4a_summary.csv",index=False)
for w in agg.world.unique():
    x=agg[agg.world==w]
    print(f"\n#### {w}: relative bias % (mean over 10 seeds) by window; coverage of (S,A,S') in brackets")
    t=x.pivot_table(index=["est","stat"],columns="W",values="rel_bias")*100
    c=x[x.est=="plugin"].groupby("W")["cov"].first().round(2).to_dict(); print("coverage:",c)
    print(t.round(2).to_string())
    print("relative SD of window estimates % (within-window noise):")
    print((x.pivot_table(index=["est","stat"],columns="W",values="rel_sd")*100).round(2).to_string())
    print("bias drift across the coverage range = max-min of relative bias over W (percentage points):")
    print((x.groupby(["est","stat"]).rel_bias.agg(lambda v:v.max()-v.min())*100).round(2).unstack().to_string())
