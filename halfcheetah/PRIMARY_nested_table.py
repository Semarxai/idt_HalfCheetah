"""Primary held-out comparison (main display): every method under the same nested leave-one-agent-out
protocol (H5). Methods: telemetry consensus (Chao-Shen), P alone (Chao-Shen), reward, MMD (H3), H(S),
dynamics-model prediction error (H13), and a fixed reward + telemetry combination (H14: alarm when the
consensus fires OR reward exceeds the same z; i.e. score_k = max(consensus k-th score, reward score)).
usage: python PRIMARY_nested_table.py point | curve | boot <chunk> <n> | bootsum"""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import H5_nested as h
O=h.O
T=pd.read_csv(f"{O}/H2_trial_scores_cs.csv")
dyn=[]
for _,r in h.MAP.iterrows():
    d=pd.read_csv(f"{O}/H13/{r.UID}__{r.file.replace('_metrics.csv','')}.csv"); cal=d[(d.window_id>=150)&(d.window_id<=290)]; po=d[d.window_id>=300]
    dyn.append(((po.pred_err-cal.pred_err.mean())/cal.pred_err.std()).max())
T["DYN"]=dyn
for k in range(1,5): T[f"comb{k}"]=np.maximum(T[f"cs{k}"],T["reward"])
T.to_csv(f"{O}/PRIMARY_trial_scores.csv",index=False); h.T=T
h.METHODS={"telemetry consensus (Chao-Shen)":["cs1","cs2","cs3","cs4"],"P alone (Chao-Shen)":["P_cs"],
           "reward + telemetry (fixed OR)":["comb1","comb2","comb3","comb4"],"dynamics model":["DYN"],
           "MMD":["MMD"],"H(S)":["HS"],"reward":["reward"]}
mode=sys.argv[1]
if mode=="point":
    P=T[~T.ctrl]; C=T[T.ctrl]
    for al in (0.10,0.25):
        rr=h.nested(T,al); print(f"=== nested, ALPHA {al:.2f}")
        for m,(d,f,ch) in rr.items():
            cc=pd.Series([f"{c}@{z}" for c,z in ch]).value_counts().head(2).to_dict(); print(f"  {m:34s} det {100*d:5.1f}%  ctrl {int(round(f*21))}/21  {cc}")
    print("\nin-sample, dynamics model: +3σ det %.1f%% ctrl %d/21; matched 5/21:"%(100*(P.DYN>3).mean(),(C.DYN>3).sum()),end=" ")
    zs=np.sort(C.DYN.values)[::-1]; zt=(zs[4]+zs[5])/2; print(f"z={zt:.2f} det {100*(P.DYN>zt).mean():.1f}%")
    print("per condition, nested ALPHA 0.25 (held-out detection %):")
    rows=[]
    for m,cols in h.METHODS.items():
        for a in h.AG:
            c,zt=h.select(T[T.UID!=a],cols,0.25); held=T[(T.UID==a)&(~T.ctrl)]
            for _,x in held.iterrows(): rows.append(dict(method=m,cond=x.cond,hit=x[c]>zt))
    D=pd.DataFrame(rows); print((D.pivot_table(index="cond",columns="method",values="hit",aggfunc="mean")*100).round(1).to_string())
elif mode=="curve":
    cur=[]
    for al in [0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50]:
        rr=h.nested(T,al); cur.append(dict(alpha=al,**{f"{m} det":round(100*rr[m][0],1) for m in h.METHODS},**{f"{m} fa":round(100*rr[m][1],1) for m in h.METHODS}))
    C=pd.DataFrame(cur); C.to_csv(f"{O}/PRIMARY_nested_curve.csv",index=False); print(C.T.to_string())
elif mode=="boot":
    cid,n=int(sys.argv[2]),int(sys.argv[3]); rng=np.random.default_rng(20260917+200+cid); out=[]
    for b in range(n):
        rr=h.nested(T,0.25,rng.choice(h.AG,len(h.AG),replace=True))
        out.append({**{f"{m}|det":rr[m][0] for m in h.METHODS},**{f"{m}|fa":rr[m][1] for m in h.METHODS}})
    pd.DataFrame(out).to_csv(f"{O}/PRIMARY_boot_{cid:02d}.csv",index=False); print("chunk",cid,"done")
elif mode=="bootsum":
    B=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{O}/PRIMARY_boot_*.csv"))]); print("resamples:",len(B))
    pt=h.nested(T,0.25); ref="telemetry consensus (Chao-Shen)"
    for m in h.METHODS:
        d=B[f"{m}|det"]*100; f=B[f"{m}|fa"]*100
        line=f"  {m:34s} {100*pt[m][0]:5.1f}% [{d.quantile(.025):5.1f},{d.quantile(.975):5.1f}]  ctrl {int(round(pt[m][1]*21))}/21 [{f.quantile(.025):4.1f},{f.quantile(.975):4.1f}]"
        if m!=ref:
            dd=(B[f"{ref}|det"]-B[f"{m}|det"])*100; ff=(B[f"{ref}|fa"]-B[f"{m}|fa"])*100
            line+=f" | consensus - this: det [{dd.quantile(.025):5.1f},{dd.quantile(.975):5.1f}] fa [{ff.quantile(.025):5.1f},{ff.quantile(.975):5.1f}]"
        print(line)
