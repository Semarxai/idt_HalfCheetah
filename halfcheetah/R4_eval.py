"""R4 evaluation: MMD configurations under the H5 nested protocol, next to P (Chao-Shen), k-of-four,
three-of-four and reward. usage: point | boot <chunk> <n> | bootsum"""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import H5_nested as h
O=h.O; T=pd.read_csv(f"{O}/PRIMARY_trial_scores.csv")
M=pd.concat([pd.read_csv(f) for f in glob.glob(f"{O}/R4/*.csv")])
M["col"]=M.apply(lambda r:f"MMD_ref{int(r.nref)}_bw{r.bw_mult}",axis=1)
W=M.pivot_table(index=["UID","cond"],columns="col",values="z").reset_index(); T=T.merge(W,on=["UID","cond"]); h.T=T
CFG=sorted(W.columns[2:])
h.METHODS={"P alone, Chao-Shen":["P_cs"],"three-of-four, Chao-Shen":["cs3"],"reward":["reward"],"MMD (H3: ref 800, bw x1)":["MMD"]}|{c:[c] for c in CFG}
m=sys.argv[1]
if m=="point":
    P=T[~T.ctrl]; C=T[T.ctrl]; rows=[]
    for al in (0.10,0.25):
        rr=h.nested(T,al)
        for k,cols in h.METHODS.items():
            c=cols[0]; zs=np.sort(C[c].values)[::-1]; zt=(zs[4]+zs[5])/2
            rows.append(dict(alpha=al,method=k,insample_at_5of21=round(100*(P[c]>zt).mean(),1),pm3sigma=f"{100*(P[c]>3).mean():.1f}% {int((C[c]>3).sum())}/21",
                             nested_det=round(100*rr[k][0],1),nested_ctrl=f"{int(round(rr[k][1]*21))}/21"))
    R=pd.DataFrame(rows); R.to_csv(f"{O}/R4_point.csv",index=False); pd.set_option("display.width",200); print(R.to_string(index=False))
elif m=="boot":
    cid,n=int(sys.argv[2]),int(sys.argv[3]); rng=np.random.default_rng(20260917+700+cid); out=[]
    for b in range(n):
        rr=h.nested(T,0.25,rng.choice(h.AG,len(h.AG),replace=True)); out.append({f"{k}|det":v[0] for k,v in rr.items()}|{f"{k}|fa":v[1] for k,v in rr.items()})
    pd.DataFrame(out).to_csv(f"{O}/R4_boot_{cid:02d}.csv",index=False); print("chunk",cid)
elif m=="bootsum":
    B=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{O}/R4_boot_*.csv"))]); print("resamples",len(B))
    for k in h.METHODS:
        d=B[f"{k}|det"]*100; line=f"  {k:28s} det [{d.quantile(.025):5.1f},{d.quantile(.975):5.1f}]"
        for ref in ("P alone, Chao-Shen","three-of-four, Chao-Shen"):
            if k!=ref and "MMD" in k:
                dd=(B[f"{ref}|det"]-B[f"{k}|det"])*100; ff=(B[f"{ref}|fa"]-B[f"{k}|fa"])*100
                line+=f" | {ref.split(',')[0]} minus: det [{dd.quantile(.025):5.1f},{dd.quantile(.975):5.1f}] fa [{ff.quantile(.025):5.1f},{ff.quantile(.975):5.1f}]"
        print(line)
