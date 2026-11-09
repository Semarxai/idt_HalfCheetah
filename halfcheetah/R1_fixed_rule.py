"""R1 (review point 1): the fixed three-of-four rule reported separately from the nested-selected k-of-four rule.
Same trial scores and nested protocol as PRIMARY. usage: point | boot <chunk> <n> | bootsum"""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import H5_nested as h
O=h.O; T=pd.read_csv(f"{O}/PRIMARY_trial_scores.csv"); h.T=T
h.METHODS={"P alone, Chao-Shen (z selected)":["P_cs"],
           "three-of-four, Chao-Shen (z selected)":["cs3"],"three-of-four, plug-in (z selected)":["cons3"],
           "four-of-four, Chao-Shen (z selected)":["cs4"],"k-of-four, Chao-Shen (k and z selected)":["cs1","cs2","cs3","cs4"],
           "reward (z selected)":["reward"]}
m=sys.argv[1]
if m=="point":
    P=T[~T.ctrl]; C=T[T.ctrl]
    print("fixed rules, no selection (+-3 sigma, in-sample):")
    for lab,c in [("three-of-four Chao-Shen","cs3"),("three-of-four plug-in","cons3"),("four-of-four Chao-Shen","cs4"),("P alone Chao-Shen","P_cs"),("reward","reward")]:
        print(f"  {lab:28s} det {100*(P[c]>3).mean():5.1f}%  ctrl {int((C[c]>3).sum())}/21")
    for al in (0.10,0.25):
        print(f"\nnested, ALPHA {al:.2f}:"); rr=h.nested(T,al)
        for k,(d,f,ch) in rr.items():
            cc=pd.Series([f"{c}@{z}" for c,z in ch]).value_counts().to_dict()
            print(f"  {k:42s} det {100*d:5.1f}%  ctrl {int(round(f*21))}/21  selections {cc}")
elif m=="boot":
    cid,n=int(sys.argv[2]),int(sys.argv[3]); rng=np.random.default_rng(20260917+400+cid); out=[]
    for b in range(n):
        rr=h.nested(T,0.25,rng.choice(h.AG,len(h.AG),replace=True)); out.append({f"{k}|det":v[0] for k,v in rr.items()}|{f"{k}|fa":v[1] for k,v in rr.items()})
    pd.DataFrame(out).to_csv(f"{O}/R1_boot_{cid:02d}.csv",index=False); print("chunk",cid)
elif m=="bootsum":
    B=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{O}/R1_boot_*.csv"))]); print("resamples",len(B))
    for k in h.METHODS:
        d=B[f"{k}|det"]*100; f=B[f"{k}|fa"]*100; line=f"  {k:42s} det [{d.quantile(.025):5.1f},{d.quantile(.975):5.1f}] fa [{f.quantile(.025):4.1f},{f.quantile(.975):4.1f}]"
        if not k.startswith("reward"):
            dd=(B[f"{k}|det"]-B["reward (z selected)|det"])*100; ff=(B[f"{k}|fa"]-B["reward (z selected)|fa"])*100
            line+=f"  | minus reward: det [{dd.quantile(.025):5.1f},{dd.quantile(.975):5.1f}] fa [{ff.quantile(.025):5.1f},{ff.quantile(.975):5.1f}]"
        print(line)
