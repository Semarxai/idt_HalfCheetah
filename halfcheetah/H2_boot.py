"""Bootstrap (nested selection inside each resample) for the Chao-Shen statistics next to the other methods.
usage: python H2_boot.py <chunk> <n>   |   python H2_boot.py sum"""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import H5_nested as h
O=h.O; T=pd.read_csv(f"{O}/H2_trial_scores_cs.csv")
h.T=T; h.METHODS={"consensus Chao-Shen":["cs1","cs2","cs3","cs4"],"P Chao-Shen":["P_cs"],"consensus plug-in":["cons1","cons2","cons3","cons4"],
                  "reward":["reward"],"MMD":["MMD"],"H(S)":["HS"]}
if sys.argv[1]=="sum":
    B=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{O}/H2_boot_*.csv"))]); print("resamples:",len(B))
    pt=h.nested(T,0.25)
    for m in h.METHODS:
        d=B[f"{m}_det"]*100; f=B[f"{m}_fa"]*100
        print(f"  {m:20s} point det {100*pt[m][0]:5.1f}% fa {int(round(pt[m][1]*21))}/21 | det [{d.quantile(.025):5.1f},{d.quantile(.975):5.1f}] fa [{f.quantile(.025):5.1f},{f.quantile(.975):5.1f}]")
    for ref in ["consensus Chao-Shen","P Chao-Shen"]:
        for m in ["reward","MMD","H(S)","consensus plug-in"]:
            dd=(B[f"{ref}_det"]-B[f"{m}_det"])*100; ff=(B[f"{ref}_fa"]-B[f"{m}_fa"])*100
            print(f"  {ref} - {m:18s} det diff [{dd.quantile(.025):5.1f},{dd.quantile(.975):5.1f}]  fa diff [{ff.quantile(.025):5.1f},{ff.quantile(.975):5.1f}]")
else:
    cid,n=int(sys.argv[1]),int(sys.argv[2]); rng=np.random.default_rng(20260917+100+cid); out=[]
    for b in range(n):
        rr=h.nested(T,0.25,rng.choice(h.AG,len(h.AG),replace=True))
        out.append({**{f"{m}_det":rr[m][0] for m in h.METHODS},**{f"{m}_fa":rr[m][1] for m in h.METHODS}})
    pd.DataFrame(out).to_csv(f"{O}/H2_boot_{cid:02d}.csv",index=False); print("chunk",cid,"done")
