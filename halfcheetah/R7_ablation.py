"""R7 (review point 7): the normalisation ablation (submission Table 3) redone with Chao-Shen estimates and
held-out intervals. Single statistics: I(S,A;S'), I/min{H(S,A),H(S')}, I/H(S,A,S'), P = I/C, H_f, H_b, Delta H,
plus reward. In-sample matched to 5/21 controls (as the submission) and nested (H5 protocol, ALPHA 0.25)
with 1,000 bootstrap resamples. usage: scores | point | boot <chunk> <n> | bootsum"""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import H5_nested as h
O=h.O
STATS={"I":lambda d,e:d[f"I_{e}"],"NMImin":lambda d,e:d[f"I_{e}"]/np.minimum(d.H_SA,d.H_Sn),"NMIjoint":lambda d,e:d[f"I_{e}"]/d.H_SASn,
       "P":lambda d,e:d[f"P_{e}"],"Hf":lambda d,e:d[f"Hf_{e}"],"Hb":lambda d,e:d[f"Hb_{e}"],"dH":lambda d,e:d[f"dH_{e}"]}
LAB={"I":"I(S,A;S') unnormalised","NMImin":"I / min{H(S,A),H(S')}","NMIjoint":"I / H(S,A,S')","P":"P = I / C","Hf":"H_f","Hb":"H_b","dH":"Delta H","reward":"windowed reward"}
m=sys.argv[1]
if m=="scores":
    rows=[]
    for _,r in h.MAP.iterrows():
        k=f"{r.UID}__{r.file.replace('_metrics.csv','')}.csv"; rec=dict(UID=r.UID,cond=r.perturbation,ctrl=r.perturbation=="no_noise (baseline)")
        for e,src in (("pi","H1H2"),("cs","H2cs")):
            d=pd.read_csv(f"{O}/{src}/{k}"); pre=d.window_id<=290; po=d.window_id>=300
            for s,f in STATS.items():
                v=f(d,e); rec[f"{s}_{e}"]=((v[po]-v[pre].mean()).abs()/v[pre].std()).max()
        rec["reward"]=pd.read_csv(f"{O}/PRIMARY_trial_scores.csv").query("UID==@r.UID and cond==@r.perturbation").reward.iloc[0]
        rows.append(rec)
    pd.DataFrame(rows).to_csv(f"{O}/R7_trial_scores.csv",index=False); print("scores written")
    sys.exit()
T=pd.read_csv(f"{O}/R7_trial_scores.csv"); h.T=T
h.METHODS={f"{LAB[s]} [{e}]":[f"{s}_{e}"] for e in ("cs","pi") for s in STATS}|{"windowed reward":["reward"]}
if m=="point":
    P=T[~T.ctrl]; C=T[T.ctrl]; rows=[]
    rr=h.nested(T,0.25)
    for k,cols in h.METHODS.items():
        c=cols[0]; zs=np.sort(C[c].values)[::-1]; zt=(zs[4]+zs[5])/2
        rows.append(dict(statistic=k,insample_z=round(zt,2),insample_det=round(100*(P[c]>zt).mean(),1),
                         nested_det=round(100*rr[k][0],1),nested_ctrl=f"{int(round(rr[k][1]*21))}/21"))
    R=pd.DataFrame(rows); R.to_csv(f"{O}/R7_point.csv",index=False); print(R.to_string(index=False))
elif m=="boot":
    cid,n=int(sys.argv[2]),int(sys.argv[3]); rng=np.random.default_rng(20260917+500+cid); out=[]
    for b in range(n):
        rr=h.nested(T,0.25,rng.choice(h.AG,len(h.AG),replace=True)); out.append({f"{k}|det":v[0] for k,v in rr.items()}|{f"{k}|fa":v[1] for k,v in rr.items()})
    pd.DataFrame(out).to_csv(f"{O}/R7_boot_{cid:02d}.csv",index=False); print("chunk",cid)
elif m=="bootsum":
    B=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{O}/R7_boot_*.csv"))]); print("resamples",len(B))
    for e in ("cs","pi"):
        ref=f"{LAB['P']} [{e}]"
        for k in [x for x in h.METHODS if x.endswith(f"[{e}]")]+["windowed reward"]:
            d=B[f"{k}|det"]*100; line=f"  {k:34s} det [{d.quantile(.025):5.1f},{d.quantile(.975):5.1f}]"
            if k!=ref:
                dd=(B[f"{ref}|det"]-B[f"{k}|det"])*100; ff=(B[f"{ref}|fa"]-B[f"{k}|fa"])*100
                line+=f"  P minus this: det [{dd.quantile(.025):5.1f},{dd.quantile(.975):5.1f}] fa [{ff.quantile(.025):5.1f},{ff.quantile(.975):5.1f}]"
            print(line)
        print()
