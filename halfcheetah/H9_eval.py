"""H9 evaluation: in-sample and nested (H5 protocol) detection per bin count x window length.
usage: python H9_eval.py point | boot <bins> <W> <n> | bootsum"""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import H5_nested as h
O=h.O; D=pd.concat([pd.read_csv(f) for f in glob.glob(f"{O}/H9/*.csv")]); D["ctrl"]=D.cond=="no_noise (baseline)"
def table(sub):
    T=sub.copy()
    for e,lab in (("pi","cons"),("cs","cs")):
        Z=np.sort(T[[f"z_P_{e}",f"z_Hf_{e}",f"z_Hb_{e}",f"z_dH_{e}"]].to_numpy(),1)[:,::-1]
        for i in range(4): T[f"{lab}{i+1}"]=Z[:,i]
    return T.rename(columns={"z_P_cs":"P_cs","z_reward":"reward"})
M={"consensus Chao-Shen":["cs1","cs2","cs3","cs4"],"P Chao-Shen":["P_cs"],"consensus plug-in":["cons1","cons2","cons3","cons4"],"reward":["reward"]}
h.METHODS=M
if sys.argv[1]=="point":
    rows=[]
    for (k,W),sub in D.groupby(["bins","W"]):
        T=table(sub); P=T[~T.ctrl]; C=T[T.ctrl]
        rec=dict(bins=k,W=W,cal_windows=int(sub.n_cal.iloc[0]),
                 ins_cons3_pi=f"{100*(P.cons3>3).mean():.1f}% {int((C.cons3>3).sum())}/21",
                 ins_cons3_cs=f"{100*(P.cs3>3).mean():.1f}% {int((C.cs3>3).sum())}/21",
                 ins_reward=f"{100*(P.reward>3).mean():.1f}% {int((C.reward>3).sum())}/21")
        for al in (0.10,0.25):
            rr=h.nested(T,al)
            for m,(d,f,ch) in rr.items(): rec[f"{m} @{al:.2f}"]=f"{100*d:.1f}% {int(round(f*21))}/21"
        rows.append(rec)
    R=pd.DataFrame(rows); R.to_csv(f"{O}/H9_summary.csv",index=False)
    pd.set_option("display.width",300); pd.set_option("display.max_columns",30)
    print(R[["bins","W","cal_windows","ins_cons3_pi","ins_cons3_cs","ins_reward"]].to_string(index=False))
    print(R[["bins","W"]+[c for c in R.columns if "@0.25" in c]].to_string(index=False))
    print(R[["bins","W"]+[c for c in R.columns if "@0.10" in c]].to_string(index=False))
elif sys.argv[1]=="boot":
    k,W,n=int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]); T=table(D[(D.bins==k)&(D.W==W)]); rng=np.random.default_rng(20260917+300+k*10000+W); out=[]
    for b in range(n):
        rr=h.nested(T,0.25,rng.choice(h.AG,len(h.AG),replace=True)); out.append({f"{m}|det":rr[m][0] for m in M}|{f"{m}|fa":rr[m][1] for m in M})
    pd.DataFrame(out).to_csv(f"{O}/H9_boot_b{k}_W{W}.csv",index=False); print("done",k,W)
elif sys.argv[1]=="bootsum":
    for f in sorted(glob.glob(f"{O}/H9_boot_*.csv")):
        B=pd.read_csv(f); d=(B["consensus Chao-Shen|det"]-B["reward|det"])*100; p=(B["P Chao-Shen|det"]-B["reward|det"])*100
        print(f"{os.path.basename(f)[8:-4]:10s} n={len(B)}  consensus CS - reward [{d.quantile(.025):5.1f},{d.quantile(.975):5.1f}]   P CS - reward [{p.quantile(.025):5.1f},{p.quantile(.975):5.1f}]")
