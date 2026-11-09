import os, numpy as np, pandas as pd
O=os.path.join(os.path.dirname(os.path.abspath(__file__)),"out"); BASE=os.environ.get("IDT_DATA_ROOT","data")
MAP=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","config","trial_label_map.csv")); MAP=MAP[~MAP.UID.isin({"ppo15","ppo18","ppo19","ppo21"})]
Q=["P","termS","termA","Cs","Cact","wS","wA","HA_S"]; rows=[]; lv=[]; idmax={"pi":0,"cs":0}
for _,r in MAP.iterrows():
    d=pd.read_csv(f"{O}/H8/{r.UID}__{r.file.replace('_metrics.csv','')}.csv"); pre=d[d.window_id<=290]; po=d[d.window_id>=300]
    for k in ("pi","cs"):
        idmax[k]=max(idmax[k],(d[f"P_{k}"]-d[f"termS_{k}"]-d[f"termA_{k}"]).abs().max())
        rec=dict(UID=r.UID,cond=r.perturbation,est=k)
        for q in Q:
            c=f"{q}_{k}"; rec[f"lvl_{q}"]=pre[c].mean(); rec[f"d_{q}"]=po[c].mean()-pre[c].mean(); rec[f"z_{q}"]=(po[c].mean()-pre[c].mean())/pre[c].std()
        rows.append(rec)
D=pd.DataFrame(rows); D.to_csv(f"{O}/H8_trial_changes.csv",index=False)
print("identity check, max |P - termS - termA| over all windows:",{k:float(v) for k,v in idmax.items()})
for k,lab in (("cs","Chao-Shen (primary)"),("pi","plug-in")):
    X=D[D.est==k]
    print(f"\n######## {lab}")
    print("healthy calibration levels (mean over 189 trials):"); print(X[[f"lvl_{q}" for q in Q]].mean().round(4).to_string())
    P=X[X.cond!="no_noise (baseline)"]
    g=P.groupby("cond")
    print("\nmean change post - pre (x1000) by condition:")
    print((g[["d_P","d_termS","d_termA","d_Cs","d_Cact","d_HA_S"]].mean()*1000).round(2).to_string())
    print("\nmean standardised change (post - pre)/SD_pre by condition:")
    print(g[["z_P","z_termS","z_termA","z_Cs","z_Cact"]].mean().round(2).to_string())
    print("\nagents (of 21) in which the quantity falls / P rises:")
    t=pd.DataFrame({"P falls":g.d_P.apply(lambda s:(s<0).sum()),"termS falls":g.d_termS.apply(lambda s:(s<0).sum()),
                    "Cs falls":g.d_Cs.apply(lambda s:(s<0).sum()),"termA rises":g.d_termA.apply(lambda s:(s>0).sum()),"Cact rises":g.d_Cact.apply(lambda s:(s>0).sum())})
    print(t.to_string())
    print("\nshare of |dP| carried by each term (mean over trials of |d_term|/(|d_termS|+|d_termA|)):")
    sh=P.assign(sS=P.d_termS.abs()/(P.d_termS.abs()+P.d_termA.abs())).groupby("cond").sS.mean()
    print(pd.DataFrame({"state term":sh.round(2),"action term":(1-sh).round(2)}).to_string())
    C=X[X.cond=="no_noise (baseline)"]; print("\ncontrols: mean change x1000 P %.2f termS %.2f termA %.2f"%(C.d_P.mean()*1000,C.d_termS.mean()*1000,C.d_termA.mean()*1000))
