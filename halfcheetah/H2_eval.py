"""H2 evaluation: Chao-Shen statistics vs plug-in - levels, bounds, in-sample and nested detection."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import H5_nested as h
O=h.O; T=h.T.copy(); lv=[]; viol=dict(P_gt_half=0,Hf_neg=0,Hb_neg=0,I_neg=0,windows=0)
add={c:[] for c in ["cs1","cs2","cs3","cs4","P_cs"]}
for _,r in h.MAP.iterrows():
    k=f"{r.UID}__{r.file.replace('_metrics.csv','')}.csv"
    c=pd.read_csv(f"{O}/H2cs/{k}"); p=pd.read_csv(f"{O}/H1H2/{k}")
    viol["windows"]+=len(c); viol["P_gt_half"]+=int((c.P_cs>0.5).sum()); viol["Hf_neg"]+=int((c.Hf_cs<0).sum())
    viol["Hb_neg"]+=int((c.Hb_cs<0).sum()); viol["I_neg"]+=int((c.I_cs<0).sum())
    pre=c[c.window_id<=290]; po=c[c.window_id>=300]; pp=p[p.window_id<=290]
    z=lambda col:((po[col]-pre[col].mean()).abs()/pre[col].std()).max()
    zz=sorted([z("P_cs"),z("Hf_cs"),z("Hb_cs"),z("dH_cs")],reverse=True)
    for i in range(4): add[f"cs{i+1}"].append(zz[i])
    add["P_cs"].append(z("P_cs"))
    lv.append(dict(P_pi=pp.P_pi.mean(),P_cs=pre.P_cs.mean(),Hf_pi=pp.Hf_pi.mean(),Hf_cs=pre.Hf_cs.mean(),
                   Hb_pi=pp.Hb_pi.mean(),Hb_cs=pre.Hb_cs.mean(),dH_pi=pp.dH_pi.mean(),dH_cs=pre.dH_cs.mean(),
                   I_pi=pp.I_pi.mean(),I_cs=pre.I_cs.mean()))
for k_,v in add.items(): T[k_]=v
T.to_csv(f"{O}/H2_trial_scores_cs.csv",index=False)
print("healthy calibration levels, mean over 189 retained trials (plug-in vs Chao-Shen):")
print(pd.DataFrame(lv).mean().round(4).to_string())
print("\nbound checks over all windows:",viol)
P=T[~T.ctrl]; C=T[T.ctrl]
print("\nin-sample, ±3 sigma, latched:")
for lab,col in [("plug-in three-of-four","cons3"),("Chao-Shen three-of-four","cs3"),("plug-in four-of-four","cons4"),("Chao-Shen four-of-four","cs4")]:
    print(f"  {lab:26s} det {100*(P[col]>3).mean():5.1f}%  ctrl {(C[col]>3).sum()}/21")
print("in-sample, single statistic at 5/21 control alarms:")
for col in ["P","P_cs"]:
    zs=np.sort(C[col].values)[::-1]; zt=(zs[4]+zs[5])/2; print(f"  {col:5s} z={zt:.2f} det {100*(P[col]>zt).mean():5.1f}%")
h.T=T; h.METHODS={"consensus plug-in":["cons1","cons2","cons3","cons4"],"consensus Chao-Shen":["cs1","cs2","cs3","cs4"],
                  "P plug-in":["P"],"P Chao-Shen":["P_cs"],"reward":["reward"]}
print("\nnested leave-one-agent-out (same protocol as H5):")
for al in (0.10,0.25):
    rr=h.nested(T,al)
    for m,(d,f,ch) in rr.items():
        cc=pd.Series([f"{c}@{z}" for c,z in ch]).value_counts().head(2).to_dict()
        print(f"  ALPHA {al:.2f}  {m:20s} det {100*d:5.1f}%  ctrl {int(round(f*21))}/21  {cc}")
