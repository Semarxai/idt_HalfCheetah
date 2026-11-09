"""E5 evaluation: KL trial scores (one-sided z, calibration windows 150-290 as for MMD, evaluation >= 300, max)
merged with the primary Chao-Shen scores; nested leave-one-agent-out (H5 protocol) point (ALPHA 0.25) and curve."""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import H5_nested as h
O=h.O; T=pd.read_csv(f"{O}/PRIMARY_trial_scores.csv")
lab={r.file.replace("_metrics.csv",""):r.perturbation for _,r in h.MAP.iterrows()}; rows=[]
for f in glob.glob(f"{O}/E5/*.npz"):
    uid,key=os.path.basename(f)[:-4].split("__",1); z=np.load(f); w=np.arange(len(z["kl_joint"])); cal=(w>=150)&(w<=290); ev=w>=300; r=dict(UID=uid,cond=lab[key])
    for c in ["kl_joint","kl_S"]:
        v=z[c]; r[c]=np.nanmax((v[ev]-np.nanmean(v[cal]))/np.nanstd(v[cal],ddof=1))
    rows.append(r)
K=pd.DataFrame(rows); T=T.merge(K,on=["UID","cond"],how="left"); assert T.kl_joint.notna().all()
T.to_csv(f"{O}/E5_trial_scores.csv",index=False); h.T=T
h.METHODS={"three-of-four (CS)":["cs3"],"P alone (CS)":["P_cs"],"KL joint (S,A,S')":["kl_joint"],"KL S":["kl_S"],"MMD":["MMD"],"H(S)":["HS"],"reward":["reward"]}
rr=h.nested(T,0.25)
for m,(d,f,ch) in rr.items(): print(f"  {m:20s} det {100*d:5.1f}%  ctrl {int(round(f*21))}/21  {pd.Series([f'{c}@{z}' for c,z in ch]).value_counts().head(2).to_dict()}")
cur=[]
for al in [0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50]:
    r2=h.nested(T,al); cur.append(dict(alpha=al,**{m:round(100*r2[m][0],1) for m in h.METHODS}))
C=pd.DataFrame(cur); C.to_csv(f"{O}/E5_curve.csv",index=False); print(C.to_string(index=False))
P=T[~T.ctrl]; print((P.groupby("cond")[["kl_joint","kl_S"]].median()).round(2).to_string())
