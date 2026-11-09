"""E3: episode-boundary sensitivity. The primary Chao-Shen trial scores recomputed from 500-step windows with the
boundary triplet (last step of an episode, whose S' is the auto-reset observation) removed from every window that
contains one (E1 'win_*' arrays). Calibration windows 0-290, evaluation >= 300, |z| with calibration mean/SD
(ddof=1), max over evaluation windows ignoring undefined windows. Nested leave-one-agent-out as in H5/PRIMARY
(ALPHA 0.25): fixed three-of-four (z selected), P alone, k-of-four (k and z selected); reward unchanged."""
import glob, os, numpy as np, pandas as pd
ZG=np.round(np.arange(1.5,30.0,0.05),2)
MAPF=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","config","trial_label_map.csv")
ORIG=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"out","H2_trial_scores_cs.csv"))
M=pd.read_csv(MAPF); lab={f.replace("_metrics.csv",""):p for f,p in zip(M.file,M.perturbation)}; rows=[]
for f in sorted(glob.glob("E1/*.npz")):
    uid,key=os.path.basename(f)[:-4].split("__",1); z=np.load(f); w=np.arange(len(z["win_P"])); cal=w<=290; ev=w>=300
    zz={}
    for c in ["P","Hf","Hb","dH"]:
        v=z["win_"+c]; zz[c]=np.nanmax(np.abs(v[ev]-np.nanmean(v[cal]))/np.nanstd(v[cal],ddof=1))
    s=sorted(zz.values(),reverse=True)
    rows.append(dict(UID=uid,cond=lab[key],ctrl=lab[key]=="no_noise (baseline)",nb1=z["win_P"].size,P_cs=zz["P"],cs1=s[0],cs2=s[1],cs3=s[2],cs4=s[3]))
T=pd.DataFrame(rows).sort_values(["UID","cond"]).reset_index(drop=True)
O=ORIG.sort_values(["UID","cond"]).reset_index(drop=True); assert (O.UID.values==T.UID.values).all() and (O.cond.values==T.cond.values).all()
T["reward"]=O.reward.values
for c in ["P_cs","cs1","cs2","cs3","cs4"]: T[c+"_orig"]=O[c].values
def select(inner,cols):
    C=inner[inner.ctrl]; P=inner[~inner.ctrl]; best=None
    for c in cols:
        cs=C[c].to_numpy(); ps=P[c].to_numpy(); fa=(cs[None,:]>ZG[:,None]).mean(1); det=(ps[None,:]>ZG[:,None]).mean(1)
        ok=np.where(fa<=0.25)[0]
        if not len(ok): continue
        i=ok[np.flatnonzero(det[ok]==det[ok].max())[-1]]; cand=(det[i],ZG[i],c)
        if best is None or cand[0]>best[0] or (cand[0]==best[0] and cand[1]>best[1]): best=cand
    return best[2],best[1]
def nested(T,meth):
    hit=[];fa=[]
    for a in sorted(T.UID.unique()):
        inner=T[T.UID!=a]; held=T[T.UID==a]; c,z=select(inner,meth)
        hit+=list(held[~held.ctrl][c]>z); fa+=list(held[held.ctrl][c]>z)
    return 100*np.mean(hit),f"{sum(fa)}/{len(fa)}"
rows=[]
for name,new,old in [("three-of-four (fixed k, z selected)",["cs3"],["cs3_orig"]),("P alone",["P_cs"],["P_cs_orig"]),
                     ("k-of-four (k and z selected)",["cs1","cs2","cs3","cs4"],["cs1_orig","cs2_orig","cs3_orig","cs4_orig"]),("reward",["reward"],["reward"])]:
    d1,f1=nested(T,new); d0,f0=nested(T,old); rows.append(dict(method=name,with_boundary_triplets=f"{d0:.1f}% ({f0})",boundary_triplets_removed=f"{d1:.1f}% ({f1})"))
R=pd.DataFrame(rows); print(R.to_string(index=False)); R.to_csv("E3_boundary_summary.csv",index=False); T.to_csv("E3_trial_scores.csv",index=False)
for c in ["P_cs","cs3"]:
    d=T[c]-T[c+"_orig"]; print(f"{c}: per-trial score change median {d.median():+.3f}, IQR [{d.quantile(.25):+.3f}, {d.quantile(.75):+.3f}], max |change| {d.abs().max():.3f}; Spearman with original {T[c].corr(T[c+'_orig'],method='spearman'):.4f}")
