"""H10a: time to first alarm on archived exposure, at the nested held-out operating point (ALPHA 0.25).
For each held-out agent the operating point (statistic set, k, z) is selected on the other agents
exactly as in H5; the held-out agent's windowed |z| series then give the first alarm window (latched
rule: alarm when k statistics have each crossed z at least once). Controls -> time to first false
alarm (survival over ~690 evaluation windows = 34.5 episodes); perturbation trials -> delay from the
start of evaluation (window 300; onset at step 15,000 lies in window 291-300)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import H5_nested as h
O=h.O; T=pd.read_csv(f"{O}/H2_trial_scores_cs.csv"); h.T=T
SETS={"three-of-four Chao-Shen":("cs",["P_cs","Hf_cs","Hb_cs","dH_cs"],["cs3"]),
      "consensus Chao-Shen":("cs",["P_cs","Hf_cs","Hb_cs","dH_cs"],["cs1","cs2","cs3","cs4"]),
      "P Chao-Shen":("cs",["P_cs"],["P_cs"]),
      "consensus plug-in":("pi",["P_pi","Hf_pi","Hb_pi","dH_pi"],["cons1","cons2","cons3","cons4"]),
      "reward":("pi",["mean_reward"],["reward"]),"H(S)":("pi",["H_S"],["HS"]),"MMD":("mmd",["mmd2"],["MMD"])}
series={}
for _,r in h.MAP.iterrows():
    k=f"{r.UID}__{r.file.replace('_metrics.csv','')}.csv"
    src={"pi":pd.read_csv(f"{O}/H1H2/{k}"),"cs":pd.read_csv(f"{O}/H2cs/{k}"),"mmd":pd.read_csv(f"{O}/H3/{k}")}
    for name,(s,cols,_) in SETS.items():
        d=src[s]; pre=d[d.window_id<=290]; po=d[d.window_id>=300].sort_values("window_id")
        Z=np.vstack([((po[c]-pre[c].mean())/pre[c].std()).to_numpy() for c in cols])
        if s!="mmd": Z=np.abs(Z)
        series[(r.UID,r.perturbation,name)]=Z
def first_alarm(Z,k,z):
    fc=[np.argmax(row>z) if (row>z).any() else np.inf for row in Z]; fc=sorted(fc); return fc[k-1] if k<=len(fc) else np.inf
rows=[]
for a in h.AG:
    inner=T[T.UID!=a]
    for name,(s,cols,scols) in SETS.items():
        col,zt=h.select(inner,scols,0.25); k=int(col[-1]) if (col[-1].isdigit() and len(cols)>1) else 1
        for cond in T[T.UID==a].cond:
            t=first_alarm(series[(a,cond,name)],k,zt)
            rows.append(dict(UID=a,cond=cond,method=name,k=k,z=zt,first_window=t))
D=pd.DataFrame(rows); D.to_csv(f"{O}/H10a_first_alarm.csv",index=False)
NW=691; EP=lambda w: w*50/1000
print("controls: share with a false alarm by evaluation episode (1 episode = 20 windows); median time to first false alarm")
C=D[D.cond=="no_noise (baseline)"]
for m,g in C.groupby("method",sort=False):
    t=g.first_window.to_numpy(); line="  ".join(f"ep{e:>4}: {100*np.mean(t<e*20):5.1f}%" for e in (1,5,10,20,34.5))
    alarmed=t[np.isfinite(t)]
    print(f"  {m:20s} {line}   alarmed {len(alarmed)}/21, median first alarm {EP(np.median(alarmed)) if len(alarmed) else float('nan'):.1f} ep")
print("\nperturbation trials: detected share and median delay from evaluation start (episodes) by method")
Pp=D[D.cond!="no_noise (baseline)"]
for m,g in Pp.groupby("method",sort=False):
    t=g.first_window.to_numpy(); det=np.isfinite(t)
    print(f"  {m:20s} detected {100*det.mean():5.1f}%  median delay {EP(np.median(t[det])):.2f} ep  IQR [{EP(np.percentile(t[det],25)):.2f}, {EP(np.percentile(t[det],75)):.2f}]  by ep1 {100*np.mean(t<20):.1f}%  by ep5 {100*np.mean(t<100):.1f}%")
print("\nmedian delay (episodes) by condition, detected trials only:")
x=Pp[np.isfinite(Pp.first_window)].assign(ep=lambda d:d.first_window*50/1000)
print(x.pivot_table(index="cond",columns="method",values="ep",aggfunc="median").round(2).to_string())
