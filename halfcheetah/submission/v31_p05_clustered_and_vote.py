import pandas as pd, numpy as np, os
from math import comb
R=pd.read_csv(os.path.join(os.environ.get("IDT_WORK","work"),"marks_290.csv"))
P=R[R.cond!="no_noise (baseline)"].copy(); C=R[R.cond=="no_noise (baseline)"].copy()
rng=np.random.default_rng(20260917); ag=sorted(R.UID.unique())
def boot(df,col,k=None):
    out=[]
    for _ in range(10000):
        pick=rng.choice(ag,len(ag),True)
        sub=pd.concat([df[df.UID==a] for a in pick])
        v=(sub.nmark>=k) if k else sub[col]
        out.append(v.mean())
    a=np.array(out); return 100*a.mean(),100*np.percentile(a,2.5),100*np.percentile(a,97.5)
m,lo,hi=boot(P,None,3); print(f"monitor detection  {m:.1f}%  95% CI [{lo:.1f},{hi:.1f}]")
m,lo,hi=boot(P,"rw");   print(f"reward  detection  {m:.1f}%  95% CI [{lo:.1f},{hi:.1f}]")
m,lo,hi=boot(C,None,3); print(f"monitor controls   {m:.1f}%  95% CI [{lo:.1f},{hi:.1f}]")
m,lo,hi=boot(C,"rw");   print(f"reward  controls   {m:.1f}%  95% CI [{lo:.1f},{hi:.1f}]")
print("\n(whole-agent bootstrap, 10,000 resamples, numpy default_rng seed 20260917)")
print("\n=== P1.3 vote audit, calibration 0-290 ===")
def rate(df,cols,k):
    return 100*np.mean(df[cols].sum(1)>=k) if cols else None
# rebuild per-statistic marks
import glob
BASE=os.environ.get("IDT_DATA_ROOT","data")
MAP=pd.read_csv(os.path.join(BASE,"redesign/02_config/trial_label_map.csv"))
MAP=MAP[~MAP.UID.isin({"ppo15","ppo18","ppo19","ppo21"})]
files={}
for p in glob.glob(os.path.join(BASE,"RL_sep_analysis/metrics","**","*_metrics.csv"),recursive=True):
    files.setdefault(os.path.basename(p),[]).append(p)
MAP["path"]=MAP.file.map(lambda b: files.get(b,[None])[0])
S=["P","Hf","Hb","delta_H"]; rows=[]
for _,r in MAP.iterrows():
    d=pd.read_csv(r.path); pre=d[d.window_id<291]; post=d[d.window_id>=300]
    e={s:(bool((np.abs(post[s].values-pre[s].mean())>3*pre[s].std(ddof=1)).any()) if pre[s].std(ddof=1)>0 else False) for s in S}
    rows.append(dict(UID=r.UID,cond=r.perturbation,**e))
M=pd.DataFrame(rows); MP=M[M.cond!="no_noise (baseline)"]; MC=M[M.cond=="no_noise (baseline)"]
for lbl,cols,k in [("three-of-four {P,Hf,Hb,dH}",S,3),
                   ("two-of-three {P,Hf,Hb}",["P","Hf","Hb"],2),
                   ("three-of-three {P,Hf,Hb}",["P","Hf","Hb"],3)]:
    print(f"  {lbl:28s} detection {rate(MP,cols,k):5.1f}%   controls {int((MC[cols].sum(1)>=k).sum())}/21")
both=MP[(MP.Hf)&(MP.Hb)]; nei=MP[(~MP.Hf)&(~MP.Hb)]
print(f"  Hf and Hb both marked in {len(both)} trials; dH also marked in {int(both.delta_H.sum())}")
print(f"  neither marked in {len(nei)}; dH marked anyway in {int(nei.delta_H.sum())}")
det=MP[MP[S].sum(1)>=3]
print(f"  detected without P marked: {int((~det.P).sum())} of {len(det)}")
