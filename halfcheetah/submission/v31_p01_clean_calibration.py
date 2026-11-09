import pandas as pd, numpy as np, glob, os
from math import comb
BASE=os.environ.get("IDT_DATA_ROOT","data")
MAP=pd.read_csv(os.path.join(BASE,"redesign/02_config/trial_label_map.csv"))
MAP=MAP[~MAP.UID.isin({"ppo15","ppo18","ppo19","ppo21"})]
files={}
for p in glob.glob(os.path.join(BASE,"RL_sep_analysis/metrics","**","*_metrics.csv"),recursive=True):
    files.setdefault(os.path.basename(p),[]).append(p)
MAP["path"]=MAP.file.map(lambda b: files.get(b,[None])[0])
S=["P","Hf","Hb","delta_H"]
def build(Tc):
    rows=[]
    for _,r in MAP.iterrows():
        d=pd.read_csv(r.path); pre=d[d.window_id<Tc]; post=d[d.window_id>=300]
        mk=sum(1 for s in S if pre[s].std(ddof=1)>0 and
               (np.abs(post[s].values-pre[s].mean())>3*pre[s].std(ddof=1)).any())
        mu,sd=pre.mean_reward.mean(),pre.mean_reward.std(ddof=1)
        rows.append(dict(UID=r.UID,cond=r.perturbation,nmark=mk,
                         rw=bool((np.abs(post.mean_reward.values-mu)>3*sd).any())))
    return pd.DataFrame(rows)
def mcn(b,c):
    n=b+c
    return min(1,2*sum(comb(n,i) for i in range(max(b,c),n+1))/2**n) if n else 1.0
def report(R,k=3,label=""):
    R=R.copy(); R["mon"]=R.nmark>=k
    P=R[R.cond!="no_noise (baseline)"]; C=R[R.cond=="no_noise (baseline)"]
    b=int((P.mon&~P.rw).sum()); c=int((~P.mon&P.rw).sum())
    rng=np.random.default_rng(20260917); ag=sorted(R.UID.unique()); dif=[]
    for _ in range(10000):
        pick=rng.choice(ag,len(ag),True)
        sub=pd.concat([P[P.UID==a] for a in pick]); dif.append(sub.mon.mean()-sub.rw.mean())
    dif=np.array(dif)
    print(f"{label}  det {100*P.mon.mean():.1f}%  rw {100*P.rw.mean():.1f}%  "
          f"ctrl {int(C.mon.sum())}/{len(C)} vs {int(C.rw.sum())}/{len(C)}  "
          f"disc {b}v{c}  p={mcn(b,c):.2g}  "
          f"clustered {100*(P.mon.mean()-P.rw.mean()):.1f} [{100*np.percentile(dif,2.5):.1f},{100*np.percentile(dif,97.5):.1f}]")
    return R,P,C
print("=== primary: calibration windows 0-290 (wholly pre-injection) ===")
R290,P290,C290=report(build(291),3,"k=3 ")
for k in (1,2,4): report(build(291),k,f"k={k} ")
print("\n=== v30 published: calibration 0-299 ===")
report(build(300),3,"k=3 ")
R290.to_csv(os.path.join(os.environ.get("IDT_WORK","work"),"marks_290.csv"),index=False)
print("\nper-condition (0-290 calibration):")
P290=P290.copy()
for c,g in P290.groupby("cond"):
    print(f"  {c:22s} monitor {100*g.mon.mean():5.1f}%  reward {100*g.rw.mean():5.1f}%  margin {100*(g.mon.mean()-g.rw.mean()):+5.1f}")
