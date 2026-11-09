import os, glob
import pandas as pd, numpy as np
from math import comb
BASE=os.environ.get("IDT_DATA_ROOT","data")
MAP=pd.read_csv(os.path.join(os.environ.get("IDT_WORK","work"),"map_resolved.csv"))
files={}
for p in glob.glob(os.path.join(BASE,"RL_sep_analysis/metrics","**","*_metrics.csv"),recursive=True):
    files.setdefault(os.path.basename(p),[]).append(p)
MAP["path"]=MAP.file.map(lambda b: files.get(b,[None])[0]); MAP=MAP[MAP.path.notna()]
cache={r.path: pd.read_csv(r.path) for _,r in MAP.iterrows()}
P=MAP[MAP.perturbation!="no_noise (baseline)"]
S=["P","Hf","Hb","delta_H"]
def trial(path,k=3,Tc=300,split=300,z=3.0):
    d=cache[path]; pre=d[d.window_id<Tc]; post=d[d.window_id>=split]
    wid=post.window_id.values
    M=[]
    for s in S:
        mu,sd=pre[s].mean(),pre[s].std(ddof=1)
        M.append((np.abs(post[s].values-mu)>z*sd) if (sd and not np.isnan(sd)) else np.zeros(len(post),bool))
    M=np.array(M)
    cum=np.cumsum(M,axis=1)>0           # has stat m deviated by window t
    cnt=cum.sum(0)
    hit=cnt>=k
    mon_t = wid[hit.argmax()] if hit.any() else None
    mu,sd=pre.mean_reward.mean(),pre.mean_reward.std(ddof=1)
    rh=np.abs(post.mean_reward.values-mu)>z*sd
    rw_t = wid[rh.argmax()] if rh.any() else None
    return (M.any(1).sum()>=k), mon_t, bool(rh.any()), rw_t
rows=[]
for _,r in P.iterrows():
    md,mt,rd,rt=trial(r.path)
    rows.append(dict(UID=r.UID,algo="sac" if r.UID.startswith("sac") else "ppo",
                     cond=r.perturbation,mon=md,mt=mt,rw=rd,rt=rt))
R=pd.DataFrame(rows); R.to_csv(os.path.join(os.environ.get("IDT_WORK","work"),"lead.csv"),index=False)
b=int((R.mon&~R.rw).sum()); c=int((~R.mon&R.rw).sum()); n=b+c
p=min(1,2*sum(comb(n,i) for i in range(max(b,c),n+1))/2**n)
print(f"detect {100*R.mon.mean():.1f}%  reward {100*R.rw.mean():.1f}%  discordant {b} vs {c}  McNemar p={p:.2g}")
B=R[R.mon&R.rw].copy(); B["lead"]=B.rt-B.mt
print(f"\nboth detected: n={len(B)}  median lead={B.lead.median()}  monitor earlier in {int((B.lead>0).sum())} of {len(B)}")
for a in ("sac","ppo"):
    Ba=B[B.algo==a]; e=int((Ba.lead>0).sum()); n2=len(Ba)
    pv=min(1,2*sum(comb(n2,i) for i in range(max(e,n2-e),n2+1))/2**n2)
    print(f"  {a}: n={n2} median={Ba.lead.median()} earlier in {e} of {n2}  two-sided sign p={pv:.3f}  one-sided={pv/2:.3f}")
    print(f"     ties(lead==0)={int((Ba.lead==0).sum())}  strictly-earlier={int((Ba.lead>0).sum())}  later={int((Ba.lead<0).sum())}")
