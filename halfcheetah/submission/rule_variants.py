import os, glob
import pandas as pd, numpy as np
BASE=os.environ.get("IDT_DATA_ROOT","data")
MAP=pd.read_csv(os.path.join(os.environ.get("IDT_WORK","work"),"map_resolved.csv"))
files={}
for p in glob.glob(os.path.join(BASE,"RL_sep_analysis/metrics","**","*_metrics.csv"),recursive=True):
    files.setdefault(os.path.basename(p),[]).append(p)
MAP["path"]=MAP.file.map(lambda b: files.get(b,[None])[0]); MAP=MAP[MAP.path.notna()]
cache={r.path: pd.read_csv(r.path) for _,r in MAP.iterrows()}
P=MAP[MAP.perturbation!="no_noise (baseline)"]; C=MAP[MAP.perturbation=="no_noise (baseline)"]
S=["P","Hf","Hb","delta_H"]
def flags(path,Tc=300,split=300,z=3.0):
    d=cache[path]; pre=d[d.window_id<Tc]; post=d[d.window_id>=split]
    M=[]
    for s in S:
        mu,sd=pre[s].mean(),pre[s].std(ddof=1)
        M.append((np.abs(post[s].values-mu)>z*sd) if (sd and not np.isnan(sd)) else np.zeros(len(post),bool))
    return np.array(M), post.window_id.values     # 4 x nwin
def rule_anywindow(path,k):        # k statistics deviate in the SAME window
    M,_=flags(path); return bool((M.sum(0)>=k).any())
def rule_anytime(path,k):          # k statistics each deviate at SOME point
    M,_=flags(path); return bool(M.any(1).sum()>=k)
for name,fn in [("same-window",rule_anywindow),("ever-during-post",rule_anytime)]:
    print(name)
    for k in (1,2,3,4):
        det=100*np.mean([fn(r.path,k) for _,r in P.iterrows()])
        ctl=[fn(r.path,k) for _,r in C.iterrows()]
        print(f"   k={k}  det={det:5.1f}%   ctrl={100*np.mean(ctl):5.1f}%  ({int(np.sum(ctl))}/21)")
