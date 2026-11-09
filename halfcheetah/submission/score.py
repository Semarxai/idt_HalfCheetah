import pandas as pd, numpy as np, glob, os, sys
from math import comb
OUT=os.path.join(os.environ.get("IDT_WORK","work"),"qbin/out")
S=["P","Hf","Hb","delta_H"]
def score(mode,k=3,z=3.0,split=300):
    rows=[]
    for f in glob.glob(os.path.join(OUT,f"*__{mode}.csv")):
        uid,pert,_=os.path.basename(f).split("__")
        d=pd.read_csv(f); pre=d[d.window_id<split]; post=d[d.window_id>=split]
        mk=0
        for s in S:
            mu,sd=pre[s].mean(),pre[s].std(ddof=1)
            if sd and not np.isnan(sd) and (np.abs(post[s].values-mu)>z*sd).any(): mk+=1
        mu,sd=pre.mean_reward.mean(),pre.mean_reward.std(ddof=1)
        rows.append(dict(uid=uid,cond=pert,mon=mk>=k,
                         rw=bool((np.abs(post.mean_reward.values-mu)>z*sd).any())))
    R=pd.DataFrame(rows); P=R[R.cond!="no_noise_baseline"]; C=R[R.cond=="no_noise_baseline"]
    b=int((P.mon&~P.rw).sum()); c=int((~P.mon&P.rw).sum()); n=b+c
    p=min(1,2*sum(comb(n,i) for i in range(max(b,c),n+1))/2**n) if n else 1
    return dict(n=len(P),det=100*P.mon.mean(),rw=100*P.rw.mean(),
                ctl=int(C.mon.sum()),ctlrw=int(C.rw.sum()),nc=len(C),b=b,c=c,p=p)
for mode in sys.argv[1:]:
    r=score(mode)
    print(f"{mode:11s} n={r['n']}  detection {r['det']:.1f}%  reward {r['rw']:.1f}%  "
          f"controls {r['ctl']}/{r['nc']} vs {r['ctlrw']}/{r['nc']}  "
          f"discordant {r['b']}v{r['c']}  p={r['p']:.2g}")
