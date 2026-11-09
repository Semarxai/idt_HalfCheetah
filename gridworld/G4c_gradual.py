"""G4c: gradual faults. Same four mechanisms as G4b, ramped linearly from zero at onset (step 15,000) to the
G4b magnitude at the end of the trial (step 50,000): lambda(t) = (t - 15,000) / 35,000.
 agent-side stochastic : executed action random with probability 0.1*lambda
 agent-side systematic : sensor offset applied to the observation at step t with probability lambda
                         (one draw per step, so S_{t+1} and the logged S'_t agree)
 env-side stochastic   : slip 0.05 -> 0.05 + 0.10*lambda outside the slip zone (kernel mixture, linear in slip)
 env-side systematic   : wind south in the right room with probability 0.2*lambda
Protocol, windows, Chao-Shen readings and thresholds (matched to 5/21 controls) as G4b. Outputs: latched
alarm rate by the end, time from onset to the first alarm, and the fault magnitude (lambda) at that time."""
import sys, hashlib, numpy as np, pandas as pd, gw_core as g, estimators as e
from G4d_benign_variation import mmd_delta, STEPS, ONSET, STR
STATS=["P","Hf","Hb","dH","HS","MMD","reward"]
def lam(t): return 0.0 if t<ONSET else (t-ONSET)/(STEPS-ONSET)
def rollout(T0,T1,st,pi,n,rng,kind,obs=None,wind=None):
    cpi=np.cumsum(pi,1); c0=np.cumsum(T0,2); c1=np.cumsum(T1,2); cs=np.cumsum(st); u=rng.random((n+1,5))
    flag=np.array([u[t,4]<lam(t) for t in range(n+1)]) if kind=="obs" else np.zeros(n+1,bool)
    S=np.empty(n,int);A=np.empty(n,int);Sn=np.empty(n,int); s=int(np.searchsorted(cs,rng.random()))
    for t in range(n):
        L=lam(t); o=obs[s] if flag[t] else s
        a=int(np.searchsorted(cpi[o],u[t,0]))
        cT=c1 if (kind=="mix" and u[t,3]<L) else c0
        sn=int(np.searchsorted(cT[s,a],u[t,1]))
        if kind=="wind" and wind[sn]>=0 and u[t,3]<0.2*L: sn=int(wind[sn])
        on=obs[sn] if flag[t+1] else sn
        S[t],A[t],Sn[t]=o,a,on
        s=int(np.searchsorted(cs,u[t,2])) if sn==g.G else sn
    return S,A,Sn
def trial(T0,T1,st,pi,W,rng,kind,obs=None,wind=None):
    S,A,Sn=rollout(T0,T1,st,pi,STEPS,rng,kind,obs,wind); sym=(S*g.nA+A)*g.nS+Sn
    ru,rc=np.unique(sym[:7500],return_counts=True); refc=dict(zip(ru,rc)); ref_sq=(rc*(rc-1)).sum()/(7500*7499)
    nw=(STEPS-W)//STR+1; rows=[]
    for i in range(nw):
        sl=slice(i*STR,i*STR+W); q=e.window_quantities(S[sl],A[sl],Sn[sl],g.nS,g.nA,"cs")
        rows.append(dict(P=q["P"],Hf=q["Hf"],Hb=q["Hb"],dH=q["dH"],HS=e.H_plugin(e.counts(S[sl])),
                         MMD=mmd_delta(sym[sl],refc,7500,ref_sq) if i*STR>=7500 else np.nan,reward=(Sn[sl]==g.G).mean()))
    D=pd.DataFrame(rows); wid=np.arange(nw); cal=wid*STR+W<=ONSET; ev=wid*STR>=ONSET; calm=cal&(wid*STR>=7500)
    Z={}
    for c in STATS:
        v=D[c].to_numpy(); m=calm if c=="MMD" else cal; mu,sd=np.nanmean(v[m]),np.nanstd(v[m],ddof=1)
        z=(v[ev]-mu)/sd if c=="MMD" else np.abs(v[ev]-mu)/sd
        Z[c]=np.fmax.accumulate(np.nan_to_num(z,nan=-np.inf))          # running (latched) max over evaluation windows
    four=np.sort(np.vstack([Z[c] for c in ["P","Hf","Hb","dH"]]),0)[::-1]; Z["three_of_four"]=four[2]
    return {k:Z[k] for k in ["three_of_four","P","HS","MMD","reward"]}, (wid[ev]*STR+W-ONSET)   # steps after onset at window end
def run(N,eps,W,tag,seeds):
    g.setup(N); pi=g.policy(eps); T0=g.kernel(g.slip_map()); st=g.start_dist()
    obs=np.arange(g.nS)
    for c,i in g.idx.items():
        if c[0]<g.WALL_X-1 and (c[0]+1,c[1]) in g.idx: obs[i]=g.idx[(c[0]+1,c[1])]
    wind=-np.ones(g.nS,int)
    for c,i in g.idx.items():
        if c[0]>g.WALL_X and (c[0],c[1]+1) in g.idx and c!=g.GOAL: wind[i]=g.idx[(c[0],c[1]+1)]
    conds={"control":dict(kind="none",T1=T0),
           "agent-side stochastic: action noise 0 -> 10%":dict(kind="mix",T1=0.9*T0+0.1*T0.mean(1,keepdims=True)),
           "agent-side systematic: sensor offset 0 -> 100%":dict(kind="obs",T1=T0,obs=obs),
           "env-side stochastic: slip 0.05 -> 0.15":dict(kind="mix",T1=g.kernel(np.where(g.slip_map()>0.1,0.30,0.15))),
           "env-side systematic: wind 0 -> 0.2":dict(kind="wind",T1=T0,wind=wind)}
    out={}
    for name,kw in conds.items():
        for sd in seeds:
            rng=np.random.default_rng(sd*1000+int(hashlib.md5(name.encode()).hexdigest(),16)%997+500)
            Z,tt=trial(T0,kw["T1"],st,pi,W,rng,kw["kind"],kw.get("obs"),kw.get("wind"))
            for k,v in Z.items(): out[f"{name}|{sd}|{k}"]=v
    out["t_after_onset"]=tt; out["world"]=np.array(tag)
    return out
if __name__=="__main__":
    part=sys.argv[1]; seeds=list(range(1,22))
    R=run(7,0.2,500,"7x7 W500",seeds) if part=="7" else run(27,0.4,200,"27x27 W200",seeds)
    np.savez_compressed(f"G4c_running_{part}.npz",**R); print("done",part)
