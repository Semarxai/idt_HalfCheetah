"""G4b: entry point x mechanism - all four cells, under the G4d trial protocol (50,000 steps, onset 15,000).
 agent-side stochastic : 10% of executed actions replaced by a random action (logged action = commanded)
 agent-side systematic : sensor offset - in the left room the observed cell is the one to its east (when that
                         cell is in the room); the policy acts on, and the log records, the observed cell
 env-side stochastic   : slip probability 0.05 -> 0.15 everywhere (zero-mean, symmetric)
 env-side systematic   : wind - in the right room, with probability 0.2 the agent is pushed one cell south
Readings (Chao-Shen): P, H_f, H_b, Delta H, state term I(S;S')/C, action term I(A;S'|S)/C; three-of-four and
P alone; H(S); MMD (delta kernel); reward. Alarm rates at thresholds matched to 5/21 controls, and the mean
signed standardised shift (evaluation mean - calibration mean)/calibration SD per reading."""
import sys, hashlib, numpy as np, pandas as pd, gw_core as g, estimators as e
from G4d_benign_variation import mmd_delta, STEPS, ONSET, STR
def rollout(T0,T1,st,pi,n,rng,obs=None,wind=None):
    cpi=np.cumsum(pi,1); c0=np.cumsum(T0,2); c1=np.cumsum(T1,2); cs=np.cumsum(st); u=rng.random((n,4))
    S=np.empty(n,int);A=np.empty(n,int);Sn=np.empty(n,int); s=int(np.searchsorted(cs,rng.random()))
    for t in range(n):
        post=t>=ONSET; o=obs[s] if (post and obs is not None) else s
        a=int(np.searchsorted(cpi[o],u[t,0])); cT=c1 if post else c0; sn=int(np.searchsorted(cT[s,a],u[t,1]))
        if post and wind is not None and wind[sn]>=0 and u[t,3]<0.2: sn=int(wind[sn])
        on=obs[sn] if (post and obs is not None) else sn
        S[t],A[t],Sn[t]=o,a,on
        s=int(np.searchsorted(cs,u[t,2])) if sn==g.G else sn
    return S,A,Sn
def trial(T0,T1,st,pi,W,rng,obs=None,wind=None):
    S,A,Sn=rollout(T0,T1,st,pi,STEPS,rng,obs,wind); sym=(S*g.nA+A)*g.nS+Sn
    ru,rc=np.unique(sym[:7500],return_counts=True); refc=dict(zip(ru,rc)); ref_sq=(rc*(rc-1)).sum()/(7500*7499)
    nw=(STEPS-W)//STR+1; rows=[]
    for i in range(nw):
        sl=slice(i*STR,i*STR+W); q=e.window_quantities(S[sl],A[sl],Sn[sl],g.nS,g.nA,"cs"); C=q["HS"]+q["HA"]+q["HSn"]
        rows.append(dict(P=q["P"],Hf=q["Hf"],Hb=q["Hb"],dH=q["dH"],termS=q["I_SSn"]/C,termA=q["I_ASn_S"]/C,HS=e.H_plugin(e.counts(S[sl])),
                         MMD=mmd_delta(sym[sl],refc,7500,ref_sq) if i*STR>=7500 else np.nan,reward=(Sn[sl]==g.G).mean()))
    D=pd.DataFrame(rows); wid=np.arange(nw); cal=wid*STR+W<=ONSET; ev=wid*STR>=ONSET; calm=cal&(wid*STR>=7500); out={}
    for c in ["P","Hf","Hb","dH","termS","termA","HS","reward"]:
        v=D[c].to_numpy(); mu,sd=np.nanmean(v[cal]),np.nanstd(v[cal],ddof=1)
        out[c]=np.nanmax(np.abs(v[ev]-mu)/sd); out["shift_"+c]=(np.nanmean(v[ev])-mu)/sd
    v=D.MMD.to_numpy(); out["MMD"]=np.nanmax((v[ev]-np.nanmean(v[calm]))/np.nanstd(v[calm],ddof=1)); out["shift_MMD"]=(np.nanmean(v[ev])-np.nanmean(v[calm]))/np.nanstd(v[calm],ddof=1)
    z=sorted([out[c] for c in ["P","Hf","Hb","dH"]],reverse=True); out["three_of_four"]=z[2]
    return out
def run(N,eps,W,tag,seeds):
    g.setup(N); pi=g.policy(eps); T0=g.kernel(g.slip_map()); st=g.start_dist()
    obs=np.arange(g.nS)
    for c,i in g.idx.items():
        if c[0]<g.WALL_X-1 and (c[0]+1,c[1]) in g.idx: obs[i]=g.idx[(c[0]+1,c[1])]
    wind=-np.ones(g.nS,int)
    for c,i in g.idx.items():
        if c[0]>g.WALL_X and (c[0],c[1]+1) in g.idx and c!=g.GOAL: wind[i]=g.idx[(c[0],c[1]+1)]
    conds={"control":dict(T1=T0),
           "agent-side stochastic: 10% action noise":dict(T1=0.9*T0+0.1*T0.mean(1,keepdims=True)),
           "agent-side systematic: sensor offset (left room)":dict(T1=T0,obs=obs),
           "env-side stochastic: slip 0.05 -> 0.15":dict(T1=g.kernel(np.where(g.slip_map()>0.1,0.30,0.15))),
           "env-side systematic: wind south (right room, p 0.2)":dict(T1=T0,wind=wind)}
    rows=[]
    for name,kw in conds.items():
        for sd in seeds:
            rng=np.random.default_rng(sd*1000+int(hashlib.md5(name.encode()).hexdigest(),16)%997)
            r=trial(T0,kw["T1"],st,pi,W,rng,kw.get("obs"),kw.get("wind")); r.update(world=tag,cond=name,seed=sd); rows.append(r)
    return pd.DataFrame(rows)
if __name__=="__main__":
    part=sys.argv[1]; seeds=list(range(1,22))
    R=run(7,0.2,500,"7x7 W500",seeds) if part=="7" else run(27,0.4,200,"27x27 W200",seeds)
    R.to_csv(f"G4b_scores_{part}.csv",index=False); print("done",part)
