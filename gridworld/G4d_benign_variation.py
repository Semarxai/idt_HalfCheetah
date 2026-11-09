"""G4d: benign in-envelope variation vs faults, under the HalfCheetah trial protocol.
A trial is one 50,000-step rollout; healthy operation until step 15,000, then either nothing (control),
a benign in-envelope change (new start region; the same policy is valid everywhere), or a fault (10%
executed-action noise). Windows W (500 primary 7x7, 200 stress 27x27), stride 50; calibration on windows
wholly before step 15,000, evaluation from the window starting at 15,000; latched max |z|.
Readings: P, H_f, H_b, Delta H (Chao-Shen; three-of-four and P alone), H(S), MMD (unbiased, delta kernel
on (s,a,s') symbols, reference = steps 0-7,499, calibration windows from 7,500), reward (goal rate).
Thresholds: per method, midway between the 5th and 6th largest of 21 control scores (5/21 controls, as the
submission); also +-3 sigma. 21 independent seeds per condition."""
import sys, hashlib, numpy as np, pandas as pd, gw_core as g, estimators as e
STEPS,ONSET,STR=50000,15000,50
def rollout(T0,T1,st0,st1,pi,n,rng):
    cpi=np.cumsum(pi,1); c0=np.cumsum(T0,2); c1=np.cumsum(T1,2); s0=np.cumsum(st0); s1=np.cumsum(st1); u=rng.random((n,3))
    S=np.empty(n,int);A=np.empty(n,int);Sn=np.empty(n,int); s=int(np.searchsorted(s0,rng.random()))
    for t in range(n):
        post=t>=ONSET; cT=c1 if post else c0; cs=s1 if post else s0
        a=int(np.searchsorted(cpi[s],u[t,0])); sn=int(np.searchsorted(cT[s,a],u[t,1])); S[t],A[t],Sn[t]=s,a,sn
        s=int(np.searchsorted(cs,u[t,2])) if sn==g.G else sn
    return S,A,Sn
def mmd_delta(x,ref_counts,mref,ref_sq):
    u,c=np.unique(x,return_counts=True); n=len(x); cr=np.array([ref_counts.get(k,0) for k in u])
    return (c*(c-1)).sum()/(n*(n-1))+ref_sq-2*(c*cr).sum()/(n*mref)
def trial(T0,T1,st0,st1,pi,W,rng):
    S,A,Sn=rollout(T0,T1,st0,st1,pi,STEPS,rng); sym=(S*g.nA+A)*g.nS+Sn
    ru,rc=np.unique(sym[:7500],return_counts=True); refc=dict(zip(ru,rc)); mref=7500; ref_sq=(rc*(rc-1)).sum()/(mref*(mref-1))
    nw=(STEPS-W)//STR+1; rows=[]
    for i in range(nw):
        sl=slice(i*STR,i*STR+W); q=e.window_quantities(S[sl],A[sl],Sn[sl],g.nS,g.nA,"cs")
        rows.append(dict(P=q["P"],Hf=q["Hf"],Hb=q["Hb"],dH=q["dH"],HS=e.H_plugin(e.counts(S[sl])),
                         MMD=mmd_delta(sym[sl],refc,mref,ref_sq) if i*STR>=7500 else np.nan,reward=(Sn[sl]==g.G).mean()))
    D=pd.DataFrame(rows); wid=np.arange(nw); cal=wid*STR+W<=ONSET; ev=wid*STR>=ONSET; calm=cal&(wid*STR>=7500)
    out={}
    for c in ["P","Hf","Hb","dH","HS","reward"]:
        v=D[c].to_numpy(); out[c]=np.nanmax(np.abs(v[ev]-np.nanmean(v[cal]))/np.nanstd(v[cal],ddof=1))
    v=D.MMD.to_numpy(); out["MMD"]=np.nanmax((v[ev]-np.nanmean(v[calm]))/np.nanstd(v[calm],ddof=1))
    z=sorted([out[c] for c in ["P","Hf","Hb","dH"]],reverse=True); out["three_of_four"]=z[2]
    return out
def run(N,eps,W,tag,seeds):
    g.setup(N); pi=g.policy(eps); T0=g.kernel(g.slip_map()); st0=g.start_dist(); left=[c for c in g.cells if c[0]<g.WALL_X]
    Tn=0.9*T0+0.1*T0.mean(1,keepdims=True); top=[(x,0) for x in range(g.WALL_X)]; bottom=[(x,g.N-1) for x in range(g.WALL_X)]
    mix=g.start_dist(); mix=0.5*mix+0.5*g.start_dist(top)
    conds={"control":(T0,st0),"benign: start top row of left room":(T0,g.start_dist(top)),
           "benign: start anywhere in left room":(T0,g.start_dist(left)),"benign: 50/50 left column + top row":(T0,mix),
           "benign: start bottom row of left room":(T0,g.start_dist(bottom)),"fault: 10% executed-action noise":(Tn,st0)}
    rows=[]
    for name,(T1,st1) in conds.items():
        for sd in seeds:
            rng=np.random.default_rng(sd*1000+int(hashlib.md5(name.encode()).hexdigest(),16)%997); r=trial(T0,T1,st0,st1,pi,W,rng); r.update(world=tag,cond=name,seed=sd); rows.append(r)
            print(tag,name,sd,flush=True)
    return pd.DataFrame(rows)
if __name__=="__main__":
    part=sys.argv[1]; seeds=list(range(1,22))
    R=run(7,0.2,500,"7x7 W500",seeds) if part=="7" else run(27,0.4,200,"27x27 W200",seeds)
    R.to_csv(f"G4d_scores_{part}.csv",index=False)
