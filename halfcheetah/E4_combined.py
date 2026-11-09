"""E4 (mcXu Q1, complementarity): combined sequential monitor and trial-level overlap, held out.
Combined: alarm when either component CUSUM (reward; telemetry = three-of-four or P) crosses its own level.
Common scale fixed on the inner agents only: short exposure - each component divided by the 75th percentile of
its inner control scores (same k); long run - each component divided by the h at which it alone meets the
long-run target on the inner runs. One threshold c on the combined score max(r/q_r, t/q_t), selected as before:
short - max inner detection s.t. inner control alarms <= 0.25 (ties -> larger c); long - lowest c with <= 25% of
inner nominal runs alarming within 100 episodes, k (shared) chosen by inner detection.
Overlap: at the single-reading held-out operating points of E2, each fault trial classed reward-only /
telemetry-only / both / neither. Bootstrap (1,000, selection inside) for combined detection and differences.
usage: python E4_combined.py [nboot]"""
import sys, glob, os, numpy as np, pandas as pd
import E2_cusum_eval as E
CG=np.round(np.arange(0.3,5.0,0.01),2); KS=E.KS; HG=E.HG
PAIRS={"reward OR three-of-four":"three-of-four","reward OR P":"P"}
def load():
    M=pd.read_csv(E.MAPF); lab={f.replace("_metrics.csv",""):p for f,p in zip(M.file,M.perturbation)}; rows=[]
    for i,f in enumerate(sorted(glob.glob("E1/*.npz"))):
        uid,key=os.path.basename(f)[:-4].split("__",1); sc=E.run_scores(np.load(f))
        r=dict(TID=i,UID=uid,cond=lab[key],ctrl=lab[key]=="no_noise (baseline)")
        for (m,k),p in sc.items(): r[f"{m}|{k}"]=p[-1]
        rows.append(r)
    return pd.DataFrame(rows)
def long_paths(ag): return E.run_scores(np.load(f"E1_long/{ag}.npz"))
def first_le(path,grid):                      # episodes after calibration at first crossing of each level in grid
    f=np.searchsorted(path,grid,side="right").astype(float); f[f>=len(path)]=np.inf; return f+1
def short_combined(T,agents,tm):
    hit=[];fa=[];uniq,mult=np.unique(agents,return_counts=True)
    for a,n in zip(uniq,mult):
        inner=pd.concat([T[T.UID==b] for b in agents if b!=a]); held=T[T.UID==a]; best=None
        for k in KS:
            qr=np.quantile(inner[inner.ctrl][f"reward|{k}"],0.75); qt=np.quantile(inner[inner.ctrl][f"{tm}|{k}"],0.75)
            sc=lambda D:np.maximum(D[f"reward|{k}"]/qr,D[f"{tm}|{k}"]/qt).to_numpy()
            c=sc(inner[inner.ctrl]); p=sc(inner[~inner.ctrl]); faR=(c[None,:]>CG[:,None]).mean(1); det=(p[None,:]>CG[:,None]).mean(1)
            ok=np.nonzero(faR<=0.25+1e-12)[0]
            if not len(ok): continue
            i=ok[np.flatnonzero(det[ok]==det[ok].max())[-1]]; cand=(det[i],CG[i],k,qr,qt)
            if best is None or cand[0]>best[0] or (cand[0]==best[0] and cand[1]>best[1]): best=cand
        _,cc,k,qr,qt=best; s=np.maximum(held[f"reward|{k}"]/qr,held[f"{tm}|{k}"]/qt)
        hit+=list((s[~held.ctrl.values]>cc).values)*n; fa+=list((s[held.ctrl.values]>cc).values)*n
    return 100*np.mean(hit),f"{sum(fa)}/{len(fa)}"
def long_combined(T,LP,agents,tm):
    hit=[];fa=[];t1=[];uniq,mult=np.unique(agents,return_counts=True)
    for a,n in zip(uniq,mult):
        inner=[b for b in agents if b!=a]; innerT=pd.concat([T[T.UID==b] for b in inner]); held=T[T.UID==a]; best=None
        for k in KS:
            lv={}
            for m in ["reward",tm]:
                f=np.mean(np.vstack([first_le(LP[b][(m,k)],HG) for b in inner])<=100,0); ok=np.nonzero(f<=0.25+1e-12)[0]
                lv[m]=HG[ok[0]] if len(ok) else None
            if None in lv.values(): continue
            comb={b:np.maximum(LP[b][("reward",k)]/lv["reward"],LP[b][(tm,k)]/lv[tm]) for b in inner+[a]}
            f=np.mean(np.vstack([first_le(comb[b],CG) for b in inner])<=100,0); ok=np.nonzero(f<=0.25+1e-12)[0]
            if not len(ok): continue
            cc=CG[ok[0]]; s=np.maximum(innerT[f"reward|{k}"]/lv["reward"],innerT[f"{tm}|{k}"]/lv[tm])
            det=(s[~innerT.ctrl.values]>cc).mean()
            if best is None or det>best[0]: best=(det,k,cc,lv,first_le(comb[a],np.array([cc]))[0])
        _,k,cc,lv,tf=best; s=np.maximum(held[f"reward|{k}"]/lv["reward"],held[f"{tm}|{k}"]/lv[tm])
        hit+=list((s[~held.ctrl.values]>cc).values)*n; fa+=list((s[held.ctrl.values]>cc).values)*n; t1.append(tf)
    t1=np.array(t1); return 100*np.mean(hit),f"{sum(fa)}/{len(fa)}",f"{(t1<=100).sum()}/{len(t1)}"
def overlap(T,R,tag):
    """R: E2 nested result (with per-agent choices in order of sorted unique agents)"""
    AG=sorted(T.UID.unique()); rows=[]
    for i,a in enumerate(AG):
        held=T[(T.UID==a)&(~T.ctrl)]
        ch={m:R[m]["ch"][i] for m in R}
        for _,t in held.iterrows():
            hr=t[f"reward|{ch['reward'][0]}"]>ch["reward"][1]
            for tm in ["three-of-four","P"]:
                ht=t[f"{tm}|{ch[tm][0]}"]>ch[tm][1]
                rows.append(dict(analysis=tag,telemetry=tm,UID=a,cond=t.cond,cls="both" if hr and ht else "reward only" if hr else "telemetry only" if ht else "neither"))
    return pd.DataFrame(rows)
if __name__=="__main__":
    nboot=int(sys.argv[1]) if len(sys.argv)>1 else 1000
    T=load(); AG=np.array(sorted(T.UID.unique())); LP={a:long_paths(a) for a in AG}
    L={a:E.long_times(a) for a in AG}
    res=[]
    for name,tm in PAIRS.items():
        d,f=short_combined(T,AG,tm); res.append(dict(analysis="short exposure",monitor=name,detection=d,control_alarms=f,heldout_runs_alarm_within_100=""))
        d,f,t=long_combined(T,LP,AG,tm); res.append(dict(analysis="long-run target",monitor=name,detection=d,control_alarms=f,heldout_runs_alarm_within_100=t))
    R=pd.DataFrame(res); pd.set_option("display.width",200); print(R.round(1).to_string(index=False)); R.to_csv("E4_combined_summary.csv",index=False)
    S=E.nested_short(T,AG); LR=E.nested_long(T,L,AG)
    O=pd.concat([overlap(T,S,"short"),overlap(T,LR,"long")]); O.to_csv("E4_overlap_trials.csv",index=False)
    tab=O.groupby(["analysis","telemetry","cls"]).size().unstack("cls").fillna(0).astype(int); print(tab.to_string()); tab.to_csv("E4_overlap_summary.csv")
    bc=O[O.telemetry=="three-of-four"].groupby(["analysis","cond","cls"]).size().unstack("cls").fillna(0).astype(int); print(bc.to_string()); bc.to_csv("E4_overlap_by_condition.csv")
    if nboot>0:
        rng=np.random.default_rng(20261110); B=[]
        for b in range(nboot):
            ags=rng.choice(AG,len(AG),replace=True)
            if len(np.unique(ags))<3: continue
            row={}
            s2=E.nested_short(T,ags); l2=E.nested_long(T,L,ags)
            for m in ["reward","three-of-four","P"]: row[f"short_{m}"]=100*np.mean(s2[m]["hit"]); row[f"long_{m}"]=100*np.mean(l2[m]["hit"])
            for name,tm in PAIRS.items():
                row[f"short_{name}"]=short_combined(T,ags,tm)[0]; row[f"long_{name}"]=long_combined(T,LP,ags,tm)[0]
            B.append(row)
            if (b+1)%50==0: pd.DataFrame(B).to_csv("E4_boot.csv",index=False); print("boot",b+1,flush=True)
        B=pd.DataFrame(B); B.to_csv("E4_boot.csv",index=False)
        for tag in ["short","long"]:
            for name,tm in PAIRS.items():
                c=B[f"{tag}_{name}"]; print(f"{tag} {name}: [{c.quantile(.025):.1f}, {c.quantile(.975):.1f}]  minus reward [{(c-B[f'{tag}_reward']).quantile(.025):+.1f}, {(c-B[f'{tag}_reward']).quantile(.975):+.1f}]  minus {tm} [{(c-B[f'{tag}_{tm}']).quantile(.025):+.1f}, {(c-B[f'{tag}_{tm}']).quantile(.975):+.1f}]")
