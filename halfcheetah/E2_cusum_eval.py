"""E2 (Dp9r Q4): CUSUM on per-episode readings, held out, separating the detector from the reading.
Readings per episode (E1): reward (episode mean), P, and the four statistics P, H_f, H_b, Delta H (Chao-Shen, 999
within-episode triplets). Standardised with episodes 0-14 (mean, SD ddof=1) of the same run; two-sided CUSUM
S+ = max(0, S+ + z - k), S- = max(0, S- - z - k) from episode 15; latched; undefined episodes skipped.
Score of a run for a given k: max over monitored episodes of max(S+, S-); alarm iff score > h.
Three-of-four: alarm when three of the four statistics have crossed h (third largest of the four scores).
(1) Short exposure (archived trials, episodes 15-49): nested leave-one-agent-out, (k, h) chosen on the other 20
    agents to maximise inner detection subject to inner control alarms <= 0.25 (ties -> larger h), as in H5.
(2) Long run: H10b nominal runs (episodes 15-999). For each held-out agent and k, the lowest h at which at most 25%
    of the other agents' nominal runs alarm within 100 episodes after calibration; k chosen by inner archived
    detection at those h. Frozen (k, h) applied to the held-out nominal run and archived trials.
Agent-level bootstrap (1,000) with selection repeated inside each resample.  usage: python E2_cusum_eval.py [nboot]"""
import sys, glob, os, numpy as np, pandas as pd
KS=[0.0,0.25,0.5,0.75,1.0,1.5,2.0]; HG=np.round(np.arange(0.25,300.0,0.25),2)
MAPF=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","config","trial_label_map.csv")
def cusum_path(x,k,cal=15):
    mu,sd=np.nanmean(x[:cal]),np.nanstd(x[:cal],ddof=1); z=(x[cal:]-mu)/sd; sp=sm=0.0; out=np.empty(len(z))
    for i,v in enumerate(z):
        if np.isfinite(v): sp=max(0.0,sp+v-k); sm=max(0.0,sm-v-k)
        out[i]=max(sp,sm)
    return np.maximum.accumulate(out)          # running max (latched)
READ={"reward":["ep_reward"],"P":["ep_P"],"three-of-four":["ep_P","ep_Hf","ep_Hb","ep_dH"]}
def run_scores(z):
    """per method, per k: running-max path (monitored episodes); three-of-four = third largest across the four paths"""
    out={}
    for m,cols in READ.items():
        for k in KS:
            paths=np.vstack([cusum_path(np.asarray(z[c],float),k) for c in cols])
            out[(m,k)]=paths[0] if len(cols)==1 else np.sort(paths,0)[::-1][2]
    return out
def load_trials():
    M=pd.read_csv(MAPF); lab={f.replace("_metrics.csv",""):p for f,p in zip(M.file,M.perturbation)}; rows=[]
    for f in sorted(glob.glob("E1/*.npz")):
        uid,key=os.path.basename(f)[:-4].split("__",1); z=np.load(f); sc=run_scores(z)
        r=dict(UID=uid,cond=lab[key],ctrl=lab[key]=="no_noise (baseline)")
        for (m,k),p in sc.items(): r[f"{m}|{k}"]=p[-1]
        rows.append(r)
    return pd.DataFrame(rows)
def select_short(inner,m):
    C=inner[inner.ctrl]; P=inner[~inner.ctrl]; best=None
    for k in KS:
        c=C[f"{m}|{k}"].to_numpy(); p=P[f"{m}|{k}"].to_numpy()
        fa=(c[None,:]>HG[:,None]).mean(1); det=(p[None,:]>HG[:,None]).mean(1); ok=np.nonzero(fa<=0.25+1e-12)[0]
        if not len(ok): continue
        i=ok[np.flatnonzero(det[ok]==det[ok].max())[-1]]; cand=(det[i],HG[i],k)
        if best is None or cand[0]>best[0] or (cand[0]==best[0] and cand[1]>best[1]): best=cand
    return best[2],best[1]
def nested_short(T,agents):
    out={m:dict(hit=[],fa=[],ch=[],per=[]) for m in READ}; uniq,mult=np.unique(agents,return_counts=True)
    for a,n in zip(uniq,mult):
        inner=pd.concat([T[T.UID==b] for b in agents if b!=a]); held=T[T.UID==a]
        for m in READ:
            k,h=select_short(inner,m); col=f"{m}|{k}"
            out[m]["hit"]+=list(held[~held.ctrl][col]>h)*n; out[m]["fa"]+=list(held[held.ctrl][col]>h)*n; out[m]["ch"].append((k,h))
            out[m]["per"]+=[(c,v>h) for c,v in zip(held[~held.ctrl].cond,held[~held.ctrl][col])]
    return out
def long_times(ag):
    z=np.load(f"E1_long/{ag}.npz"); sc=run_scores(z); t={}
    for (m,k),p in sc.items():
        f=np.searchsorted(p,HG,side="right").astype(float); f[f>=len(p)]=np.inf
        t[(m,k)]=f+1          # episodes after calibration at first alarm (episode 15 -> 1); inf = none in 985
    return t
def nested_long(T,L,agents):
    out={m:dict(hit=[],fa=[],t100=[],tfirst=[],ch=[],per=[]) for m in READ}; uniq,mult=np.unique(agents,return_counts=True)
    for a,n in zip(uniq,mult):
        inner_ag=[b for b in agents if b!=a]; innerT=pd.concat([T[T.UID==b] for b in inner_ag]); held=T[T.UID==a]
        for m in READ:
            best=None
            for k in KS:
                f=np.mean(np.vstack([L[b][(m,k)] for b in inner_ag])<=100,0); ok=np.nonzero(f<=0.25+1e-12)[0]
                if not len(ok): continue
                h=HG[ok[0]]; det=(innerT[~innerT.ctrl][f"{m}|{k}"]>h).mean()
                if best is None or det>best[0]: best=(det,k,h,ok[0])
            _,k,h,i=best; col=f"{m}|{k}"
            out[m]["hit"]+=list(held[~held.ctrl][col]>h)*n; out[m]["fa"]+=list(held[held.ctrl][col]>h)*n
            out[m]["tfirst"].append(L[a][(m,k)][i]); out[m]["ch"].append((k,h))
            out[m]["per"]+=[(c,v>h) for c,v in zip(held[~held.ctrl].cond,held[~held.ctrl][col])]
    return out
def summarise(R,tag):
    rows=[]
    for m,r in R.items():
        d=dict(analysis=tag,method=m,detection=100*np.mean(r["hit"]),control_alarms=f"{sum(r['fa'])}/{len(r['fa'])}",
               choices=str(pd.Series([f"k{k}/h{h}" for k,h in r["ch"]]).value_counts().head(3).to_dict()))
        if "tfirst" in r and r["tfirst"]:
            t=np.array(r["tfirst"]); d.update(heldout_runs_alarm_within_100=f"{(t<=100).sum()}/{len(t)}",runs_no_alarm=f"{np.isinf(t).sum()}/{len(t)}",
                 median_first_alarm=np.median(np.where(np.isinf(t),985,t)))
        rows.append(d)
    return pd.DataFrame(rows)
if __name__=="__main__":
    nboot=int(sys.argv[1]) if len(sys.argv)>1 else 1000
    T=load_trials(); T.to_csv("E2_trial_scores.csv",index=False); AG=np.array(sorted(T.UID.unique()))
    S=nested_short(T,AG); out=[summarise(S,"short exposure (archived controls, 6/21-type budget)")]
    longs=sorted(os.path.basename(f)[:-4] for f in glob.glob("E1_long/*.npz")); haveL=len(longs)==21
    if haveL:
        L={a:long_times(a) for a in AG}; LR=nested_long(T,L,AG); out.append(summarise(LR,"long-run target (<=25% of nominal runs alarm within 100 episodes)"))
    Rsum=pd.concat(out); pd.set_option("display.width",250); pd.set_option("display.max_colwidth",80); print(Rsum.round(1).to_string(index=False)); Rsum.to_csv("E2_summary.csv",index=False)
    per=[]
    for tag,R in [("short",S)]+([("long",LR)] if haveL else []):
        for m,r in R.items():
            for c,h in r["per"]: per.append(dict(analysis=tag,method=m,cond=c,hit=h))
    P=pd.DataFrame(per).pivot_table(index="cond",columns=["analysis","method"],values="hit",aggfunc="mean")*100
    print(P.round(1).to_string()); P.to_csv("E2_by_condition.csv")
    if haveL:
        EPS=[1,5,10,25,50,100,150,200,300,500,700,985]
        sv=pd.DataFrame({m:[np.mean(np.array(LR[m]["tfirst"])<=e)*100 for e in EPS] for m in READ},index=EPS); sv.index.name="episodes"
        print(sv.round(1).to_string()); sv.to_csv("E2_long_survival.csv")
    if nboot>0:
        rng=np.random.default_rng(20261109); B=[]
        for b in range(nboot):
            ags=rng.choice(AG,len(AG),replace=True)
            if len(np.unique(ags))<3: continue
            rs=nested_short(T,ags); row={f"short_{m}":100*np.mean(rs[m]["hit"]) for m in READ}
            if haveL:
                rl=nested_long(T,L,ags); row.update({f"long_{m}":100*np.mean(rl[m]["hit"]) for m in READ})
            B.append(row)
            if (b+1)%100==0: pd.DataFrame(B).to_csv("E2_boot.csv",index=False); print("boot",b+1,flush=True)
        B=pd.DataFrame(B); B.to_csv("E2_boot.csv",index=False)
        for tag in ["short"]+(["long"] if haveL else []):
            for m in READ: print(f"{tag} {m:14s} [{B[f'{tag}_{m}'].quantile(.025):.1f}, {B[f'{tag}_{m}'].quantile(.975):.1f}]")
            for m in ["three-of-four","P"]:
                d=B[f"{tag}_{m}"]-B[f"{tag}_reward"]; print(f"{tag} {m} - reward [{d.quantile(.025):+.1f}, {d.quantile(.975):+.1f}]")
