"""H5 (+ H11, H12): nested leave-one-agent-out evaluation, identical for every method.
Outer loop: one held-out agent. Inner loop: the other agents choose the operating point by one fixed
criterion declared in advance - the threshold (and, for the consensus, k) that maximises inner
detection subject to an inner control false-alarm rate <= ALPHA; ties go to the larger threshold.
The held-out agent's healthy period calibrates its own statistics; its control and fault outcomes
never enter the selection. Intervals: agent-level bootstrap with the whole nested procedure repeated
inside each resample. Curves: the same procedure over a range of ALPHA."""
import os, numpy as np, pandas as pd
R=os.environ.get("IDT_DATA_ROOT","data"); O=os.path.join(os.path.dirname(os.path.abspath(__file__)),"out")
MAP=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","config","trial_label_map.csv")); MAP=MAP[~MAP.UID.isin({"ppo15","ppo18","ppo19","ppo21"})]
# ---------- per-trial scores (max |z| over evaluation windows >= 300) ----------
if not os.path.isdir(f"{O}/H1H2") and os.path.exists(f"{O}/H5_trial_scores.csv"):
    T=pd.read_csv(f"{O}/H5_trial_scores.csv")   # per-trial scores shipped with the repository
else:
    rows=[]
    for _,r in MAP.iterrows():
        k=f"{r.UID}__{r.file.replace('_metrics.csv','')}.csv"
        d=pd.read_csv(f"{O}/H1H2/{k}"); m=pd.read_csv(f"{O}/H3/{k}")
        pre=d[d.window_id<=290]; po=d[d.window_id>=300]
        z=lambda c:((po[c]-pre[c].mean()).abs()/pre[c].std()).max()
        zz=sorted([z("P_pi"),z("Hf_pi"),z("Hb_pi"),z("dH_pi")],reverse=True)
        cal=m[m.window_id<=290]; mp=m[m.window_id>=300]
        rows.append(dict(UID=r.UID,cond=r.perturbation,ctrl=r.perturbation=="no_noise (baseline)",
            cons1=zz[0],cons2=zz[1],cons3=zz[2],cons4=zz[3],P=z("P_pi"),reward=z("mean_reward"),HS=z("H_S"),
            MMD=((mp.mmd2-cal.mmd2.mean())/cal.mmd2.std()).max()))
    T=pd.DataFrame(rows); T.to_csv(f"{O}/H5_trial_scores.csv",index=False)
AG=np.array(sorted(T.UID.unique()))
METHODS={"consensus":["cons1","cons2","cons3","cons4"],"reward":["reward"],"MMD":["MMD"],"H(S)":["HS"],"P alone":["P"]}
ZGRID=np.round(np.arange(1.5,30.0,0.05),2)
def precompute(T):
    """per agent: for each score column, matrix [n_trials] of scores; ctrl flags"""
    return {a:(T[T.UID==a]) for a in T.UID.unique()}
def select(inner, cols, alpha):
    """inner: DataFrame of inner trials. returns (col, z) maximising detection s.t. FA<=alpha"""
    C=inner[inner.ctrl]; P=inner[~inner.ctrl]; best=None
    for c in cols:
        cs=C[c].to_numpy(); ps=P[c].to_numpy()
        fa=(cs[None,:]>ZGRID[:,None]).mean(1); det=(ps[None,:]>ZGRID[:,None]).mean(1)
        ok=np.where(fa<=alpha)[0]
        if len(ok)==0: continue
        i=ok[np.argmax(det[ok][::-1])*-1-1] if False else ok[np.flatnonzero(det[ok]==det[ok].max())[-1]]  # max det, largest z on ties
        cand=(det[i],ZGRID[i],c)
        if best is None or cand[0]>best[0] or (cand[0]==best[0] and cand[1]>best[1]): best=cand
    return best[2],best[1]
def nested(T, alpha, agents=None):
    """returns dict method -> (det, fa, choices). agents: array of agent ids (with repeats for bootstrap)"""
    if agents is None: agents=AG
    parts=[T[T.UID==a].assign(_rep=i) for i,a in enumerate(agents)]
    TT=pd.concat(parts,ignore_index=True); out={}
    for meth,cols in METHODS.items():
        hit=[];fal=[];ch=[]
        for a in np.unique(agents):
            inner=TT[TT.UID!=a]; held=TT[TT.UID==a]
            c,zt=select(inner,cols,alpha); ch.append((c,zt))
            hit+=list(held[~held.ctrl][c]>zt); fal+=list(held[held.ctrl][c]>zt)
        out[meth]=(np.mean(hit),np.mean(fal),ch)
    return out
if __name__=="__main__":
    import sys
    ALPHA=0.25; mode=sys.argv[1] if len(sys.argv)>1 else "point"
    if mode=="point":
        res=nested(T,ALPHA)
        print(f"=== nested leave-one-agent-out, ALPHA={ALPHA}: 21 folds, 168 perturbation trials, 21 controls ===")
        for meth,(det,fa,ch) in res.items():
            cc=pd.Series([f"{c}@{z}" for c,z in ch]).value_counts().head(3).to_dict()
            print(f"  {meth:10s} detection {100*det:5.1f}%   control alarms {int(round(fa*21))}/21   choices {cc}")
    elif mode=="curve":
        cur=[]
        for al in [0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50]:
            rr=nested(T,al); cur.append(dict(alpha=al,**{f"{m}_det":100*rr[m][0] for m in METHODS},**{f"{m}_fa":100*rr[m][1] for m in METHODS}))
        C=pd.DataFrame(cur); C.to_csv(f"{O}/H5_nested_curve.csv",index=False); print(C.round(1).to_string(index=False))
    elif mode=="boot":           # python H5_nested.py boot <chunk_id> <n>
        cid,n=int(sys.argv[2]),int(sys.argv[3]); rng=np.random.default_rng(20260917+cid); out=[]
        for b in range(n):
            ags=rng.choice(AG,len(AG),replace=True); rr=nested(T,ALPHA,ags)
            out.append({**{f"{m}_det":rr[m][0] for m in METHODS},**{f"{m}_fa":rr[m][1] for m in METHODS}})
        pd.DataFrame(out).to_csv(f"{O}/H5_boot_{cid:02d}.csv",index=False); print("chunk",cid,"done")
    elif mode=="bootsum":
        import glob
        B=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{O}/H5_boot_*.csv"))]); print("resamples:",len(B))
        for m in METHODS:
            d=B[f"{m}_det"]*100; f=B[f"{m}_fa"]*100
            line=f"  {m:10s} detection [{d.quantile(.025):5.1f},{d.quantile(.975):5.1f}]  false alarm [{f.quantile(.025):5.1f},{f.quantile(.975):5.1f}]"
            if m!="consensus":
                dd=(B["consensus_det"]-B[f"{m}_det"])*100; ff=(B["consensus_fa"]-B[f"{m}_fa"])*100
                line+=f" | consensus-{m}: det [{dd.quantile(.025):5.1f},{dd.quantile(.975):5.1f}] fa [{ff.quantile(.025):5.1f},{ff.quantile(.975):5.1f}]"
            print(line)
