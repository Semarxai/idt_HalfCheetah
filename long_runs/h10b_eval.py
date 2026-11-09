"""H10b evaluation: thresholds set at a long-run false-alarm target, held out, applied to the archived faults.
Long run per agent: calibration windows 0-290 (steps 0-15,000), monitoring windows >= 300 (985 episodes);
latched alarm. Time to first false alarm in episodes after calibration = (window end step - 15,000)/1000;
runs without an alarm are censored at 985 episodes and counted at 985 (conservative).
Target (decision of 8 Nov, after the code check and before the final comparison): at most 25% of nominal runs
alarm within 100 episodes after calibration. Held out: for each agent, the lowest z at which at most 5 of the other
20 agents' long runs alarm within 100 episodes (fraction <= 0.25 of the inner runs in bootstrap resamples); that z is frozen and applied to (a) the held-out agent's long run,
(b) its archived control and 8 fault trials (trial scores: max |z| over evaluation windows; three-of-four = third
largest of the four Chao-Shen statistics). Methods: three-of-four (P, H_f, H_b, Delta H), P alone, reward.
Agent-level bootstrap (1,000 resamples) with the threshold selection repeated inside each resample.
usage: python h10b_eval.py [nboot]"""
import sys, glob, os, numpy as np, pandas as pd
CAL,EV0,NEP=290,300,985.0
ZG=np.round(np.arange(1.5,200.0,0.05),2)
TRIALS=os.environ.get("H2_SCORES",os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","halfcheetah","out","H2_trial_scores_cs.csv"))
def first_cross(z):
    """z: per-window |z| over monitoring windows (nan = undefined window). Returns, for each threshold in ZG,
    the index of the first window with z > threshold (inf if none)."""
    rm=np.fmax.accumulate(np.nan_to_num(z,nan=-np.inf)); idx=np.searchsorted(rm,ZG,side="right").astype(float)
    idx[idx>=len(rm)]=np.inf; return idx
def times(agent):
    D=pd.read_csv(f"readings/readings_{agent}.csv.gz"); out={}
    end=(D.window_id.to_numpy()*50+500-15000)/1000.0; ev=D.window_id.to_numpy()>=EV0; ep=end[ev]
    fc={}
    for c in ["P_cs","Hf_cs","Hb_cs","dH_cs","mean_reward"]:
        v=D[c].to_numpy(); cal=v[:CAL+1]; mu,sd=np.nanmean(cal),np.nanstd(cal,ddof=1)
        fc[c]=first_cross(np.abs(v[ev]-mu)/sd)
    three=np.sort(np.vstack([fc[c] for c in ["P_cs","Hf_cs","Hb_cs","dH_cs"]]),0)[2]
    for m,f in [("three-of-four",three),("P alone",fc["P_cs"]),("reward",fc["mean_reward"])]:
        t=np.full(len(ZG),NEP); ok=np.isfinite(f); t[ok]=ep[f[ok].astype(int)]; out[m]=t     # episodes; censored = 985
        out[m+"_cens"]=~ok
    return out
METH={"three-of-four":"cs3","P alone":"P_cs","reward":"reward"}
def select(Tmat):            # Tmat: [n_inner, len(ZG)] -> lowest z with at most 25% of inner runs alarming within 100 episodes
    f=(Tmat<=100).mean(0); ok=np.nonzero(f<=0.25+1e-12)[0]; return ok[0] if len(ok) else len(ZG)-1
def nested(L,TR,agents):
    res={m:dict(hit=[],fa=[],tfa=[],cens=[],z=[]) for m in METH}
    uniq,mult=np.unique(agents,return_counts=True)
    for a,k in zip(uniq,mult):
        inner=[b for b in agents if b!=a]
        for m,col in METH.items():
            i=select(np.vstack([L[b][m] for b in inner])); z=ZG[i]
            held=TR[TR.UID==a]; res[m]["hit"]+=list(held[~held.ctrl][col]>z)*k; res[m]["fa"]+=list(held[held.ctrl][col]>z)*k
            res[m]["tfa"].append(L[a][m][i]); res[m]["cens"].append(L[a][m+"_cens"][i]); res[m]["z"].append(z)
    return res
if __name__=="__main__":
    nboot=int(sys.argv[1]) if len(sys.argv)>1 else 1000
    AG=sorted(os.path.basename(f)[9:-7] for f in glob.glob("readings/readings_*.csv.gz")); print("agents:",len(AG))
    L={a:times(a) for a in AG}; TR=pd.read_csv(TRIALS); TR=TR[TR.UID.isin(AG)]
    R=nested(L,TR,np.array(AG)); rows=[]
    for m in METH:
        r=R[m]; t=np.array(r["tfa"]); c=np.array(r["cens"])
        rows.append(dict(method=m,z_median=np.median(r["z"]),z_range=f"{min(r['z']):.2f}-{max(r['z']):.2f}",
            heldout_alarm_within_100ep=f"{(t<=100).sum()}/{len(t)}",heldout_median_time=np.median(t),heldout_mean_time=t.mean(),censored=f"{c.sum()}/{len(c)}",
            archived_detection=100*np.mean(r["hit"]),
            archived_control_alarms=f"{sum(r['fa'])}/{len(r['fa'])}"))
    S=pd.DataFrame(rows); pd.set_option("display.width",250); print(S.round(2).to_string(index=False)); S.to_csv("h10b_summary.csv",index=False)
    per=pd.DataFrame([dict(agent=a,method=m,z=R[m]["z"][i],time_to_first_FA=R[m]["tfa"][i],censored=R[m]["cens"][i])
                      for m in METH for i,a in enumerate(AG)]); per.to_csv("h10b_per_agent.csv",index=False)
    cond=[]
    for m,col in METH.items():
        for i,a in enumerate(AG):
            held=TR[(TR.UID==a)&(~TR.ctrl)]
            for _,t in held.iterrows(): cond.append(dict(method=m,cond=t.cond,hit=t[col]>R[m]["z"][i]))
    C=pd.DataFrame(cond).pivot_table(index="cond",columns="method",values="hit",aggfunc="mean")*100
    print(C.round(1).to_string()); C.to_csv("h10b_by_condition.csv")
    # survival: share of held-out long runs with a false alarm by episode e
    EPS=[1,5,10,25,50,75,100,150,200,300,400,500,600,700,800,900,985]
    sv=pd.DataFrame({m:[np.mean((np.array(R[m]["tfa"])<=e)&~np.array(R[m]["cens"]))*100 for e in EPS] for m in METH},index=EPS)
    sv.index.name="episodes"; print(sv.round(1).to_string()); sv.to_csv("h10b_survival.csv")
    if nboot>0:
        rng=np.random.default_rng(20261108); B=[]
        for b in range(nboot):
            ags=rng.choice(AG,len(AG),replace=True)
            if len(np.unique(ags))<3: continue
            rr=nested(L,TR,ags); B.append({m:np.mean(rr[m]["hit"]) for m in METH})
        B=pd.DataFrame(B)*100; B.to_csv("h10b_boot.csv",index=False)
        for m in METH: print(f"{m:14s} detection [{B[m].quantile(.025):.1f}, {B[m].quantile(.975):.1f}]")
        for m in ["three-of-four","P alone"]:
            d=B[m]-B["reward"]; print(f"{m} - reward [{d.quantile(.025):+.1f}, {d.quantile(.975):+.1f}]")
