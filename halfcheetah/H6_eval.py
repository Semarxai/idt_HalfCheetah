"""H6 evaluation: results by cohort - retained (21), + mapped screened (24), + all logged screened (28)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import H5_nested as h
O=h.O; BASE=os.environ.get("IDT_DATA_ROOT","data")
M0=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","config","trial_label_map.csv"))[["file","UID","perturbation"]]
M1=pd.read_csv(f"{O}/H6_extended_label_map.csv")[["file","UID","perturbation"]]
M=pd.concat([M0,M1],ignore_index=True)
def scores(r):
    k=f"{r.UID}__{r.file.replace('_metrics.csv','')}.csv"
    d=pd.read_csv(f"{O}/H1H2/{k}"); c=pd.read_csv(f"{O}/H2cs/{k}"); m=pd.read_csv(f"{O}/H3/{k}")
    def z(df,col):
        pre=df[df.window_id<=290]; po=df[df.window_id>=300]; return ((po[col]-pre[col].mean()).abs()/pre[col].std()).max()
    zp=sorted([z(d,s) for s in ["P_pi","Hf_pi","Hb_pi","dH_pi"]],reverse=True); zc=sorted([z(c,s) for s in ["P_cs","Hf_cs","Hb_cs","dH_cs"]],reverse=True)
    cal=m[m.window_id<=290]; mp=m[m.window_id>=300]; pre=d[d.window_id<=290]
    return dict(UID=r.UID,cond=r.perturbation,ctrl=r.perturbation=="no_noise (baseline)",
                cons1=zp[0],cons2=zp[1],cons3=zp[2],cons4=zp[3],cs1=zc[0],cs2=zc[1],cs3=zc[2],cs4=zc[3],
                P=z(d,"P_pi"),P_cs=z(c,"P_cs"),reward=z(d,"mean_reward"),HS=z(d,"H_S"),
                MMD=((mp.mmd2-cal.mmd2.mean())/cal.mmd2.std()).max(),
                cal_reward_mean=pre.mean_reward.mean(),cal_reward_cv=pre.mean_reward.std()/abs(pre.mean_reward.mean()))
T=pd.DataFrame([scores(r) for _,r in M.iterrows()]); T.to_csv(f"{O}/H6_trial_scores_all.csv",index=False)
SCREEN_MAPPED={"ppo15","ppo18","ppo19"}; SCREEN_LOGGED={"ppo06","ppo20","ppo21","ppo35"}
cohorts={"retained (paper)":T[~T.UID.isin(SCREEN_MAPPED|SCREEN_LOGGED)],
         "+ mapped screened (15,18,19)":T[~T.UID.isin(SCREEN_LOGGED)],
         "all logged seeds":T}
h.METHODS={"consensus Chao-Shen":["cs1","cs2","cs3","cs4"],"P Chao-Shen":["P_cs"],"consensus plug-in":["cons1","cons2","cons3","cons4"],
           "reward":["reward"],"MMD":["MMD"],"H(S)":["HS"]}
for name,TT in cohorts.items():
    ag=np.array(sorted(TT.UID.unique())); P=TT[~TT.ctrl]; C=TT[TT.ctrl]
    print(f"\n=== {name}: {len(ag)} agents ({sum(a.startswith('sac') for a in ag)} SAC, {sum(a.startswith('ppo') for a in ag)} PPO), {len(P)} perturbation trials, {len(C)} controls")
    print(f"  in-sample ±3σ: consensus plug-in 3-of-4 det {100*(P.cons3>3).mean():.1f}% ctrl {(C.cons3>3).sum()}/{len(C)} | reward det {100*(P.reward>3).mean():.1f}% ctrl {(C.reward>3).sum()}/{len(C)}")
    rr=h.nested(TT,0.25,ag)
    for m,(d,f,ch) in rr.items(): print(f"  nested ALPHA 0.25  {m:20s} det {100*d:5.1f}%  ctrl {int(round(f*len(C)))}/{len(C)}")
print("\n=== control trials of screened seeds (max |z| over evaluation windows) ===")
cols=["UID","cons3","cs1","P_cs","reward","MMD","cal_reward_mean","cal_reward_cv"]
print(T[T.ctrl&T.UID.isin(SCREEN_MAPPED|SCREEN_LOGGED)][cols].round(2).to_string(index=False))
r=T[T.ctrl&~T.UID.isin(SCREEN_MAPPED|SCREEN_LOGGED)]
print(f"\nretained controls: median calibration reward CV {r.cal_reward_cv.median():.3f} (range {r.cal_reward_cv.min():.3f}-{r.cal_reward_cv.max():.3f}); "
      f"PPO retained median {r[r.UID.str.startswith('ppo')].cal_reward_cv.median():.3f}")
