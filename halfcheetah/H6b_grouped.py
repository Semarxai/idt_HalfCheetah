"""H6b: seed-screening sensitivity with ppo19 and ppo36 held out together. H10b gate 1 showed both runs are
evaluations of one checkpoint (PPO seed 19), so a leave-one-agent-out fold that holds out one of them while the
other stays in the selection set leaks. Here both carry one group label ("ppo19+36") and are held out together;
all their trials count. Scores: out/H6_trial_scores_all.csv (unchanged). Protocol: H5 nested, ALPHA 0.25."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import H5_nested as h
O=h.O; T=pd.read_csv(f"{O}/H6_trial_scores_all.csv")
SCREEN_MAPPED={"ppo15","ppo18","ppo19"}; SCREEN_LOGGED={"ppo06","ppo20","ppo21","ppo35"}
h.METHODS={"three-of-four Chao-Shen (fixed k)":["cs3"],"k-of-four Chao-Shen":["cs1","cs2","cs3","cs4"],"P Chao-Shen":["P_cs"],
           "reward":["reward"],"MMD":["MMD"],"H(S)":["HS"]}
rows=[]
for name,TT in {"+ mapped screened (15,18,19)":T[~T.UID.isin(SCREEN_LOGGED)],"all logged seeds":T}.items():
    for grouped in [False,True]:
        X=TT.copy()
        if grouped: X.loc[X.UID.isin({"ppo19","ppo36"}),"UID"]="ppo19+36"
        ag=np.array(sorted(X.UID.unique())); rr=h.nested(X,0.25,ag); nC=X.ctrl.sum()
        for m,(d,f,ch) in rr.items(): rows.append(dict(cohort=name,holdout="ppo19+ppo36 together" if grouped else "separate (as H6)",groups=len(ag),method=m,detection=round(100*d,1),control_alarms=f"{int(round(f*nC))}/{nC}"))
R=pd.DataFrame(rows); R.to_csv(f"{O}/H6b_grouped.csv",index=False)
print(R.pivot_table(index=["cohort","method"],columns="holdout",values="detection",aggfunc="first").to_string())
print(R.pivot_table(index=["cohort","method"],columns="holdout",values="control_alarms",aggfunc="first").to_string())
