"""R7b: detection-false-alarm curves for every row of the normalisation ablation (submission Table 3), from the
saved R7 trial scores. Nested leave-one-agent-out (H5 protocol) at ALPHA 0.05-0.50; plug-in and Chao-Shen."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import H5_nested as h
O=h.O; T=pd.read_csv(f"{O}/R7_trial_scores.csv"); h.T=T
LAB={"I":"I(S,A;S') unnormalised","NMImin":"I / min{H(S,A),H(S')}","NMIjoint":"I / H(S,A,S')","P":"P = I / C","Hf":"H_f","Hb":"H_b","dH":"Delta H"}
h.METHODS={f"{LAB[s]} [{e}]":[f"{s}_{e}"] for e in ("cs","pi") for s in LAB}|{"windowed reward":["reward"]}
rows=[]
for al in [0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50]:
    rr=h.nested(T,al)
    for m,(d,f,ch) in rr.items(): rows.append(dict(alpha=al,statistic=m,detection=round(100*d,1),control_alarms=int(round(f*21))))
R=pd.DataFrame(rows); R.to_csv(f"{O}/R7b_curves.csv",index=False)
print(R.pivot_table(index="statistic",columns="alpha",values="detection",sort=False).to_string())
print(R.pivot_table(index="statistic",columns="alpha",values="control_alarms",sort=False).to_string())
