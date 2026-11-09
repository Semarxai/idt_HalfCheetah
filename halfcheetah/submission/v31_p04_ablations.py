import pandas as pd, numpy as np, glob, os
BASE=os.environ.get("IDT_DATA_ROOT","data")
MAP=pd.read_csv(os.path.join(BASE,"redesign/02_config/trial_label_map.csv"))
MAP=MAP[~MAP.UID.isin({"ppo15","ppo18","ppo19","ppo21"})]
files={}
for p in glob.glob(os.path.join(BASE,"RL_sep_analysis/metrics","**","*_metrics.csv"),recursive=True):
    files.setdefault(os.path.basename(p),[]).append(p)
MAP["path"]=MAP.file.map(lambda b: files.get(b,[None])[0])
TC,SPLIT=291,300
rows=[]
for _,r in MAP.iterrows():
    d=pd.read_csv(r.path)
    d["rawMI"]=d.MI_SA_Snext
    d["NMImin"]=d.MI_SA_Snext/np.minimum(d.H_S_A,d.H_Snext).replace(0,np.nan)
    d["NMIjoint"]=d.MI_SA_Snext/d.H_S_A_Snext.replace(0,np.nan)
    d["Ponly"]=d.P
    pre=d[d.window_id<TC]; post=d[d.window_id>=SPLIT]
    out=dict(UID=r.UID,cond=r.perturbation)
    for s in ["rawMI","NMImin","NMIjoint","Ponly","P","Hf","Hb","delta_H","mean_reward"]:
        mu,sd=pre[s].mean(),pre[s].std(ddof=1)
        out[s]=bool((np.abs(post[s].values-mu)>3*sd).any()) if (sd and not np.isnan(sd)) else False
    rows.append(out)
R=pd.DataFrame(rows)
P=R[R.cond!="no_noise (baseline)"]; C=R[R.cond=="no_noise (baseline)"]
R["k4"]=R[["P","Hf","Hb","delta_H"]].sum(1)>=3
P4=R[R.cond!="no_noise (baseline)"]; C4=R[R.cond=="no_noise (baseline)"]
print("single-statistic detectors, identical calibration (windows 0-290), +/-3 sigma, latched\n")
print(f"{'statistic':34s} {'detection':>10s} {'control alarms':>15s}")
lbl={"rawMI":"raw I(S,A;S')  (unnormalised)",
     "NMImin":"I/min{H(S,A),H(S')}  (standard NMI)",
     "NMIjoint":"I/H(S,A,S')  (joint-normalised)",
     "Ponly":"P = I/[H(S)+H(A)+H(S')]",
     "Hf":"H_f alone","Hb":"H_b alone","delta_H":"dH alone"}
for s in ["rawMI","NMImin","NMIjoint","Ponly","Hf","Hb","delta_H"]:
    print(f"{lbl[s]:34s} {100*P[s].mean():9.1f}% {int(C[s].sum()):11d}/{len(C)}")
print(f"{'three-of-four consensus':34s} {100*P4.k4.mean():9.1f}% {int(C4.k4.sum()):11d}/{len(C4)}")
print(f"{'windowed reward':34s} {100*P.mean_reward.mean():9.1f}% {int(C.mean_reward.sum()):11d}/{len(C)}")
