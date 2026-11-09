import pandas as pd, numpy as np, glob, os
BASE=os.environ.get("IDT_DATA_ROOT","data")
MAP=pd.read_csv(os.path.join(BASE,"redesign/02_config/trial_label_map.csv"))
MAP=MAP[~MAP.UID.isin({"ppo15","ppo18","ppo19","ppo21"})]
files={}
for p in glob.glob(os.path.join(BASE,"RL_sep_analysis/metrics","**","*_metrics.csv"),recursive=True):
    files.setdefault(os.path.basename(p),[]).append(p)
MAP["path"]=MAP.file.map(lambda b: files.get(b,[None])[0])
TC,SPLIT=291,300
STATS=["rawMI","NMImin","NMIjoint","P","Hf","Hb","delta_H","mean_reward"]
cache=[]
for _,r in MAP.iterrows():
    d=pd.read_csv(r.path)
    d["rawMI"]=d.MI_SA_Snext
    d["NMImin"]=d.MI_SA_Snext/np.minimum(d.H_S_A,d.H_Snext).replace(0,np.nan)
    d["NMIjoint"]=d.MI_SA_Snext/d.H_S_A_Snext.replace(0,np.nan)
    pre=d[d.window_id<TC]; post=d[d.window_id>=SPLIT]
    e={}
    for s in STATS:
        mu,sd=pre[s].mean(),pre[s].std(ddof=1)
        e[s]=(np.abs(post[s].values-mu)/sd).max() if (sd and not np.isnan(sd)) else 0.0
    cache.append(dict(UID=r.UID,cond=r.perturbation,**e))
T=pd.DataFrame(cache)
P=T[T.cond!="no_noise (baseline)"]; C=T[T.cond=="no_noise (baseline)"]
print("matched false-alarm budget: z chosen per statistic to give 5 of 21 control alarms\n")
print(f"{'statistic':36s} {'z':>6s} {'ctrl':>6s} {'detection':>11s}")
for s in STATS:
    zs=np.sort(C[s].values)[::-1]
    z=(zs[4]+zs[5])/2 if len(zs)>5 else zs[4]      # threshold admitting exactly 5 controls
    ctl=int((C[s]>z).sum()); det=100*(P[s]>z).mean()
    print(f"{s:36s} {z:6.2f} {ctl:5d}/21 {det:10.1f}%")
print("\nfor reference, three-of-four at +/-3 sigma: 69.0% detection, 5/21 controls")
