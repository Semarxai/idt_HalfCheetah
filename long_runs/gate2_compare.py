"""H10b gate 2: a fresh 50-episode nominal run per agent (new environment seed) against that agent's archived
control. Criterion declared before inspection: for P, H_f, H_b, Delta H (Chao-Shen) and mean reward, the
calibration-window mean (windows 0-290) of the new run lies within 1 calibration-window SD of the archived
control's mean. Also reported: SD ratio and the archived-calibrated max |z| of the new run (windows >= 300)."""
import glob, numpy as np, pandas as pd, readings as rd
from gate1_actions import CTRL, R
STATS=["P_cs","Hf_cs","Hb_cs","dH_cs","mean_reward"]; rows=[]
for ag in [a for a in CTRL if not a.startswith("ppo19")]:
    d=pd.read_csv(glob.glob(f"{R}/**/{CTRL[ag]}.csv.gz",recursive=True)[0])
    old=rd.window_readings(d[[f"s_{i}" for i in range(17)]].to_numpy(),d[[f"a_{i}" for i in range(6)]].to_numpy(),
                           d[[f"s_next_{i}" for i in range(17)]].to_numpy(),d.reward.to_numpy())
    z=np.load(f"runs/gate/{ag}/seg00.npz"); new=rd.window_readings(z["S"].astype(float),z["A"].astype(float),z["Sn"].astype(float),z["R"].astype(float))
    r=dict(agent=ag,reward_step_arch=d.reward.mean(),reward_step_new=float(z["R"].mean())); ok=True
    for c in STATS:
        mo,so=old[c][:291].mean(),old[c][:291].std(ddof=1); mn,sn=new[c][:291].mean(),new[c][:291].std(ddof=1)
        dz=(mn-mo)/so; r[f"{c}_dmean_sd"]=dz; r[f"{c}_sd_ratio"]=sn/so
        r[f"{c}_maxz_new_vs_arch"]=np.nanmax(np.abs(new[c][300:]-mo)/so); ok&=abs(dz)<=1
    r["pass"]=ok; rows.append(r)
D=pd.DataFrame(rows); D.to_csv("gate2_compare.csv",index=False); pd.set_option("display.width",250)
print(D[["agent","reward_step_arch","reward_step_new"]+[f"{c}_dmean_sd" for c in STATS]+[f"{c}_sd_ratio" for c in STATS]+["pass"]].round(3).to_string(index=False))
print("pass:",int(D["pass"].sum()),"/",len(D))
