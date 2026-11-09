import os
"""H10b gate 1: each frozen checkpoint must reproduce the logged (deterministic) actions of that agent's archived
control run from the logged (normalised) observations. PPO tried with tanh and ReLU hidden activations."""
import glob, numpy as np, pandas as pd, policies as P
R=os.environ.get("IDT_DATA_ROOT","data")+"/RL_sep_analysis/raw_runs"
CTRL={"ppo02":"ppo_seed2_no_noise_20251226_110411","ppo04":"ppo_seed4_no_noise_20251226_191400","ppo08":"ppo_seed8_no_noise_20251226_190002",
"ppo11":"ppo_seed11_no_noise_20260101_150453","ppo13":"ppo_seed13_no_noise_20260101_175542","ppo17":"ppo_seed17_no_noise_20260101_203448",
"ppo27":"ppo_seed27_no_noise_20260102_215842","ppo30":"ppo_seed30_no_noise_20260102_220217","ppo33":"ppo_seed33_no_noise_20260102_221810",
"ppo36":"ppo_seed19_no_noise_20260102_223225","ppo19(screened)":"ppo_seed19_no_noise_20260101_180703",
"sac01":"sac_seed1_no_noise_20251226_104913","sac03":"sac_seed3_no_noise_20251223_201715","sac05":"sac_seed5_no_noise_20251226_105937",
"sac07":"sac_seed7_no_noise_20251226_180114","sac09":"sac_seed9_no_noise_20251226_204046","sac10":"sac_seed10_no_noise_20260101_141459",
"sac11":"sac_seed11_no_noise_20260101_154004","sac13":"sac_seed13_no_noise_20260101_174239","sac14":"sac_seed14_no_noise_20260101_174624",
"sac18":"sac_seed18_no_noise_20260101_144353","sac19":"sac_seed19_no_noise_20260101_175009"}
if __name__=="__main__":
    rows=[]
    for ag,f in CTRL.items():
        d=pd.read_csv(glob.glob(f"{R}/**/{f}.csv.gz",recursive=True)[0],nrows=5000)
        X=d[[f"s_{i}" for i in range(17)]].to_numpy(); A=d[[f"a_{i}" for i in range(6)]].to_numpy()
        key="ppo36" if ag.startswith("ppo19") else ag; r=dict(agent=ag,file=f)
        for act in (["tanh","relu"] if key.startswith("ppo") else ["-"]):
            pol,_=P.load_policy(key,act); r[f"maxdiff_{act}"]=float(np.abs(pol(X)-A).max())
        rows.append(r)
    D=pd.DataFrame(rows); D.to_csv("gate1_actions.csv",index=False); pd.set_option("display.width",200); print(D.to_string(index=False))
