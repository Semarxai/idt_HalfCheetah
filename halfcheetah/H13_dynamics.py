"""H13: predictive dynamics-model baseline (one-step S' from (S, A); in the spirit of Haider et al. 2023).
Per trial, an MLP (2 x 64, ReLU) is fitted on that trial's own steps 0-7,499 (healthy, z-scored with the
shared reference baseline, as for MMD). Window statistic: mean squared one-step prediction error over the
window's 500 steps. Calibration mean/SD on windows 150-290 (no overlap with training steps); evaluation on
windows >= 300; one-sided, latched - the same treatment as MMD."""
import sys, os, glob, zlib, numpy as np, pandas as pd, warnings
from sklearn.neural_network import MLPRegressor
warnings.filterwarnings("ignore")
BASE=os.environ.get("IDT_DATA_ROOT","data"); A=BASE+"/RL_sep_analysis"; O=os.path.join(os.path.dirname(os.path.abspath(__file__)),"out")
OUT=f"{O}/H13"; os.makedirs(OUT,exist_ok=True)
sys.path.insert(0,A+"/analysis"); import analyze_trajectory as at
base=at.load_baseline(A+"/analysis/baselines/baseline.json")
M=pd.concat([pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","config","trial_label_map.csv"))[["file","UID","perturbation"]],
             pd.read_csv(f"{O}/H6_extended_label_map.csv")[["file","UID","perturbation"]]])
raw={os.path.basename(p).replace(".csv.gz",""):p for p in glob.glob(A+"/raw_runs/**/*.csv.gz",recursive=True)}
XC=[f"s_{i}_z" for i in range(17)]+[f"a_{i}_z" for i in range(6)]; YC=[f"s_next_{i}_z" for i in range(17)]
W,STR=500,50
for _,r in M.iterrows():
    key=r.file.replace("_metrics.csv",""); out=f"{OUT}/{r.UID}__{key}.csv"
    if os.path.exists(out): continue
    z=at.z_score_normalize(pd.read_csv(raw[key]),base); X=z[XC].to_numpy(float); Y=z[YC].to_numpy(float)
    mdl=MLPRegressor(hidden_layer_sizes=(64,64),max_iter=150,early_stopping=True,random_state=zlib.crc32(key.encode())%2**31)
    mdl.fit(X[:7500],Y[:7500]); e=((mdl.predict(X)-Y)**2).mean(1)
    nw=(len(X)-W)//STR+1
    pd.DataFrame({"window_id":np.arange(nw),"pred_err":[e[i*STR:i*STR+W].mean() for i in range(nw)]}).to_csv(out,index=False)
    print(r.UID,key,flush=True)
print("DONE",flush=True)
