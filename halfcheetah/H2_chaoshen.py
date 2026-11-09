"""H2: Chao-Shen (coverage-adjusted) entropy estimates per window, frozen after grid-world validation (G1).
All quantities from one coherent set of entropies; bound violations counted, never clipped."""
import sys, os, glob, numpy as np, pandas as pd
BASE=os.environ.get("IDT_DATA_ROOT","data"); A=BASE+"/RL_sep_analysis"
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"out","H2cs"); os.makedirs(OUT,exist_ok=True)
sys.path.insert(0,A+"/analysis"); import analyze_trajectory as at
base=at.load_baseline(A+"/analysis/baselines/baseline.json")
MAP=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","config","trial_label_map.csv"))
raw={os.path.basename(p).replace(".csv.gz",""):p for p in glob.glob(A+"/raw_runs/**/*.csv.gz",recursive=True)}
W,STR=500,50
def H_cs(c):
    n=c.sum(); f1=(c==1).sum()
    if f1==n: f1=n-1
    pa=(1-f1/n)*c/n; return -np.sum(pa*np.log2(pa)/(1-(1-pa)**n))
for _,r in MAP.iterrows():
    key=r.file.replace("_metrics.csv",""); out=f"{OUT}/{r.UID}__{key}.csv"
    if os.path.exists(out) or key not in raw: continue
    df=pd.read_csv(raw[key]); df=at.z_score_normalize(df,base); df=at.discretize(df,base); df=at.compose_system(df)
    res={}
    for col,tag in [("S","S"),("A","A"),("S_next","Sn"),("S_A","SA"),("S_A_Snext","SASn")]:
        x=pd.factorize(df[col])[0]; nw=(len(x)-W)//STR+1
        res[f"H_{tag}"]=np.array([H_cs(np.unique(x[i*STR:i*STR+W],return_counts=True)[1]) for i in range(nw)])
    d=pd.DataFrame(res); d["window_id"]=np.arange(len(d))
    I=d.H_SA+d.H_Sn-d.H_SASn; d["I_cs"]=I; d["P_cs"]=I/(d.H_S+d.H_A+d.H_Sn)
    d["Hf_cs"]=d.H_SASn-d.H_SA; d["Hb_cs"]=d.H_SASn-d.H_Sn; d["dH_cs"]=d.Hf_cs-d.Hb_cs
    d.to_csv(out,index=False); print(r.UID,key,flush=True)
print("DONE",flush=True)
