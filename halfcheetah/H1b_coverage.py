"""H1 (completion): per-window support K, K/W, singleton f1 and doubleton f2 shares,
Good-Turing sample coverage 1 - f1/W (estimated unseen mass f1/W), and Chao1 support estimate,
for S, A, S', (S,A), (S,A,S'). Same pipeline and baseline as H1H2_coverage_bias.py."""
import sys, os, glob, numpy as np, pandas as pd
BASE=os.environ.get("IDT_DATA_ROOT","data"); A=BASE+"/RL_sep_analysis"
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"out","H1b"); os.makedirs(OUT,exist_ok=True)
sys.path.insert(0,A+"/analysis"); import analyze_trajectory as at
base=at.load_baseline(A+"/analysis/baselines/baseline.json")
MAP=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","config","trial_label_map.csv"))
raw={os.path.basename(p).replace(".csv.gz",""):p for p in glob.glob(A+"/raw_runs/**/*.csv.gz",recursive=True)}
W,STR=500,50
for _,r in MAP.iterrows():
    key=r.file.replace("_metrics.csv",""); out=f"{OUT}/{r.UID}__{key}.csv"
    if os.path.exists(out) or key not in raw: continue
    df=pd.read_csv(raw[key]); df=at.z_score_normalize(df,base); df=at.discretize(df,base); df=at.compose_system(df)
    res={}
    for col,tag in [("S","S"),("A","A"),("S_next","Sn"),("S_A","SA"),("S_A_Snext","SASn")]:
        x=pd.factorize(df[col])[0]; nw=(len(x)-W)//STR+1
        K=np.empty(nw,int); f1=np.empty(nw,int); f2=np.empty(nw,int)
        for i in range(nw):
            _,c=np.unique(x[i*STR:i*STR+W],return_counts=True); K[i]=len(c); f1[i]=(c==1).sum(); f2[i]=(c==2).sum()
        res[f"K_{tag}"]=K; res[f"f1_{tag}"]=f1; res[f"f2_{tag}"]=f2
        res[f"cov_{tag}"]=1-f1/W
        res[f"chao1_{tag}"]=K+np.where(f2>0,f1**2/(2*np.maximum(f2,1)),f1*(f1-1)/2)
    d=pd.DataFrame(res); d["window_id"]=np.arange(len(d)); d.to_csv(out,index=False); print(r.UID,key,flush=True)
print("DONE",flush=True)
