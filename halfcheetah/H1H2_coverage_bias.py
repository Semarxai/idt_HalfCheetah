"""H1 + H2: per-window coverage diagnostics and Miller-Madow bias-corrected statistics.
Recomputes symbols from raw step logs with the published pipeline (analyze_trajectory,
shared reference baseline.json); plug-in values reproduce the stored metrics exactly."""
import sys, os, glob, numpy as np, pandas as pd
BASE=os.environ.get("IDT_DATA_ROOT","data"); A=BASE+"/RL_sep_analysis"
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"out","H1H2"); os.makedirs(OUT,exist_ok=True)
sys.path.insert(0,A+"/analysis"); import analyze_trajectory as at
base=at.load_baseline(A+"/analysis/baselines/baseline.json")
MAP=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","config","trial_label_map.csv"))
raw={os.path.basename(p).replace(".csv.gz",""):p for p in glob.glob(A+"/raw_runs/**/*.csv.gz",recursive=True)}
W,STR,LN2=500,50,np.log(2)
def win_stats(codes):
    # codes: int array per step; returns per-window plug-in H, MM H, occupied m, singletons
    n=len(codes); nw=(n-W)//STR+1; H=np.empty(nw); Hmm=np.empty(nw); m=np.empty(nw,int); s1=np.empty(nw,int)
    for i in range(nw):
        _,c=np.unique(codes[i*STR:i*STR+W],return_counts=True); p=c/W
        H[i]=-(p*np.log2(p)).sum(); m[i]=len(c); s1[i]=(c==1).sum(); Hmm[i]=H[i]+(m[i]-1)/(2*W*LN2)
    return H,Hmm,m,s1
for _,r in MAP.iterrows():
    key=r.file.replace("_metrics.csv",""); out=f"{OUT}/{r.UID}__{key}.csv"
    if os.path.exists(out) or key not in raw: continue
    df=pd.read_csv(raw[key]); df=at.z_score_normalize(df,base); df=at.discretize(df,base); df=at.compose_system(df)
    res={}
    for col,tag in [("S","S"),("A","A"),("S_next","Sn"),("S_A","SA"),("S_A_Snext","SASn")]:
        H,Hmm,m,s1=win_stats(pd.factorize(df[col])[0])
        res[f"H_{tag}"]=H; res[f"HMM_{tag}"]=Hmm; res[f"m_{tag}"]=m; res[f"single_{tag}"]=s1
    d=pd.DataFrame(res); d["window_id"]=np.arange(len(d))
    for k,pre in [("pi","H_"),("mm","HMM_")]:
        HS,HA,HSn,HSA,HT=(d[pre+t] for t in ["S","A","Sn","SA","SASn"])
        I=HSA+HSn-HT; d[f"I_{k}"]=I; d[f"P_{k}"]=I/(HS+HA+HSn)
        d[f"Hf_{k}"]=HT-HSA; d[f"Hb_{k}"]=HT-HSn; d[f"dH_{k}"]=d[f"Hf_{k}"]-d[f"Hb_{k}"]
    d["mean_reward"]=pd.read_csv([p for p in glob.glob(A+"/metrics/**/"+r.file,recursive=True)][0]).mean_reward.values[:len(d)]
    d.to_csv(out,index=False); print(r.UID,key,flush=True)
print("DONE",flush=True)
