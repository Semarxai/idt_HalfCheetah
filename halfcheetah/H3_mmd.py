"""H3: same-stream MMD baseline on HalfCheetah.
Input: the monitor's own 40-d stream (S, A, S') z-scored with the shared reference baseline, before binning.
Per trial: reference = 800 steps drawn from steps 0-7,499; unbiased MMD^2 with an RBF kernel
(median-heuristic bandwidth on the reference) between each window (W=500, stride 50, 300 steps
sampled) and the reference. Calibration of mean/SD on windows 150-290 (no overlap with the reference
steps); evaluation on windows >= 300, as for every other statistic. Seed per trial from its UID."""
import sys, os, glob, zlib, numpy as np, pandas as pd
BASE=os.environ.get("IDT_DATA_ROOT","data"); A=BASE+"/RL_sep_analysis"
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"out","H3"); os.makedirs(OUT,exist_ok=True)
sys.path.insert(0,A+"/analysis"); import analyze_trajectory as at
base=at.load_baseline(A+"/analysis/baselines/baseline.json")
MAP=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","config","trial_label_map.csv"))
raw={os.path.basename(p).replace(".csv.gz",""):p for p in glob.glob(A+"/raw_runs/**/*.csv.gz",recursive=True)}
COLS=[f"s_{i}_z" for i in range(17)]+[f"a_{i}_z" for i in range(6)]+[f"s_next_{i}_z" for i in range(17)]
W,STR,NREF,NWIN=500,50,800,300
def sqd(U,V): return (U*U).sum(1)[:,None]+(V*V).sum(1)[None,:]-2*U@V.T
for _,r in MAP.iterrows():
    key=r.file.replace("_metrics.csv",""); out=f"{OUT}/{r.UID}__{key}.csv"
    if os.path.exists(out) or key not in raw: continue
    rng=np.random.default_rng(zlib.crc32((r.UID+key).encode()))
    X=at.z_score_normalize(pd.read_csv(raw[key]),base)[COLS].to_numpy(float)
    Rf=X[rng.choice(7500,NREF,replace=False)]
    D=sqd(Rf,Rf); bw2=np.median(D[np.triu_indices(NREF,1)]); g=1/(2*bw2)
    Kyy=np.exp(-g*D); yy=(Kyy.sum()-np.trace(Kyy))/(NREF*(NREF-1))
    nw=(len(X)-W)//STR+1; res=[]
    for w in range(150,nw):
        Xw=X[w*STR:w*STR+W]; Xw=Xw[rng.choice(W,NWIN,replace=False)]
        Kxx=np.exp(-g*sqd(Xw,Xw)); Kxy=np.exp(-g*sqd(Xw,Rf))
        res.append((w,(Kxx.sum()-np.trace(Kxx))/(NWIN*(NWIN-1))+yy-2*Kxy.mean()))
    pd.DataFrame(res,columns=["window_id","mmd2"]).to_csv(out,index=False); print(r.UID,key,flush=True)
print("DONE",flush=True)
