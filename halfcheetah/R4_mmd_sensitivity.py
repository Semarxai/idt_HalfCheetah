"""R4 (review point 4): MMD sensitivity - reference size {800, 2000} x RBF bandwidth {0.5, 1, 2} x median
heuristic. Same input, windows, window sample (300 steps), calibration (150-290) and evaluation (>=300) as H3.
Reference drawn from steps 0-7,499; per-trial seed from its UID. Output out/R4/<trial>.csv: per configuration,
max one-sided z over evaluation windows."""
import sys, os, glob, zlib, numpy as np, pandas as pd
BASE=os.environ.get("IDT_DATA_ROOT","data"); A=BASE+"/RL_sep_analysis"; O=os.path.join(os.path.dirname(os.path.abspath(__file__)),"out")
OUT=f"{O}/R4"; os.makedirs(OUT,exist_ok=True)
sys.path.insert(0,A+"/analysis"); import analyze_trajectory as at
base=at.load_baseline(A+"/analysis/baselines/baseline.json")
MAP=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","config","trial_label_map.csv")); MAP=MAP[~MAP.UID.isin({"ppo15","ppo18","ppo19","ppo21"})]
raw={os.path.basename(p).replace(".csv.gz",""):p for p in glob.glob(A+"/raw_runs/**/*.csv.gz",recursive=True)}
COLS=[f"s_{i}_z" for i in range(17)]+[f"a_{i}_z" for i in range(6)]+[f"s_next_{i}_z" for i in range(17)]
W,STR,NW=500,50,300; MULT=(0.5,1.0,2.0)
def sqd(U,V): return np.maximum((U*U).sum(1)[:,None]+(V*V).sum(1)[None,:]-2*U@V.T,0)
SH,NS=(int(sys.argv[1]),int(sys.argv[2])) if len(sys.argv)>2 else (0,1)
for ii,(_,r) in enumerate(MAP.iterrows()):
    if ii%NS!=SH: continue
    key=r.file.replace("_metrics.csv",""); out=f"{OUT}/{r.UID}__{key}.csv"
    if os.path.exists(out): continue
    X=at.z_score_normalize(pd.read_csv(raw[key]),base)[COLS].to_numpy(np.float32); nw=(len(X)-W)//STR+1
    rows=[]
    for nref in (800,2000):
        rng=np.random.default_rng(zlib.crc32((r.UID+key+str(nref)).encode()))
        Rf=X[rng.choice(7500,nref,replace=False)]; Drr=sqd(Rf,Rf); med=np.median(Drr[np.triu_indices(nref,1)])
        gam=[1/(2*med*m*m) for m in MULT]; yy=[(np.exp(-g*Drr).sum()-nref)/(nref*(nref-1)) for g in gam]
        vals=np.empty((len(MULT),nw-150))
        for j,w in enumerate(range(150,nw)):
            Xw=X[w*STR:w*STR+W][rng.choice(W,NW,replace=False)]; Dxx=sqd(Xw,Xw); Dxr=sqd(Xw,Rf)
            for i,g in enumerate(gam): vals[i,j]=(np.exp(-g*Dxx).sum()-NW)/(NW*(NW-1))+yy[i]-2*np.exp(-g*Dxr).mean()
        wid=np.arange(150,nw); cal=wid<=290; ev=wid>=300
        for i,m in enumerate(MULT):
            v=vals[i]; rows.append(dict(UID=r.UID,cond=r.perturbation,nref=nref,bw_mult=m,z=float(np.max((v[ev]-v[cal].mean())/v[cal].std(ddof=1)))))
    pd.DataFrame(rows).to_csv(out,index=False); print(r.UID,key,flush=True)
print("DONE",flush=True)
