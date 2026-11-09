"""H6: screened seeds. Builds an extended label map for PPO runs whose logs exist but are not in
trial_label_map.csv (raw dirs ppo6, ppo20_p, ppo21, ppo35_p; file-name convention as in the map:
*_00_percent = 1%), then computes for every such trial, from raw logs with the published pipeline:
plug-in and Chao-Shen statistics, windowed reward (identical to the stored mean_reward, checked to 4e-9),
and the H3 MMD statistic. Outputs go to the same out/H1H2, out/H2cs, out/H3 folders."""
import sys, os, glob, re, zlib, numpy as np, pandas as pd
BASE=os.environ.get("IDT_DATA_ROOT","data"); A=BASE+"/RL_sep_analysis"; O=os.path.join(os.path.dirname(os.path.abspath(__file__)),"out")
sys.path.insert(0,A+"/analysis"); import analyze_trajectory as at
base=at.load_baseline(A+"/analysis/baselines/baseline.json")
COND=[("act_noise_00","act_noise_01%"),("act_noise_01","act_noise_01%"),("act_noise_03","act_noise_03%"),("act_noise_04","act_noise_04%"),
      ("force_torso_10N","force_torso_10N"),("force_torso_5N","force_torso_5N"),("gravity_110","gravity_110%"),("no_noise","no_noise (baseline)"),
      ("obs_noise_00","obs_noise_01%"),("obs_noise_01","obs_noise_01%"),("obs_noise_03","obs_noise_03%")]
rows=[]
for d,uid in [("ppo6","ppo06"),("ppo20_p","ppo20"),("ppo21","ppo21"),("ppo35_p","ppo35")]:
    for p in sorted(glob.glob(f"{A}/raw_runs/ppo/{d}/*.csv.gz")):
        b=os.path.basename(p)
        for k,c in COND:
            if f"_{k}" in b: rows.append(dict(file=b.replace(".csv.gz","_metrics.csv"),UID=uid,perturbation=c,path=p)); break
X=pd.DataFrame(rows); X.to_csv(f"{O}/H6_extended_label_map.csv",index=False)
W,STR,LN2=500,50,np.log(2)
def H_cs(c):
    n=c.sum(); f1=(c==1).sum()
    if f1==n: f1=n-1
    pa=(1-f1/n)*c/n; return -np.sum(pa*np.log2(pa)/(1-(1-pa)**n))
COLS=[f"s_{i}_z" for i in range(17)]+[f"a_{i}_z" for i in range(6)]+[f"s_next_{i}_z" for i in range(17)]
def sqd(U,V): return (U*U).sum(1)[:,None]+(V*V).sum(1)[None,:]-2*U@V.T
for d in ["H1H2","H2cs","H3"]: os.makedirs(f"{O}/{d}",exist_ok=True)
for _,r in X.iterrows():
    key=r.file.replace("_metrics.csv",""); name=f"{r.UID}__{key}.csv"
    if all(os.path.exists(f"{O}/{d}/{name}") for d in ["H1H2","H2cs","H3"]): continue
    raw=pd.read_csv(r.path); z=at.z_score_normalize(raw,base); df=at.compose_system(at.discretize(z,base))
    nw=(len(df)-W)//STR+1; pi={};cs={}
    for col,tag in [("S","S"),("A","A"),("S_next","Sn"),("S_A","SA"),("S_A_Snext","SASn")]:
        x=pd.factorize(df[col])[0]; Hp=np.empty(nw); Hc=np.empty(nw)
        for i in range(nw):
            c=np.unique(x[i*STR:i*STR+W],return_counts=True)[1]; p_=c/W; Hp[i]=-(p_*np.log2(p_)).sum(); Hc[i]=H_cs(c)
        pi[f"H_{tag}"]=Hp; cs[f"H_{tag}"]=Hc
    for D,k,dest in [(pi,"pi","H1H2"),(cs,"cs","H2cs")]:
        o=pd.DataFrame(D); o["window_id"]=np.arange(nw); I=o.H_SA+o.H_Sn-o.H_SASn
        o[f"I_{k}"]=I; o[f"P_{k}"]=I/(o.H_S+o.H_A+o.H_Sn); o[f"Hf_{k}"]=o.H_SASn-o.H_SA; o[f"Hb_{k}"]=o.H_SASn-o.H_Sn; o[f"dH_{k}"]=o[f"Hf_{k}"]-o[f"Hb_{k}"]
        if dest=="H1H2": o["mean_reward"]=[raw.reward.values[i*STR:i*STR+W].mean() for i in range(nw)]
        o.to_csv(f"{O}/{dest}/{name}",index=False)
    rng=np.random.default_rng(zlib.crc32((r.UID+key).encode())); Xz=z[COLS].to_numpy(float)
    Rf=Xz[rng.choice(7500,800,replace=False)]; Dm=sqd(Rf,Rf); g=1/(2*np.median(Dm[np.triu_indices(800,1)]))
    Kyy=np.exp(-g*Dm); yy=(Kyy.sum()-np.trace(Kyy))/(800*799); res=[]
    for w in range(150,nw):
        Xw=Xz[w*STR:w*STR+W][rng.choice(W,300,replace=False)]; Kxx=np.exp(-g*sqd(Xw,Xw))
        res.append((w,(Kxx.sum()-np.trace(Kxx))/(300*299)+yy-2*np.exp(-g*sqd(Xw,Rf)).mean()))
    pd.DataFrame(res,columns=["window_id","mmd2"]).to_csv(f"{O}/H3/{name}",index=False); print(r.UID,key,flush=True)
print("DONE",len(X),"trials",flush=True)
