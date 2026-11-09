"""H9: sensitivity to bin count (3, 4, 5) and window length (250, 500, 1000).
Same z-scoring and reference as the published pipeline (shared baseline.json). The published edges are
equal-width over the reference range [lo, hi]; lo = 2e1 - e2 and hi = 2e2 - e1 are recovered exactly from
the 3-bin edges and divided into k equal bins (k = 3 reproduces the published symbols). Stride 50.
Calibration: windows lying wholly before step 15,000; evaluation: windows starting at or after it (>= 300).
Per trial and configuration: max |z| over evaluation windows for P, H_f, H_b, Delta H (plug-in and
Chao-Shen) and windowed reward. Output: out/H9/<trial>.csv (one row per configuration)."""
import sys, os, glob, json, numpy as np, pandas as pd
BASE=os.environ.get("IDT_DATA_ROOT","data"); A=BASE+"/RL_sep_analysis"; O=os.path.join(os.path.dirname(os.path.abspath(__file__)),"out")
OUT=f"{O}/H9"; os.makedirs(OUT,exist_ok=True)
b=json.load(open(A+"/analysis/baselines/baseline.json"))
MAP=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","config","trial_label_map.csv")); MAP=MAP[~MAP.UID.isin({"ppo15","ppo18","ppo19","ppo21"})]
raw={os.path.basename(p).replace(".csv.gz",""):p for p in glob.glob(A+"/raw_runs/**/*.csv.gz",recursive=True)}
smu,ssd=np.array(b["s_mean"]),np.where(np.array(b["s_std"])>0,b["s_std"],1.0); amu,asd=np.array(b["a_mean"]),np.where(np.array(b["a_std"])>0,b["a_std"],1.0)
SE,AE=np.array(b["s_bin_edges"]),np.array(b["a_bin_edges"])
def edges(E,k):
    if k==3: return [E[i] for i in range(len(E))]          # published edges, exactly
    lo=2*E[:,0]-E[:,1]; hi=2*E[:,1]-E[:,0]; return [lo[i]+(hi[i]-lo[i])*np.arange(1,k)/k for i in range(len(E))]
def binned(Z,E):  # bin index 0..k-1 = number of edges strictly below the value (published rule for k=3)
    return np.stack([np.searchsorted(E[i],Z[:,i],side="left") for i in range(Z.shape[1])],1)
def code(Bm,k): return (Bm*(k**np.arange(Bm.shape[1]))).sum(1)
def H_pi(c): p=c/c.sum(); return -(p*np.log2(p)).sum()
def H_cs(c):
    n=c.sum(); f1=(c==1).sum()
    if f1==n: f1=n-1
    pa=(1-f1/n)*c/n; return -np.sum(pa*np.log2(pa)/(1-(1-pa)**n))
for _,r in MAP.iterrows():
    key=r.file.replace("_metrics.csv",""); out=f"{OUT}/{r.UID}__{key}.csv"
    if os.path.exists(out): continue
    d=pd.read_csv(raw[key])
    Zs=(d[[f"s_{i}" for i in range(17)]].to_numpy()-smu)/ssd; Za=(d[[f"a_{i}" for i in range(6)]].to_numpy()-amu)/asd
    Zn=(d[[f"s_next_{i}" for i in range(17)]].to_numpy()-smu)/ssd; rew=d.reward.to_numpy(); rows=[]
    for k in (3,4,5):
        se,ae=edges(SE,k),edges(AE,k); S=code(binned(Zs,se),k); Ac=code(binned(Za,ae),k); Sn=code(binned(Zn,se),k)
        SA=pd.factorize(pd.Series(list(zip(S,Ac))))[0]; SAS=pd.factorize(pd.Series(list(zip(SA,Sn))))[0]
        streams=[pd.factorize(x)[0] for x in (S,Ac,Sn)]+[SA,SAS]
        for W in (250,500,1000):
            nw=(len(S)-W)//50+1; cal_last=(15000-W)//50
            Hs={e:np.empty((nw,5)) for e in ("pi","cs")}
            for i in range(nw):
                for j,x in enumerate(streams):
                    c=np.unique(x[i*50:i*50+W],return_counts=True)[1]; Hs["pi"][i,j]=H_pi(c); Hs["cs"][i,j]=H_cs(c)
            mr=np.array([rew[i*50:i*50+W].mean() for i in range(nw)])
            rec=dict(UID=r.UID,cond=r.perturbation,bins=k,W=W,n_cal=cal_last+1)
            pre=slice(0,cal_last+1); post=slice(300,nw)
            zf=lambda v:float(np.nanmax(np.abs(v[post]-np.nanmean(v[pre]))/np.nanstd(v[pre],ddof=1)))  # zero-budget windows (P undefined) skipped, as in H2/H5
            for e in ("pi","cs"):
                HS,HA,HSn,HSA,HT=Hs[e].T; I=HSA+HSn-HT
                stats={"P":I/(HS+HA+HSn),"Hf":HT-HSA,"Hb":HT-HSn,"dH":(HT-HSA)-(HT-HSn)}
                for s,v in stats.items(): rec[f"z_{s}_{e}"]=zf(v)
            rec["z_reward"]=zf(mr); rows.append(rec)
    pd.DataFrame(rows).to_csv(out,index=False); print(r.UID,key,flush=True)
print("DONE",flush=True)
