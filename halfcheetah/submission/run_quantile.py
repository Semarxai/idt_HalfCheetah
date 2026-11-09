"""Recompute the four statistics with QUANTILE bins instead of equal-width.
Everything else identical: same trials, same W=500/stride 50, same calibration window.
Resumable: run repeatedly until it prints DONE."""
import os, re, sys, json, glob, time
import numpy as np, pandas as pd

BASE=os.environ.get("IDT_DATA_ROOT","data")
RAW=os.path.join(BASE,"RL_sep_analysis/raw_runs")
OUT=os.path.join(os.environ.get("IDT_WORK","work"),"qbin/out"); os.makedirs(OUT,exist_ok=True)
MAP=pd.read_csv(os.path.join(BASE,"redesign/02_config/trial_label_map.csv"))
MAP=MAP[~MAP.UID.isin({"ppo15","ppo18","ppo19","ppo21"})]
raws={}
for p in glob.glob(os.path.join(RAW,"**","*.csv.gz"),recursive=True):
    raws.setdefault(os.path.basename(p),[]).append(p)
MAP["raw"]=MAP.file.str.replace("_metrics.csv",".csv.gz",regex=False).map(
    lambda b: raws.get(b,[None])[0])

BODY={"back_leg":[2,3,4,11,12,13],"front_leg":[5,6,7,14,15,16],"tip":[0,1,8,9,10]}
W,STRIDE,SPLIT,NB=500,50,300,3
CAL_STEPS=15000                      # episodes 0-14

def ent(s):
    v=s.value_counts(normalize=True).values
    return float(-(v*np.log2(v)).sum())

SCOL=[f"s_{i}" for i in range(17)]; ACOL=[f"a_{i}" for i in range(6)]
NCOL=[f"s_next_{i}" for i in range(17)]

def load_json_baseline(fp):
    import json
    b=json.load(open(fp))
    return dict(mu_s=np.array(b["s_mean"]),sd_s=np.array(b["s_std"]),
                mu_a=np.array(b["a_mean"]),sd_a=np.array(b["a_std"]),
                es=np.array(b["s_bin_edges"]).T, ea=np.array(b["a_bin_edges"]).T)

def make_baseline(control_path, mode):
    c=pd.read_csv(control_path).iloc[:CAL_STEPS]
    mu_s,sd_s=c[SCOL].mean().values, c[SCOL].std().values
    mu_a,sd_a=c[ACOL].mean().values, c[ACOL].std().values
    zs=(c[SCOL].values-mu_s)/np.where(sd_s>0,sd_s,1e-12)
    za=(c[ACOL].values-mu_a)/np.where(sd_a>0,sd_a,1e-12)
    def edges(z):
        if mode=="quantile":
            return np.percentile(z,[100/NB,200/NB],axis=0)
        lo,hi=z.min(0),z.max(0); r=hi-lo
        return np.stack([lo+r/3, lo+2*r/3])
    return dict(mu_s=mu_s,sd_s=sd_s,mu_a=mu_a,sd_a=sd_a,es=edges(zs),ea=edges(za))

def process(path, mode, bl):
    d=pd.read_csv(path)
    scol,acol,ncol=SCOL,ACOL,NCOL
    Z=lambda X,mu,sd:(X.values-mu)/np.where(sd>0,sd,1e-12)
    zs,za,zn=Z(d[scol],bl["mu_s"],bl["sd_s"]),Z(d[acol],bl["mu_a"],bl["sd_a"]),Z(d[ncol],bl["mu_s"],bl["sd_s"])
    es,ea=bl["es"],bl["ea"]
    bin_=lambda z,e:(z>e[0]).astype(np.int8)+(z>e[1]).astype(np.int8)+1
    bs,ba,bn=bin_(zs,es),bin_(za,ea),bin_(zn,es)
    def j(B,idx):
        out=B[:,idx[0]].astype("U1")
        for i in idx[1:]:
            out=np.char.add(out,B[:,i].astype("U1"))
        return pd.Series(out)
    S = j(bs,BODY["back_leg"])+"|"+j(bs,BODY["front_leg"])+"|"+j(bs,BODY["tip"])
    Sn= j(bn,BODY["back_leg"])+"|"+j(bn,BODY["front_leg"])+"|"+j(bn,BODY["tip"])
    A = j(ba,list(range(6)))
    rew=d["reward"].values
    rows=[]
    n=(len(d)-W)//STRIDE+1
    for i in range(n):
        a,b=i*STRIDE,i*STRIDE+W
        s_,a_,n_=S[a:b],A[a:b],Sn[a:b]
        HS,HA,HN=ent(s_),ent(a_),ent(n_)
        HSA=ent(s_+"||"+a_); HAN=ent(a_+"||"+n_); HSAN=ent(s_+"||"+a_+"||"+n_)
        MI=HSA+HN-HSAN
        C=HS+HA+HN
        P = 0.0 if C<=0 else MI/C        # matches the published pipeline's guard
        rows.append((i,HS,HA,HN,P,HSAN-HSA,HSAN-HN,
                     (HSAN-HSA)-(HSAN-HN), rew[a:b].mean()))
    return pd.DataFrame(rows,columns=["window_id","H_S","H_A","H_Snext","P","Hf","Hb","delta_H","mean_reward"])

mode=sys.argv[1] if len(sys.argv)>1 else "quantile"
budget=float(sys.argv[2]) if len(sys.argv)>2 else 150
t0=time.time(); done=0
CTRL={u:g[g.perturbation=="no_noise (baseline)"].raw.iloc[0]
      for u,g in MAP[MAP.raw.notna()].groupby("UID")
      if (g.perturbation=="no_noise (baseline)").any()}
cache={}
for uid,g in MAP[MAP.raw.notna()].groupby("UID"):
    todo=[]
    for _,r in g.iterrows():
        tag=f"{uid}__{r.perturbation.replace(' ','_').replace('(','').replace(')','').replace('%','pct')}__{mode}.csv"
        if not os.path.exists(os.path.join(OUT,tag)): todo.append((tag,r.raw))
    if not todo: continue
    if time.time()-t0>budget: print(f"paused, {done} this batch"); sys.exit(0)
    if mode.startswith("shared"):
        bl=load_json_baseline(os.path.join(BASE,"RL_sep_analysis/analysis/baselines",
            "baseline.json" if mode=="shared" else "sac3_baseline_3bins.json"))
    else:
        bl=make_baseline(CTRL[uid],mode)
    for tag,raw in todo:
        if time.time()-t0>budget: print(f"paused, {done} this batch"); sys.exit(0)
        process(raw,mode,bl).to_csv(os.path.join(OUT,tag),index=False); done+=1
print(f"DONE mode={mode}, {done} this batch, {len(os.listdir(OUT))} files total")
