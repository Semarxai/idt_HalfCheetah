"""H8: exact decomposition of the coupling share on one coherent set of entropy estimates.
I(S,A;S') = I(S;S') + I(A;S'|S), hence
P = I(S;S')/C + I(A;S'|S)/C = (H(S)/C)*Cs + (H(S'|S)/C)*Cact,
Cs = I(S;S')/H(S) (situation closure), Cact = I(A;S'|S)/H(S'|S) (action term), C = H(S)+H(A)+H(S').
This script adds H(S,S') per window (plug-in and Chao-Shen); all other entropies come from out/H1H2 and out/H2cs."""
import sys, os, glob, numpy as np, pandas as pd
BASE=os.environ.get("IDT_DATA_ROOT","data"); A=BASE+"/RL_sep_analysis"; O=os.path.join(os.path.dirname(os.path.abspath(__file__)),"out")
OUT=f"{O}/H8"; os.makedirs(OUT,exist_ok=True)
sys.path.insert(0,A+"/analysis"); import analyze_trajectory as at
base=at.load_baseline(A+"/analysis/baselines/baseline.json")
MAP=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","config","trial_label_map.csv")); MAP=MAP[~MAP.UID.isin({"ppo15","ppo18","ppo19","ppo21"})]
raw={os.path.basename(p).replace(".csv.gz",""):p for p in glob.glob(A+"/raw_runs/**/*.csv.gz",recursive=True)}
W,STR=500,50
def H_cs(c):
    n=c.sum(); f1=(c==1).sum()
    if f1==n: f1=n-1
    pa=(1-f1/n)*c/n; return -np.sum(pa*np.log2(pa)/(1-(1-pa)**n))
for _,r in MAP.iterrows():
    key=r.file.replace("_metrics.csv",""); name=f"{r.UID}__{key}.csv"; out=f"{OUT}/{name}"
    if os.path.exists(out): continue
    df=at.compose_system(at.discretize(at.z_score_normalize(pd.read_csv(raw[key]),base),base))
    x=pd.factorize(df["S_Snext"])[0]; nw=(len(x)-W)//STR+1; Hp=np.empty(nw); Hc=np.empty(nw)
    for i in range(nw):
        c=np.unique(x[i*STR:i*STR+W],return_counts=True)[1]; p=c/W; Hp[i]=-(p*np.log2(p)).sum(); Hc[i]=H_cs(c)
    pi=pd.read_csv(f"{O}/H1H2/{name}"); cs=pd.read_csv(f"{O}/H2cs/{name}"); d=pd.DataFrame({"window_id":np.arange(nw)})
    for k,src,HSS in [("pi",pi,Hp),("cs",cs,Hc)]:
        HS,HA,HSn,HSA,HT=src.H_S,src.H_A,src.H_Sn,src.H_SA,src.H_SASn; C=HS+HA+HSn
        I_SSn=HS+HSn-HSS; HSn_S=HSS-HS; I_ASn_S=HSn_S-(HT-HSA)
        d[f"P_{k}"]=(HSA+HSn-HT)/C; d[f"termS_{k}"]=I_SSn/C; d[f"termA_{k}"]=I_ASn_S/C
        d[f"Cs_{k}"]=I_SSn/HS; d[f"Cact_{k}"]=I_ASn_S/HSn_S; d[f"wS_{k}"]=HS/C; d[f"wA_{k}"]=HSn_S/C
        d[f"I_SSn_{k}"]=I_SSn; d[f"I_ASn_S_{k}"]=I_ASn_S; d[f"HA_S_{k}"]=HSA-HS
    d.to_csv(out,index=False); print(r.UID,key,flush=True)
print("DONE",flush=True)
