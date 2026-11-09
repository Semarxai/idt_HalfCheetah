"""E1 (Dp9r Q4 + episode-boundary check): one pass over the archived raw step logs of the 21 primary agents.
(a) Per-episode readings for CUSUM: Chao-Shen P, H_f, H_b, Delta H on the 999 triplets that stay inside each
    1000-step episode (the last triplet of an episode, whose S' is the auto-reset observation, is excluded);
    mean reward over the episode's 1000 steps.
(b) Boundary sensitivity: the published 500-step windows (stride 50) recomputed with the boundary triplet
    removed from every window that contains one (Chao-Shen; same symbols and estimator otherwise).
Symbols: published pipeline (baseline.json, 3 bins) as integer codes (bijection of the published strings).
Output per trial: out/E1/<UID>__<file>.npz. Idempotent; usage: python E1_episode_readings.py <chunk> <nchunks>"""
import os, sys, glob, json, numpy as np, pandas as pd
ROOT=os.environ.get("IDT_DATA_ROOT","data"); A=ROOT+"/RL_sep_analysis"
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"out","E1"); os.makedirs(OUT,exist_ok=True)
B=json.load(open(A+"/analysis/baselines/baseline.json"))
MAP=pd.read_csv(os.environ.get("IDT_MAP",ROOT+"/redesign/02_config/trial_label_map.csv")); MAP=MAP[~MAP.UID.isin({"ppo15","ppo18","ppo19","ppo21"})]
raw={os.path.basename(p).replace(".csv.gz",""):p for p in glob.glob(A+"/raw_runs/**/*.csv.gz",recursive=True)}
W,STR=500,50
def bins(X,mean,std,edges):
    std=np.where(np.array(std)>0,std,1.0); Z=(X-np.array(mean))/std; E=np.array(edges)
    return 1+(Z>E[:,0]).astype(np.int64)+(Z>E[:,1]).astype(np.int64)
def code(Bm):
    c=np.zeros(len(Bm),np.int64)
    for j in range(Bm.shape[1]): c=c*3+(Bm[:,j]-1)
    return c
def H_cs(x):
    c=np.unique(x,return_counts=True)[1]; n=c.sum(); f1=(c==1).sum()
    if f1==n: f1=n-1
    pa=(1-f1/n)*c/n; return -np.sum(pa*np.log2(pa)/(1-(1-pa)**n))
def quantities(sym,idx):
    H={t:H_cs(sym[t][idx]) for t in sym}; I=H["SA"]+H["Sn"]-H["SASn"]; C=H["S"]+H["A"]+H["Sn"]
    P=I/C if C>0 else np.nan; Hf=H["SASn"]-H["SA"]; Hb=H["SASn"]-H["Sn"]; return P,Hf,Hb,Hf-Hb
if __name__=="__main__":
    ch,nch=int(sys.argv[1]),int(sys.argv[2]); rows=MAP.iloc[ch::nch]
    for _,r in rows.iterrows():
        key=r.file.replace("_metrics.csv",""); f=f"{OUT}/{r.UID}__{key}.npz"
        if os.path.exists(f) or key not in raw: continue
        d=pd.read_csv(raw[key])
        S=d[[f"s_{i}" for i in range(17)]].to_numpy(); Ac=d[[f"a_{i}" for i in range(6)]].to_numpy(); Sn=d[[f"s_next_{i}" for i in range(17)]].to_numpy()
        s=code(bins(S,B["s_mean"],B["s_std"],B["s_bin_edges"])); a=code(bins(Ac,B["a_mean"],B["a_std"],B["a_bin_edges"])); sn=code(bins(Sn,B["s_mean"],B["s_std"],B["s_bin_edges"]))
        sym=dict(S=s,A=a,Sn=sn,SA=s*729+a,SASn=(s*729+a)*3**17+sn)
        ep=d.episode.to_numpy(); t=d.t.to_numpy(); last=np.r_[ep[1:]!=ep[:-1],True]      # last step of each episode
        n=len(d); E=np.unique(ep); per=np.full((len(E),4),np.nan); rew=np.empty(len(E))
        for j,e in enumerate(E):
            m=np.nonzero(ep==e)[0]; inside=m[~last[m]]; per[j]=quantities(sym,inside); rew[j]=d.reward.to_numpy()[m].mean()
        nw=(n-W)//STR+1; win=np.full((nw,4),np.nan); nb=np.zeros(nw,int)
        for i in range(nw):
            idx=np.arange(i*STR,i*STR+W); keep=idx[~last[idx]]; nb[i]=W-len(keep); win[i]=quantities(sym,keep)
        np.savez(f+".tmp.npz",episode=E,ep_P=per[:,0],ep_Hf=per[:,1],ep_Hb=per[:,2],ep_dH=per[:,3],ep_reward=rew,
                 win_P=win[:,0],win_Hf=win[:,1],win_Hb=win[:,2],win_dH=win[:,3],win_nboundary=nb,steps=n,last_t=t[last])
        os.replace(f+".tmp.npz",f); print(r.UID,key,flush=True)
    print("chunk",ch,"done",flush=True)
