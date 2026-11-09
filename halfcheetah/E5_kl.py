"""E5 (iTwq Q4): histogram KL on the binned alphabet. Per 500-step window (stride 50), KL(window || reference) with
add-1/2 smoothing over the union of symbols seen in the window or the reference; reference = steps 0-7,499 of the
same run (as for MMD, H3); symbols as in the published pipeline (baseline.json, 3 bins). Two alphabets: the joint
triplet (S, A, S') and S alone. Output per trial: out/E5/<UID>__<file>.npz. usage: python E5_kl.py <chunk> <nchunks>"""
import os, sys, glob, json, numpy as np, pandas as pd
ROOT=os.environ.get("IDT_DATA_ROOT","data"); A=ROOT+"/RL_sep_analysis"
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"out","E5"); os.makedirs(OUT,exist_ok=True)
B=json.load(open(A+"/analysis/baselines/baseline.json"))
MAP=pd.read_csv(os.environ.get("IDT_MAP",ROOT+"/redesign/02_config/trial_label_map.csv")); MAP=MAP[~MAP.UID.isin({"ppo15","ppo18","ppo19","ppo21"})]
raw={os.path.basename(p).replace(".csv.gz",""):p for p in glob.glob(A+"/raw_runs/**/*.csv.gz",recursive=True)}
W,STR,REF=500,50,7500
def bins(X,mean,std,edges):
    std=np.where(np.array(std)>0,std,1.0); Z=(X-np.array(mean))/std; E=np.array(edges)
    return 1+(Z>E[:,0]).astype(np.int64)+(Z>E[:,1]).astype(np.int64)
def code(Bm):
    c=np.zeros(len(Bm),np.int64)
    for j in range(Bm.shape[1]): c=c*3+(Bm[:,j]-1)
    return c
def kl(win,ref_u,ref_c):
    u,c=np.unique(win,return_counts=True); allu=np.union1d(u,ref_u)
    pw=np.zeros(len(allu)); pr=np.zeros(len(allu)); pw[np.searchsorted(allu,u)]=c; pr[np.searchsorted(allu,ref_u)]=ref_c
    pw=(pw+0.5)/(pw.sum()+0.5*len(allu)); pr=(pr+0.5)/(pr.sum()+0.5*len(allu)); return np.sum(pw*np.log2(pw/pr))
if __name__=="__main__":
    ch,nch=int(sys.argv[1]),int(sys.argv[2])
    for _,r in MAP.iloc[ch::nch].iterrows():
        key=r.file.replace("_metrics.csv",""); f=f"{OUT}/{r.UID}__{key}.npz"
        if os.path.exists(f) or key not in raw: continue
        d=pd.read_csv(raw[key])
        s=code(bins(d[[f"s_{i}" for i in range(17)]].to_numpy(),B["s_mean"],B["s_std"],B["s_bin_edges"]))
        a=code(bins(d[[f"a_{i}" for i in range(6)]].to_numpy(),B["a_mean"],B["a_std"],B["a_bin_edges"]))
        sn=code(bins(d[[f"s_next_{i}" for i in range(17)]].to_numpy(),B["s_mean"],B["s_std"],B["s_bin_edges"]))
        out={}
        for tag,x in [("joint",(s*729+a)*3**17+sn),("S",s)]:
            ru,rc=np.unique(x[:REF],return_counts=True); nw=(len(x)-W)//STR+1
            out[tag]=np.array([kl(x[i*STR:i*STR+W],ru,rc) if i*STR>=REF else np.nan for i in range(nw)])
        np.savez(f+".tmp.npz",kl_joint=out["joint"],kl_S=out["S"]); os.replace(f+".tmp.npz",f); print(r.UID,key,flush=True)
    print("chunk",ch,"done",flush=True)
