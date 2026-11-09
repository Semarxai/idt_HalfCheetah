import os
"""Per-window readings for HalfCheetah step logs, vectorised but identical to the published pipeline:
z-score with baseline.json, 3 bins (bin = 1 + [x>e0] + [x>e1], i.e. <= edges as in analyze_trajectory.discretize),
symbols S, A, S' and their joints as integer codes (a bijection of the published string symbols, so all
entropies are unchanged), Chao-Shen entropies per window (W 500, stride 50), mean reward per window."""
import json, numpy as np
B=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","config","baseline.json")))
W,STR=500,50
def bins(X,mean,std,edges):
    std=np.where(np.array(std)>0,std,1.0); Z=(X-np.array(mean))/std; E=np.array(edges)
    return 1+(Z>E[:,0]).astype(np.int64)+(Z>E[:,1]).astype(np.int64)
def code(Bm):
    c=np.zeros(len(Bm),np.int64)
    for j in range(Bm.shape[1]): c=c*3+(Bm[:,j]-1)
    return c
def symbols(S,A,Sn):
    s=code(bins(S,B["s_mean"],B["s_std"],B["s_bin_edges"])); a=code(bins(A,B["a_mean"],B["a_std"],B["a_bin_edges"]))
    sn=code(bins(Sn,B["s_mean"],B["s_std"],B["s_bin_edges"])); n3=3**17
    return dict(S=s,A=a,Sn=sn,SA=s*729+a,SASn=(s*729+a)*n3+sn)
def H_cs(c):
    n=c.sum(); f1=(c==1).sum()
    if f1==n: f1=n-1
    pa=(1-f1/n)*c/n; return -np.sum(pa*np.log2(pa)/(1-(1-pa)**n))
def H_pi(c):
    p=c/c.sum(); return -np.sum(p*np.log2(p))
def window_readings(S,A,Sn,R):
    y=symbols(S,A,Sn); n=len(S); nw=(n-W)//STR+1; out={}
    for t in ["S","A","Sn","SA","SASn"]:
        x=y[t]; out[f"H_{t}"]=np.array([H_cs(np.unique(x[i*STR:i*STR+W],return_counts=True)[1]) for i in range(nw)])
    out["Hpi_S"]=np.array([H_pi(np.unique(y["S"][i*STR:i*STR+W],return_counts=True)[1]) for i in range(nw)])
    I=out["H_SA"]+out["H_Sn"]-out["H_SASn"]; C=out["H_S"]+out["H_A"]+out["H_Sn"]
    with np.errstate(invalid="ignore",divide="ignore"): out["P_cs"]=I/C
    out["Hf_cs"]=out["H_SASn"]-out["H_SA"]; out["Hb_cs"]=out["H_SASn"]-out["H_Sn"]; out["dH_cs"]=out["Hf_cs"]-out["Hb_cs"]
    cr=np.concatenate([[0],np.cumsum(R)]); out["mean_reward"]=np.array([(cr[i*STR+W]-cr[i*STR])/W for i in range(nw)])
    out["window_id"]=np.arange(nw); return out
