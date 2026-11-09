"""E1 long runs: per-episode readings of the H10b nominal runs (1,000 episodes per agent), defined exactly as for
the archived trials: Chao-Shen P, H_f, H_b, Delta H on the 999 within-episode triplets (the last triplet, whose S'
is the auto-reset observation, is excluded); mean reward over the 1000 steps. usage: python E1_long.py <agents>"""
import sys, os, numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import readings as rd
def H_cs(x):
    c=np.unique(x,return_counts=True)[1]; n=c.sum(); f1=(c==1).sum()
    if f1==n: f1=n-1
    pa=(1-f1/n)*c/n; return -np.sum(pa*np.log2(pa)/(1-(1-pa)**n))
def quantities(sym,idx):      # identical to E1_episode_readings.quantities
    H={t:H_cs(sym[t][idx]) for t in sym}; I=H["SA"]+H["Sn"]-H["SASn"]; C=H["S"]+H["A"]+H["Sn"]
    P=I/C if C>0 else np.nan; Hf=H["SASn"]-H["SA"]; Hb=H["SASn"]-H["Sn"]; return P,Hf,Hb,Hf-Hb
os.makedirs("E1_long",exist_ok=True)
for ag in sys.argv[1:]:
    f=f"E1_long/{ag}.npz"
    if os.path.exists(f): continue
    Z=[np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),"runs","long",ag,f"seg{s:02d}.npz")) for s in range(10)]
    S=np.concatenate([z["S"] for z in Z]).astype(float); A=np.concatenate([z["A"] for z in Z]).astype(float)
    Sn=np.concatenate([z["Sn"] for z in Z]).astype(float); R=np.concatenate([z["R"] for z in Z]).astype(float)
    sym=rd.symbols(S,A,Sn); per=np.empty((1000,4)); rew=R.reshape(1000,1000).mean(1)
    for e in range(1000): per[e]=quantities(sym,np.arange(e*1000,e*1000+999))
    np.savez(f+".tmp.npz",ep_P=per[:,0],ep_Hf=per[:,1],ep_Hb=per[:,2],ep_dH=per[:,3],ep_reward=rew); os.replace(f+".tmp.npz",f); print(ag,"done",flush=True)
