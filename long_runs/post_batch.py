"""H10b: per-window readings of the completed long runs (segments concatenated: 1,000,000 steps,
19,991 windows of W 500 / stride 50). Saves readings_<agent>.csv.gz (all windows; Chao-Shen entropies,
P, H_f, H_b, Delta H, plug-in H(S), mean reward) and a per-agent summary."""
import sys, os, numpy as np, pandas as pd, readings as rd
os.makedirs("readings",exist_ok=True); rows=[]
for ag in sys.argv[1:]:
    Z=[np.load(f"runs/long/{ag}/seg{s:02d}.npz") for s in range(10)]
    S=np.concatenate([z["S"] for z in Z]).astype(float); A=np.concatenate([z["A"] for z in Z]).astype(float)
    Sn=np.concatenate([z["Sn"] for z in Z]).astype(float); R=np.concatenate([z["R"] for z in Z]).astype(float)
    o=rd.window_readings(S,A,Sn,R); D=pd.DataFrame(o); D.to_csv(f"readings/readings_{ag}.csv.gz",index=False)
    ep=R.reshape(-1,1000).sum(1)
    rows.append(dict(agent=ag,steps=len(R),windows=len(D),reward_per_step=R.mean(),episode_return_median=np.median(ep),
                     episodes_below_90pct_median=int((ep<0.9*np.median(ep)).sum()),undefined_P_windows=int(np.isnan(D.P_cs).sum())))
    print(ag,"done",flush=True)
pd.DataFrame(rows).to_csv(f"readings/summary_{'_'.join(sys.argv[1:])}.csv",index=False); print(pd.DataFrame(rows).round(3).to_string(index=False))
