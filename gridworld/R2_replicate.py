"""R2 (review point 2): replicate G1-G3 across 10 independent random seeds; report mean, SD and range of the
key quantities. Uses the unchanged G1/G2/G3 functions; only the random generator is reseeded."""
import sys, numpy as np, pandas as pd, importlib
import gw_core as g, estimators as e
import G1_truth_vs_estimate as G1
src2=open("G2_occupancy_vs_mechanism.py").read().split("R=pd.concat")[0]; G2=type(sys)("G2"); exec(src2,G2.__dict__)
src3=open("G3_accounting_vs_mmd.py").read().rsplit("\nR=pd.concat(",1)[0]
G3=type(sys)("G3"); exec(src3,G3.__dict__)
SEEDS=list(range(101,111)); part=sys.argv[1]
if part=="g1":
    rows=[]
    for s in SEEDS:
        G1.rng=np.random.default_rng(s)
        G1.main(27,(0.4,),[200,500],f"rep_{s}"); R=pd.read_csv(f"G1_results_rep_{s}.csv")
        G1.main(7,(0.2,0.0),[500],f"rep7_{s}"); R=pd.concat([R,pd.read_csv(f"G1_results_rep7_{s}.csv")])
        R["seed"]=s; rows.append(R)
    A=pd.concat(rows); A.to_csv("R2_G1_reps.csv",index=False)
    A["rel_bias"]=A.bias/A.truth.replace(0,np.nan)
    sel=A[A.stat.isin(["P","Hf","Hb","I_ASn_S"])]
    print(sel.groupby(["N","eps","W","est","stat"]).agg(truth=("truth","first"),bias_mean=("bias","mean"),bias_sd=("bias","std"),
          rel_mean=("rel_bias","mean"),rel_min=("rel_bias","min"),rel_max=("rel_bias","max"),cov=("cov_trip","mean")).round(4).to_string())
    print("out-of-bound windows total by estimator:",A.groupby("est").out_of_bound.sum().to_dict())
elif part=="g2":
    out=[]
    for s in SEEDS:
        G2.rng=np.random.default_rng(s); R=pd.concat([G2.run(7,0.2,500,200,"7x7 W=500"),G2.run(27,0.4,200,300,"27x27 W=200")]); R["seed"]=s; out.append(R)
    A=pd.concat(out); A.to_csv("R2_G2_reps.csv",index=False)
    k=A[A.stat.isin(["P","Hf","Hb","I_SSn","I_ASn_S","HS","reward"])]
    t=k.groupby(["world","construction","case","stat"]).d.agg(["mean","std","min","max"]).round(2)
    print(t.to_string())
elif part.startswith("g3"):
    lo,hi=(0,5) if part=="g3a" else (5,10); out=[]
    for s in SEEDS[lo:hi]:
        G3.rng=np.random.default_rng(s); G3.G2.rng=G3.rng
        R=pd.concat([G3.run(7,0.2,500,150,"7x7 W=500"),G3.run(27,0.4,200,200,"27x27 W=200")]); R["seed"]=s; out.append(R)
    pd.concat(out).to_csv(f"R2_G3_reps_{part}.csv",index=False); print("done",part)
elif part=="g3sum":
    A=pd.concat([pd.read_csv("R2_G3_reps_g3a.csv"),pd.read_csv("R2_G3_reps_g3b.csv")])
    print(A[A.stat.isin(["MMD","MMD_rbf","P","Hf","HS"])].groupby(["world","construction","case","stat"]).d.agg(["mean","std","min","max"]).round(2).to_string())
