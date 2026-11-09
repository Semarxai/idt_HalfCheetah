import pandas as pd, numpy as np
R=pd.concat([pd.read_csv("G4d_scores_7.csv"),pd.read_csv("G4d_scores_27.csv")])
M={"P alone (CS)":"P","three-of-four (CS)":"three_of_four","H(S)":"HS","MMD (delta)":"MMD","reward":"reward"}
rows=[]
for w,X in R.groupby("world",sort=False):
    C=X[X.cond=="control"]
    for lab,c in M.items():
        zs=np.sort(C[c].values)[::-1]; zt=(zs[4]+zs[5])/2
        for cond,Y in X.groupby("cond",sort=False):
            rows.append(dict(world=w,method=lab,cond=cond,matched=f"{100*(Y[c]>zt).mean():.0f}%",pm3=f"{100*(Y[c]>3).mean():.0f}%",z=round(zt,2)))
T=pd.DataFrame(rows); T.to_csv("G4d_summary.csv",index=False)
pd.set_option("display.width",250)
for w in T.world.unique():
    x=T[T.world==w]
    print(f"\n#### {w} — alarm rate per condition (21 trials each); threshold matched to 5/21 controls")
    print(x.pivot_table(index="cond",columns="method",values="matched",aggfunc="first",sort=False)[list(M)].to_string())
    print("thresholds:",x.groupby("method").z.first().to_dict())
    print(f"at +-3 sigma:")
    print(x.pivot_table(index="cond",columns="method",values="pm3",aggfunc="first",sort=False)[list(M)].to_string())
