import pandas as pd, numpy as np
R=pd.concat([pd.read_csv("G4b_scores_7.csv"),pd.read_csv("G4b_scores_27.csv")])
M={"three-of-four (CS)":"three_of_four","P alone (CS)":"P","H(S)":"HS","MMD":"MMD","reward":"reward"}
pd.set_option("display.width",250)
out=[]
for w,X in R.groupby("world",sort=False):
    C=X[X.cond=="control"]; th={}
    for lab,c in M.items():
        zs=np.sort(C[c].values)[::-1]; th[lab]=(zs[4]+zs[5])/2
    t=pd.DataFrame({lab:X.groupby("cond",sort=False)[c].apply(lambda v,z=th[lab]:f"{100*(v>z).mean():.0f}%") for lab,c in M.items()})
    print(f"\n#### {w}: alarm rate (21 trials each), thresholds matched to 5/21 controls"); print(t.to_string())
    print("thresholds:",{k:round(v,2) for k,v in th.items()})
    s=X.groupby("cond",sort=False)[[f"shift_{c}" for c in ["P","Hf","Hb","dH","termS","termA","HS","MMD","reward"]]].mean().round(2)
    s.columns=[c.replace("shift_","") for c in s.columns]
    print("mean signed shift (evaluation mean - calibration mean)/calibration SD:"); print(s.to_string())
    t["world"]=w; s["world"]=w; out.append((t,s))
pd.concat([o[0] for o in out]).to_csv("G4b_alarm_rates.csv"); pd.concat([o[1] for o in out]).to_csv("G4b_shifts.csv")
