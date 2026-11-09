import numpy as np, pandas as pd
M={"three-of-four":"three_of_four","P alone":"P","H(S)":"HS","MMD":"MMD","reward":"reward"}
CK=[0.25,0.5,0.75,1.0]; rates=[]; delays=[]
for part in ["7","27"]:
    Z=np.load(f"G4c_running_{part}.npz"); t=Z["t_after_onset"]; L=t/35000; w=str(Z["world"])
    conds=sorted({k.split("|")[0] for k in Z.files if "|" in k},key=lambda c:(c!="control",c))
    seeds=sorted({int(k.split("|")[1]) for k in Z.files if "|" in k})
    for lab,c in M.items():
        fin=np.sort([Z[f"control|{s}|{c}"][-1] for s in seeds])[::-1]; th=(fin[4]+fin[5])/2
        for cond in conds:
            first=[]
            for s in seeds:
                v=Z[f"{cond}|{s}|{c}"]; hit=np.nonzero(v>th)[0]; first.append(L[hit[0]] if len(hit) else np.nan)
            first=np.array(first); r=dict(world=w,cond=cond,method=lab,threshold=round(th,2))
            for ck in CK: r[f"by lambda {ck}"]=f"{100*np.mean(first<=ck+1e-9):.0f}%"
            rates.append(r)
            d=first[~np.isnan(first)]
            delays.append(dict(world=w,cond=cond,method=lab,detected=f"{len(d)}/21",
                 median_lambda_at_alarm=round(np.median(d),3) if len(d) else np.nan,
                 IQR=f"{np.percentile(d,25):.2f}-{np.percentile(d,75):.2f}" if len(d) else "",
                 median_steps_after_onset=int(np.median(d)*35000) if len(d) else np.nan))
R=pd.DataFrame(rates); D=pd.DataFrame(delays); pd.set_option("display.width",250); pd.set_option("display.max_rows",200)
print(R.to_string(index=False)); print(); print(D.to_string(index=False))
R.to_csv("G4c_alarm_by_magnitude.csv",index=False); D.to_csv("G4c_first_alarm.csv",index=False)
