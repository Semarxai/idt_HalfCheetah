"""G1: exact population quantities vs plug-in, Miller-Madow and Chao-Shen (coverage-adjusted)
estimates, over window lengths spanning HalfCheetah's sparse regime to well-sampled.
Windows are contiguous segments of one rollout, so temporal dependence is that of real operation."""
import sys, numpy as np, pandas as pd, gw_core as g, estimators as e
rng=np.random.default_rng(20261108)
def fast_rollout(T,pi,start,n):
    cpi=np.cumsum(pi,1); cT=np.cumsum(T,2); cst=np.cumsum(start); u=rng.random((n,3))
    S=np.empty(n,int);A=np.empty(n,int);Sn=np.empty(n,int); s=int(np.searchsorted(cst,rng.random()))
    for t in range(n):
        a=int(np.searchsorted(cpi[s],u[t,0])); sn=int(np.searchsorted(cT[s,a],u[t,1]))
        S[t],A[t],Sn[t]=s,a,sn; s=int(np.searchsorted(cst,u[t,2])) if sn==g.G else sn
    return S,A,Sn
def main(N,EPS,WS,tag):
  g.setup(N)
  KEYS=["P","I","Hf","Hb","dH","I_SSn","I_ASn_S","HA_S","HT","HSA"]
  rows=[]
  for eps in EPS:
      T=g.kernel(g.slip_map()); pi=g.policy(eps); p,start=g.population(T,pi); truth=g.quantities(p)
      for W in WS:
          nwin=max(10,min(300,300000//W)); S,A,Sn=fast_rollout(T,pi,start,nwin*W+2000); S,A,Sn=S[2000:],A[2000:],Sn[2000:]
          for est in ("plugin","mm","cs"):
              vals=[];kw=[];cov=[]
              for k in range(nwin):
                  sl=slice(k*W,(k+1)*W); q=e.window_quantities(S[sl],A[sl],Sn[sl],g.nS,g.nA,est); vals.append(q)
                  if est=="plugin":
                      sas=(S[sl]*g.nA+A[sl])*g.nS+Sn[sl]; r,c=e.coverage(sas); kw.append(r); cov.append(c)
              V=pd.DataFrame(vals)
              for key in KEYS:
                  rows.append(dict(eps=eps,W=W,est=est,stat=key,truth=truth[key],mean=V[key].mean(),sd=V[key].std(),
                                   bias=V[key].mean()-truth[key],rmse=np.sqrt(((V[key]-truth[key])**2).mean()),
                                   out_of_bound=int(((V[key]<0)&(key not in("dH",))).sum()+((key=="P")*(V[key]>0.5)).sum()),
                                   n_windows=nwin,KW_trip=np.mean(kw) if kw else np.nan,cov_trip=np.mean(cov) if cov else np.nan))
  R=pd.DataFrame(rows); R["N"]=N; R.to_csv(f"G1_results_{tag}.csv",index=False)
  # fill coverage columns for corrected estimators from plugin rows
  cv=R[R.est=="plugin"].groupby(["eps","W"])[["KW_trip","cov_trip"]].first()
  print(cv.round(3).to_string())
  for eps in EPS:
      print(f"\n=== eps={eps}: bias of P, Hf, I(A;S'|S) by estimator (truth in header) ===")
      sub=R[(R.eps==eps)&(R.stat.isin(["P","Hf","Hb","I_ASn_S"]))]
      tr=sub.groupby("stat").truth.first().round(4).to_dict(); print("truth:",tr)
      print(sub.pivot_table(index="W",columns=["stat","est"],values="bias").round(3).to_string())
      print("out-of-bound windows (any stat):"); print(R[R.eps==eps].groupby(["est"]).out_of_bound.sum().to_string())

if __name__=="__main__":
    main(7,(0.2,0.0),[50,100,200,350,500,1000,2000,5000,20000],"primary_7x7")
    main(27,(0.4,),[100,200,350,500,1000,2000,5000],"stress_27x27")
