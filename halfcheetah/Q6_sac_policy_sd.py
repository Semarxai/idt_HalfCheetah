"""Compare evaluation action noise (sigma 0.01) with each SAC agent's own training-time action spread.
The SAC actor is read from the saved checkpoint without torch (weights parsed from policy.pth) and evaluated in
numpy at every logged state of the agent's unperturbed run. Identity check: tanh(mean) must reproduce the
logged (deterministic, commanded) action."""
import zipfile, pickle, io, sys, glob, os, json, numpy as np, pandas as pd
U=os.environ.get("IDT_DATA_ROOT","data")+"/RL_sep_analysis"
class _Storage:
    def __init__(s,key,dtype): s.key,s.dtype=key,dtype
def load_pth(zbytes):
    z=zipfile.ZipFile(io.BytesIO(zbytes)); root=[n for n in z.namelist() if n.endswith("data.pkl")][0].rsplit("/",1)[0]
    DT={"FloatStorage":np.float32,"DoubleStorage":np.float64,"LongStorage":np.int64,"IntStorage":np.int32,"BoolStorage":np.bool_,"HalfStorage":np.float16}
    class U_(pickle.Unpickler):
        def find_class(s,mod,name):
            if mod=="torch._utils" and name=="_rebuild_tensor_v2":
                def rb(st,off,size,stride,*a):
                    buf=np.frombuffer(z.read(f"{root}/data/{st.key}"),dtype=st.dtype)
                    if len(size)==0: return buf[off]
                    return np.lib.stride_tricks.as_strided(buf[off:],shape=size,strides=[x*buf.itemsize for x in stride]).copy()
                return rb
            if mod=="torch" and name.endswith("Storage"): return name
            if mod=="collections" and name=="OrderedDict":
                import collections; return collections.OrderedDict
            if mod=="torch" and name=="device": return lambda *a: None
            return super().find_class(mod,name)
        def persistent_load(s,pid):
            _,stype,key,loc,n=pid; return _Storage(key,DT[stype if isinstance(stype,str) else stype])
    return U_(io.BytesIO(z.read(f"{root}/data.pkl"))).load()
def actor(sd):
    keys=sorted([k for k in sd if k.startswith("actor.latent_pi") and k.endswith("weight")],key=lambda k:int(k.split(".")[2]))
    Ws=[(sd[k],sd[k.replace("weight","bias")]) for k in keys]
    def f(x):
        h=x
        for W,b in Ws: h=np.maximum(h@W.T+b,0)
        mu=h@sd["actor.mu.weight"].T+sd["actor.mu.bias"]; ls=np.clip(h@sd["actor.log_std.weight"].T+sd["actor.log_std.bias"],-20,2)
        return mu,ls
    return f,[W.shape for W,_ in Ws]
rows=[]; rng=np.random.default_rng(20261108)
for s in [1,3,5,7,9,10,11,13,14,18,19]:
    z=zipfile.ZipFile(f"{U}/checkpoints/_all/SAC_Seeds/sac_claude_seed{s}.zip"); sd=load_pth(z.read("policy.pth"))
    f,arch=actor(sd)
    log=glob.glob(f"{U}/raw_runs/sac/**/sac_seed{s}_no_noise_*.csv.gz",recursive=True)[0]
    d=pd.read_csv(log); X=d[[f"s_{i}" for i in range(17)]].to_numpy(np.float32); A=d[[f"a_{i}" for i in range(6)]].to_numpy()
    mu,ls=f(X); a_det=np.tanh(mu); match=np.abs(a_det-A).max()
    idx=rng.choice(len(X),2000,replace=False); eps=rng.standard_normal((200,2000,6))
    samp=np.tanh(mu[idx][None]+np.exp(ls[idx])[None]*eps)             # training-time (stochastic) actions
    sd_exec=samp.std(0)                                                # per-state, per-dim SD of executed action
    rows.append(dict(seed=s,arch=str(arch),max_abs_det_vs_logged=float(match),
        gauss_sd_median=float(np.median(np.exp(ls))),gauss_sd_p10=float(np.percentile(np.exp(ls),10)),gauss_sd_p90=float(np.percentile(np.exp(ls),90)),
        action_sd_median=float(np.median(sd_exec)),action_sd_p10=float(np.percentile(sd_exec,10)),action_sd_p90=float(np.percentile(sd_exec,90)),
        ratio_median_to_0p01=float(np.median(sd_exec)/0.01)))
R=pd.DataFrame(rows); R.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"out","Q6_sac_policy_sd.csv"),index=False)
pd.set_option("display.width",200); print(R.round(4).to_string(index=False))
print("\nacross agents: median training-time action SD %.4f (range of agent medians %.4f-%.4f); ratio to 0.01: %.1f-%.1f"%(
    R.action_sd_median.median(),R.action_sd_median.min(),R.action_sd_median.max(),R.ratio_median_to_0p01.min(),R.ratio_median_to_0p01.max()))
