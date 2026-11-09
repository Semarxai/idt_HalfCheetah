import os
"""Numpy re-implementation of the frozen SB3 policies (no torch) and VecNormalize statistics."""
import zipfile, pickle, io, json, numpy as np
U=os.environ.get("IDT_CHECKPOINTS",os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","checkpoints"))
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
AGENTS={**{f"sac{s:02d}":("sac",s) for s in [1,3,5,7,9,10,11,13,14,18,19]},
        **{f"ppo{s:02d}":("ppo",s) for s in [2,4,8,11,13,17,27,30,33]},"ppo36":("ppo",19)}
def paths(algo,s):
    d=f"{U}/{algo}"          # layout checkpoints/sac, checkpoints/ppo
    if not os.path.isdir(d):  # repository layout: checkpoints_sac/sac, checkpoints_ppo/ppo
        d=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",f"checkpoints_{algo}",algo)
    return f"{d}/{algo}_claude_seed{s}.zip", f"{d}/vec_normalize_{algo}_claude_seed{s}.pkl"
class _D:
    def __init__(s,*a,**k): pass
    def __setstate__(s,st): s.__dict__.update(st if isinstance(st,dict) else {"st":st})
class _UP(pickle.Unpickler):
    def find_class(s,mod,name):
        if mod.startswith(("stable_baselines3","gymnasium","gym")): return _D
        return super().find_class(mod,name)
def load_norm(p):
    o=_UP(open(p,"rb")).load(); return o.obs_rms.mean.astype(np.float64),o.obs_rms.var.astype(np.float64),o.clip_obs,o.epsilon
def load_policy(agent,act=None):
    algo,s=AGENTS[agent]; zp,pp=paths(algo,s); z=zipfile.ZipFile(zp); sd=load_pth(z.read("policy.pth"))
    if algo=="sac":
        keys=sorted([k for k in sd if k.startswith("actor.latent_pi") and k.endswith("weight")],key=lambda k:int(k.split(".")[2]))
        Ws=[(sd[k],sd[k.replace("weight","bias")]) for k in keys]
        def f(x):
            h=x
            for W,b in Ws: h=np.maximum(h@W.T+b,0)
            return np.tanh(h@sd["actor.mu.weight"].T+sd["actor.mu.bias"])
    else:
        Ws=[(sd["mlp_extractor.policy_net.0.weight"],sd["mlp_extractor.policy_net.0.bias"]),(sd["mlp_extractor.policy_net.2.weight"],sd["mlp_extractor.policy_net.2.bias"])]
        g=(lambda h:np.maximum(h,0)) if act=="relu" else np.tanh
        def f(x):
            h=x
            for W,b in Ws: h=g(h@W.T+b)
            return np.clip(h@sd["action_net.weight"].T+sd["action_net.bias"],-1,1)
    return f, load_norm(pp)
