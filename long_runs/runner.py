"""H10b runner: nominal (unperturbed) evaluation of a frozen agent, replicating the archived evaluation loop:
gymnasium HalfCheetah-v4, VecNormalize observation normalisation (stored statistics, training off, clip 10,
reward not normalised), deterministic policy, 1000-step episodes. As in the archived DummyVecEnv loop, the
logged s_next of an episode's last step is the auto-reset observation, and the next episode starts from a
second reset. Runs are written in segments of 100 episodes (100,000 steps); each segment starts from a reset
seeded by (base seed, segment index), so a resumed run is identical to an uninterrupted one."""
import os, sys, time, warnings, numpy as np
warnings.filterwarnings("ignore")
import gymnasium as gym, policies as P
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"runs"); SEG=100_000
AG_LIST=["sac01","sac03","sac05","sac07","sac09","sac10","sac11","sac13","sac14","sac18","sac19",
         "ppo02","ppo04","ppo08","ppo11","ppo13","ppo17","ppo27","ppo30","ppo33","ppo36"]
def base_seed(agent,tag): return 9_100_000+1000*AG_LIST.index(agent)+(0 if tag=="long" else 500)
def run_segment(agent,tag,seg,nsteps):
    act="relu" if agent in ("ppo02","ppo04","ppo08","ppo11","ppo13") else "tanh"
    pol,(mu,var,clip,eps)=P.load_policy(agent,act); sd=np.sqrt(var+eps)
    norm=lambda o: np.clip((o-mu)/sd,-clip,clip).astype(np.float32).astype(np.float64)
    env=gym.make("HalfCheetah-v4"); raw,_=env.reset(seed=base_seed(agent,tag)*100+seg); obs=norm(raw)
    S=np.empty((nsteps,17),np.float32);A=np.empty((nsteps,6),np.float32);R=np.empty(nsteps,np.float32);Sn=np.empty((nsteps,17),np.float32)
    t=0
    for k in range(nsteps):
        a=pol(obs[None])[0].astype(np.float32)
        raw,r,term,trunc,_=env.step(a); t+=1
        if term or trunc or t==1000:
            raw2,_=env.reset(); nxt=norm(raw2)                     # auto-reset observation, logged as s_next
            raw3,_=env.reset(); start=norm(raw3); t=0             # explicit reset by the evaluation loop
        else: nxt=norm(raw); start=nxt
        S[k],A[k],R[k],Sn[k]=obs,a,r,nxt; obs=start
    env.close(); return S,A,R,Sn
def run(agent,tag,nseg,seglen=SEG):
    d=f"{OUT}/{tag}/{agent}"; os.makedirs(d,exist_ok=True)
    for seg in range(nseg):
        f=f"{d}/seg{seg:02d}.npz"
        if os.path.exists(f): continue
        t0=time.time(); S,A,R,Sn=run_segment(agent,tag,seg,seglen)
        np.savez(f+".tmp.npz",S=S,A=A,R=R,Sn=Sn); os.replace(f+".tmp.npz",f)
        print(f"{agent} {tag} seg{seg:02d} done {time.time()-t0:.0f}s mean reward/step {R.mean():.3f}",flush=True)
if __name__=="__main__":
    tag=sys.argv[1]; nseg=int(sys.argv[2]); seglen=int(sys.argv[3]); agents=sys.argv[4:]
    for a in agents: run(a,tag,nseg,seglen)
