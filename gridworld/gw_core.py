"""Compact grid world with an exactly computable population record.
p(s,a,s') = mu(s) pi(a|s) T(s'|s,a), mu = stationary distribution of the recording chain
(after the goal is reached the next recorded state is drawn from the start distribution)."""
import numpy as np
from collections import deque
MOVES=[(0,-1),(1,0),(0,1),(-1,0)]               # N,E,S,W
nA=4
def setup(n=7):
    """Two rooms joined by a door in an internal wall; start column on the left, goal bottom-right.
    Slip zone: the top-left block of the right room."""
    global N,WALL_X,DOOR_Y,GOAL,STARTS,cells,idx,nS,G
    N=n; WALL_X=n//2; DOOR_Y=n//2; GOAL=(n-1,n-1); STARTS=[(0,y) for y in range(n)]
    cells=[(x,y) for y in range(n) for x in range(n) if not (x==WALL_X and y!=DOOR_Y)]
    idx={c:i for i,c in enumerate(cells)}; nS=len(cells); G=idx[GOAL]
setup(7)
def slip_map(base=0.05,zone=0.30):
    s=np.full(nS,base)
    for c,i in idx.items():
        if WALL_X< c[0] <=WALL_X+max(2,N//4) and c[1] < max(3,N//3): s[i]=zone
    return s
def step_cell(c,a):
    nx,ny=c[0]+MOVES[a][0],c[1]+MOVES[a][1]
    return (nx,ny) if (nx,ny) in idx else c
def kernel(slip,side=0.5):
    """side = share of a slip that goes to the clockwise perpendicular (0.5 = symmetric)."""
    T=np.zeros((nS,nA,nS))
    for c,i in idx.items():
        for a in range(nA):
            T[i,a,idx[step_cell(c,a)]]+=1-slip[i]
            T[i,a,idx[step_cell(c,(a+1)%4)]]+=slip[i]*side
            T[i,a,idx[step_cell(c,(a+3)%4)]]+=slip[i]*(1-side)
    return T
def bfs_dist():
    d={GOAL:0}; q=deque([GOAL])
    while q:
        c=q.popleft()
        for a in range(4):
            n=step_cell(c,a)
            if n not in d: d[n]=d[c]+1; q.append(n)
    return np.array([d[c] for c in cells])
def policy(eps):
    d=bfs_dist(); pi=np.full((nS,nA),eps/nA)
    for c,i in idx.items():
        best=min(range(4),key=lambda a:(d[idx[step_cell(c,a)]],a)); pi[i,best]+=1-eps
    return pi
def start_dist(cells_list=None):
    st=np.zeros(nS); cl=STARTS if cells_list is None else cells_list
    st[[idx[c] for c in cl]]=1/len(cl); return st
def population(T,pi,start=None):
    start=start_dist() if start is None else start
    M=np.einsum('sa,sat->st',pi,T); K=M.copy()
    K[:,G]=0; K+=np.outer(M[:,G],start); K[G]=start          # goal is never a recorded S
    w,v=np.linalg.eig(K.T); mu=np.real(v[:,np.argmin(abs(w-1))]); mu=mu/mu.sum(); mu[G]=0; mu/=mu.sum()
    return mu[:,None,None]*pi[:,:,None]*T, start             # p(s,a,s')
def H(p):
    p=p[p>0]; return -(p*np.log2(p)).sum()
def quantities(p):
    pS=p.sum((1,2)); pA=p.sum((0,2)); pSn=p.sum((0,1)); pSA=p.sum(2); pSSn=p.sum(1)
    HS,HA,HSn,HSA,HT,HSSn=H(pS),H(pA),H(pSn),H(pSA),H(p),H(pSSn)
    I=HSA+HSn-HT; IS=HS+HSn-HSSn
    return dict(HS=HS,HA=HA,HSn=HSn,HSA=HSA,HT=HT,I=I,P=I/(HS+HA+HSn),Hf=HT-HSA,Hb=HT-HSn,
                dH=(HT-HSA)-(HT-HSn),I_SSn=IS,I_ASn_S=I-IS,HA_S=HSA-HS)
def rollout(T,pi,start,n,rng):
    s=rng.choice(nS,p=start); S=np.empty(n,int);A=np.empty(n,int);Sn=np.empty(n,int)
    for t in range(n):
        a=rng.choice(nA,p=pi[s]); sn=rng.choice(nS,p=T[s,a]); S[t],A[t],Sn[t]=s,a,sn
        s=rng.choice(nS,p=start) if sn==G else sn
    return S,A,Sn
