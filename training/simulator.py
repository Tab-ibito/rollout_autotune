"""Vectorized closed-loop throughput surrogate. Units are env steps, GPU replicas and seconds."""
import numpy as np
from rollout_autotune.dit import ACTIONS, INPUT

DELTA = np.asarray(ACTIONS)
LENGTH = 32


class Worlds:
    def __init__(self,n,rng):
        u = lambda a,b:rng.uniform(a,b,n)
        self.worker_rate = u(18,65)
        self.cpu_rate = u(115,370)
        self.replica_rate = u(55,145)
        self.efficiency = u(.25,.93)
        self.contention = u(.03,.32)
        self.cpu_penalty = u(.012,.07)
        self.cpu_knee = rng.integers(7,25,n)
        self.gpu_cap = u(100,340)
        self.mem_cap = u(22,50)
        self.replica_mem = u(2,5)
        self.worker_mem = u(.18,.60)
        self.block = u(1500,6000)
        self.load = u(3,30)
        self.drain = u(.3,5)
        self.cpu_switch = u(.1,2.5)

    def observe(self,k,g,phase,noise):
        offered = k*self.worker_rate*(1-.15*phase) / (1+self.cpu_penalty*np.maximum(k-self.cpu_knee,0))
        compute = np.minimum(self.gpu_cap,self.replica_rate*(1+self.efficiency*(g-1))/(1+self.contention*(g-1)**1.3))
        cpu = self.cpu_rate / (1+self.cpu_penalty*np.maximum(k-self.cpu_knee,0))
        rate = np.minimum(np.minimum(cpu,offered),compute)*noise
        sm = np.clip(rate/self.gpu_cap,0,1)
        wait = np.clip(1-rate/(k*self.worker_rate*(1-.15*phase)),0,1)
        slow = 1+self.contention*(g-1)*sm + .07*np.maximum(k/self.cpu_knee-1,0)
        return rate,wait,sm,slow


def make_data(worlds,rng,prior=None):
    """Generate two shuffled independent rollouts per world; returns packed histories."""
    n = len(worlds.block)
    slots = 2*LENGTH
    x = np.zeros((n,slots,INPUT),np.float32)
    masks = np.zeros((n,slots,5),bool)
    actions = np.zeros((n,slots),np.int64)
    rewards = np.zeros((n,slots),np.float32)
    terminal = np.zeros((n,slots),bool)
    lengths = rng.integers(10,LENGTH+1,(2,n))
    ix = np.arange(n)
    for segment in range(2):
        offset = lengths[0]*segment
        end = lengths[segment]
        if prior is None:
            fixed = rng.random(n)<.5
        else:
            fixed = np.ones(n,bool)
        k = np.where(fixed,12,rng.integers(4,20,n))
        g = np.where(fixed,3,rng.integers(1,5,n))
        # The initial placement is feasible for the sampled memory envelope.
        g = np.where(k*worlds.worker_mem+g*worlds.replica_mem <= worlds.mem_cap,g,1)
        k = np.minimum(k,((worlds.mem_cap-g*worlds.replica_mem)/worlds.worker_mem).astype(int))
        old = np.full(n,-1)
        cost = np.zeros(n)
        reward = np.zeros(n)
        refused = np.zeros((n,33,5),bool)
        perturb = (rng.random(n)<.8) & (prior is None)
        wl = np.where(perturb,rng.uniform(.18,.35,n),.25)
        wh = np.where(perturb,rng.uniform(.45,.75,n),.60)
        sl = np.where(perturb,rng.uniform(.45,.70,n),.60)
        sh = np.where(perturb,rng.uniform(.75,.95,n),.85)
        active_rate,wait,sm,slow = worlds.observe(k,g,0,rng.uniform(.97,1.03,n))
        for t in range(int(end.max())):
            alive = t<end
            idx = ix[alive]
            slot = offset[idx]+t
            tok = np.column_stack((k/32,g/4,wait,sm,np.minimum(slow,8)/4,
                                   np.log1p(active_rate)/6,np.log1p(np.maximum(end-t,0)*worlds.block)/12,
                                   np.ones((n,3)),np.full(n,t==0),
                                   old[:,None]==np.arange(5),cost/30,reward/60)).astype(np.float32)
            ck = k[:,None]+DELTA[:,0]
            cg = g[:,None]+DELTA[:,1]
            mask = (ck>=1)&(ck<=32)&(cg>=1)&(cg<=4)&(cg<=ck)
            mask &= ~refused[ix[:,None],ck.clip(0,32),cg.clip(0,4)]
            if prior is None or prior == 'threshold':
                a = np.select([(g>1)&(slow>1.30),(wait>wh)&(sm>sh),(wait<wl)&(sm<sl),
                               (wait>wh)&(sm<sl)&((g==1)|(slow<1.10))],[4,2,1,3],default=0)
                a = np.where(mask[ix,a],a,0)
                if prior is None:
                    random_action = np.where(mask,rng.random((n,5)),-1).argmax(axis=1)
                    a = np.where(perturb & (rng.random(n)<.15),random_action,a)
            else:
                # The evaluator only sees this task's past and current observation.
                x[idx,slot]=tok[idx]
                a = np.zeros(n,dtype=int)
                a[idx]=prior(x,idx,slot,mask[idx],t)
            nk,ng = ck[ix,a],cg[ix,a]
            rejected = nk*worlds.worker_mem+ng*worlds.replica_mem>worlds.mem_cap
            refused[ix[rejected],nk[rejected],ng[rejected]]=True
            cost = np.where(rejected,.5,(ng>g)*worlds.load+(ng<g)*worlds.drain+(nk!=k)*worlds.cpu_switch)
            k,g = np.where(rejected,k,nk),np.where(rejected,g,ng)
            active_rate,wait,sm,slow = worlds.observe(k,g,(t+1)/end,rng.uniform(.97,1.03,n))
            reward = -(worlds.block/active_rate+cost)
            x[idx,slot]=tok[idx]
            masks[idx,slot]=mask[idx]
            actions[idx,slot]=a[idx]
            rewards[idx,slot]=reward[idx]/60
            terminal[idx,slot]=t+1==end[idx]
            old=a
    return x,masks,actions,rewards,terminal
