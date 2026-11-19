"""Train the two causal networks and export the policy-only inference artifact."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from rollout_autotune import Config
from rollout_autotune.dit import ACTIONS, INPUT, CONTEXT, FORMAT, CPUModel, action_mask, features, token
from model import Transformer
from simulator import Worlds,make_data


def real_history(windows_path, changes_path):
    rows=[json.loads(l) for l in windows_path.read_text().splitlines()]
    changes={r['after_episodes']:r['seconds'] for r in map(json.loads,changes_path.read_text().splitlines())}
    x=np.zeros((1,CONTEXT,INPUT),np.float32)
    masks=np.zeros((1,CONTEXT,5),bool)
    actions=np.zeros((1,CONTEXT),np.int64)
    rewards=np.zeros((1,CONTEXT),np.float32)
    done=np.zeros((1,CONTEXT),bool)
    assert 1<len(rows)<=CONTEXT+1 and rows[-1]['next'] is None
    total=sum(r['episodes'] for r in rows)
    used=steps=0
    old=None
    cost=reward=0.
    for t,(r,nxt) in enumerate(zip(rows,rows[1:])):
        used+=r['episodes']
        steps+=r['steps']
        c=Config(**r['config'])
        target=Config(**r['next'])
        assert target==Config(**nxt['config'])
        state=features(c.k,c.g,r['wait'],r['sm'],r['slowdown'],r['steps']/r['formal_s'],
                       (total-used)*steps/used,t==0)
        x[0,t]=token(state,old,cost,reward)
        masks[0,t]=action_mask(c,Config(32,4),set())
        old=ACTIONS.index((target.k-c.k,target.g-c.g))
        cost=changes.get(used,0.)
        reward=-(nxt['seconds']+cost)
        actions[0,t]=old
        rewards[0,t]=reward/60
    done[0,len(rows)-2]=True
    return x,masks,actions,rewards,done


def td_targets(r,terminal,valid,v,gamma,lam):
    following=torch.zeros_like(v[:,-1])
    out=torch.zeros_like(v)
    next_values=F.pad(v[:,1:],(0,1))
    for t in reversed(range(v.shape[1])):
        following=r[:,t]+gamma*(~terminal[:,t])*((1-lam)*next_values[:,t]+lam*following)
        following=torch.where(valid[:,t],following,0.)
        out[:,t]=following
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--worlds',type=int,default=25000)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--seed',type=int,default=82)
    p.add_argument('--real-windows',type=Path)
    p.add_argument('--real-changes',type=Path)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    critic,actor=[Transformer(n).to(device) for n in (1,5)]
    data=make_data(Worlds(args.worlds,rng),rng)
    nr=0
    if args.real_windows:
        real=real_history(args.real_windows,args.real_changes)
        nr=int(real[1].any(-1).sum())
        data=tuple(np.concatenate((a,b)) for a,b in zip(data,real))
    names=('tokens','masks','actions','rewards','done')
    counts=Counter(data[2][data[1].any(-1)].tolist())
    print('dataset',args.worlds*2,'synthetic trajectories,',nr,'real decisions',counts,flush=True)
    tensors=[torch.from_numpy(d).to(device) for d in data]
    del data
    x,masks,actions,rewards,done=tensors
    valid=masks.any(-1)
    gamma,lam,beta=.90,.90,.4
    losses={}
    batches=64
    for phase,net in [('value',critic),('policy',actor)]:
        optimizer=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-2)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs*((len(x)+batches-1)//batches))
        if phase=='policy':
            critic.eval().requires_grad_(False)
            with torch.no_grad():
                values=torch.cat([critic(a).squeeze(-1) for a in x.split(batches)])
                advantage=rewards+gamma*(~done)*F.pad(values[:,1:],(0,1))-values
        for epoch in range(args.epochs):
            if phase=='value':
                critic.eval()
                with torch.no_grad():
                    # Freeze bootstrap targets for each epoch. Resets terminate the TD recursion.
                    v=torch.cat([critic(a).squeeze(-1) for a in x.split(batches)])
                    targets=torch.cat([td_targets(rewards[i:i+512],done[i:i+512],valid[i:i+512],
                                                  v[i:i+512],gamma,lam) for i in range(0,len(x),512)])
            net.train()
            epoch_loss=[]
            for ids in torch.randperm(len(x),device=device).split(batches):
                h=net(x[ids])
                ix=valid[ids]
                if phase=='value':
                    loss=F.mse_loss(h.squeeze(-1)[ix],targets[ids][ix])
                else:
                    logits=h[ix].masked_fill(~masks[ids][ix],-torch.inf)
                    adv=advantage[ids][ix]
                    logw=(adv/(adv.std(unbiased=False)+1e-6)/beta).clamp(np.log(.05),np.log(20.))
                    weight=logw.exp()
                    loss=(weight*F.cross_entropy(logits,actions[ids][ix],reduction='none')).sum()/weight.sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                optimizer.step()
                schedule.step()
                epoch_loss.append(loss.detach())
            val=torch.stack(epoch_loss).mean().item()
            losses.setdefault(phase,[]).append(val)
            print(phase,epoch+1,val,flush=True)
        torch.save(net.state_dict(),args.out/(phase+'.pt'))
    actor.eval()
    path=args.out/'policy.npz'
    np.savez(path,format_version=FORMAT,**{k:v.cpu().numpy() for k,v in actor.state_dict().items()})
    # Compare every token including bounded-context eviction to an uncached full forward.
    cpu=CPUModel(path)
    cpu_actor=Transformer(5).eval()
    cpu_actor.load_state_dict(actor.state_dict())
    probe=rng.normal(size=(72,INPUT)).astype(np.float32)
    with torch.no_grad():
        err=max(float(np.max(np.abs(cpu.logits(t)-cpu_actor(torch.from_numpy(probe[max(0,i-63):i+1])[None])[0,-1].numpy())))
                for i,t in enumerate(probe))
    assert err<1e-4,err
    def validation(policy):
        vr=np.random.default_rng(args.seed+9001)
        worlds=Worlds(128,vr)
        _,m,_,r,d=make_data(worlds,vr,prior=policy)
        first=np.cumsum(d,axis=1)-d==0
        return -(r*first*m.any(-1)).sum(axis=1)*60
    sample_rng=np.random.default_rng(args.seed+9002)
    @torch.no_grad()
    def choose(x,idx,slot,mask,t):
        history=torch.from_numpy(x[idx,:max(slot)+1]).to(device)
        logits=actor(history)[torch.arange(len(idx),device=device),torch.as_tensor(slot,device=device)].cpu().numpy()
        logits=np.where(mask,logits,-np.inf)
        if t<3:
            probs=np.exp((logits-logits.max(1,keepdims=True))/.5)
            return (sample_rng.random(len(idx))[:,None]>(probs/probs.sum(1,keepdims=True)).cumsum(1)).sum(1).clip(max=4)
        return logits.argmax(1)
    base=validation('threshold')
    new=validation(choose)
    report=dict(seed=args.seed,synthetic_worlds=args.worlds,synthetic_trajectories=2*args.worlds,
                real_transitions=nr,train_decisions=int(valid.sum()),actions=dict(counts),
                gamma=gamma,td_lambda=lam,beta=beta,loss=losses,
                actor_parameters=sum(p.numel() for p in actor.parameters()),
                cpu_logit_max_error=err,device=device,elapsed_s=time.monotonic()-start,
                validation=dict(held_out_synthetic_worlds=len(base),threshold_mean_s=float(base.mean()),
                                policy_mean_s=float(new.mean()),fraction_faster=float((new<base).mean())))
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
