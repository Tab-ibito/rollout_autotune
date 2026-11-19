"""CPU-only step-fused DIT policy. Each instance owns one evaluation's history."""
import argparse
import json
import math
import sys
from collections import deque
from dataclasses import asdict

import numpy as np

from . import Config


ACTIONS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
LAYERS, WIDTH, HEADS, INNER, CONTEXT = 4, 128, 4, 352, 64
INPUT = 18
FORMAT = 1


def features(k, g, wait, sm, slowdown, rate, work_left, reset):
    values = [k / 32, g / 4, wait or 0, sm or 0, min(slowdown or 0, 8) / 4,
              math.log1p(rate) / 6, math.log1p(work_left) / 12, wait is not None,
              sm is not None, slowdown is not None, reset]
    return np.asarray(values, dtype=np.float32)


def action_mask(config, limits, rejected):
    return np.array([
        1 <= config.k + dk <= limits.k and 1 <= config.g + dg <= min(config.k + dk, limits.g)
        and (config.k + dk, config.g + dg) not in rejected
        for dk, dg in ACTIONS], dtype=bool)


def token(state, previous, cost, reward):
    x = np.zeros(INPUT, dtype=np.float32)
    x[:11] = state
    if previous is not None:
        x[11 + previous] = 1
    x[-2:] = (cost / 30, reward / 60)
    return x


def norm(x, gain):
    return x * (1.0 / np.sqrt(np.mean(x*x) + 1e-6)) * gain


class CPUModel:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as checkpoint:
            if int(checkpoint['format_version']) != FORMAT:
                raise ValueError('unsupported policy format')
            self.w = {k: checkpoint[k] for k in checkpoint.files if k != 'format_version'}
        self.cache = [[] for _ in range(LAYERS)]
        self.history = deque(maxlen=CONTEXT)
        self.freq = np.float32(10000) ** (-np.arange(0, WIDTH//HEADS, 2, dtype=np.float32)/(WIDTH//HEADS))

    def _step(self, x):
        w = self.w
        pos = len(self.cache[0])
        cos, sin = np.cos(pos*self.freq), np.sin(pos*self.freq)
        def rotate(a):
            p,q = a[:,::2],a[:,1::2]
            return np.stack((p*cos-q*sin,p*sin+q*cos),axis=-1).reshape(a.shape)
        x = w['embed.weight'] @ x + w['embed.bias']
        for i, cache in enumerate(self.cache):
            prefix = f'blocks.{i}.'
            def param(name):
                return w[prefix + name + '.weight']
            q,k,v = (param('qkv') @ norm(x,param('n1'))).reshape(3,HEADS,-1)
            cache.append((rotate(k), v))
            keys,values = np.asarray(cache).transpose(1,2,0,3)
            scores = np.einsum('hd,htd->ht', rotate(q),keys) / np.sqrt(np.float32(WIDTH//HEADS))
            weights = np.exp(scores-scores.max(axis=1,keepdims=True))
            weights /= weights.sum(axis=1,keepdims=True)
            x = x + param('proj') @ np.einsum('ht,htd->hd',weights,values).ravel()
            gate,up = np.split(param('up') @ norm(x,param('n2')),2)
            x = x + param('down') @ ((gate / (1 + np.exp(-np.clip(gate,-70,70)))) * up)
        return w['head.weight'] @ norm(x,w['final.weight']) + w['head.bias']

    def logits(self, x):
        self.history.append(x.copy())
        if len(self.cache[0]) == CONTEXT:
            # Recompute retained tokens so old context cannot leak through their cached states.
            self.cache = [[] for _ in range(LAYERS)]
            for old in list(self.history)[:-1]:
                self._step(old)
        return self._step(x)


class Session:
    def __init__(self, path, limits=Config(32,4), seed=7):
        self.model = CPUModel(path)
        self.current = Config()
        if limits.k < self.current.k or limits.g < self.current.g:
            raise ValueError('initial configuration exceeds limits')
        self.limits = limits
        self.rng = np.random.default_rng(seed)
        self.rejected = set()
        self.pending = None
        self.previous = None
        self.windows = 0

    def propose(self, *, wait, sm, slowdown, rate, work_left, elapsed_s, switch_s):
        """elapsed_s covers the settled observation; switch_s is disjoint transition time."""
        if self.pending is not None:
            raise RuntimeError('acknowledge the previous proposal first')
        for v in (wait, sm):
            if v is not None and not 0 <= v <= 1:
                raise ValueError('expected fractional telemetry or null')
        for v in (rate, work_left, elapsed_s, switch_s, 0 if slowdown is None else slowdown):
            if not math.isfinite(v) or v < 0:
                raise ValueError('invalid telemetry')
        c = self.current
        state = features(c.k,c.g,wait,sm,slowdown,rate,work_left,self.windows==0)
        x = token(state,self.previous,switch_s if self.windows else 0,
                  -(elapsed_s+switch_s) if self.windows else 0)
        logits = self.model.logits(x)
        mask = action_mask(c,self.limits,self.rejected)
        if wait is None or sm is None:
            mask[1:] = False
        scores = np.where(mask,logits,-np.inf)
        if self.windows < 3:
            p = np.exp((scores-scores.max())/0.5)
            action = int(self.rng.choice(len(p),p=p/p.sum()))
        else:
            action = int(np.argmax(scores))
        dk,dg = ACTIONS[action]
        self.pending = action,Config(c.k+dk,c.g+dg)
        self.windows += 1
        return self.pending[1]

    def acknowledge(self, accepted):
        if type(accepted) is not bool or self.pending is None:
            raise ValueError('expected a pending proposal and a boolean acknowledgement')
        action,target = self.pending
        if accepted:
            self.current = target
        elif action:
            self.rejected.add((target.k,target.g))
        self.previous = action
        self.pending = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weights')
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--max-workers',type=int,default=32)
    parser.add_argument('--max-replicas',type=int,default=4)
    args = parser.parse_args()
    session = Session(args.weights,Config(args.max_workers,args.max_replicas),args.seed)
    def exchange(op,c):
        print(json.dumps(dict(op=op,**asdict(c))),flush=True)
        return json.loads(sys.stdin.readline())
    if exchange('configure',session.current) is not True:
        raise RuntimeError('initial (12,3) refused')
    while (window := exchange('observe',session.current)) is not None:
        target = session.propose(**window)
        session.acknowledge(True if target==session.current else exchange('configure',target))


if __name__=='__main__':
    main()
