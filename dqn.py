import os
import sys
import time
from collections import deque

import numpy as np

from env import MODEL, N_ACTIONS, OBS_SIZE, episode_score, run_episode

STEPS=200_000
HIDDEN=(128, 128)
GAMMA=0.99
N_STEP=5
LR=5e-4
BATCH=64
BUFFER_SIZE=100_000
LEARN_START=2_000
TRAIN_EVERY=4
TARGET_SYNC=1_000
EPS_START=1.0
EPS_END=0.05
EPS_DECAY=60_000
GRAD_CLIP=10.0
RANDOM_START=0.5   # chance a training episode starts at a random place in the level
BURST=8            # a random action is repeated for 1..BURST steps (a pipe needs jump held ~8 steps)
EVAL_EVERY=5_000
LOG_EVERY=1_000

# random actions favour going right, otherwise Mario dies at the first Goomba
EXPLORE_P=np.array([0.02, 0.14, 0.26, 0.14, 0.30, 0.06, 0.08, 0.03, 0.02, 0.02])
EXPLORE_P=EXPLORE_P/EXPLORE_P.sum()


class MLP:
    def __init__(self, sizes, rng=None):
        rng=rng if rng is not None else np.random.default_rng(0)
        self.sizes=tuple(sizes)
        self.weights=[(rng.standard_normal((i, o))*np.sqrt(2.0/i)).astype(np.float32) for i, o in zip(sizes[:-1], sizes[1:])]
        self.biases=[np.zeros(o, dtype=np.float32) for o in sizes[1:]]
        self.acts=None

    def forward(self, x, train=False):
        single=x.ndim==1
        a=x[None, :] if single else x
        acts=[a]

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            a=a@w+b
            if i<len(self.weights)-1:
                a=np.maximum(a, 0.0)
            acts.append(a)

        if train:
            self.acts=acts
        return a[0] if single else a

    # call right after forward(train=True); returns gradients in params() order
    def backward(self, grad_out):
        n=len(self.weights)
        grads_w=[None]*n
        grads_b=[None]*n
        g=grad_out

        for i in reversed(range(n)):
            grads_w[i]=self.acts[i].T@g
            grads_b[i]=g.sum(axis=0)
            if i>0:
                g=(g@self.weights[i].T)*(self.acts[i]>0)

        return grads_w+grads_b

    def params(self):
        return self.weights+self.biases

    def copy_from(self, other):
        for mine, theirs in zip(self.params(), other.params()):
            mine[...]=theirs

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        arrays={'sizes':np.array(self.sizes)}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f'w{i}']=w
            arrays[f'b{i}']=b
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        data=np.load(path)
        net=cls([int(s) for s in data['sizes']])
        for i in range(len(net.weights)):
            net.weights[i][...]=data[f'w{i}']
            net.biases[i][...]=data[f'b{i}']
        return net


class Adam:
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params=params
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.eps=eps
        self.t=0
        self.m=[np.zeros_like(p) for p in params]
        self.v=[np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t+=1
        c1=1-self.beta1**self.t
        c2=1-self.beta2**self.t

        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m*=self.beta1
            m+=(1-self.beta1)*g
            v*=self.beta2
            v+=(1-self.beta2)*g*g
            p-=self.lr*(m/c1)/(np.sqrt(v/c2)+self.eps)


def load_model():
    if not os.path.isfile(MODEL):
        sys.exit('No model yet. Run: python train.py')

    net=MLP.load(MODEL)
    if net.sizes[0]!=OBS_SIZE:
        sys.exit('Model does not fit this game version. Run: python train.py')
    return net


class ReplayBuffer:
    def __init__(self, capacity, rng):
        self.rng=rng
        self.cap=capacity
        self.pos=0
        self.size=0
        self.s=np.zeros((capacity, OBS_SIZE), dtype=np.float32)
        self.s2=np.zeros((capacity, OBS_SIZE), dtype=np.float32)
        self.a=np.zeros(capacity, dtype=np.int64)
        self.r=np.zeros(capacity, dtype=np.float32)
        self.terminal=np.zeros(capacity, dtype=np.float32)
        self.discount=np.ones(capacity, dtype=np.float32)

    def add(self, s, a, r, s2, terminal, discount):
        i=self.pos
        self.s[i]=s
        self.s2[i]=s2
        self.a[i]=a
        self.r[i]=r
        self.terminal[i]=terminal
        self.discount[i]=discount
        self.pos=(i+1)%self.cap
        self.size=min(self.size+1, self.cap)

    def sample(self, n):
        i=self.rng.integers(0, self.size, n)
        return self.s[i], self.a[i], self.r[i], self.s2[i], self.terminal[i], self.discount[i]


class DQN:
    def __init__(self, env, eval_env, resume=False, seed=0):
        self.env=env
        self.eval_env=eval_env
        self.rng=np.random.default_rng(seed)
        self.resume=resume

        sizes=(OBS_SIZE, *HIDDEN, N_ACTIONS)
        self.online=load_model() if resume else MLP(sizes, self.rng)
        self.target=MLP(sizes, self.rng)
        self.target.copy_from(self.online)
        self.opt=Adam(self.online.params(), LR/10 if resume else LR)   # gentle updates so continuing does not wreck a good model
        self.buffer=ReplayBuffer(BUFFER_SIZE, self.rng)

        self.pending=deque()   # recent (obs, action, reward) waiting to become n-step transitions
        self.burst_action=0
        self.burst_left=0

    def act_greedy(self, obs):
        return int(np.argmax(self.online.forward(obs)))

    def act(self, obs, eps):
        if self.burst_left>0:
            self.burst_left-=1
            return self.burst_action

        # bursts last (1+BURST)/2 steps, so start them less often to keep the random share at eps
        if self.rng.random()<eps/((1+BURST)/2):
            self.burst_action=int(self.rng.choice(N_ACTIONS, p=EXPLORE_P))
            self.burst_left=int(self.rng.integers(0, BURST))
            return self.burst_action

        return self.act_greedy(obs)

    def reset_episode(self):
        self.burst_left=0
        self.pending.clear()

        if self.rng.random()<RANDOM_START:
            column=int(self.rng.choice(self.env.start_columns))
            return self.env.reset(start_x=column*32), False
        return self.env.reset(), True

    def store(self, obs, action, reward, next_obs, terminated, truncated):
        self.pending.append((obs, action, reward))

        if terminated or truncated:
            while self.pending:
                self.flush_oldest(next_obs, terminated)
        elif len(self.pending)==N_STEP:
            self.flush_oldest(next_obs, False)

    def flush_oldest(self, next_obs, terminal):
        total=sum(GAMMA**k*r for k, (_, _, r) in enumerate(self.pending))
        obs, action, _=self.pending.popleft()
        self.buffer.add(obs, action, total, next_obs, terminal, GAMMA**(len(self.pending)+1))

    def learn(self):
        s, a, r, s2, terminal, discount=self.buffer.sample(BATCH)
        rows=np.arange(BATCH)

        # double DQN: online net picks the action, target net scores it
        best=np.argmax(self.online.forward(s2), axis=1)
        y=r+discount*(1-terminal)*self.target.forward(s2)[rows, best]

        q=self.online.forward(s, train=True)
        td=q[rows, a]-y
        loss=float(np.mean(np.where(np.abs(td)<=1, 0.5*td*td, np.abs(td)-0.5)))   # huber

        grad_q=np.zeros_like(q)
        grad_q[rows, a]=np.clip(td, -1, 1)/BATCH
        grads=self.online.backward(grad_q)

        norm=float(np.sqrt(sum((g*g).sum() for g in grads)))
        if norm>GRAD_CLIP:
            grads=[g*GRAD_CLIP/norm for g in grads]
        self.opt.step(grads)

        if not np.isfinite(loss):
            raise FloatingPointError('loss is NaN/inf: lower LR or the reward scale')
        return loss, float(q.mean())

    def evaluate(self, step, best):
        info=run_episode(self.eval_env, self.act_greedy)
        score=episode_score(info)

        if score>best:
            best=score
            self.online.save(MODEL)

        print(f'  eval step={step} distance={info["distance"]} score={info["score"]} won={info["won"]} best={best:.0f}', flush=True)
        return best

    def train(self, total_steps=STEPS):
        best=self.evaluate(0, -np.inf) if self.resume else -np.inf# never overwrite a better model

        obs, from_start=self.reset_episode()
        recent=deque(maxlen=20)
        wins=0
        loss=q_mean=0.0
        started=time.time()

        for step in range(1, total_steps+1):
            eps=EPS_END if self.resume else EPS_START+(EPS_END-EPS_START)*min(1.0, step/EPS_DECAY)
            action=self.act(obs, eps)
            next_obs, reward, terminated, truncated, info=self.env.step(action)
            self.store(obs, action, reward, next_obs, terminated, truncated)
            obs=next_obs

            if terminated or truncated:
                if from_start:   # random-start episodes
                    wins+=int(info['won'])
                    recent.append(info['distance'])
                obs, from_start=self.reset_episode()

            if self.buffer.size>=LEARN_START and step%TRAIN_EVERY==0:
                loss, q_mean=self.learn()
            if step%TARGET_SYNC==0:
                self.target.copy_from(self.online)
            if step%EVAL_EVERY==0:
                best=self.evaluate(step, best)
            if step%LOG_EVERY==0:
                print(f'step={step} eps={eps:.2f} loss={loss:.3f} mean_q={q_mean:.2f} '
                      f'avg_dist={np.mean(recent) if recent else 0:.0f} wins={wins} {int(step/(time.time()-started))}/s', flush=True)

        return self.evaluate(total_steps, best)
