"""Batched closed-loop rollouts of a (possibly controlled) frozen model in the pinpad world.

The model only chooses actions; positions come from the environment. Each row records `sub[row]`, the abstract
action (pair index) the environment currently requires *when that row was appended* — i.e. the subgoal of the
next action — which ground-truth controllers use as their label (paper C.3), and which a group-rate controller
reads at the group's release row.

judge='task'  : success iff the given task is completed (post-training evaluation; sparse reward).
judge='any'   : success iff the visited colour sequence completes *some* pretraining task, failure as soon as it
                is not a prefix of any (base competence on its own pretraining distribution).
"""
import random
import numpy as np
import torch
import pinpad as P


def _prefix_ok(seq, tasks):
    return any(t[:len(seq)] == seq for t in tasks)


@torch.no_grad()
def rollout(model, world, n, seed, task=None, judge='task', ctrl_factory=None, device='mps', temp=1.0,
            layouts=None, max_steps=None):
    """Returns dict(success (n,), steps (n,), ids, sub, colours_visited)."""
    rng = random.Random(seed)
    tasks = [P.parse(t) for t in P.PRETRAIN_TASKS]
    T_max = (max_steps or world.T)
    states, seqs = [], []
    for i in range(n):
        st, seq = P.env_reset(world, rng, None if layouts is None else layouts[i])
        st['visited'] = []
        states.append(st); seqs.append(seq)
    H = len(seqs[0])
    L = H + 2 * T_max + 1
    ids = torch.zeros(n, L, dtype=torch.long)
    for i, s in enumerate(seqs): ids[i, :H] = torch.tensor(s)
    sub = torch.full((n, L), -1, dtype=torch.long)
    cur = H                                                     # next free column (same for all envs)
    first = (lambda st: (task[st['k']] // 2) if task is not None and st['k'] < len(task) else -1)
    sub[:, H - 1] = torch.tensor([first(st) for st in states])
    done = np.zeros(n, bool); success = np.zeros(n, bool); steps = np.zeros(n, int)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    for t in range(T_max):
        # context padded (PAD, sub = -1) to a multiple of 32: the model is causal, so logits at cur-1 are unchanged,
        # and MPS sees ~7 shapes instead of ~100 (one cached graph per shape exhausted memory under the cap)
        Tb = min(L, -(-cur // 32) * 32)
        x = ids[:, :Tb].to(device)
        ctrl = ctrl_factory(x, sub[:, :Tb].to(device)) if ctrl_factory else None
        logits = model(x, ctrl=ctrl)['logits'][:, cur - 1, P.ACT0:P.ACT0 + 4].float().cpu()
        probs = torch.softmax(logits / temp, -1)
        acts = torch.multinomial(probs, 1, generator=gen)[:, 0].numpy()
        ids[:, cur] = torch.tensor(P.ACT0 + acts)
        sub[:, cur] = sub[:, cur - 1]
        newpos = []
        for i, st in enumerate(states):
            if done[i]:
                newpos.append(P.PAD); continue
            if judge == 'task':
                tok = P.env_step(world, st, int(acts[i]), task)
            else:
                tok = _judge_any_step(world, st, int(acts[i]), tasks)
            steps[i] += 1
            newpos.append(tok)
            if st['done']:
                done[i] = True; success[i] = st['success']
        ids[:, cur + 1] = torch.tensor(newpos)
        sub[:, cur + 1] = torch.tensor([first(st) if not done[i] else -1 for i, st in enumerate(states)])
        cur += 2
        if done.all(): break
    if device == 'mps': torch.mps.empty_cache()
    return dict(success=success, steps=steps, ids=ids[:, :cur], sub=sub[:, :cur],
                visited=[st.get('visited', []) for st in states])


def _judge_any_step(world, st, a, tasks):
    y = world.step_cell(st['pos'], a, st['walls'])
    st['steps'] += 1
    if y != st['pos'] and y in st['colour_at']:
        st['visited'].append(st['colour_at'][y])
        if not _prefix_ok(st['visited'], tasks): st['done'] = True
        elif any(t == st['visited'] for t in tasks): st['done'] = st['success'] = True
    st['pos'] = y
    if st['steps'] >= world.T: st['done'] = True
    return world.pos_tok(y)
