"""Token-form gridworld-pinpad (ETA metacontroller paper, Appendix A.1), as an autoregressive token stream.

World: G x G grid, O uniquely coloured cells, W walls, agent; 4 cardinal actions; bumping a wall or the border
is a no-op. A colour is *visited* when the agent moves onto it from a different cell. An episode ends on
success (all task colours visited in order), on failure (a colour visited out of order), or after T steps.

Token stream of one episode (no task token: the subgoal is only inferable from behaviour):
    BOS  C_0 c_0 .. C_{O-1} c_{O-1}  WALL w_0 .. WALL w_{W-1}  SEP  p_0  a_1 p_1  a_2 p_2 ...  a_n p_n  END
  c_i / w_j / p_t are position tokens, each header cell tagged by its object token (pilot finding: with slot-only
  identity, RoPE models could not bind colour to position — probes stayed at chance).
Scored targets: actions (weight 1), positions (weight lam_obs), END (weight 1); the header is never scored.

Ground truth kept per token (never an input): the current target colour and the abstract action (pair index:
colours (0,1)->0, (2,3)->1, (4,5)->2, (6,7)->3 for O = 8), plus whether the token starts a new abstract action.
"""
import random
from collections import deque
import numpy as np

PAD, BOS, SEP, END = 0, 1, 2, 3
ACT0 = 4                                   # U, D, L, R = 4..7
COL0 = 8                                   # colour tokens C0..C{O-1} = 8..8+O-1, then WALL
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Table A1 (pretraining tasks) and the post-training task, for O = 8.
PRETRAIN_TASKS = ['0-1-4-5-0-1', '0-1-4-5-2-3', '0-1-6-7-2-3', '2-3-0-1-4-5', '2-3-6-7-2-3', '2-3-6-7-4-5',
                  '4-5-0-1-4-5', '4-5-0-1-6-7', '4-5-2-3-6-7', '6-7-2-3-0-1', '6-7-2-3-6-7', '6-7-4-5-0-1',
                  '0-1-6-7-4-5', '2-3-0-1-6-7', '4-5-2-3-0-1', '6-7-4-5-2-3']
POST_TASK = '0-1-2-3-4-5-6-7-0-1-2-3'


def parse(task):
    return [int(c) for c in task.split('-')]


class World:
    def __init__(self, G=7, O=8, W=4, T=100):
        self.G, self.O, self.W, self.T = G, O, W, T
        self.WALL = COL0 + O
        self.POS0 = self.WALL + 1
        self.vocab = self.POS0 + G * G

    def pos_tok(self, cell):
        return self.POS0 + cell[0] * self.G + cell[1]

    def tok_cell(self, tok):
        k = tok - self.POS0
        return (k // self.G, k % self.G)

    def layout(self, rng):
        cells = [(r, c) for r in range(self.G) for c in range(self.G)]
        while True:
            pick = rng.sample(cells, self.O + self.W + 1)
            colours, walls, agent = pick[:self.O], pick[self.O:self.O + self.W], pick[-1]
            if self.connected(colours, walls, agent):
                return colours, walls, agent

    def connected(self, colours, walls, agent=None):
        """Every colour (and the agent) can reach every colour, with the other colours blocking."""
        for t in range(self.O):
            D = self.dist_map(colours[t], colours, walls)
            for s in [c for i, c in enumerate(colours) if i != t] + ([agent] if agent else []):
                if not any(self.step_cell(s, a, walls) in D for a in range(4)):
                    return False
        return True

    def step_cell(self, cell, a, walls):
        r, c = cell[0] + MOVES[a][0], cell[1] + MOVES[a][1]
        if not (0 <= r < self.G and 0 <= c < self.G) or (r, c) in walls:
            return cell
        return (r, c)

    def dist_map(self, target, colours, walls):
        """BFS distances to `target`; other colour cells are blocked (stepping on them fails the task)."""
        blocked = set(walls) | (set(colours) - {target})
        D = {target: 0}
        q = deque([target])
        while q:
            x = q.popleft()
            for dr, dc in MOVES:
                y = (x[0] + dr, x[1] + dc)
                if 0 <= y[0] < self.G and 0 <= y[1] < self.G and y not in blocked and y not in D:
                    D[y] = D[x] + 1; q.append(y)
        return D


def header_tokens(world, colours, walls, agent):
    """BOS (C_i pos_i)_i (WALL pos_j)_j SEP p_0 — each cell is tagged with its object identity, as in the
    paper's observation (one-hot object per cell), so binding colour to position is a content lookup."""
    seq = [BOS]
    for i, c in enumerate(colours): seq += [COL0 + i, world.pos_tok(c)]
    for w in walls: seq += [world.WALL, world.pos_tok(w)]
    return seq + [SEP, world.pos_tok(agent)]


def expert_episode(world, task, rng, eps=0.0, layout=None):
    """Roll out the (stochastic, shortest-path) expert. Returns tokens and per-token ground truth, or None."""
    colours, walls, agent = layout if layout is not None else world.layout(rng)
    seq = header_tokens(world, colours, walls, agent)
    header = len(seq) - 1                         # index of p_0
    tgt_col = [-1] * len(seq); abs_act = [-1] * len(seq)
    colour_at = {c: i for i, c in enumerate(colours)}
    k, pos, steps = 0, agent, 0
    dmaps = {}
    while k < len(task) and steps < world.T:
        goal = task[k]
        if goal not in dmaps: dmaps[goal] = world.dist_map(colours[goal], colours, walls)
        D = dmaps[goal]
        safe = []
        for a in range(4):
            y = world.step_cell(pos, a, walls)
            if y in colour_at and y != pos and colour_at[y] != goal: continue      # would fail the task
            safe.append((a, y))
        if not safe: return None
        if eps > 0 and rng.random() < eps:
            a, y = rng.choice(safe)
        else:
            dcur = D.get(pos, 10 ** 6) if pos not in colour_at or colour_at[pos] == goal else min(
                (D[y] + 1 for _, y in safe if y in D), default=10 ** 6)
            good = [(a, y) for a, y in safe if y in D and D[y] < dcur and y != pos]
            if not good: return None
            a, y = rng.choice(good)
        # the action token and the resulting position token both belong to the current subgoal
        seq += [ACT0 + a, world.pos_tok(y)]
        tgt_col += [goal, goal]; abs_act += [goal // 2, goal // 2]
        steps += 1
        if y != pos and y in colour_at:
            if colour_at[y] == goal: k += 1
            else: return None
        pos = y
    if k < len(task): return None
    seq.append(END); tgt_col.append(-1); abs_act.append(-1)
    return dict(tokens=np.array(seq, np.int16), tgt_col=np.array(tgt_col, np.int8), abs_act=np.array(abs_act, np.int8),
                header=header, task=task)


def env_reset(world, rng, layout=None):
    colours, walls, agent = layout if layout is not None else world.layout(rng)
    seq = header_tokens(world, colours, walls, agent)
    return dict(colours=colours, walls=walls, pos=agent, k=0, steps=0, done=False, success=False,
                colour_at={c: i for i, c in enumerate(colours)}), seq


def env_step(world, st, a, task):
    """Apply action a (0..3). Returns the new position token; sets done/success."""
    y = world.step_cell(st['pos'], a, st['walls'])
    st['steps'] += 1
    if y != st['pos'] and y in st['colour_at']:
        if st['colour_at'][y] == task[st['k']]:
            st['k'] += 1
            if st['k'] == len(task): st['done'] = st['success'] = True
        else:
            st['done'] = True
    st['pos'] = y
    if st['steps'] >= world.T: st['done'] = True
    return world.pos_tok(y)


def generate(n, seed, eps, tasks=PRETRAIN_TASKS, world=None):
    world = world or World()
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        task = parse(rng.choice(tasks))
        ep = expert_episode(world, task, rng, eps=eps)
        if ep is not None: out.append(ep)
    return out


def pack(eps_list, path):
    """Store a list of episodes as one flat int16 array + offsets (+ ground truth)."""
    toks = np.concatenate([e['tokens'] for e in eps_list])
    tc = np.concatenate([e['tgt_col'] for e in eps_list]); aa = np.concatenate([e['abs_act'] for e in eps_list])
    lens = np.array([len(e['tokens']) for e in eps_list], np.int32)
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    task_ids = np.array([PRETRAIN_TASKS.index('-'.join(map(str, e['task']))) if '-'.join(map(str, e['task'])) in PRETRAIN_TASKS else -1
                         for e in eps_list], np.int16)
    np.savez_compressed(path, tokens=toks, tgt_col=tc, abs_act=aa, offsets=off, task_ids=task_ids,
                        header=np.array([e['header'] for e in eps_list], np.int16))
