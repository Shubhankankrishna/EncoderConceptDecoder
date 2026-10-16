"""Gridworld-pinpad (Kobayashi et al. 2025, arXiv 2512.20605, Appendix A.1), tokenised for the v4-vs-LLM tiny experiment.

Environment (paper A.1.1-A.1.2): G x G grid (G = 7), O = 8 uniquely coloured cells, W = 4 walls, episode limit 100 steps.
4 actions (N, E, S, W); moving into a wall or off the grid is a no-op. A coloured cell is visited when the agent moves
onto it from a different cell. Visiting the wrong colour ends the episode (failure); visiting all task colours in order
is success. Colours, walls and the start cell are sampled without overlap at the start of every episode.
Tasks: the 16 pretraining tasks of Table A1 (compositions of the abstract subgoals 0-1, 2-3, 4-5, 6-7); held-out = the
other 20 three-pair compositions without a repeated consecutive pair; long = the post-training task (12 colours).
Expert (C.1.1): exact shortest path by BFS, uniform over the optimal actions (stochastic tie-breaking), epsilon = 0.
The path to the current target avoids every other coloured cell (stepping on one would end the episode).

Tokens (the task is latent; it never appears in the sequence):
  header  COLOR_0 pos_0 ... COLOR_7 pos_7  WALL pos x4  START pos  SEP     (27 tokens)
  steps   ACT_a POS_next  per step
  end     END after the last task colour is visited
Labels per token (for probes / split metrics, never fed to the model): subgoal pair (0-3) and target colour (0-7) of
the step whose action or position the token belongs to; -1 on the header.
"""
import random
from collections import deque
import numpy as np

G, O, W, TMAX = 7, 8, 4, 100
PAD, END, SEP, WALL, START = 0, 1, 2, 3, 4
COLOR0 = 5                      # 5..12
ACT0 = 13                       # 13..16  N E S W
POS0 = 17                       # 17..65
V = 72                          # 66 used, rounded up
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACT_NAMES = "NESW"

TRAIN_TASKS = ["0-1-4-5-0-1", "0-1-4-5-2-3", "0-1-6-7-2-3", "2-3-0-1-4-5", "2-3-6-7-2-3", "2-3-6-7-4-5",
               "4-5-0-1-4-5", "4-5-0-1-6-7", "4-5-2-3-6-7", "6-7-2-3-0-1", "6-7-2-3-6-7", "6-7-4-5-0-1",
               "0-1-6-7-4-5", "2-3-0-1-6-7", "4-5-2-3-0-1", "6-7-4-5-2-3"]
TRAIN_TASKS = [tuple(int(x) for x in t.split("-")) for t in TRAIN_TASKS]
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
ALL3 = [PAIRS[a] + PAIRS[b] + PAIRS[c] for a in range(4) for b in range(4) for c in range(4) if a != b and b != c]
HELDOUT_TASKS = [t for t in ALL3 if t not in TRAIN_TASKS]
LONG_TASK = (0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3)
assert len(TRAIN_TASKS) == 16 and len(set(TRAIN_TASKS)) == 16 and all(t in ALL3 for t in TRAIN_TASKS)
assert len(HELDOUT_TASKS) == 20 and not set(HELDOUT_TASKS) & set(TRAIN_TASKS)


def pos_tok(cell):
    return POS0 + cell


def step(cell, a, walls):
    r, c = divmod(cell, G)
    dr, dc = MOVES[a]
    r2, c2 = r + dr, c + dc
    if not (0 <= r2 < G and 0 <= c2 < G):
        return cell
    n = r2 * G + c2
    return cell if n in walls else n


def bfs_dist(target, blocked):
    """Distance to target from every cell over cells not in `blocked` (target itself is allowed). -1 = unreachable."""
    dist = np.full(G * G, -1, np.int64)
    dist[target] = 0
    q = deque([target])
    while q:
        u = q.popleft()
        r, c = divmod(u, G)
        for dr, dc in MOVES:
            r2, c2 = r + dr, c + dc
            if 0 <= r2 < G and 0 <= c2 < G:
                n = r2 * G + c2
                if dist[n] < 0 and n not in blocked:
                    dist[n] = dist[u] + 1
                    q.append(n)
    return dist


def optimal_actions(cell, target, walls, colors):
    """Set of actions on a shortest path from cell to target avoiding walls and every other coloured cell."""
    blocked = set(walls) | {colors[c] for c in range(O) if colors[c] != target and colors[c] != cell}
    dist = bfs_dist(target, blocked)            # the agent may stand on a just-visited colour: it may leave it
    if dist[cell] <= 0:
        return [], dist
    out = []
    for a in range(4):
        n = step(cell, a, walls)
        if n != cell and n not in blocked and dist[n] == dist[cell] - 1:
            out.append(a)
    return out, dist


def sample_layout(rng):
    cells = rng.sample(range(G * G), O + W + 1)
    return cells[:O], set(cells[O:O + W]), cells[O + W]


def expert_episode(task, rng, max_tries=200):
    """Returns dict(tokens, sub, tgt, opt, layout) for one successful expert episode on a fresh layout."""
    for _ in range(max_tries):
        colors, walls, start = sample_layout(rng)
        cell, acts, cells, targets, opts = start, [], [], [], []
        ok = True
        for tc in task:
            target = colors[tc]
            while cell != target:
                A, dist = optimal_actions(cell, target, walls, colors)
                if not A or len(acts) >= TMAX:
                    ok = False
                    break
                a = rng.choice(A)
                opts.append(A); acts.append(a); targets.append(tc)
                cell = step(cell, a, walls)
                cells.append(cell)
            if not ok:
                break
        if ok:
            return encode(colors, walls, start, acts, cells, targets, opts, task)
    raise RuntimeError("no feasible layout")


def header_tokens(colors, walls, start):
    h = []
    for c in range(O):
        h += [COLOR0 + c, pos_tok(colors[c])]
    for w in sorted(walls):
        h += [WALL, pos_tok(w)]
    h += [START, pos_tok(start), SEP]
    return h


def encode(colors, walls, start, acts, cells, targets, opts, task):
    h = header_tokens(colors, walls, start)
    tok, sub, tgt, kind, opt = list(h), [-1] * len(h), [-1] * len(h), [0] * len(h), [None] * len(h)
    for a, c, tc, A in zip(acts, cells, targets, opts):
        tok += [ACT0 + a, pos_tok(c)]
        sub += [tc // 2] * 2; tgt += [tc] * 2; kind += [1, 2]; opt += [A, None]
    tok.append(END); sub.append(-1); tgt.append(-1); kind.append(3); opt.append(None)
    return dict(tokens=tok, sub=sub, tgt=tgt, kind=kind, opt=opt, task=task,
                layout=dict(colors=list(colors), walls=sorted(walls), start=start))


def boundary_flags(ep):
    """For every ACT token: 1 if it is the first action after a visit (or the episode's first action) whose target is
    the FIRST colour of a pair (the next pair is latent), 2 if the target is the second colour of a pair (determined by
    the subgoal structure), else 0 (inside a segment). Header / position tokens get -1."""
    out = [-1] * len(ep["tokens"])
    prev_tgt = None
    for i, (k, tc) in enumerate(zip(ep["kind"], ep["tgt"])):
        if k != 1:
            continue
        if tc != prev_tgt:
            out[i] = 1 if tc % 2 == 0 else 2
        else:
            out[i] = 0
        prev_tgt = tc
    return out


class PinpadStream:
    """Training rows of packed expert episodes on the 16 pretraining tasks, fresh layouts every episode.
    Depends only on the seed (identical rows for every architecture)."""
    def __init__(self, T, k, B, seed=0, tasks=TRAIN_TASKS):
        import data_v4 as D
        self.D, self.T, self.k, self.B = D, T, k, B
        self.rng = random.Random(10_000 + seed)
        self.tasks = tasks
        self.buf_t, self.buf_d = [], []

    def _row(self):
        while not self.buf_t:
            seqs = [expert_episode(self.rng.choice(self.tasks), self.rng)["tokens"] for _ in range(48)]
            t, d, _ = self.D.pack_sequences(seqs, self.T, self.k)
            self.buf_t, self.buf_d = list(t[:-1]), list(d[:-1])
        return self.buf_t.pop(0), self.buf_d.pop(0)

    def next(self):
        tt, dd = zip(*[self._row() for _ in range(self.B)])
        return np.stack(tt), np.stack(dd)


def eval_episodes(kind, n, seed):
    rng = random.Random({"heldin": 1, "heldout": 2, "long": 3, "probe": 4}[kind] * 1_000_003 + seed)
    tasks = {"heldin": TRAIN_TASKS, "heldout": HELDOUT_TASKS, "long": [LONG_TASK], "probe": TRAIN_TASKS}[kind]
    return [expert_episode(tasks[i % len(tasks)], rng) for i in range(n)]


class Ident:
    """Engram canonicalisation for pinpad: identity (the text Canon maps 8,192 BPE ids)."""
    def __call__(self, ids):
        return ids


def decode(tokens):
    out = []
    for t in tokens:
        if t == PAD: out.append(".")
        elif t == END: out.append("END")
        elif t == SEP: out.append("|")
        elif t == WALL: out.append("W")
        elif t == START: out.append("S")
        elif COLOR0 <= t < COLOR0 + O: out.append(f"c{t - COLOR0}")
        elif ACT0 <= t < ACT0 + 4: out.append(ACT_NAMES[t - ACT0])
        elif POS0 <= t < POS0 + G * G: out.append(f"{divmod(t - POS0, G)}")
    return " ".join(out)


# ----------------------------------------------------------------------------- closed-loop environment (rollouts)
class Env:
    """Task-free environment for rollouts of a generative model: records the order of coloured-cell visits.
    The rollout is judged afterwards (does the visit order form a pretraining task? pair integrity?)."""
    def __init__(self, layout):
        self.colors, self.walls, self.cell = layout["colors"], set(layout["walls"]), layout["start"]
        self.visits, self.steps = [], 0
        self.color_at = {c: i for i, c in enumerate(self.colors)}

    def act(self, a):
        n = step(self.cell, a, self.walls)
        if n != self.cell and n in self.color_at:
            self.visits.append(self.color_at[n])
        self.cell = n; self.steps += 1
        return n


if __name__ == "__main__":
    rng = random.Random(0)
    ep = expert_episode(TRAIN_TASKS[0], rng)
    print(len(ep["tokens"]), decode(ep["tokens"]))
    lens = [len(expert_episode(rng.choice(TRAIN_TASKS), rng)["tokens"]) for _ in range(500)]
    print("episode tokens mean %.1f max %d" % (np.mean(lens), max(lens)))
    lens = [len(expert_episode(LONG_TASK, rng)["tokens"]) for _ in range(100)]
    print("long task tokens mean %.1f max %d" % (np.mean(lens), max(lens)))
    print("held-out tasks", len(HELDOUT_TASKS), HELDOUT_TASKS[:4])
