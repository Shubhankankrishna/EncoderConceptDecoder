"""V9 data additions (draft §10.4-§10.5): the colour-count task and the S6 calibration split. The frozen V8
generator, verifier and splits are untouched; colour-count has its own parser dispatch, verifier and manifest.

Colour-count: header [GNAV, COL c*, DONE, SEP]; body of L colour tokens (L in 8, 12, ..., 40), uniform over 3
colours; response [NODE0 + count(c*), DONE, SEP, SEP] with count <= 23 (resampled otherwise); response-only loss.
"""
import hashlib, json
import numpy as np
import torch
import v9paths
from v9paths import DATA, SPEC, digest, read_json, write_json
from data import PAD, EOS, GNAV, SEP, TO, DONE, NODE0, COL0, N_COL, N_LABELS, gen_symbolic, parse_prompt, verify, Mixer, Text, fixed_text

CC_DIR = v9paths.V9 / 'data_v9'
CC_SEEDS = dict(dev=97400, calibration=97401, test=97402)
CC_COUNTS = dict(dev=512, calibration=512, test=2048)
CALIB_SEED = 97500
CALIB_SUITES = [('standard', 10, 'graph', False, 'shuffle', 'random'), ('lookup', 10, 'lookup', False, 'shuffle', 'random'), ('compose', 10, 'compose', False, 'shuffle', 'random'),
                ('pointer', 10, 'pointer', False, 'shuffle', 'random'), ('hard10', 10, 'graph', True, 'shuffle', 'random'), ('bind', 10, 'bind', False, 'shuffle', 'random'),
                ('shifted', 10, 'graph', False, 'shifted', 'ordered')]


def is_colourcount(prompt):
    p = list(map(int, prompt[:4]))
    return len(p) == 4 and p[0] == GNAV and COL0 <= p[1] < COL0 + N_COL and p[2] == DONE and p[3] == SEP


def parse_cc(prompt):
    """Colour-count parser; raises on any other format (including every V8 task)."""
    p = list(map(int, prompt))
    if not is_colourcount(p) or (len(p) - 4) % 4 or not (8 <= len(p) - 4 <= 40) or not all(COL0 <= t < COL0 + N_COL for t in p[4:]):
        raise ValueError('not a colour-count prompt')
    target = p[1] - COL0; body = [t - COL0 for t in p[4:]]
    return target, body


def verify_cc(prompt, response):
    target, body = parse_cc(prompt); want = [NODE0 + body.count(target), DONE, SEP, SEP]
    r = list(map(int, response)); ok = r == want
    return dict(success=ok, reason='accepted' if ok else 'wrong_count', format_ok=len(r) == 4)


def dispatch_verify(prompt, response):
    """Task dispatch: colour-count by its unique header, every other prompt through the frozen V8 verifier."""
    return verify_cc(prompt, response) if is_colourcount(prompt) else verify(prompt, response)


def canonical_cc(prompt):
    target, body = parse_cc(prompt)
    return hashlib.sha256(json.dumps(['colourcount', target, body]).encode()).hexdigest()


def gen_cc(rng, context=192):
    while True:
        target = int(rng.randint(N_COL)); L = int(rng.choice(np.arange(8, 41, 4))); body = rng.randint(N_COL, size=L)
        count = int((body == target).sum())
        if count > 23:
            continue
        prompt = [GNAV, COL0 + target, DONE, SEP] + [COL0 + int(b) for b in body]
        response = [NODE0 + count, DONE, SEP, SEP]
        x = np.full(context, PAD, np.int64); docs = np.full(context, -1, np.int64); m = np.zeros(context, np.float32)
        n = len(prompt) + 4; x[:n] = prompt + response; docs[:n] = 0; m[len(prompt) - 1:n - 1] = 1
        return x, m, docs, dict(task='colourcount', prompt_len=len(prompt), response_len=4, canonical=canonical_cc(prompt))


def build():
    """Colour-count dev/calibration/test manifests (content-hash disjoint) and the calibration split of V8 tasks."""
    CC_DIR.mkdir(exist_ok=True)
    if (CC_DIR / 'manifest.json').exists():
        raise SystemExit('V9 data already built')
    seen = set(); man = dict(colourcount={}, calibration={})
    for split in ('test', 'dev', 'calibration'):
        rng = np.random.RandomState(CC_SEEDS[split]); rows = []
        while len(rows) < CC_COUNTS[split]:
            r = gen_cc(rng)
            if r[3]['canonical'] in seen:
                continue
            seen.add(r[3]['canonical']); rows.append(r)
        save(CC_DIR / f'colourcount_{split}.npz', rows); man['colourcount'][split] = dict(n=len(rows), seed=CC_SEEDS[split], sha256=digest(CC_DIR / f'colourcount_{split}.npz'))
    (CC_DIR / 'colourcount_eval_hashes.txt').write_text('\n'.join(sorted(seen)) + '\n')
    excluded = set((DATA / 'evaluation_symbolic_hashes.txt').read_text().split()); cal_seen = set()
    for i, (name, n, task, hard, layout, labels) in enumerate(CALIB_SUITES):
        rng = np.random.RandomState(CALIB_SEED + i); rows = []
        while len(rows) < 256:
            r = gen_symbolic(rng, n, task, hard, 4, layout, 192, labels)
            if r[3]['canonical'] in excluded or r[3]['canonical'] in cal_seen:
                continue
            cal_seen.add(r[3]['canonical']); rows.append(r)
        save(CC_DIR / f'calibration_{name}.npz', rows); man['calibration'][name] = dict(n=256, seed=CALIB_SEED + i, sha256=digest(CC_DIR / f'calibration_{name}.npz'))
    man['calibration_text'] = 'base and continual TRAIN documents, fixed_text seed 97600 (old-task inputs for gate telemetry, never evaluation)'
    man['eval_hashes_sha256'] = digest(CC_DIR / 'colourcount_eval_hashes.txt')
    write_json(CC_DIR / 'manifest.json', man)
    return man


def save(path, rows):
    np.savez(path, x=np.stack([r[0] for r in rows]), m=np.stack([r[1] for r in rows]), docs=np.stack([r[2] for r in rows]), prompt_len=np.array([r[3]['prompt_len'] for r in rows]))


def load_npz(name):
    return np.load(CC_DIR / f'{name}.npz')


class CCMixer:
    """Phase-data mixer for colour-count with the frozen Mixer's interface; excludes dev/calibration/test items."""
    def __init__(self, seed, batch=64, context=192):
        self.rng = np.random.RandomState(seed); self.batch_size = batch; self.context = context
        self.excluded = set((CC_DIR / 'colourcount_eval_hashes.txt').read_text().split())
        self.seen_docs = set(); self.seen_symbols = set(); self.seen_tasks = {}; self.targets = {}; self.slots = 0; self.nonpad = 0; self.source_targets = {'edu': 0, 'ts': 0}

    def batch(self, device='cpu', single_text=False):
        rows = []; metas = []
        while len(rows) < self.batch_size:
            x, m, d, meta = gen_cc(self.rng, self.context)
            if meta['canonical'] in self.excluded:
                continue
            rows.append((x, m, d)); metas.append(meta); self.seen_symbols.add(meta['canonical'])
            self.seen_tasks['colourcount'] = self.seen_tasks.get('colourcount', 0) + 1; self.targets['colourcount'] = self.targets.get('colourcount', 0) + int(m.sum()); self.nonpad += int((d >= 0).sum())
        xs, ms, ds = map(np.stack, zip(*rows)); self.slots += xs.size
        return dict(x=torch.as_tensor(xs, device=device), m=torch.as_tensor(ms, device=device), docs=torch.as_tensor(ds, device=device),
                    domain=torch.ones(len(rows), dtype=torch.long, device=device), metas=metas, sources=['symbolic'] * len(rows))

    def state(self):
        return dict(rng=self.rng.get_state(), seen_symbols=sorted(self.seen_symbols), seen_tasks=self.seen_tasks, targets=self.targets, slots=self.slots, nonpad=self.nonpad)

    def restore(self, s):
        self.rng.set_state(s['rng']); self.seen_symbols = set(s['seen_symbols'])
        for k in ('seen_tasks', 'targets', 'slots', 'nonpad'):
            setattr(self, k, s[k])


def phase_mixer(mode, seed, batch=64):
    return CCMixer(seed, batch) if mode == 'colourcount' else Mixer(seed, batch, mode)


if __name__ == '__main__':
    print(json.dumps(build())[:500])
