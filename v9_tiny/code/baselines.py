"""S4 baselines (draft §8.2), from the same promoted checkpoint in code-free mode, same problems per round and the
same training verifier calls as expert iteration (L3):
  rft       token rejection-sampling fine-tuning: sample complete answers at T = 1, keep verified successes,
            fine-tune a rank-16 LoRA on q and v of the three decoder blocks (24,576 parameters);
  grpo      group-relative policy gradient on the same adapter (group of 8 samples per problem);
  reinforce REINFORCE on the selector with the verifier reward (codes sampled from pi, greedy decoding).
"""
import argparse, copy, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import v9paths
from v9paths import digest, read_json, write_json
from model import D, chunk_info
from runtime import DEVICE, atomic_save, nll
from data import DONE, PAD, verify
import hncp, hgen, train_s4
from hgen import CH, classify

RANK, SCALE = 16, 2.


class QVLoRA(nn.Module):
    """Wraps a frozen Block.qkv: adds rank-16 LoRA deltas to the q and v outputs."""
    def __init__(self, base):
        super().__init__(); self.base = base
        self.Aq = nn.Parameter(torch.randn(RANK, D) * .02); self.Bq = nn.Parameter(torch.zeros(D, RANK))
        self.Av = nn.Parameter(torch.randn(RANK, D) * .02); self.Bv = nn.Parameter(torch.zeros(D, RANK))

    def forward(self, a):
        q, k, v = self.base(a).chunk(3, -1)
        q = q + SCALE * F.linear(F.linear(a, self.Aq), self.Bq); v = v + SCALE * F.linear(F.linear(a, self.Av), self.Bv)
        return torch.cat([q, k, v], -1)


def add_lora(hs):
    for blk in hs.f.dec:
        if not isinstance(blk.qkv, QVLoRA):
            blk.qkv = QVLoRA(blk.qkv).to(blk.o.weight.device)
    ps = [p for blk in hs.f.dec for p in (blk.qkv.Aq, blk.qkv.Bq, blk.qkv.Av, blk.qkv.Bv)]
    for p in ps:
        p.requires_grad_(True)
    return ps


def lora_state(hs):
    return {f'dec{i}.{n}': getattr(blk.qkv, n).detach().cpu() for i, blk in enumerate(hs.f.dec) for n in ('Aq', 'Bq', 'Av', 'Bv')}


def load_lora(hs, state):
    add_lora(hs)
    for i, blk in enumerate(hs.f.dec):
        for n in ('Aq', 'Bq', 'Av', 'Bv'):
            getattr(blk.qkv, n).data.copy_(state[f'dec{i}.{n}'].to(blk.o.weight.device))


@torch.no_grad()
def sample_answers(engine, items, temp=1., code_policy='free', hop_limit=20, gen=None):
    """Complete answers, token-sampled at temperature temp (code-free plans), or code-sampled from pi with greedy
    tokens (code_policy='sample'). Full recomputation. Returns per item (tokens, codes, success)."""
    hs = engine.hs; dev = engine.device; B = len(items); gen = gen or torch.Generator().manual_seed(0)
    toks = [np.array(it.prompt, dtype=np.int64) for it in items]; codes = [np.full(64, -1, dtype=np.int64) for _ in items]
    state = [(it.start, 0) for it in items]; live = list(range(B)); out = [None] * B
    while live:
        x, docs, L = engine.batch([toks[i] for i in live])
        ca = torch.full((len(live), x.shape[1] // CH + 1), -1, dtype=torch.long, device=dev)
        if code_policy == 'sample' and engine.K > 1:
            st = engine.state_heads(x, docs, L, need_codes=False)
            k = torch.multinomial(st['pi_logits'].float().softmax(-1).cpu(), 1, generator=gen).squeeze(1)
            for r, i in enumerate(live):
                J = len(toks[i]) // CH - 1; codes[i][J] = int(k[r])
        for r, i in enumerate(live):
            c = torch.as_tensor(codes[i][:ca.shape[1]], device=dev); ca[r, :len(c)] = c
        if code_policy == 'sample':
            chunk = engine.decode_chunk(x, docs, L, ca).cpu().numpy()
        else:
            h, c, cd = engine.context(x, docs); H = hs.heads(c, cd, need_codes=False); chat, a = H['chat0'], H['a0']
            for t in range(CH):
                if t:
                    h = hs.f.encode(x, docs)
                logits = engine.logits_at(engine.hidden(h, chat, a, docs), L + t - 1).float()
                tok = logits.argmax(-1) if temp == 0 else torch.multinomial((logits / temp).softmax(-1).cpu(), 1, generator=gen).squeeze(1).to(dev)
                rows = torch.arange(len(live), device=dev); x[rows, L + t] = tok; docs[rows, L + t] = 0
            chunk = x.gather(1, L[:, None] + torch.arange(CH, device=dev)[None]).cpu().numpy()
        nxt = []
        for r, i in enumerate(live):
            ch = tuple(int(v) for v in chunk[r]); toks[i] = np.r_[toks[i], ch].astype(np.int64)
            cl = classify(items[i], *state[i], ch)
            if cl['kind'] == 'done' or not cl['legal'] or state[i][1] + 1 >= hop_limit:
                out[i] = (toks[i], codes[i].copy(), bool(verify(items[i].prompt, toks[i][items[i].P:])['success']))
            else:
                state[i] = (cl['node'], state[i][1] + 1); nxt.append(i)
        live = nxt
    return out


def seq_batch(answers, dev):
    L = max(len(t) for t, *_ in answers); cap = (L + CH - 1) // CH * CH
    x = torch.full((len(answers), cap), PAD, dtype=torch.long, device=dev); docs = torch.full_like(x, -1); m = torch.zeros(x.shape, device=dev)
    for i, (t, c, ok, P) in enumerate(answers):
        x[i, :len(t)] = torch.as_tensor(t, device=dev); docs[i, :len(t)] = 0; m[i, P - 1:len(t) - 1] = 1
    return x, docs, m


def token_logprob(hs, x, docs, m):
    """Per-row sum of response log-probs under code-free plans (adapter active)."""
    logits = hs(x, docs)
    return -(nll(logits, x, m)).sum(1)


def train_round(hs, method, params, answers, epochs=2, lr=1e-3, bs=64):
    """rft: maximise log p of accepted answers. grpo: sum_i A_i log p(answer_i), A normalised within each problem's group.
    reinforce: sum_i (o_i - mean) log pi(codes_i) on the selector."""
    dev = next(hs.parameters()).device; opt = torch.optim.Adam(params, lr=lr); losses = []
    if method == 'rft':
        data = [(t, c, ok, P, 1.) for t, c, ok, P, g in answers if ok]
    elif method == 'grpo':
        groups = {}
        for a in answers:
            groups.setdefault(a[4], []).append(a)
        data = []
        for g, rows in groups.items():
            r = np.array([float(ok) for _, _, ok, _, _ in rows]); sd = r.std()
            if sd == 0:
                continue                                    # no learning signal in an all-equal group
            for (t, c, ok, P, _), adv in zip(rows, (r - r.mean()) / sd):
                data.append((t, c, ok, P, float(adv)))
    else:
        r = np.array([float(ok) for *_, ok, _, _ in [(a[0], a[1], a[2], a[3], a[4]) for a in answers]]); base = r.mean() if len(r) else 0.
        data = [(t, c, ok, P, float(ok) - base) for t, c, ok, P, g in answers]
    if not data:
        return dict(examples=0, loss=None)
    for _ in range(epochs):
        order = np.random.permutation(len(data))
        for s in range(0, len(data), bs):
            batch = [data[i] for i in order[s:s + bs]]; x, docs, m = seq_batch([(t, c, ok, P) for t, c, ok, P, _ in batch], dev)
            w = torch.tensor([float(a) for *_, a in batch], dtype=torch.float32, device=dev)
            with torch.enable_grad():
                opt.zero_grad()
                if method == 'reinforce':
                    h, cc, cd, _ = hs.context(x, docs); H = hs.heads(cc, cd, need_codes=False)
                    lp = H['pi_logits'].float().log_softmax(-1); M = lp.shape[1]
                    ca = torch.full((len(batch), M), -1, dtype=torch.long, device=dev)
                    for i, (t, c, *_r) in enumerate(batch):
                        cc_ = torch.as_tensor(c[:M], device=dev); ca[i, :len(cc_)] = cc_
                    used = ca >= 0; sel = lp.gather(-1, ca.clamp_min(0)[..., None]).squeeze(-1) * used
                    loss = -(w * sel.sum(1)).mean()
                else:
                    loss = -(w * token_logprob(hs, x, docs, m)).mean()
                loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.); opt.step()
            losses.append(float(loss.detach()))
    return dict(examples=len(data), loss=float(np.mean(losses)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True); ap.add_argument('--out', required=True); ap.add_argument('--method', required=True, choices=['rft', 'grpo', 'reinforce'])
    ap.add_argument('--ei', required=True, help='expert-iteration run folder: per-round problem seeds, sizes and verifier calls are matched')
    ap.add_argument('--seed', type=int, required=True); ap.add_argument('--group', type=int, default=8)
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hs, _ = hncp.load(a.source, DEVICE); engine = hgen.Engine(hs)
    if a.ei.startswith('default'):
        # No expert-iteration schedule to match (concept branch not run): declared default (I16), 16 samples per problem.
        small = a.ei == 'default-smoke'; sizes = [8, 8] if small else [512, 4096, 4096, 4096, 4096]
        schedule = [(i, n, 16 * n, 86000 + 100 * a.seed + i) for i, n in enumerate(sizes)]
    else:
        ei = read_json(Path(a.ei) / 'state.json'); eicfg = read_json(Path(a.ei) / 'config.json'); schedule = []
        for i in ei['done']:
            s = read_json(Path(a.ei) / f'iter{i}' / 'summary.json'); schedule.append((i, s['problems'], s['verifier_calls'], 86000 + 100 * eicfg['seed'] + i))
    params = add_lora(hs) if a.method in ('rft', 'grpo') else list(hs.codes.selector.parameters())
    for p in params:
        p.requires_grad_(True)
    gen = torch.Generator().manual_seed(88000 + a.seed); rounds = []; t0 = time.monotonic()
    for i, n, calls, pseed in schedule:
        items = train_s4.fresh_problems(n, pseed)                                     # the same problems as the EI round
        answers = []; used = 0; g = 0
        while used < calls:
            for j0 in range(0, len(items), 64):
                if used >= calls:
                    break
                batch = items[j0:j0 + 64]
                reps = a.group if a.method == 'grpo' else 1
                group_items = [it for it in batch for _ in range(reps)][:max(1, calls - used)]
                res = sample_answers(engine, group_items, 1., 'sample' if a.method == 'reinforce' else 'free', gen=gen)
                for q, (it, (t, c, ok)) in enumerate(zip(group_items, res)):
                    answers.append((t, c, ok, it.P, (g, j0 + q // reps)))
                used += len(group_items)
            g += 1
        fit = train_round(hs, a.method, params, answers)
        rounds.append(dict(round=i, problems=n, verifier_calls=used, target_calls=calls, accepted=int(sum(ok for _, _, ok, _, _ in answers)), fit=fit))
        write_json(out / 'rounds.json', rounds); print(json.dumps(rounds[-1]), flush=True)
    state = lora_state(hs) if a.method in ('rft', 'grpo') else hs.codes.selector.state_dict()
    for p in params:
        p.requires_grad_(False)
    atomic_save(out / 'adapter.pt', dict(method=a.method, state=state, source=a.source, trainable=sum(p.numel() for p in params)))
    write_json(out / 'train_summary.json', dict(status='completed', completed_steps=len(rounds), method=a.method, trainable_parameters=sum(p.numel() for p in params),
                                                verifier_calls=sum(r['verifier_calls'] for r in rounds), wall_s=time.monotonic() - t0, source_sha256=digest(a.source)))


if __name__ == '__main__':
    main()
