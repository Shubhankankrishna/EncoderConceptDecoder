"""S2 measurements (draft §6.6): everything the register rows G2.1-G2.4 read, per model, with per-item vectors.

Development data only. One grounded state per item per distribution, so bootstrap over items is bootstrap
over states. Thresholds are not here; they are in register_s2.py (draft §12.3).
"""
import argparse, json, math, time
import numpy as np
import torch
from torch.nn import functional as F
import v9paths
from v9paths import DATA, RUNS, write_json, read_json, digest
from model import D, chunk_info, shift, Core, freeze
from runtime import DEVICE, nll, ncp_weights, balanced, load_system
from data import PAD, SEP, TO, DONE, NODE0, COL0, N_COL, N_LABELS, fixed_text
import hncp, hgen, ceiling
from hgen import CH, Engine, Item, classify

N_STATES = 512
SEED = 9600


def load_suite(name, n=None):
    a = np.load(DATA / f'dev_{name}.npz'); n = min(n or len(a['x']), len(a['x']))
    return [Item(a['x'][i], a['m'][i], int(a['prompt_len'][i])) for i in range(n)], a


# ---------------------------------------------------------------- teacher-forced likelihoods (G2.1(a), G2.2(a))
@torch.no_grad()
def tf_rows(hs, x, docs, m, mode, gen, mixture=False):
    """Per-row (sum NLL, targets) under a plan mode: codefree | oracle | argmax | sample. Also k*, pi at scored chunks."""
    h, c, cd = hs.f.encode(x, docs), None, None
    c = hs.f.pool(h); cd, valid = chunk_info(docs)
    H = hs.heads(c, cd, need_codes=hs.codes is not None)
    cm = ncp_weights(dict(docs=docs, m=m))
    info = {}
    if hs.codes is None or mode == 'codefree':
        chat, a = H['chat0'], H['a0']
    else:
        d = (H['chatk'][:, :-1] - c[:, 1:, None]).square().mean(-1)
        kstar = d.argmin(-1); pi = H['pi_logits'][:, :-1].float().softmax(-1)
        if mode == 'oracle':
            k = kstar
        elif mode == 'argmax':
            k = pi.argmax(-1)
        elif mode == 'sample':
            k = torch.multinomial(pi.reshape(-1, hs.k).cpu(), 1, generator=gen).view(pi.shape[:2]).to(pi.device)
        codes = torch.full(cd.shape, -1, dtype=torch.long, device=x.device)
        codes[:, :-1] = torch.where(cm > 0, k, torch.full_like(k, -1))
        chat, a = hs.plan(H, codes)
        info = dict(kstar=kstar, pi=pi, cm=cm, d=d)
    logits = hs.f.decode(h, chat, a, docs)
    tok = nll(logits, x, m)
    return tok.sum(1), m[:, :-1].sum(1), info


@torch.no_grad()
def mixture_rows(hs, x, docs, m):
    """Per-chunk mixture log sum_k pi_k p(chunk | k), other chunks' codes at selector-argmax (exact per chunk)."""
    h = hs.f.encode(x, docs); c = hs.f.pool(h); cd, _ = chunk_info(docs)
    H = hs.heads(c, cd); cm = ncp_weights(dict(docs=docs, m=m)); M = c.shape[1]
    pi = H['pi_logits'][:, :-1].float().softmax(-1); am = pi.argmax(-1)
    total = torch.zeros(len(x), device=x.device)
    for j in range(M - 1):
        if not bool((cm[:, j] > 0).any()):
            continue
        lps = []
        for k in range(hs.k):
            codes = torch.full(cd.shape, -1, dtype=torch.long, device=x.device)
            codes[:, :-1] = torch.where(cm > 0, am, torch.full_like(am, -1)); codes[:, j] = torch.where(cm[:, j] > 0, torch.full_like(am[:, j], k), codes[:, j])
            chat, a = hs.plan(H, codes)
            z = h + shift(chat, docs) + hs.f.gamma * shift(a, docs)
            for blk in hs.f.dec:
                z = blk(z, docs)
            lo, hi = 4 * j + 3, 4 * j + 7
            lg = (hs.f.nf(z[:, lo:hi]) @ hs.f.emb.weight.t()).float().log_softmax(-1)
            tgt = x[:, lo + 1:hi + 1]; mask = m[:, lo:hi]
            lps.append((lg.gather(-1, tgt[..., None]).squeeze(-1) * mask).sum(1))
        mix = torch.logsumexp(torch.stack(lps, 1) + pi[:, j].log(), 1)
        total += torch.where(cm[:, j] > 0, -mix, torch.zeros_like(mix))
    return total


def nll_suite(hs, modes, gen, graph_n=2048, text_n=256, batch=64, mixture_graph=256, mixture_text=64):
    out = dict(graph={}, text={})
    a = np.load(DATA / 'dev_standard.npz'); n = min(graph_n, len(a['x']))
    calib = dict(conf=[], correct=[], logloss=[], wins=torch.zeros(max(hs.k, 1)), item=[])
    for mode in modes:
        nums, dens = [], []
        for i in range(0, n, batch):
            x = torch.tensor(a['x'][i:i + batch], device=DEVICE); docs = torch.tensor(a['docs'][i:i + batch], device=DEVICE); m = torch.tensor(a['m'][i:i + batch], device=DEVICE)
            s, t, info = tf_rows(hs, x, docs, m, mode, gen)
            nums += s.tolist(); dens += t.tolist()
            if mode == 'argmax' and info:
                collect_calibration(calib, info, i, 'graph')
        out['graph'][mode] = dict(per_item=(np.array(nums) / np.maximum(dens, 1)).tolist(), nll=float(np.sum(nums) / np.sum(dens)))
    if hs.codes is not None and mixture_graph:
        x = torch.tensor(a['x'][:mixture_graph], device=DEVICE); docs = torch.tensor(a['docs'][:mixture_graph], device=DEVICE); m = torch.tensor(a['m'][:mixture_graph], device=DEVICE)
        vals = torch.cat([mixture_rows(hs, x[i:i + batch], docs[i:i + batch], m[i:i + batch]) for i in range(0, mixture_graph, batch)])
        out['graph']['mixture'] = dict(per_item=(vals.cpu().numpy() / m[:, :-1].sum(1).cpu().numpy()).tolist(), n=mixture_graph)
    for src in ('edu', 'ts'):
        tb, ids = fixed_text('dev', text_n, source=src)
        for mode in modes:
            nums, dens = [], []
            for i in range(0, text_n, batch):
                x = tb['x'][i:i + batch].to(DEVICE); docs = tb['docs'][i:i + batch].to(DEVICE); m = tb['m'][i:i + batch].to(DEVICE)
                s, t, info = tf_rows(hs, x, docs, m, mode, gen)
                nums += s.tolist(); dens += t.tolist()
                if mode == 'argmax' and info:
                    collect_calibration(calib, info, i, 'text_' + src)
            out['text'].setdefault(mode, {})[src] = dict(numerators=nums, denominators=dens, nll=float(np.sum(nums) / np.sum(dens)), document_ids=[str(v) for v in ids])
        if hs.codes is not None and mixture_text:
            x = tb['x'][:mixture_text].to(DEVICE); docs = tb['docs'][:mixture_text].to(DEVICE); m = tb['m'][:mixture_text].to(DEVICE)
            vals = mixture_rows(hs, x, docs, m)
            out['text'].setdefault('mixture', {})[src] = dict(numerators=vals.tolist(), denominators=m[:, :-1].sum(1).tolist(), n=mixture_text)
    if hs.codes is not None and calib['conf']:
        out['calibration'] = calibration_summary(calib, hs.k)
    return out


def collect_calibration(calib, info, offset, domain):
    sc = info['cm'] > 0; pi = info['pi'][sc]; ks = info['kstar'][sc]
    conf, pred = pi.max(-1)
    calib['conf'] += conf.tolist(); calib['correct'] += (pred == ks).float().tolist()
    calib['logloss'] += (-pi.gather(-1, ks[:, None]).squeeze(-1).clamp_min(1e-12).log()).tolist()
    calib['wins'] += torch.bincount(ks.cpu(), minlength=len(calib['wins'])).float()
    rows = sc.nonzero()[:, 0] + offset; calib['item'] += [f'{domain}{int(r)}' for r in rows]


def ece(conf, correct, min_bin=50):
    order = np.argsort(conf); n = len(conf); bins = max(1, min(15, n // min_bin)); edges = np.linspace(0, n, bins + 1).astype(int)
    tot = 0.; diag = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = order[lo:hi]; c = conf[idx].mean(); a = correct[idx].mean(); tot += (hi - lo) / n * abs(c - a); diag.append(dict(n=int(hi - lo), confidence=float(c), accuracy=float(a)))
    return float(tot), diag


def calibration_summary(calib, K, resamples=2000, seed=SEED):
    conf = np.array(calib['conf']); correct = np.array(calib['correct']); items = np.array(calib['item'])
    value, diagram = ece(conf, correct)
    uniq, inv = np.unique(items, return_inverse=True); rng = np.random.RandomState(seed); groups = [np.nonzero(inv == g)[0] for g in range(len(uniq))]
    boots = []
    for _ in range(resamples):
        pick = rng.randint(len(groups), size=len(groups)); idx = np.concatenate([groups[g] for g in pick])
        boots.append(ece(conf[idx], correct[idx])[0])
    wins = calib['wins'].numpy(); share = wins / max(wins.sum(), 1)
    return dict(ece=value, ece_interval=[float(np.quantile(boots, .025)), float(np.quantile(boots, .975))], reliability=diagram, positions=len(conf),
                kstar_logloss=float(np.mean(calib['logloss'])), win_share=share.tolist(), live_codes=int((share >= 1 / (10 * K)).sum()), bootstrap='cluster over items, 2,000 resamples')


# ---------------------------------------------------------------- greedy success (G2.2(d)) and D_self rollouts
def eval_batch(eng):
    """I20: MPS driver memory grows with batch x K x distinct lengths; K > 1 decodes score K code rows per sequence, so batch 16 (plateau 2 GB on hard14) where dense K = 1 keeps 64 (I19); recurrent K = 1 also uses 16 (I22)."""
    return 64 if eng.K == 1 and eng.hs.core_kind == 'dense' else 16


def greedy_suite(eng, name, n=None, batch=None, policy='argmax'):
    batch = batch or eval_batch(eng); items, a = load_suite(name, n); P = items[0].P; out = dict(success=[], reasons=[], responses=[])
    for i in range(0, len(items), batch):
        g = eng.greedy([it.prompt for it in items[i:i + batch]], max_new=a['x'].shape[1] - P, policy=policy)
        for j, v in enumerate(g['verification']):
            out['success'].append(bool(v['success'])); out['reasons'].append(v['reason']); out['responses'].append(g['x'][j, P:int(g['lengths'][j])].tolist())
    out['rate'] = float(np.mean(out['success'])); out['n'] = len(items)
    return out


def self_states(eng, items, batch=None, seed=SEED):
    """D_self: selector-argmax rollouts up to the demonstration's response length or DONE; one legal-history state per item."""
    batch = batch or eval_batch(eng); rng = np.random.RandomState(seed); states = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]; P = chunk[0].P
        g = eng.greedy([it.prompt for it in chunk], max_new=max(len(it.response) for it in chunk), record_states=True,
                       state_limit=[len(it.hops) for it in chunk])
        for j, it in enumerate(chunk):
            toks = g['x'][j]; cands = []
            for st in g['states'][j]:
                L = st['length']; hops = [tuple(int(v) for v in toks[q:q + CH]) for q in range(P, L, CH)]
                cur, n, legal = it.walk(hops)
                if legal and not any(hc[0] == DONE for hc in hops):
                    cands.append(dict(tokens=toks[:L].copy(), codes=st['codes'], cur=cur, hops=n))
            states.append(dict(item=i + j, **cands[rng.randint(len(cands))]))
    return states


def gold_states(eng, items, seed=SEED):
    rng = np.random.RandomState(seed + 1); states = []
    for idx, it in enumerate(items):
        nh = len(it.hops) - 1  # demonstration hops before the terminator
        h = rng.randint(nh + 1)
        toks = np.concatenate([it.prompt] + [np.array(c) for c in it.hops[:h]]) if h else it.prompt.copy()
        cur, n, legal = it.walk(it.hops[:h]); assert legal
        nxt = N_COL if h == nh else it.hops[h][0] - COL0
        states.append(dict(item=idx, tokens=toks, codes=None, cur=cur, hops=n, demo_next=nxt, demo_colours=[c[0] - COL0 for c in it.hops[:h]]))
    return states


def wrong_states(eng, items, self_st, seed=SEED):
    """D_wrong: a D_self state after one forced illegal or non-progress hop (claimed node becomes current)."""
    rng = np.random.RandomState(seed + 2); out = []
    for st in self_st:
        it = items[st['item']]; cur = st['cur']
        nonprog = [c for c in range(N_COL) if c not in it.progress_options(cur)]
        if nonprog and rng.rand() < .5:
            col = nonprog[rng.randint(len(nonprog))]; dst = it.edges[(cur, col)]; kind = 'non_progress'
        else:
            col = rng.randint(N_COL); true = it.edges[(cur, col)]; dst = (true + 1 + rng.randint(N_LABELS - 1)) % N_LABELS; kind = 'illegal'
        chunk = np.array([COL0 + col, TO, NODE0 + dst, SEP])
        out.append(dict(item=st['item'], tokens=np.concatenate([st['tokens'], chunk]), codes=None, cur=dst, hops=st['hops'] + 1, forced=kind))
    return out


def fill_codes(eng, states):
    """Plan records for prefixes not produced by a rollout: selector-argmax at every coded chunk (causal, one pass)."""
    if eng.K == 1:
        return
    todo = [s for s in states if s['codes'] is None]
    for i in range(0, len(todo), 256):
        part = todo[i:i + 256]; x, docs, L = eng.batch([s['tokens'] for s in part])
        with torch.no_grad():
            h, c, cd = eng.context(x, docs); H = eng.hs.heads(c, cd, need_codes=False)
        am = H['pi_logits'].argmax(-1)
        for r, s in enumerate(part):
            P = eng_P(s); codes = torch.full((x.shape[1] // CH,), -1, dtype=torch.long)
            J = len(s['tokens']) // CH - 1
            for j in range(P // CH - 1, J):
                codes[j] = int(am[r, j])
            s['codes'] = codes


def eng_P(s):
    return s['P']


# ---------------------------------------------------------------- G2.4 and swap: decode every code at every state
@torch.no_grad()
def decode_codes(eng, states, items, port0=False, batch=512):
    """For each state and code k: the decoded chunk, its action class, and (unless port0) the pooled real concept."""
    K = max(eng.K, 1); rows = [(si, k) for si in range(len(states)) for k in range(K)]; res = [[None] * K for _ in states]
    for i in range(0, len(rows), batch):
        part = rows[i:i + batch]; x, docs, L = eng.batch([states[si]['tokens'] for si, _ in part])
        codes = torch.full((len(part), x.shape[1] // CH), -1, dtype=torch.long, device=eng.device)
        for r, (si, k) in enumerate(part):
            sc = states[si]['codes']
            if sc is not None:
                n = min(len(sc), codes.shape[1]); codes[r, :n] = sc[:n].to(eng.device)
            if eng.K > 1:
                codes[r, int(L[r]) // CH - 1] = k
        chunk = eng.decode_chunk(x, docs, L, codes, port0=port0)
        real = None if port0 else eng.pooled_next(x, docs, L)
        for r, (si, k) in enumerate(part):
            st = states[si]; cl = classify(items[st['item']], st['cur'], st['hops'], chunk[r].tolist())
            cl['chunk'] = chunk[r].tolist()
            if real is not None:
                cl['real'] = real[r].cpu()
            res[si][k] = cl
    return res


@torch.no_grad()
def candidate_concepts(eng, states, batch=1024):
    """73 candidate chunks (3 colours x 24 nodes + DONE), each encoded in the state's own context and pooled."""
    cands = [np.array([COL0 + c, TO, NODE0 + n, SEP]) for c in range(N_COL) for n in range(N_LABELS)] + [np.array([DONE, SEP, SEP, SEP])]
    rows = [(si, q) for si in range(len(states)) for q in range(len(cands))]; out = torch.zeros(len(states), len(cands), D)
    for i in range(0, len(rows), batch):
        part = rows[i:i + batch]; x, docs, L = eng.batch([np.concatenate([states[si]['tokens'], cands[q]]) for si, q in part])
        h = eng.f.encode(x, docs); c = eng.f.pool(h); J = L // CH - 1
        v = c[torch.arange(len(part), device=c.device), J].cpu()
        for r, (si, q) in enumerate(part):
            out[si, q] = v[r]
    return out


def g24(eng, states, items, dec, port, cand, heads):
    """Per-state G2.4 quantities. heads: per-state chatk (K, D) and pi (K,) at the state's last chunk."""
    K = max(eng.K, 1); rec = []
    for si, st in enumerate(states):
        it = items[st['item']]; prog = it.progress_options(st['cur']); r = dec[si]
        legal_actions = sorted({d['action'] for d in r if d['legal'] and d['kind'] == 'hop'})
        acts = [d['action'] for d in r]
        stop_legal = st['hops'] >= 1 and st['cur'] == it.goal
        row = dict(item=st['item'], progress_options=len(prog), distinct_legal=len(legal_actions), duplicate_rate=1 - len(set(acts)) / K,
                   useful=any(d['kind'] == 'hop' and d['legal'] and d['action'] in prog for d in r), stop_legal=stop_legal,
                   stop_covered=any(d['kind'] == 'done' for d in r) if stop_legal else None,
                   false_stop_codes=[bool(d['kind'] == 'done' and not d['legal']) for d in r],
                   actions=acts, legal=[bool(d['legal']) for d in r], kinds=[d['kind'] for d in r])
        if heads is not None:
            pi = heads['pi'][si]; am = int(np.argmax(pi)); row['argmax_code'] = am; row['pi'] = [float(v) for v in pi]
            row['argmax_false_stop'] = bool(r[am]['kind'] == 'done' and not r[am]['legal'])
            # 4. predicted vs executed: nearest of 73 candidates to chat^(k), among legal decodes.
            agree = []; margins = []; real_gap = []
            for k in range(K):
                d = r[k]
                if not d['legal']:
                    continue
                dist = (cand[si] - heads['chatk'][si][k][None]).square().mean(-1)
                order = torch.argsort(dist); nearest = int(order[0])
                decoded = 72 if d['kind'] == 'done' else d['action'] * N_LABELS + d['node']
                agree.append(nearest == decoded); margins.append(float(dist[order[1]] - dist[order[0]]))
                own = float((heads['chatk'][si][k] - d['real']).square().mean())
                other = float(torch.cat([dist[:decoded], dist[decoded + 1:]]).min())
                real_gap.append(dict(predicted_to_real=own, nearest_other=other))
            row.update(pred_exec_agree=agree, pred_exec_margin=margins, real_gap=real_gap)
            row['port_changes'] = [bool(port[si][k]['chunk'] != r[k]['chunk']) for k in range(K)]
            row['sensitivity'] = float(np.mean([acts[k] != acts[am] for k in range(K) if k != am])) if K > 1 else 0.
            changed = [k for k in range(K) if k != am and acts[k] != acts[am]]
            row['changed_legal'] = [bool(r[k]['legal']) for k in changed]
        rec.append(row)
    return rec


@torch.no_grad()
def state_head_values(eng, states):
    out = dict(pi=[], chatk=[], u0=[])
    for i in range(0, len(states), 256):
        part = states[i:i + 256]; x, docs, L = eng.batch([s['tokens'] for s in part])
        st = eng.state_heads(x, docs, L, need_codes=True)
        if eng.K > 1:
            out['pi'] += list(st['pi_logits'].float().softmax(-1).cpu().numpy()); out['chatk'] += list(st['chatk'].cpu())
        else:
            out['pi'] += [np.ones(1)] * len(part); out['chatk'] += list(st['chat0'][:, None].cpu())
        out['u0'] += list(st['u0'].cpu())
    return out


# ---------------------------------------------------------------- G2.3 imagined vs executed
@torch.no_grad()
def g23(eng, states, items, rule, n_max=4, seed=SEED, batch=128):
    """Code sequence of length n_max fixed in advance (argmax along the imagined path, or uniform); imagined vs executed."""
    rng = np.random.RandomState(seed + (0 if rule == 'argmax' else 7)); K = max(eng.K, 1); recs = []
    for i in range(0, len(states), batch):
        part = states[i:i + batch]; B = len(part)
        x, docs, L = eng.batch([s['tokens'] for s in part], cap=max(len(s['tokens']) for s in part) + (n_max + 1) * CH)
        codes = torch.full((B, x.shape[1] // CH), -1, dtype=torch.long, device=eng.device)
        for r, s in enumerate(part):
            if s['codes'] is not None:
                n = min(len(s['codes']), codes.shape[1]); codes[r, :n] = s['codes'][:n].to(eng.device)
        st = eng.state_heads(x, docs, L)
        h, c, cd, J0 = st['h'], st['c'], st['cd'], st['J']
        # Imagined path and the code sequence.
        seq = []; preds = []; imag_dec = []; imag_c = []
        H0 = st['H']
        for t in range(n_max):
            if t == 0:
                pi = st['pi_logits'] if K > 1 else None; chk = st['chatk'] if K > 1 else st['chat0'][:, None]
            else:
                Hi, Jt = eng.imagine(c, cd, J0, torch.stack(preds, 1))
                pi = eng.at(Hi['pi_logits'], Jt) if K > 1 else None
                chk = eng.at(Hi['chatk'], Jt) if K > 1 else eng.at(Hi['chat0'], Jt)[:, None]
            if t > 0:
                imag_dec.append(pi.argmax(-1) if K > 1 else torch.zeros(B, dtype=torch.long, device=eng.device))
            k = (pi.argmax(-1) if rule == 'argmax' else torch.tensor(rng.randint(K, size=B), device=eng.device)) if K > 1 else torch.zeros(B, dtype=torch.long, device=eng.device)
            seq.append(k); pr = chk[torch.arange(B, device=eng.device), k] if K > 1 else chk[:, 0]; preds.append(pr); imag_c.append(pr)
        Hi, Jt = eng.imagine(c, cd, J0, torch.stack(preds, 1))
        imag_dec.append(eng.at(Hi['pi_logits'], Jt).argmax(-1) if K > 1 else torch.zeros(B, dtype=torch.long, device=eng.device))
        # Executed path: same code sequence, every chunk decoded, re-encoded, pooled.
        Lx = L.clone(); exec_dec = []; chunks = []; real = []
        for t in range(n_max):
            if K > 1:
                codes[torch.arange(B, device=eng.device), Lx // CH - 1] = seq[t]
            ch = eng.decode_chunk(x, docs, Lx, codes); chunks.append(ch.cpu()); real.append(eng.pooled_next(x, docs, Lx).cpu())
            Lx = Lx + CH
            se = eng.state_heads(x, docs, Lx, need_codes=False)
            exec_dec.append(se['pi_logits'].argmax(-1) if K > 1 else torch.zeros(B, dtype=torch.long, device=eng.device))
        for r, s in enumerate(part):
            it = items[s['item']]; cur = s['cur']; hops = s['hops']; alive = True; row = dict(item=s['item'], steps=[])
            for t in range(n_max):
                cl = classify(it, cur, hops, chunks[t][r].tolist())
                step = dict(n=t + 1, alive=alive, kind=cl['kind'], legal=bool(cl['legal']), progress=bool(cl.get('progress', False)),
                            agree=bool(int(imag_dec[t][r]) == int(exec_dec[t][r])) if alive else None,
                            distance=float((imag_c[t][r].cpu() - real[t][r]).square().mean()))
                row['steps'].append(step)
                if not alive:
                    continue
                if cl['kind'] == 'hop' and cl['legal']:
                    cur = cl['node']; hops += 1
                else:
                    alive = False  # DONE or illegal ends the executed path
            recs.append(row)
    return recs


def g23_summary(recs, n_max=4):
    out = {}
    for n in range(1, n_max + 1):
        legal = [all(s['legal'] or (s['kind'] == 'done' and s['legal']) for s in r['steps'][:n] if s['alive']) for r in recs]
        agree = [r['steps'][n - 1]['agree'] for r in recs if r['steps'][n - 1]['alive']]
        out[n] = dict(legality=float(np.mean(legal)), legality_vector=legal, agreement=float(np.mean(agree)) if agree else None, agreement_vector=agree,
                      distance=float(np.mean([r['steps'][n - 1]['distance'] for r in recs])), progress=float(np.mean([r['steps'][n - 1]['progress'] for r in recs])))
    return out


# ---------------------------------------------------------------- E5-form drift (G2.3(c)), code-free drift (§7.1)
@torch.no_grad()
def open_loop_nll(model_heads, f, x, docs, m, coded_fn):
    """E5 form: every chunk after c0 replaced by the model's own committed prediction, decoder reads true tokens.
    Codes are used only at chunks with a scored next chunk (§6.2); elsewhere the code-free prediction is fed."""
    h = f.encode(x, docs); c = f.pool(h); cd, _ = chunk_info(docs); M = c.shape[1]; c_in = c.clone()
    cm = torch.zeros(cd.shape, device=x.device); cm[:, :-1] = ncp_weights(dict(docs=docs, m=m))
    for j in range(1, M):
        H = model_heads(c_in[:, :j], cd[:, :j], cm[:, :j])
        c_in[:, j] = coded_fn(H, j - 1, cm[:, j - 1])
    H = model_heads(c_in, cd, cm)
    chat, a = H['plan']
    logits = f.decode(h, chat, a, docs)
    return nll(logits, x, m).sum(1), m[:, :-1].sum(1)


def drift_suite(hs, teacher, n_graph=256, n_text=128, batch=32):
    """E5-form open-loop NLL increase for this model, and code-free drift of u0/A0 against the frozen teacher."""
    f = hs.f; res = {}
    def own(H, j, coded):
        if hs.codes is None:
            return H['chat0'][:, j]
        k = H['pi_logits'][:, j].argmax(-1)
        return torch.where(coded[:, None] > 0, H['chatk'][torch.arange(len(k)), j, k], H['chat0'][:, j])
    def student_heads(c, cd, cm):
        H = hs.heads(c, cd)
        if hs.codes is not None:
            am = H['pi_logits'].argmax(-1); H['plan'] = hs.plan(H, torch.where(cm > 0, am, torch.full_like(am, -1)))
        else:
            H['plan'] = (H['chat0'], H['a0'])
        return H
    def teacher_heads(c, cd, cm):
        u = teacher.foundation.concept(c, cd); return dict(chat0=teacher.foundation.pq(u), plan=(teacher.foundation.pq(u), teacher.foundation.port(u)))
    a = np.load(DATA / 'dev_standard.npz')
    sets = [('graph', torch.tensor(a['x'][:n_graph]), torch.tensor(a['docs'][:n_graph]), torch.tensor(a['m'][:n_graph]))]
    for src in ('edu', 'ts'):
        tb, _ = fixed_text('dev', n_text, source=src); sets.append(('text_' + src, tb['x'], tb['docs'], tb['m']))
    drift_u = []; drift_a = []
    for name, X, Dc, Mk in sets:
        rows = {}
        for label, heads_fn, coded in (('model', student_heads, own), ('teacher', teacher_heads, lambda H, j, cmj: H['chat0'][:, j])):
            num_o = num_t = den = 0.
            for i in range(0, len(X), batch):
                x = X[i:i + batch].to(DEVICE); d = Dc[i:i + batch].to(DEVICE); m = Mk[i:i + batch].to(DEVICE)
                so, t = open_loop_nll(heads_fn, f, x, d, m, coded)
                h = f.encode(x, d); c = f.pool(h); cd, valid = chunk_info(d)
                cmf = torch.zeros(cd.shape, device=x.device); cmf[:, :-1] = ncp_weights(dict(docs=d, m=m)); H = heads_fn(c, cd, cmf)
                st = nll(f.decode(h, *H['plan'], d), x, m).sum(1)
                num_o += float(so.sum()); num_t += float(st.sum()); den += float(t.sum())
                if label == 'model':
                    uT = teacher.foundation.concept(c, cd); aT = teacher.foundation.port(uT); Hs = hs.heads(c, cd, need_codes=False)
                    w = valid.float()
                    drift_u.append(float(((Hs['u0'] - uT).square().mean(-1) * w).sum() / w.sum())); drift_a.append(float(((Hs['a0'] - aT).square().mean(-1) * w).sum() / w.sum()))
            rows[label] = dict(open_loop=num_o / den, teacher_forced=num_t / den, increase=(num_o - num_t) / den)
        res[name] = rows
    res['code_free_drift'] = dict(u0_mse=float(np.mean(drift_u)), a0_mse=float(np.mean(drift_a)))
    return res


# ---------------------------------------------------------------- recurrent engaged depth (diagnostic)
@torch.no_grad()
def engaged_depth(hs, n=64):
    if hs.core_kind != 'recurrent':
        return None
    a = np.load(DATA / 'dev_standard.npz'); x = torch.tensor(a['x'][:n], device=DEVICE); docs = torch.tensor(a['docs'][:n], device=DEVICE)
    h = hs.f.encode(x, docs); c = hs.f.pool(h); cd, valid = chunk_info(docs); core = hs.core
    e = core.in_proj(c); hh, l = core.h0.expand_as(e) + 0.0, core.l0.expand_as(e); trace = []
    for visit in range(6):
        if visit in (0, 1, 3, 4):
            l = core.post_L(core.fL(l + hh + e + 0.0, cd))
        else:
            hh = core.post_H(core.fH(hh + l, cd))
        trace.append(hh.clone())
    w = valid.float()
    def readout(v):
        u = core.B_u(core.out_n(v)); return hs.f.heads(u).view(*u.shape[:2], 8, 32).float().log_softmax(-1)
    deltas = [float(((trace[i] - trace[i - 1]).norm(dim=-1) * w).sum() / w.sum()) for i in (2, 5)]
    cos = float((F.cosine_similarity(trace[2], trace[5], dim=-1) * w).sum() / w.sum())
    p3, p6 = readout(trace[2]), readout(trace[5])
    kl = float(((p6.exp() * (p6 - p3)).sum(-1).mean(-1) * w).sum() / w.sum())
    return dict(h_change_visit3=deltas[0], h_change_visit6=deltas[1], cosine_h3_h6=cos, logit_lens_kl_h6_vs_h3=kl)


# ---------------------------------------------------------------- P-blur-2 in S2 (diagnostic)
def pblur2(items, gold, dec, heads):
    rows = []
    for si, st in enumerate(gold):
        it = items[st['item']]
        if it.task != 'graph':
            continue
        p = ceiling.next_action_distribution(it.prompt, st['demo_colours'])
        ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        k = int(np.argmax(heads['pi'][si])) if heads is not None and len(heads['pi'][si]) > 1 else 0
        d = dec[si][k]
        rows.append(dict(entropy=ent, illegal=bool(d['kind'] == 'hop' and not d['legal']), malformed=d['kind'] == 'malformed'))
    if not rows:
        return None
    e = np.array([r['entropy'] for r in rows]); y = np.array([r['illegal'] for r in rows], float)
    slope = float(np.polyfit(e, y, 1)[0]) if e.std() > 0 else None
    return dict(n=len(rows), illegal_rate=float(y.mean()), slope_vs_label_entropy=slope)


# ---------------------------------------------------------------- driver
def measure(path, out, teacher_path, quick=False):
    t0 = time.time(); hs, ck = hncp.load(path, DEVICE); hs.eval(); eng = Engine(hs)
    teacher, _ = load_system(teacher_path, DEVICE); freeze(teacher)
    gen = torch.Generator().manual_seed(SEED)
    nst = 32 if quick else N_STATES
    modes = ['codefree'] if hs.codes is None else ['oracle', 'argmax', 'sample']
    res = dict(source=str(path), checkpoint_sha256=digest(path), teacher=str(teacher_path), teacher_sha256=digest(teacher_path), k=hs.k, core=hs.core_kind,
               metadata=hs.metadata(), split='dev')
    res['nll'] = nll_suite(hs, modes, gen, graph_n=128 if quick else 2048, text_n=32 if quick else 256, mixture_graph=16 if quick else 256, mixture_text=8 if quick else 64)
    res['greedy'] = {name: greedy_suite(eng, name, n=(64 if quick else (2048 if name == 'standard' else 512))) for name in ('standard', 'hard10')}
    res['greedy']['hard14'] = greedy_suite(eng, 'hard14', n=32 if quick else 256)
    res['states'] = {}
    for name in ('standard', 'hard10', 'hard14'):
        if name == 'hard14' and quick:
            continue
        items, _ = load_suite(name, nst if name != 'hard14' else 128)
        P = items[0].P
        dsel = self_states(eng, items); dgold = gold_states(eng, items); dwrong = wrong_states(eng, items, dsel)
        dist = {}
        dfirst = [dict(item=i, tokens=it.prompt.copy(), codes=None, cur=it.start, hops=0) for i, it in enumerate(items)]
        for label, st in (('first', dfirst), ('self', dsel), ('gold', dgold), ('wrong', dwrong)):
            for s in st:
                s['P'] = P
            fill_codes(eng, st)
            heads = state_head_values(eng, st)
            qb = eval_batch(eng)      # I19/I20: MPS driver memory grows with every new (batch, length) shape; 64 plateaus near 3.5 GB where 256 exceeded the 20 GB cap. Batch size does not change results
            dec = decode_codes(eng, st, items, batch=qb or 256)
            port = decode_codes(eng, st, items, port0=True, batch=qb or 256) if eng.K > 1 else dec
            cand = candidate_concepts(eng, st, batch=qb or 256) if eng.K > 1 else None
            rec = g24(eng, st, items, dec, port, cand, heads if eng.K > 1 else None)
            entry = dict(n=len(st), g24=rec)
            if label == 'gold':
                cov = []
                for si, s in enumerate(st):
                    it = items[s['item']]
                    if eng.K > 1:
                        top2 = np.argsort(-np.asarray(heads['pi'][si]))[:2]; acts = {dec[si][k]['action'] for k in top2 if dec[si][k]['kind'] != 'malformed'}
                    else:
                        acts = {dec[si][0]['action']}
                    ceil = ceiling.top2_mass(ceiling.next_action_distribution(it.prompt, s['demo_colours'])) if name == 'standard' else 1.0
                    cov.append(dict(item=s['item'], covered=bool(s['demo_next'] in acts), ceiling=ceil))
                entry['coverage'] = cov
                entry['pblur2'] = pblur2(items, st, dec, heads if eng.K > 1 else None) if name == 'standard' else None
            if name != 'hard14' and label != 'first':
                entry['g23'] = {rule: g23(eng, st, items, rule, batch=qb or 128) for rule in (('argmax', 'uniform') if eng.K > 1 else ('argmax',))}
            dist[label] = entry
        res['states'][name] = dist
    res['drift'] = drift_suite(hs, teacher, n_graph=32 if quick else 256, n_text=16 if quick else 128)
    res['engaged_depth'] = engaged_depth(hs)
    # Determinism: same-code regeneration changes nothing.
    items, _ = load_suite('standard', 16)
    g1 = eng.greedy([it.prompt for it in items], max_new=68); g2 = eng.greedy([it.prompt for it in items], max_new=68)
    res['determinism'] = bool(np.array_equal(g1['x'], g2['x']))
    res['seconds'] = time.time() - t0
    write_json(out, res)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--teacher', required=True); ap.add_argument('--out', required=True); ap.add_argument('--quick', action='store_true')
    a = ap.parse_args(); torch.set_num_threads(4)
    r = measure(a.source, a.out, a.teacher, a.quick); print(json.dumps(dict(seconds=r['seconds'], standard=r['greedy']['standard']['rate'])))
