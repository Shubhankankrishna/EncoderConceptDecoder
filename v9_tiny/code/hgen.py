"""H-NCP inference engine (draft §4.2, §6.6, §9.3): grounded states, chunk decoding under plan records,
selector-argmax rollouts, decoder-free (imagined) transitions. Full recomputation, no caches: every result is
the full causal forward of the prefix, so cached-vs-full parity holds by construction.

Plan records: codes[r, j] in {-1 (code-free), 0..K-1}; the plan used to decode chunk j+1 is (chat_j^(k), A_j^(k)).
"""
from collections import deque
import numpy as np
import torch
import v9paths
from model import D, chunk_info, shift
from data import PAD, EOS, GNAV, SEP, TO, DONE, NODE0, COL0, N_COL, N_LABELS, T, parse_prompt, verify

CH = 4
LOOKUP_TASKS = ('lookup', 'compose', 'bind')


def bfs_to_goal(edges, goal):
    """Hop distance from every label to the goal along coloured edges."""
    rev = {}
    for (src, col), dst in edges.items():
        rev.setdefault(dst, []).append(src)
    dist = {goal: 0}; q = deque([goal])
    while q:
        v = q.popleft()
        for u in rev.get(v, ()):
            if u not in dist:
                dist[u] = dist[v] + 1; q.append(u)
    return dist


class Item:
    """Parsed symbolic item: prompt, demonstration, graph facts."""
    def __init__(self, x, m, p):
        self.prompt = np.asarray(x[:p]); r = int(m.sum()); self.response = np.asarray(x[p:p + r]); self.P = p
        self.task, self.edges, self.start, self.goal, self.colour = parse_prompt(self.prompt)
        self.dist = bfs_to_goal(self.edges, self.goal) if self.task == 'graph' else None
        self.hops = [tuple(int(v) for v in self.response[i:i + CH]) for i in range(0, len(self.response), CH)]

    def walk(self, chunks):
        """Replay hop chunks from the start: returns (current node, hops, all legal so far)."""
        cur = self.start; legal = True; n = 0
        for ch in chunks:
            if ch[0] == DONE:
                break
            col, to, dst, sep = ch
            if not (COL0 <= col < COL0 + N_COL and to == TO and NODE0 <= dst < NODE0 + N_LABELS and sep == SEP) or self.edges.get((cur, col - COL0)) != dst - NODE0:
                legal = False; break
            cur = dst - NODE0; n += 1
        return cur, n, legal

    def progress_options(self, cur):
        if self.dist is None or cur not in self.dist:
            return []
        return [c for c in range(N_COL) if self.dist.get(self.edges[(cur, c)], 1 << 30) < self.dist[cur]]


def classify(item, cur, hops_done, chunk):
    """Action class of one decoded chunk at node cur: ('hop', colour, legal) / ('done', legal) / ('malformed',)."""
    chunk = tuple(int(v) for v in chunk)
    if chunk == (DONE, SEP, SEP, SEP):
        return dict(kind='done', legal=bool(hops_done >= 1 and cur == item.goal), action=N_COL)
    col, to, dst, sep = chunk
    if COL0 <= col < COL0 + N_COL and to == TO and NODE0 <= dst < NODE0 + N_LABELS and sep == SEP:
        legal = item.edges.get((cur, col - COL0)) == dst - NODE0
        nxt = dst - NODE0
        progress = bool(legal and item.dist is not None and item.dist.get(nxt, 1 << 30) < item.dist.get(cur, 1 << 30))
        return dict(kind='hop', legal=bool(legal), action=col - COL0, node=nxt, progress=progress)
    return dict(kind='malformed', legal=False, action=-1)


class Engine:
    def __init__(self, hs):
        self.hs = hs; self.f = hs.f; self.K = hs.k; self.device = next(hs.parameters()).device

    # ---- primitives -------------------------------------------------------------------------------------
    def batch(self, rows, cap=None):
        """rows: list of token arrays (grounded prefixes). Returns x, docs, lengths with capacity for one more chunk."""
        lengths = torch.tensor([len(r) for r in rows], device=self.device)
        cap = cap or int(lengths.max()) + 2 * CH
        cap = (cap + CH - 1) // CH * CH
        x = torch.full((len(rows), cap), PAD, dtype=torch.long, device=self.device); docs = torch.full_like(x, -1)
        for i, r in enumerate(rows):
            x[i, :len(r)] = torch.as_tensor(np.asarray(r), device=self.device); docs[i, :len(r)] = 0
        return x, docs, lengths

    def context(self, x, docs):
        h = self.f.encode(x, docs); c = self.f.pool(h); cd, valid = chunk_info(docs)
        return h, c, cd

    def hidden(self, h, chat, a, docs):
        z = h + shift(chat, docs) + self.f.gamma * shift(a, docs)
        for blk in self.f.dec:
            z = blk(z, docs)
        return self.f.nf(z)

    def logits_at(self, hid, pos):
        return hid[torch.arange(len(pos), device=hid.device), pos] @ self.f.emb.weight.t()

    @staticmethod
    def at(t, j):
        return t[torch.arange(len(j), device=t.device), j]

    # ---- grounded heads ---------------------------------------------------------------------------------
    @torch.no_grad()
    def state_heads(self, x, docs, lengths, need_codes=True):
        """Heads at the last complete chunk J = len/4 - 1 of each row."""
        h, c, cd = self.context(x, docs)
        H = self.hs.heads(c, cd, need_codes=need_codes)
        J = lengths // CH - 1
        out = dict(J=J, u0=self.at(H['u0'], J), a0=self.at(H['a0'], J), chat0=self.at(H['chat0'], J), h=h, c=c, cd=cd, H=H)
        if self.K > 1:
            out['pi_logits'] = self.at(H['pi_logits'], J)
            if need_codes:
                out['chatk'] = self.at(H['chatk'], J); out['ak'] = self.at(H['ak'], J); out['uk'] = self.at(H['uk'], J)
        return out

    @torch.no_grad()
    def decode_chunk(self, x, docs, lengths, codes, port0=False, force_first=None):
        """Greedy-decode the next 4 tokens of every row in place, under the plan records in `codes`.

        codes[r, J_r] must already hold the plan code for the last complete chunk. port0 decodes with
        (chat^(k), A^(0)) (G2.4 item 5). force_first forces the first token (token-tree children, §9.6)."""
        h, c, cd = self.context(x, docs)
        M = c.shape[1]
        need = self.K > 1 and bool((codes[:, :M] >= 0).any())
        H = self.hs.heads(c, cd, need_codes=need)
        cm = codes[:, :M]
        chat, a = self.hs.plan(H, cm) if self.K > 1 else (H['chat0'], H['a0'])
        if port0 and self.K > 1:
            a = H['a0']
        for i in range(CH):
            if i:
                h = self.f.encode(x, docs)
            hid = self.hidden(h, chat, a, docs)
            logits = self.logits_at(hid, lengths + i - 1)
            tok = logits.argmax(-1)
            if i == 0 and force_first is not None:
                tok = torch.where(force_first >= 0, force_first, tok)
            rows = torch.arange(len(x), device=x.device)
            x[rows, lengths + i] = tok; docs[rows, lengths + i] = 0
        return x[:, :].gather(1, (lengths[:, None] + torch.arange(CH, device=x.device)[None]))

    @torch.no_grad()
    def chunk_logprob(self, x, docs, lengths, codes):
        """Log-probability of the 4 tokens already at positions len..len+3 under the plan records (teacher-forced)."""
        h, c, cd = self.context(x, docs)
        M = c.shape[1]
        H = self.hs.heads(c, cd, need_codes=self.K > 1 and bool((codes[:, :M] >= 0).any()))
        chat, a = self.hs.plan(H, codes[:, :M]) if self.K > 1 else (H['chat0'], H['a0'])
        hid = self.hidden(h, chat, a, docs); total = torch.zeros(len(x), device=x.device)
        for i in range(CH):
            lp = self.logits_at(hid, lengths + i - 1).float().log_softmax(-1)
            total += lp.gather(1, x.gather(1, (lengths + i)[:, None])).squeeze(1)
        return total

    @torch.no_grad()
    def pooled_next(self, x, docs, lengths):
        """True pooled concept of the chunk at positions len..len+3 (re-encode after decoding)."""
        h = self.f.encode(x, docs); c = self.f.pool(h)
        return self.at(c, lengths // CH)

    # ---- decoder-free transitions -------------------------------------------------------------------------
    @torch.no_grad()
    def imagine(self, c, cd, J, preds):
        """Feed committed predictions as the concept inputs at J+1 .. J+n (preds: (B, n, D)); heads at J+n."""
        n = preds.shape[1]; B = len(J); M = int(J.max()) + n + 1
        c_in = torch.zeros(B, M, D, device=c.device); cd_in = torch.full((B, M), -1, dtype=torch.long, device=c.device)
        m0 = min(c.shape[1], M)
        c_in[:, :m0] = c[:, :m0]; cd_in[:, :m0] = cd[:, :m0]
        ar = torch.arange(M, device=c.device)[None]
        cd_in = torch.where(ar <= J[:, None], cd_in, torch.full_like(cd_in, -1))
        rows = torch.arange(B, device=c.device)
        for t in range(n):
            c_in[rows, J + 1 + t] = preds[:, t]; cd_in[rows, J + 1 + t] = 0
        H = self.hs.heads(c_in, cd_in)
        return H, J + n

    # ---- rollouts ----------------------------------------------------------------------------------------
    def coded(self, P, j):
        """Codes are chosen only at chunks whose next chunk is a response chunk: j >= P/4 - 1 (§6.2)."""
        return j >= P // CH - 1

    @torch.no_grad()
    def greedy(self, prompts, max_new, policy='argmax', stop_lookup=None, record_states=False, state_limit=None, seed=0):
        """Selector-argmax free generation (G2.2(d)) or D_self rollouts; frozen generate()'s stopping rules.

        prompts: list of arrays with one common length P. policy: 'argmax' | 'sample' | callable(heads)->codes.
        Returns tokens, lengths, codes, per-row verification and (optionally) grounded states."""
        P = len(prompts[0]); B = len(prompts)
        limit = min(T, P + max_new)
        x, docs, lengths = self.batch(prompts, cap=limit + CH)
        codes = torch.full((B, x.shape[1] // CH), -1, dtype=torch.long, device=self.device)
        lookup = torch.tensor([parse_prompt(p)[0] in LOOKUP_TASKS for p in prompts], device=self.device) if stop_lookup is None else stop_lookup
        done = torch.zeros(B, dtype=torch.bool, device=self.device); final = torch.full((B,), limit, device=self.device)
        states = [[] for _ in range(B)]; g = torch.Generator().manual_seed(seed)
        while not bool(done.all()):
            act = (~done).nonzero().flatten()
            xa, da, la = x[act], docs[act], lengths[act]
            ca = codes[act]
            if self.K > 1:
                st = self.state_heads(xa, da, la, need_codes=callable(policy))
                if policy == 'argmax':
                    k = st['pi_logits'].argmax(-1)
                elif policy == 'sample':
                    k = torch.multinomial(st['pi_logits'].float().softmax(-1).cpu(), 1, generator=g).squeeze(1).to(self.device)
                else:
                    k = policy(st)
                ca[torch.arange(len(act), device=self.device), la // CH - 1] = k
            if record_states:
                for i, r in enumerate(act.tolist()):
                    if state_limit is None or len(states[r]) < state_limit[r]:
                        states[r].append(dict(length=int(la[i]), codes=ca[i].clone()))
            chunk = self.decode_chunk(xa, da, la, ca)
            x[act], docs[act], codes[act] = xa, da, ca
            lengths[act] = la + CH
            new_done = lookup[act] | (chunk[:, 0] == DONE) | (la + CH >= limit)
            if state_limit is not None:
                new_done |= torch.tensor([len(states[r]) >= state_limit[r] for r in act.tolist()], device=self.device)
            final[act] = torch.where(new_done, la + CH, final[act])
            done[act] = new_done
        xx = x.cpu().numpy(); ln = final.cpu().numpy()
        ver = [verify(xx[i, :P], xx[i, P:int(ln[i])]) for i in range(B)]
        return dict(x=xx, lengths=ln, codes=codes.cpu(), verification=ver, states=states, prompt_len=P)


def stop_legal(item, cur, hops):
    """The verifier's STOP rule (data.py:85): DONE is legal iff the current node is the goal after >= 1 hop.
    Also the tau_psi label rule (draft §8.2); it reads only the grounded prefix, never a reference continuation."""
    return bool(hops >= 1 and cur == item.goal)
