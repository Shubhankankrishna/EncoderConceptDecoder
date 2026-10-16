"""V9 H-NCP (draft §6): K committed next-concept hypotheses at the core's final block application B_H.

Ownership is V8 REFIT's (§6.1): the foundation (encoder, embedding/tied head, decoder, Q, PQ heads, port) is
frozen; a new core (DenseCore copy of C1-C4, or V8's recurrent Core) is trained. V9 adds only the code
embeddings e_k, the write W_z and the selector W_pi, and only when K >= 2 (at K = 1 nothing is instantiated).
"""
import copy, math
import torch
from torch import nn
from torch.nn import functional as F
import v9paths
from model import D, HEADS, S, CODES, Block, DenseCore, Core, System, Foundation, rms, positions, rotate, chunk_info, shift, freeze, count
from runtime import balanced, nll, ncp_weights, load_system

CODE_DIM = 16


class Codes(nn.Module):
    """e_k in R^16, W_z in R^(128x16), selector W_pi in R^(Kx128) + b_pi. 3,208 parameters at K = 8."""
    def __init__(self, k):
        super().__init__()
        if k < 2:
            raise ValueError('K = 1 instantiates no code parameters (draft §6.2)')
        self.k = k
        self.e = nn.Parameter(torch.randn(k, CODE_DIM))
        self.W_z = nn.Linear(CODE_DIM, D, bias=False)
        nn.init.normal_(self.W_z.weight, std=.02)
        self.selector = nn.Linear(D, k)
        nn.init.zeros_(self.selector.weight); nn.init.zeros_(self.selector.bias)

    def deltas(self, e=None):
        return self.W_z(self.e if e is None else e)


def block_codes(block, x, docs, delta):
    """B_H with the code entering only the query and the residual (§6.2).

    x: (b, t, D) block input; delta: (K, D) or (b, t, K, D) code writes. Keys and values come from the
    unperturbed stream, position j's own included. Returns (b, t, K, D). LoRA banks are applied to the
    unperturbed keys/values and to the perturbed queries when present (S6-H)."""
    b, t, _ = x.shape
    if delta.dim() == 2:
        delta = delta[None, None].expand(b, t, *delta.shape)
    kk = delta.shape[2]; hd = D // HEADS
    pos = positions(docs)
    a = block.n1(x)
    qkv, _ = block.project(a, docs, None)
    _, k, v = qkv.view(b, t, 3, HEADS, hd).permute(2, 0, 3, 1, 4)
    k = rotate(k, pos)
    xk = x[:, :, None, :] + delta
    ak = block.n1(xk)
    qkvk, _ = block.project(ak.reshape(b, t * kk, D), docs.repeat_interleave(kk, 1), None)
    q = qkvk[..., :D].reshape(b, t, kk, HEADS, hd).permute(0, 2, 3, 1, 4).reshape(b * kk, HEADS, t, hd)
    q = rotate(q, pos.repeat_interleave(kk, 0))
    k = k[:, None].expand(b, kk, HEADS, t, hd).reshape(b * kk, HEADS, t, hd)
    v = v[:, None].expand(b, kk, HEADS, t, hd).reshape(b * kk, HEADS, t, hd)
    causal = torch.arange(t, device=x.device)[None, :] <= torch.arange(t, device=x.device)[:, None]
    mask = (docs[:, :, None] == docs[:, None, :]) & causal
    att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.repeat_interleave(kk, 0)[:, None])
    att = att.transpose(1, 2).reshape(b, kk, t, D).permute(0, 2, 1, 3)
    y = xk + block.o(att)
    g, u = block.w12(block.n2(y)).chunk(2, -1)
    return y + block.w3(F.silu(g) * u)


def core_stream(core, c, cd):
    """Run the core up to B_H. Returns (x, block, finish): B_H's input stream, the block, and the readout."""
    if isinstance(core, DenseCore):
        x = c
        for blk in core.blocks[:-1]:
            x = blk(x, cd)
        def finish(z):
            u = core.norm(z)
            return u, core.W_A(rms(u))
        return x, core.blocks[-1], finish
    if isinstance(core, Core):
        e = core.in_proj(c)
        h, l = core.h0.expand_as(e) + 0.0, core.l0.expand_as(e)
        for visit in range(5):  # L, L, H, L, L
            if visit in (0, 1, 3, 4):
                l = core.post_L(core.fL(l + h + e + 0.0, cd))
            else:
                h = core.post_H(core.fH(h + l, cd))
        def finish(z):
            hh = core.out_n(core.post_H(z))
            return core.B_u(hh), core.B_A(hh)
        return h + l, core.fH, finish
    raise TypeError(type(core))


def pq_any(f, u):
    """Teacher-frozen soft-PQ expectation for any leading shape (NCP paper eq. 9)."""
    p = f.heads(u).view(*u.shape[:-1], S, CODES).float().softmax(-1)
    return torch.einsum('...sv,svd->...sd', p, f.Q).reshape(*u.shape[:-1], D)


class HNCP(nn.Module):
    def __init__(self, foundation, core_kind='dense', k=1):
        super().__init__()
        self.f = foundation
        self.core = DenseCore(foundation) if core_kind == 'dense' else Core()
        self.codes = Codes(k) if k > 1 else None
        self.k = k; self.core_kind = core_kind

    @property
    def is_flat(self):
        return False

    def context(self, x, docs):
        h = self.f.encode(x, docs)
        cd, valid = chunk_info(docs)
        return h, self.f.pool(h), cd, valid

    def heads(self, c, cd, need_codes=True, e=None):
        x, blk, finish = core_stream(self.core, c, cd)
        u0, a0 = finish(blk(x, cd))
        out = dict(x=x, u0=u0, a0=a0, chat0=self.f.pq(u0))
        if self.codes is not None:
            out['pi_logits'] = self.codes.selector(u0)
            if need_codes:
                uk, ak = finish(block_codes(blk, x, cd, self.codes.deltas(e)))
                out.update(uk=uk, ak=ak, chatk=pq_any(self.f, uk))
        return out

    def plan(self, H, codes):
        """codes: (b, m) long, -1 = code-free. Returns the decoder plans (chat, a)."""
        if self.codes is None or codes is None:
            return H['chat0'], H['a0']
        idx = codes.clamp_min(0)[..., None, None].expand(*codes.shape, 1, D)
        chat = torch.gather(H['chatk'], 2, idx).squeeze(2); a = torch.gather(H['ak'], 2, idx).squeeze(2)
        use = (codes >= 0)[..., None]
        return torch.where(use, chat, H['chat0']), torch.where(use, a, H['a0'])

    def forward(self, x, docs, codes=None):
        h, c, cd, _ = self.context(x, docs)
        if c.shape[1] == 0:
            return self.f.decode(h, c, c, docs)
        H = self.heads(c, cd, need_codes=codes is not None and bool((codes >= 0).any()))
        chat, a = self.plan(H, codes)
        return self.f.decode(h, chat, a, docs)

    def as_system(self):
        """K = 1 export: the exact frozen-V8 System (foundation + core) for the frozen evaluator."""
        if self.codes is not None:
            raise ValueError('only K = 1 is a V8 System')
        s = System(copy.deepcopy(self.f)); s.core = copy.deepcopy(self.core); return s

    def metadata(self):
        return dict(kind='hncp', core=self.core_kind, k=self.k, code_parameters=count(self.codes) if self.codes is not None else 0,
                    core_parameters=count(self.core), active_parameters=count(self.f) - count(self.f.con) - count(self.f.con_norm) - count(self.f.W_A) + count(self.core) + (count(self.codes) if self.codes is not None else 0))


def build(teacher_path, core_kind='dense', k=1, device='cpu'):
    """Student from a frozen S1 teacher checkpoint (§6.1). The foundation is a copy; the teacher stays separate."""
    teacher, ck = load_system(teacher_path, device)
    if teacher.is_flat or teacher.core is not None or teacher.adapters is not None:
        raise ValueError('S2 teacher must be an S1 hierarchy foundation')
    freeze(teacher)
    f = copy.deepcopy(teacher.foundation); freeze(f)
    hs = HNCP(f, core_kind, k).to(device)
    for p in hs.core.parameters():
        p.requires_grad_(True)
    if hs.codes is not None:
        for p in hs.codes.parameters():
            p.requires_grad_(True)
    return hs, teacher


def save(path, hs, extra):
    from runtime import atomic_save
    atomic_save(path, dict(metadata=hs.metadata(), model=hs.state_dict(), **extra))


def load(path, device='cpu'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = ck['metadata']
    f = Foundation(); hs = HNCP(f, m['core'], m['k'])
    for i in range(m.get('phases', 0)):
        raise NotImplementedError('LSL banks on H-NCP are loaded by lsl.py')
    hs.load_state_dict(ck['model'], strict=True); freeze(hs)
    return hs.to(device), ck
