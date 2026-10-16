"""Deliberate-defect mutants for test_v4 (draft §8.5 step 3: every acceptance test needs a mutant that fails it).

Each mutant rewrites ONE line of a function's source (string replace, exec in the defining module) and runs the tests
that should catch it. A mutant that SURVIVES (all its tests pass) means the harness is blind to that defect.
"""
import inspect
import textwrap
import torch

import model_v4 as M4
import hrm_decoder as HD
import test_v4 as TV

torch.set_default_dtype(torch.float64)


def patch(owner, name, old, new, module):
    """old/new: one string each, or lists of strings applied in order."""
    fn = getattr(owner, name)
    src = textwrap.dedent(inspect.getsource(fn))
    olds, news = (old, new) if isinstance(old, list) else ([old], [new])
    for o, n_ in zip(olds, news):
        assert o in src, (name, o)
        src = src.replace(o, n_, 1)
    ns = dict(module.__dict__)
    exec(src, ns)
    setattr(owner, name, ns[name])
    return lambda: setattr(owner, name, fn)


MUTANTS = [
    ("plan index off by one (future chunk)", (HD, "plan_index", "j = (t + 1) // k - 1", "j = ((t + 1) // k).clamp_max(T // k - 1)", HD),
     ["test_causality", "test_decoder"]),
    ("pooler ignores the pad mask", (M4.PoolAttnFFN, "forward", "allow = r | ~r.any(-1, keepdim=True)",
                                     "allow = torch.ones_like(r)", M4), ["test_pool_init", "test_empty_and_pads"]),
    ("head trains the coarse codebook (no detach)", (M4.ConceptCoreV4, "head", "E0 = self.E0.detach()", "E0 = self.E0", M4),
     ["test_codebook_isolation"]),
    ("fine NCP in the plan's frame q - cbar", (M4.ConceptCoreV4, "ncp_terms",
                                               "frame = q[:, :-1] - self.E0.detach()[n0[:, 1:]]",
                                               "frame = q[:, :-1] - lc[:, :-1].softmax(-1) @ self.E0.detach()", M4),
     ["test_frames"]),
    ("decoder Engram read mixes the previous position", (M4.DecoderV4, "forward",
                                                         "D = D + rd.write(h) * F.silu(a * rd.Wv(e))",
                                                         "D = D + rd.write(h) * F.silu(a * (rd.Wv(e) + F.pad(rd.Wv(e), (0, 0, 1, 0))[:, :-1]))", M4),
     ["test_decoder"]),
    ("S layers looped too", (M4.EncoderV4, "_fwd", 'reps = loops if layer.kind == "F" else 1', "reps = loops", M4),
     ["test_reach_L"]),
    ("fine restart from residuals of the OLD E0", (M4.ConceptCoreV4, "restart_dead",
                                                   ["x = c.detach().reshape(-1, self.d)", "n0 = torch.cdist(x, self.E0).argmin(-1)", "r = (x - self.E0[n0])"],
                                                   ["x = c.detach().reshape(-1, self.d); old0 = self.E0.clone()", "n0 = torch.cdist(x, old0).argmin(-1)", "r = (x - old0[n0])"], M4),
     ["test_restart"]),
    ("core attention not causal", (M4.ConceptCoreV4, "run", "_, mask = build_masks(cdoc, 1)",
                                   "mask = (cdoc[:, :, None] == cdoc[:, None, :]).unsqueeze(1)", M4), ["test_causality"]),
    ("VQ sends gradient into the encoder", (M4.ConceptCoreV4, "vq_terms", "c = c_hat.detach()", "c = c_hat", M4),
     ["test_codebook_isolation"]),
    ("shared-FFN branch dead (gain held at 0)", (M4.EncoderV4, "_fwd", "None if shared is None else self.shared_gain[li]",
                                                 "None if shared is None else 0 * self.shared_gain[li]", M4), ["test_alive"]),
    ("decoder reads a FUTURE Engram row", (M4.DecoderV4, "forward", "D = D + rd.write(h) * F.silu(a * rd.Wv(e))",
                                           "D = D + rd.write(h) * F.silu(a * rd.Wv(torch.roll(e, -1, 1)))", M4),
     ["test_causality"]),
    ("gated conv: gate held at 1 (= content-blind conv)", (M4.PoolGConv, "forward",
                                                           "gate = 2 * torch.sigmoid(self.Wg(rms(m)).view(B, M, k, H) + self.bg.t())",
                                                           "gate = torch.ones(B, M, k, H, dtype=m.dtype) + 0 * self.bg.sum()", M4),
     ["test_pool_selects"]),
    ("pooler crosses chunk boundaries (softmax over the row)", (M4.PoolAttnFFN, "forward",
                                                                "a = s.masked_fill(~allow[..., None], float(\"-inf\")).softmax(2)",
                                                                "a = s.masked_fill(~allow[..., None], float(\"-inf\")).flatten(1, 2).softmax(1).view_as(s)", M4),
     ["test_causality", "test_pool_init"]),
]


def main():
    rows = []
    for name, (owner, fn, old, new, mod), tests in MUTANTS:
        undo = patch(owner, fn, old, new, mod)
        caught = []
        try:
            for t in tests:
                try:
                    getattr(TV, t)()
                except Exception as ex:                                  # noqa: BLE001
                    caught.append(f"{t}: {type(ex).__name__} {str(ex)[:90]}")
        finally:
            undo()
        status = "KILLED" if caught else "SURVIVED"
        rows.append((name, status, caught))
        print(f"{status:8s} | {name} | " + (" ; ".join(caught) if caught else "no test failed"), flush=True)
    n_k = sum(r[1] == "KILLED" for r in rows)
    print(f"MUTANTS KILLED {n_k}/{len(rows)}")
    assert n_k == len(rows)


if __name__ == "__main__":
    main()
