"""Deliberate defects for the v4-vs-LLM tests: each mutant must make at least one named test fail (by assertion).

python mutants_vs.py
"""
import contextlib
import sys
import numpy as np
import torch

import model_llm as ML
import pinpad as PP
import data_v4 as D
import run_vs as R
import test_vs as TV
import speed_v3

speed_v3.enable_block_attention(60)


@contextlib.contextmanager
def patch(obj, name, new):
    old = getattr(obj, name)
    setattr(obj, name, new)
    try:
        yield
    finally:
        setattr(obj, name, old)


def m_swa_full():                      # S layers attend over the full causal context (no window)
    orig = ML.build_masks
    return _two(patch(ML, "build_masks", lambda doc, w: (orig(doc, 10 ** 6)[0], orig(doc, w)[1])),
                patch(speed_v3, "_CFG", dict(block=0)))


def m_no_causal():
    def bm(doc, w):
        same = (doc[:, :, None] == doc[:, None, :]).unsqueeze(1)
        return same, same
    return _two(patch(ML, "build_masks", bm), patch(speed_v3, "_CFG", dict(block=0)))


def m_no_doc():
    def bm(doc, w):
        T = doc.shape[1]; i = torch.arange(T)
        causal = (i[None, :] <= i[:, None])[None, None].expand(doc.shape[0], 1, T, T)
        return causal & ((i[:, None] - i[None, :]) < w)[None, None], causal
    return _two(patch(ML, "build_masks", bm), patch(speed_v3, "_CFG", dict(block=0)))


@contextlib.contextmanager
def _two(a, b):
    with a, b:
        yield


def m_schedule_ignores_M():
    orig = ML.LLMStack.make_schedule
    return patch(ML.LLMStack, "make_schedule", lambda self, M=None, enc_loop=None: orig(self, None, enc_loop))


def m_expert_ignores_colours():
    def oa(cell, target, walls, colors):
        dist = PP.bfs_dist(target, set(walls))
        if dist[cell] <= 0:
            return [], dist
        return [a for a in range(4) if PP.step(cell, a, walls) != cell and dist[PP.step(cell, a, walls)] == dist[cell] - 1], dist
    return patch(PP, "optimal_actions", oa)


def m_position_lag():                   # position token = cell BEFORE the move
    orig = PP.encode
    def enc(colors, walls, start, acts, cells, targets, opts, task):
        prev = [start] + cells[:-1]
        return orig(colors, walls, start, acts, prev, targets, opts, task)
    return patch(PP, "encode", enc)


def m_flags_swapped():
    orig = PP.boundary_flags
    return patch(PP, "boundary_flags", lambda ep: [{1: 2, 2: 1}.get(x, x) for x in orig(ep)])


def m_metric_misaligned():              # each action scored against the NEXT step's optimal set
    orig = R.pin_rows
    def pr(eps, T):
        t, d, lab, opt = orig(eps, T)
        return t, d, lab, {(r, i): opt.get((r, i + 2), A) for (r, i), A in opt.items()}   # next step's optimal set
    return patch(R, "pin_rows", pr)


def m_llm_skips_init_batches():
    src = R.main
    def main(argv=None):
        a = R.get_args(argv)
        return src(argv) if a.arch == "v4" else _main_no_skip(argv)
    return patch(R, "main", main)


def _main_no_skip(argv):
    import types
    code = open(R.__file__).read().replace("stream.next(); stream.next()", "pass")
    mod = types.ModuleType("run_vs_mut"); mod.__file__ = R.__file__
    exec(compile(code, R.__file__, "exec"), mod.__dict__)
    mod.final_eval = lambda m, a, S: {}
    mod.D = D                                                            # share the recording to_batch patch
    return mod.main(argv)


def m_val_fillers_from_train():
    orig = D.filler_sentences
    def fs(store, tok, n_docs=3000, seed=0):
        if store.n < 10000:                                             # the val store -> use the train store instead
            return orig(D.TextStore("train"), tok, n_docs=60000, seed=seed)
        return orig(store, tok, n_docs, seed)
    return patch(D, "filler_sentences", fs)


def m_ffn_256():
    orig = ML.build_model
    return patch(ML, "build_model", lambda size="tiny", **o: orig(size, **dict(o, ff=256)))


MUTANTS = [
    ("S layers see the full context", m_swa_full, "test_window"),
    ("no causal mask", m_no_causal, "test_causal_and_docs"),
    ("no document mask", m_no_doc, "test_causal_and_docs"),
    ("middle block ignores M", m_schedule_ignores_M, "test_loops"),
    ("expert walks over other colours", m_expert_ignores_colours, "test_pinpad_expert"),
    ("position token lags the move", m_position_lag, "test_pinpad_expert"),
    ("boundary flags swapped", m_flags_swapped, "test_boundary_flags"),
    ("action metric misaligned", m_metric_misaligned, "test_metrics_oracle"),
    ("LLM runner skips the 2 init batches' consumption", m_llm_skips_init_batches, "test_runner_rows"),
    ("val fillers from the train store", m_val_fillers_from_train, "test_filler_fix"),
    ("LLM FFN 256 (param mismatch)", m_ffn_256, "test_param_match"),
]

if __name__ == "__main__":
    killed = 0
    for name, mk, test in MUTANTS:
        fn = getattr(TV, test)
        try:
            with mk():
                fn()
            print(f"SURVIVED  {name}  ({test} passed)", flush=True)
        except AssertionError as e:
            killed += 1
            print(f"KILLED    {name}  by {test}: {str(e)[:90]}", flush=True)
        except Exception as e:
            print(f"CRASHED   {name}  in {test}: {type(e).__name__}: {str(e)[:90]}", flush=True)
    print(f"{killed}/{len(MUTANTS)} mutants killed by assertions")
    sys.exit(0 if killed == len(MUTANTS) else 1)
