"""Train HRM THINKER v3 (19M, draft_v3.md r2). One process, one device; resumable; pause-aware; 48 h in ONE schedule.

  python train.py --run_dir ../runs/v3_48h --hours 48 --milestone_hours 24
  python train.py --run_dir ../runs/v3_48h --resume ../runs/v3_48h/ckpt_last.pt

Per micro-batch: main trajectory M = 3 (1/3, 1/3, 1/3) + a shortcut S ~ {1: 0.5, 2a: 0.25, 2b: 0.25} (own Generator,
checkpointed), batched with the main passes at concept rate; full backprop; one backward per micro-batch.
Loss (full-batch denominators; all micro-batches of an optimizer step are fetched first):
  L = NTP(u3) + 0.1 NCP(u3) + 0.05 NCP(u_s) + 0.1 L_scons(rms S_s, sg rms S3) + VQ
  NTP over valid targets; NCP over valid pairs (nonempty current chunk, complete target chunk, same doc);
  L_scons over nonempty chunks; VQ over complete chunks.
LR: warm-up by step, cosine by elapsed TRAINING time (pauses and the out-of-process 24 h evaluation excluded); a per-step
LR trace is stored (lr_trace.npy + checkpoint). Milestone (24 h): save and exit 5; the launcher evaluates and resumes.
Exit codes: 0 done, 3 guard stop, 4 paused, 5 milestone.
"""
import argparse, contextlib, json, math, os, random, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F

import model as M
import speed_v3 as SV
from model import build_model, chunk_masks, param_groups, param_table, rms
from concept_core_v2 import ConceptVocab
from data import ShardStream, target_mask, make_batch, Canon
from evaluate import Evaluator

V1_EVAL_STEPS = (185, 265, 365, 554, 766, 1009, 1217, 1402, 1593, 1792, 1992, 2190, 2376, 2559, 2764, 2944, 3121, 3304,
                 3484, 3669, 3853, 4039, 4050)
SHORT_P = (("1", 0.5), ("2a", 0.25), ("2b", 0.25))


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def nz(x):
    return float("nan") if x is None else x


def lr_at(step, elapsed, a):
    warm = min(1.0, (step + 1) / a.warmup)
    prog = min(1.0, elapsed / (a.hours * 3600))
    return a.lr * a.lr_scale * warm * (a.lr_min_frac + (1 - a.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def sample_short(gen):
    r = torch.rand((), generator=gen).item()
    acc = 0.0
    for name, p in SHORT_P:
        acc += p
        if r < acc:
            return name
    return SHORT_P[-1][0]


@torch.no_grad()
def restart_dead_now(vocab, c, complete, opt):
    """Re-seed codes unused since the last restart from COMPLETE real chunks; reset their Adam moments."""
    x = c.detach()[complete].reshape(-1, vocab.S, vocab.ds)
    dead_total, hits_total = 0, float(vocab.hits.sum())
    used = (vocab.hits > 0).float().mean(1)
    stt = opt.state.get(vocab.e, {})
    for j in range(vocab.S):
        dead = vocab.hits[j] == 0
        nd = int(dead.sum())
        if nd and len(x):
            vocab.e[j, dead] = x[torch.randint(len(x), (nd,), device=x.device), j]
            for k in ("exp_avg", "exp_avg_sq"):
                if k in stt:
                    stt[k][j, dead] = 0
            dead_total += nd
    vocab.hits.zero_()
    return dict(restart_dead=dead_total, hits_total=hits_total, codes_used_min_seg=float(used.min()),
                codes_used_mean=float(used.mean()), candidates=int(len(x)))


def rng_state():
    st = dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state())
    if torch.backends.mps.is_available():
        st["mps"] = torch.mps.get_rng_state()
    return st


def set_rng_state(st):
    random.setstate(st["py"]); np.random.set_state(st["np"]); torch.set_rng_state(st["torch"])
    if "mps" in st and torch.backends.mps.is_available():
        torch.mps.set_rng_state(st["mps"])


def save_ckpt(path_dir, state, step, keep=2, best=False):
    tmp = os.path.join(path_dir, "ckpt.tmp")
    torch.save(state, tmp)
    final = os.path.join(path_dir, f"ckpt_step{step:06d}.pt")
    os.replace(tmp, final)
    for name, flag in (("ckpt_last.pt", True), ("ckpt_best.pt", best)):
        if flag:
            dst = os.path.join(path_dir, name); t = dst + ".tmp"
            if os.path.exists(t):
                os.remove(t)
            os.link(final, t); os.replace(t, dst)
    steps = sorted(f for f in os.listdir(path_dir) if f.startswith("ckpt_step"))
    for f in steps[:-keep]:
        os.remove(os.path.join(path_dir, f))


def phys_footprint_gib():
    try:
        import ctypes
        buf = (ctypes.c_uint64 * 32)()
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        if lib.proc_pid_rusage(os.getpid(), 2, ctypes.byref(buf)) != 0:
            return -1.0
        return buf[2 + 7] / 2**30
    except Exception:
        return -1.0


def swap_used_mb():
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    try:
        return float(s.split("used =")[1].split("M")[0])
    except Exception:
        return -1.0


def batch_counts(batches):
    """Full-batch denominators from ids/doc only (before any forward)."""
    nt = npair = ncomp = nne = 0
    for ids, doc, _ in batches:
        nt += int(target_mask(ids, doc).sum())
        ne, co, po = chunk_masks(ids, doc)
        npair += int(po.sum()); ncomp += int(co.sum()); nne += int(ne.sum())
    return dict(t=max(nt, 1), p=max(npair, 1), c=max(ncomp, 1), ne=max(nne, 1))


def micro_step(m, ids, doc, ei, short, a, N, telem=False):
    """One micro-batch: forward (main + batched shortcut), weighted sum over full-batch denominators, one backward.
    Returns detached SUMS (divide by N for means)."""
    core = m.core
    tok, c_hat, eaux = m.enc(ids, doc, ei)
    cdoc = doc[:, ::4]
    nonempty, complete, pair_ok = chunk_masks(ids, doc)
    vq_s, n = core.vq_sum(c_hat, complete, count_hits=True)
    S3, states, Ss = core.run(c_hat, cdoc, a.main_sched, short=short)
    u = core.out_norm(S3)
    lc, cp = core.head(u)
    ncp_s = core.ncp_sum(lc, n, pair_ok)
    logits, _ = m.dec(tok, u, cp, doc, m.enc.embed.weight)
    tgt = ids[:, 1:].masked_fill(~target_mask(ids, doc), -100)
    ntp_s = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100, reduction="sum")
    if short is not None:
        us = core.out_norm(Ss)
        lcs, _ = core.head(us, pred=False)
        ncps_s = core.ncp_sum(lcs, n, pair_ok)
        scons_s = ((rms(Ss) - rms(S3).detach()).pow(2).mean(-1) * nonempty).sum()
    else:
        ncps_s = scons_s = torch.zeros((), device=ids.device)
    loss = ntp_s / N["t"] + a.lam_ncp * ncp_s / N["p"] + a.lam_s * ncps_s / N["p"] + a.lam_c * scons_s / N["ne"] + vq_s / N["c"]
    loss.backward()                                                  # no /accum: N is the whole optimizer batch
    out = dict(ntp=ntp_s.detach(), ncp=ncp_s.detach(), ncp_s=ncps_s.detach(), scons=scons_s.detach(), vq=vq_s.detach(),
               c_hat=c_hat.detach(), complete=complete, engram_alpha=eaux["engram_alpha"].mean().detach())
    if telem:
        out["states"] = [s.detach() for s in states]
    return out


@torch.no_grad()
def loop_geometry(states):
    """Per-pass norms, update norms, angle(dS2, dS3), rho3 = |S3| / sum |dS_i|, relative updates (draft §4.7)."""
    d = [states[0]] + [states[i] - states[i - 1] for i in range(1, len(states))]
    nS = [float(s.norm(dim=-1).mean()) for s in states]
    nd = [float(x.norm(dim=-1).mean()) for x in d]
    ang = float(torch.rad2deg(torch.acos(F.cosine_similarity(d[1], d[2], -1).clamp(-1, 1))).mean()) if len(d) >= 3 else None
    rho = float((states[-1].norm(dim=-1) / sum(x.norm(dim=-1) for x in d).clamp_min(1e-9)).mean())
    return dict(S_norm=nS, dS_norm=nd, angle23=ang, rho3=rho, rel_dS=[x / max(nd[0], 1e-9) for x in nd[1:]])


@torch.no_grad()
def table_norms(core):
    if core.table is None:
        return {}
    t = core.table                                                  # (6 pairs, 4 layers, 7 sites, d)
    out = {}
    names = {v: f"{k[0]}{k[1]}" for k, v in M.PAIRS.items()}
    for p in range(t.shape[0]):
        out[f"t{names[p]}"] = dict(gC=float(t[p, :, [0, 3]].norm(dim=-1).mean()), gS=float(t[p, :, [1, 4]].norm(dim=-1).mean()),
                                  a=float(t[p, :, [2, 5, 6]].norm(dim=-1).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "run_m2_25M", "data")))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--size", default="full")                       # full | small (CPU smoke)
    ap.add_argument("--micro_bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--lr_scale", type=float, default=1.0)
    ap.add_argument("--lr_min_frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--hours", type=float, default=48.0)
    ap.add_argument("--milestone_hours", type=float, default=24.0)
    ap.add_argument("--est_total_steps", type=int, default=7600)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--init_only", action="store_true")              # save the step-0 checkpoint (G3 audit) and exit
    ap.add_argument("--lam_ncp", type=float, default=0.1)
    ap.add_argument("--lam_s", type=float, default=0.05)
    ap.add_argument("--lam_c", type=float, default=0.1)
    ap.add_argument("--no_shortcut", action="store_true")           # arms A1 / A3 / T3 (draft r2 §9.4)
    ap.add_argument("--no_table", action="store_true")
    ap.add_argument("--shared_ff", type=int, default=-1)             # -1 = config default; 0 = no shared FFN
    ap.add_argument("--core_ff", type=int, default=-1)               # own-FFN arm: 992
    ap.add_argument("--main_sched", default="3")                     # A1: "1"
    ap.add_argument("--eval_R", default="3")
    ap.add_argument("--eval_every_min", type=float, default=60.0)
    ap.add_argument("--eval_at_v1_steps", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--traj_seed", type=int, default=1234)
    ap.add_argument("--heartbeat", type=int, default=1)
    ap.add_argument("--no_final_eval", action="store_true")
    ap.add_argument("--quick_eval_n", type=int, default=128)
    ap.add_argument("--resume", default="")
    ap.add_argument("--mem_frac_guard", type=float, default=0.90)
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--div_min_wall_min", type=float, default=30.0)
    ap.add_argument("--mem_restart_gib", type=float, default=24.0)
    ap.add_argument("--mem_restart_growth_gib", type=float, default=12.0)
    ap.add_argument("--debug_restart_step", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")      # tests only: 1 CPU thread + deterministic algorithms
    ap.add_argument("--block_attn", type=int, default=0)             # speed path 1 (0 = v1 full-mask SDPA)
    ap.add_argument("--step_memo", type=int, default=1)              # speed path 2 (bit-identical)
    ap.add_argument("--compile", default="none")                    # none | core | all (A/B; see DECISIONS)
    ap.add_argument("--dash_file", default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "run_m2_25M", "webapp", "data", "v3_live.json")))
    a = ap.parse_args()
    os.makedirs(a.run_dir, exist_ok=True)
    dev = torch.device(a.device)
    if a.deterministic:
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)

    over = dict(table=not a.no_table)
    if a.shared_ff >= 0:
        over["shared_ff"] = a.shared_ff
    if a.core_ff > 0:
        over["core_ff"] = a.core_ff
    m, ecfg, dcfg = build_model(a.size, **over)
    m = m.to(dev)
    groups, _ = param_groups(m, 0.1)
    opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95), eps=1e-8)
    st = dict(step=0, elapsed=0.0, best=float("inf"), cursor=0, milestone_done=False)
    traj_gen = torch.Generator().manual_seed(a.traj_seed)
    lr_trace = []

    def load(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        for k in ("step", "elapsed", "best", "cursor", "milestone_done"):
            st[k] = ck.get(k, st[k])
        for k, v in ck.get("args", {}).items():
            if k not in ("run_dir", "resume", "device", "dash_file", "data_dir", "debug_restart_step", "max_steps",
                         "no_final_eval", "deterministic", "init_only"):
                setattr(a, k, v)
        set_rng_state(ck["rng"])
        traj_gen.set_state(ck["traj_rng"])
        lr_trace[:] = list(ck.get("lr_trace", []))
        return ck

    if a.resume:
        load(a.resume)
        print(f"{now()} resumed from {a.resume}: step {st['step']}, elapsed {st['elapsed']/3600:.2f} h, cursor {st['cursor']}, "
              f"lr_scale {a.lr_scale}, micro_bs {a.micro_bs} x accum {a.accum}, milestone_done {st['milestone_done']}", flush=True)
    json.dump(dict(vars(a), params={k: v for k, v in param_table(m).items()}, time=now()),
              open(os.path.join(a.run_dir, "config.json"), "w"), indent=1)
    rec_max = torch.mps.recommended_max_memory() if dev.type == "mps" else 0

    # ---- codebook k-means++ init from COMPLETE real chunks of the first optimizer batch (>= 4 x 128 candidates)
    if st["step"] == 0 and not a.resume:
        tok_rows = np.load(os.path.join(a.data_dir, "train", "tok.npy"), mmap_mode="r")
        doc_rows = np.load(os.path.join(a.data_dir, "train", "doc.npy"), mmap_mode="r")
        order = np.random.RandomState(a.seed).permutation(tok_rows.shape[0])
        canon = Canon(a.data_dir)
        m.eval(); cs = []; i = 0; need = 4 * m.core.vocab.N
        with torch.no_grad():
            while i < a.accum or sum(len(c) for c in cs) < need:
                idx = order[i * a.micro_bs:(i + 1) * a.micro_bs]
                ids, doc, ei = make_batch(np.asarray(tok_rows[idx]), np.asarray(doc_rows[idx]), canon, ecfg)
                _, co, _ = chunk_masks(ids, doc)
                c = m.enc(ids.to(dev), doc.to(dev), ei.to(dev))[1].float().cpu()
                cs.append(c[co]); i += 1
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(a.seed + 12345)
            tmp = ConceptVocab(ecfg.dc, m.core.vocab.S, m.core.vocab.N)
            tmp.init_from(torch.cat(cs))
        m.core.vocab.e.data.copy_(tmp.e.data.to(dev))
        print(f"{now()} codebook k-means++ init from {sum(len(c) for c in cs)} complete chunks ({i} micro-batches)", flush=True)
        del cs, tmp
    m.train()

    SV.enable_block_attention(a.block_attn)
    if a.step_memo:
        def micro(*args_, **kw):
            with SV.step_memo():
                return micro_step(*args_, **kw)
    else:
        micro = micro_step
    if a.compile != "none":                                          # A/B only; keeps state_dict names unchanged
        t0 = time.time()

        for l in list(m.core.layers) + (list(m.enc.layers) + list(m.dec.layers) if a.compile == "all" else []):
            SV.compile_layer(l, lambda l_: l_.training)                  # compiled only in training; k_norm eager (V3-006)
        print(f"{now()} torch.compile({a.compile}) wrapped in {time.time() - t0:.1f}s (compiles lazily)", flush=True)

    stream = ShardStream(a.data_dir, "train", ecfg, a.micro_bs, seed=a.seed, cursor=st["cursor"], prefetch=a.accum + 4)
    ev = Evaluator(a.data_dir, ecfg, dev, batch_rows=4)
    tlog = open(os.path.join(a.run_dir, "train_log.jsonl"), "a")
    elog = open(os.path.join(a.run_dir, "log.jsonl"), "a")
    glog = open(os.path.join(a.run_dir, "GUARDS.jsonl"), "a")
    t_start = time.time() - st["elapsed"]
    last_eval_wall = time.time()
    tok_count, t_win, ema = 0, time.time(), None
    hist, eval_secs = [], [0.0]
    KEYS = ("loss", "ntp", "ncp", "ncp_s", "scons", "vq", "engram_alpha")
    new_acc = lambda: dict({k: 0.0 for k in KEYS}, n=0, gn=0.0, gn_cb=0.0, short={"1": 0, "2a": 0, "2b": 0})
    acc = new_acc()
    dash, evals_dash = [], []
    geo_last, open_log = {}, {}

    def elapsed():
        return time.time() - t_start

    def write_dash(train_rec=None, eval_rec=None):
        if not a.dash_file:
            return
        try:
            if train_rec:
                dash.append(train_rec); del dash[:-2000]
            if eval_rec:
                pr, cb, lp, tel = eval_rec["probes"], eval_rec["codebook"], eval_rec.get("loop", {}), eval_rec["telemetry"]
                evals_dash.append(dict(step=eval_rec["step"], tag=eval_rec.get("tag"), wall_h=eval_rec.get("wall_h"),
                                       val=eval_rec["val_lang_mean"], ts=eval_rec["val_tinystories"]["nats"],
                                       wk=eval_rec["val_wikitext"]["nats"], bpb_ts=eval_rec["val_tinystories"].get("bpb"),
                                       bpb_wk=eval_rec["val_wikitext"].get("bpb"),
                                       **{k[3:]: v.get("exact") for k, v in pr.items()},
                                       swap=eval_rec["recall1_plan_swap"]["exact"], ncp=cb.get("ncp_ce"), ncp_M1=eval_rec.get("ncp_M1"),
                                       VE=cb.get("var_explained"), codes_min=cb.get("codes_used_frac_min"),
                                       perplexity=cb.get("perplexity_mean"), G_share=tel["crc_mix_e0_L_G"][2],
                                       mix=tel["crc_mix_e0_L_G"], Wv1=tel.get("engram_Wv_norm"), Wv2=tel.get("engram2_Wv_norm"),
                                       alpha=tel.get("engram_alpha_mean"), shared_w2=tel.get("shared_w2_norm"),
                                       table_norm=tel.get("table_norm"), angle23=lp.get("angle23"), rho3=lp.get("rho3"),
                                       rel_step=lp.get("rel_step"), S_norm=lp.get("state_norm"), grad_ratio=lp.get("grad_ratio"),
                                       eval_sec=eval_rec.get("eval_sec"), is_best=eval_rec.get("is_best")))
            last = dash[-1] if dash else {}
            H = {k: [[r["step"], r[k]] for r in dash if r.get(k) is not None] for k in
                 ("loss", "ntp", "ncp", "ncp_s", "scons", "vq", "tok_s", "gn_preclip", "gn_codebook", "angle23", "rho3",
                  "lr", "epochs", "footprint_gib", "swap_mb", "mps_driver_gib", "mps_alloc_gib", "engram_alpha", "wall_h")}
            H["val"] = [[e["step"], e["val"]] for e in evals_dash]
            H["recall"] = [[e["step"], e["recall1"]] for e in evals_dash if e.get("recall1") is not None]
            H["table"] = [[r["step"], r["table"]] for r in dash if r.get("table")]
            H["rel_dS"] = [[r["step"], r["geometry"].get("rel_dS")] for r in dash if r.get("geometry")]
            el = elapsed()
            n_tot = int(st["step"] + (a.hours * 3600 - el) * st["step"] / el) if st["step"] >= 50 and el > 0 else a.est_total_steps
            out = dict(phase="v3", step=st["step"], time=now(), t=time.time(), wall_h=el / 3600, hours=a.hours,
                       milestone_hours=a.milestone_hours, milestone_done=st["milestone_done"], total_steps=n_tot,
                       lr=last.get("lr"), gn=last.get("gn_preclip"), tok_s=last.get("tok_s"), loss=last.get("loss"),
                       ntp=last.get("ntp"), epochs=last.get("epochs"), slots=st["step"] * a.accum * a.micro_bs * 1024,
                       best_val=st["best"] if st["best"] < 1e9 else None, history=H, evals=evals_dash,
                       geometry=last.get("geometry"), table=last.get("table"), short_counts=last.get("short_counts"),
                       config=dict(micro_bs=a.micro_bs, accum=a.accum, lr=a.lr, warmup=a.warmup, block_attn=a.block_attn,
                                   compile=a.compile, lam=[a.lam_ncp, a.lam_s, a.lam_c]), opened=open_log)
            with open(a.dash_file + ".tmp", "w") as f:
                json.dump(out, f)
            os.replace(a.dash_file + ".tmp", a.dash_file)
        except Exception as e:
            print(f"{now()} dash write failed: {e}", flush=True)

    def do_eval(tag):
        m.eval(); t0 = time.time()
        res = ev.quick(m, R=a.eval_R, n_lang=a.quick_eval_n, n_probe=a.quick_eval_n)
        res.update(step=st["step"], tag=tag, wall_h=elapsed() / 3600, slots=st["step"] * a.accum * a.micro_bs * 1024,
                   eval_sec=time.time() - t0, time=now(), lr_scale=a.lr_scale)
        is_best = res["val_lang_mean"] < st["best"]
        if is_best:
            st["best"] = res["val_lang_mean"]
        res["is_best"] = is_best
        elog.write(json.dumps(res) + "\n"); elog.flush()
        g = res.get("loop", {})
        print(f"{now()} EVAL[{tag}] step {st['step']} val_lang {res['val_lang_mean']:.4f} "
              f"(ts {res['val_tinystories']['nats']:.3f} wk {res['val_wikitext']['nats']:.3f}) probes " +
              " ".join(f"{k[3:]}={v['exact']:.3f}" for k, v in res['probes'].items()) +
              f" swap_recall1={res['recall1_plan_swap']['exact']:.3f} codes_min={res['codebook']['codes_used_frac_min']:.2f}"
              f" VE={res['codebook']['var_explained']:.3f} ncp={res['codebook']['ncp_ce']:.3f} ncp_M1={res.get('ncp_M1', float('nan')):.3f}"
              f" angle23={nz(g.get('angle23')):.2f} rho3={nz(g.get('rho3')):.3f}"
              f" G_share={res['telemetry']['crc_mix_e0_L_G'][2]:.3f} ({res['eval_sec']:.0f}s)", flush=True)
        if dev.type == "mps":
            torch.mps.empty_cache()
        m.train(); eval_secs[0] += time.time() - t0
        write_dash(eval_rec=res)
        return is_best

    def checkpoint(is_best):
        state = dict(model=m.state_dict(), opt=opt.state_dict(), step=st["step"], elapsed=elapsed(), best=st["best"],
                     cursor=stream.cursor, milestone_done=st["milestone_done"], rng=rng_state(),
                     traj_rng=traj_gen.get_state(), lr_trace=list(lr_trace), args=vars(a))
        save_ckpt(a.run_dir, state, st["step"], keep=a.keep, best=is_best)
        np.save(os.path.join(a.run_dir, "lr_trace.npy"), np.asarray(lr_trace, dtype=np.float64))

    def guard_event(kind, **kw):
        rec = dict(kind=kind, step=st["step"], time=now(), wall_h=elapsed() / 3600, **kw)
        glog.write(json.dumps(rec) + "\n"); glog.flush()
        print(f"{now()} GUARD {rec}", flush=True)

    def projection():
        el = elapsed()
        n_tot = int(st["step"] + (a.hours * 3600 - el) * st["step"] / el) if st["step"] >= 50 and el > 0 else a.est_total_steps
        print(f"{now()} start: step {st['step']}/{n_tot} (cosine over {n_tot}) [steps projected at the current speed; "
              f"the schedule itself runs on training time]", flush=True)

    projection()
    print(f"{now()} config: step {st['step']}, budget {a.hours} h, milestone {a.milestone_hours} h, micro_bs {a.micro_bs} x accum "
          f"{a.accum} x 1024, lr {a.lr} (scale {a.lr_scale}), warmup {a.warmup}, shortcut {'off' if a.no_shortcut else SHORT_P}, "
          f"lam ncp/s/c {a.lam_ncp}/{a.lam_s}/{a.lam_c}, eval R {a.eval_R}, compile {a.compile}", flush=True)
    stop_reason = None
    proc_start, mem0 = time.time(), [None]
    while True:
        if os.path.exists(os.path.join(a.run_dir, "PAUSE")):
            stop_reason = "pause"; break
        if (a.max_steps and st["step"] >= a.max_steps) or a.init_only:
            stop_reason = "max_steps"; break
        if elapsed() >= a.hours * 3600:
            stop_reason = "time_budget"; break
        if a.milestone_hours and not st["milestone_done"] and elapsed() >= a.milestone_hours * 3600:
            stop_reason = "milestone"; break
        lr = lr_at(st["step"], elapsed(), a)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        batches = [stream.next() for _ in range(a.accum)]              # whole optimizer batch first (denominators)
        N = batch_counts(batches)
        log_step = (st["step"] + 1) % a.log_every == 0
        sums, shorts = None, []
        for bi, (ids, doc, ei) in enumerate(batches):
            short = None if a.no_shortcut else sample_short(traj_gen)
            shorts.append(short)
            ids, doc, ei = ids.to(dev), doc.to(dev), ei.to(dev)
            out = micro(m, ids, doc, ei, short, a, N, telem=log_step and bi == a.accum - 1)
            vec = torch.stack([out["ntp"], out["ncp"], out["ncp_s"], out["scons"], out["vq"], out["engram_alpha"] / a.accum])
            sums = vec if sums is None else sums + vec
            tok_count += ids.numel()
        last_c, last_complete = out["c_hat"], out["complete"]
        gn_cb = m.core.vocab.e.grad.detach().norm() if m.core.vocab.e.grad is not None else torch.zeros((), device=dev)
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        s = sums.tolist()
        losses = dict(ntp=s[0] / N["t"], ncp=s[1] / N["p"], ncp_s=s[2] / N["p"], scons=s[3] / N["ne"], vq=s[4] / N["c"],
                      engram_alpha=s[5])
        losses["loss"] = losses["ntp"] + a.lam_ncp * losses["ncp"] + a.lam_s * losses["ncp_s"] + a.lam_c * losses["scons"] + losses["vq"]
        bad = not math.isfinite(losses["loss"]) or not torch.isfinite(gn)
        if bad:
            nf = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            print(f"{now()} NONFINITE at step {st['step']}: losses {losses}, gn {gn.item()}, grads {nf[:8]} ({len(nf)})", flush=True)
        else:
            opt.step()
            st["step"] += 1
            lr_trace.append(lr)
            if st["step"] % 100 == 0:
                r = restart_dead_now(m.core.vocab, last_c, last_complete, opt)
                r.update(step=st["step"]); tlog.write(json.dumps(dict(r, kind="restart")) + "\n"); tlog.flush()
            if st["step"] in (100, 200, 300):                          # did the zero-initialised pieces open?
                ow = dict(step=st["step"], Wv1=float(m.enc.engram.Wv.weight.norm()), Wv2=float(m.enc.engram2.Wv.weight.norm()))
                if m.core.table is not None:
                    ow["table"] = float(m.core.table.norm())
                if m.core.shared is not None:
                    ow["shared_w2"] = float(m.core.shared.w2.weight.norm())
                ow["still_zero"] = [k for k, v in ow.items() if k != "step" and v == 0.0]
                tlog.write(json.dumps(dict(ow, kind="opened")) + "\n"); tlog.flush()
                open_log[str(st["step"])] = ow
                print(f"{now()} OPENED {ow}", flush=True)
            for k in KEYS:
                acc[k] += losses[k]
            for sname in shorts:
                if sname:
                    acc["short"][sname] += 1
            acc["n"] += 1; acc["gn"] += gn.item(); acc["gn_cb"] += gn_cb.item()
            if log_step and "states" in out and len(out["states"]) >= 3:
                geo_last = loop_geometry(out["states"])
            ema = losses["ntp"] if ema is None else 0.98 * ema + 0.02 * losses["ntp"]
            if a.heartbeat:
                try:
                    hb = dict(step=st["step"], t=time.time(), time=now(), wall_h=elapsed() / 3600, lr=lr, gn=float(gn), ema=ema, **losses)
                    hp = os.path.join(a.run_dir, "live.json")
                    with open(hp + ".tmp", "w") as f:
                        json.dump(hb, f)
                    os.replace(hp + ".tmp", hp)
                except Exception:
                    pass
        diverged = bad
        if not bad and st["step"] % a.log_every == 0:
            hist.append((time.time(), ema))
            recent = [l for (w, l) in hist if time.time() - w < 3600]
            if len(recent) >= 2 and elapsed() > a.div_min_wall_min * 60 and ema > 3 * min(recent):
                diverged = True
        if diverged:
            last = os.path.join(a.run_dir, "ckpt_last.pt")
            guard_event("divergence", nonfinite=bad, ema=ema, lr_scale_old=a.lr_scale)
            if not os.path.exists(last):
                stop_reason = "divergence_no_ckpt"; break
            stream.close()
            old = a.lr_scale
            load(last); a.lr_scale = old * 0.5
            t_start = time.time() - st["elapsed"]
            stream = ShardStream(a.data_dir, "train", ecfg, a.micro_bs, seed=a.seed, cursor=st["cursor"], prefetch=a.accum + 4)
            hist.clear(); ema = None; acc = new_acc(); tok_count, t_win = 0, time.time()
            guard_event("reloaded", from_step=st["step"], lr_scale_new=a.lr_scale)
            continue
        if st["step"] % a.log_every == 0 and acc["n"]:
            dt = time.time() - t_win - eval_secs[0]; eval_secs[0] = 0.0
            n = acc["n"]
            rec = dict(step=st["step"], time=now(), wall_h=elapsed() / 3600, lr=lr, **{k: acc[k] / n for k in KEYS},
                       gn_preclip=acc["gn"] / n, gn_codebook=acc["gn_cb"] / n, short_counts=acc["short"],
                       tok_s=tok_count / max(dt, 1e-9), epochs=stream.epochs(), ntp_ema=ema, swap_mb=swap_used_mb(),
                       micro_bs=a.micro_bs, accum=a.accum, progress=min(1.0, elapsed() / (a.hours * 3600)),
                       footprint_gib=phys_footprint_gib(), geometry=geo_last, angle23=geo_last.get("angle23"),
                       rho3=geo_last.get("rho3"), table=table_norms(m.core), N=N)
            if dev.type == "mps":
                rec.update(mps_alloc_gib=torch.mps.current_allocated_memory() / 2**30,
                           mps_driver_gib=torch.mps.driver_allocated_memory() / 2**30)
            tlog.write(json.dumps(rec) + "\n"); tlog.flush()
            ang = geo_last.get("angle23")
            print(f"{now()} step {st['step']} loss {rec['loss']:.4f} ntp {rec['ntp']:.4f} ncp {rec['ncp']:.3f} ncp_s {rec['ncp_s']:.3f} "
                  f"scons {rec['scons']:.4f} vq {rec['vq']:.4f} gn {rec['gn_preclip']:.2f} lr {lr:.2e} "
                  f"ang23 {ang if ang is not None else float('nan'):.2f} rho3 {geo_last.get('rho3', float('nan')):.3f} "
                  f"{rec['tok_s']:.0f} tok/s", flush=True)
            write_dash(train_rec=rec)
            acc = new_acc(); tok_count, t_win = 0, time.time()
            if dev.type == "mps" and rec_max and torch.mps.current_allocated_memory() > a.mem_frac_guard * rec_max and a.micro_bs > 1:
                checkpoint(False)
                guard_event("memory", alloc_gib=torch.mps.current_allocated_memory() / 2**30, micro_bs_old=a.micro_bs, accum_old=a.accum)
                a.micro_bs //= 2; a.accum *= 2
                cur = stream.cursor; stream.close()
                stream = ShardStream(a.data_dir, "train", ecfg, a.micro_bs, seed=a.seed, cursor=cur, prefetch=a.accum + 4)
                torch.mps.empty_cache()
        # ---- memory-leak guard (V-011, carried): checkpoint and re-exec the same command with --resume
        if st["step"] % a.log_every == 0 and not bad:
            fp = phys_footprint_gib()
            if mem0[0] is None and fp > 0:
                mem0[0] = fp
            forced = a.debug_restart_step and st["step"] == a.debug_restart_step and not os.environ.get("HRM_REEXEC")
            grown = mem0[0] is not None and fp - mem0[0] > a.mem_restart_growth_gib
            old = time.time() - proc_start > 20 * 60
            if forced or (a.mem_restart_gib and old and grown and fp > a.mem_restart_gib):
                checkpoint(False)
                guard_event("memory_restart", footprint_gib=round(fp, 2), threshold_gib=a.mem_restart_gib)
                stream.close()
                for f_ in (tlog, elog, glog):
                    f_.flush()
                argv, i = list(sys.argv), 0
                while i < len(argv):
                    if argv[i] == "--resume":
                        del argv[i:i + 2]
                    else:
                        i += 1
                argv += ["--resume", os.path.join(a.run_dir, "ckpt_last.pt")]
                os.environ["HRM_REEXEC"] = "1"
                print(f"{now()} RE-EXEC after checkpoint at step {st['step']} (footprint {fp:.1f} GiB)", flush=True)
                sys.stdout.flush(); sys.stderr.flush()
                os.execv(sys.executable, [sys.executable] + argv)
        step_due = a.eval_at_v1_steps and st["step"] in V1_EVAL_STEPS and not bad
        due = step_due or (time.time() - last_eval_wall >= a.eval_every_min * 60)
        if due and st["step"] > 0 and not bad:
            is_best = do_eval("v1step" if step_due else "hourly")
            checkpoint(is_best)
            last_eval_wall = time.time()
            projection()
    stream.close()
    if stop_reason in ("max_steps", "time_budget"):
        checkpoint(False if a.no_final_eval else do_eval("final"))
    elif stop_reason == "milestone":
        st["milestone_done"] = True
        checkpoint(do_eval("milestone"))
    elif stop_reason == "pause":
        checkpoint(False)
    print(f"{now()} STOP ({stop_reason}) at step {st['step']}, {elapsed()/3600:.2f} h", flush=True)
    sys.exit({"max_steps": 0, "time_budget": 0, "pause": 4, "milestone": 5}.get(stop_reason, 3))


if __name__ == "__main__":
    main()
