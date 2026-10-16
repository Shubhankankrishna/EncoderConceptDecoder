"""Train HRM THINKER v2 (19M). One process, one device; resumable; pause-aware; 48 h in ONE schedule (v1 lesson D-041).

  python train.py --run_dir ../runs/main --hours 48 --milestone_hours 24
  python train.py --run_dir ../runs/main --resume ../runs/main/ckpt_last.pt

Per optimizer step: K ~ U{8..12} core passes (same K for every micro-batch of the step), windows of 4 (remainder merged
into the last), carry detached between windows, per-window backward (c_hat and tok are leaves whose gradients accumulate;
one backward into the encoder at the end of the micro-batch).
Loss = NTP + 0.1 * mean_r NCP(u_r) + 0.1 * consistency(final window) + VQ + 0.01 LB + 0.001 z
       + lambda_h * NextLat-h + lambda_KL * NextLat-KL.
LR: warm-up (steps) then cosine to lr_min_frac * lr over the TRAINING-TIME budget (progress = elapsed / hours; paused time
and the out-of-process 24 h evaluation do not count), so the schedule ends exactly at 48 h whatever the throughput.
Milestone (24 h): the trainer saves and exits with code 5; the launcher runs final_eval.py on ckpt_last and resumes.
Exit codes: 0 done, 3 guard stop, 4 paused (code/pause.sh), 5 milestone.
"""
import argparse, json, math, os, random, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F

from model import build_model, forward, nextlat_losses, param_groups, param_table
from concept_core_v2 import ConceptVocab
from data import ShardStream, target_mask, make_batch, Canon
from evaluate import Evaluator


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def lr_at(step, elapsed, a):
    warm = min(1.0, (step + 1) / a.warmup)
    prog = min(1.0, elapsed / (a.hours * 3600))
    return a.lr * a.lr_scale * warm * (a.lr_min_frac + (1 - a.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def windows(K, w=4):
    """Pass ranges (1-based, inclusive) of the detached windows: windows of w, remainder merged into the last."""
    n = max(1, K // w)
    out, s = [], 1
    for i in range(n):
        e = K if i == n - 1 else s + w - 1
        out.append((s, e)); s = e + 1
    return out


@torch.no_grad()
def restart_dead_now(vocab, c):
    x = c.detach().reshape(-1, vocab.S, vocab.ds)
    dead_total, hits_total = 0, float(vocab.hits.sum())
    used = (vocab.hits > 0).float().mean(1)
    for j in range(vocab.S):
        dead = vocab.hits[j] == 0
        nd = int(dead.sum())
        if nd:
            vocab.e[j, dead] = x[torch.randint(len(x), (nd,), device=x.device), j]
            dead_total += nd
    vocab.hits.zero_()
    return dict(restart_dead=dead_total, hits_total=hits_total, codes_used_min_seg=float(used.min()),
                codes_used_mean=float(used.mean()))


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
    """This process's physical memory footprint (macOS proc_pid_rusage ri_phys_footprint, includes compressed pages),
    the number Activity Monitor shows. -1 if unavailable."""
    try:
        import ctypes
        buf = (ctypes.c_uint64 * 32)()                               # rusage_info_v2: uuid (16 B) then uint64 fields
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        if lib.proc_pid_rusage(os.getpid(), 2, ctypes.byref(buf)) != 0:
            return -1.0
        return buf[2 + 7] / 2**30                                    # uuid = 2 words; ri_phys_footprint = 8th field
    except Exception:
        return -1.0


def swap_used_mb():
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    try:
        return float(s.split("used =")[1].split("M")[0])
    except Exception:
        return -1.0


def micro_step(m, ids, doc, ei, K, a, accum, amp):
    """One micro-batch of D6 training with per-window backward. Returns detached statistics (device tensors / floats)."""
    core = m.core
    fdt = core.Wq.weight.dtype                                          # loss dtype: fp32 (fp64 in the exact tests)
    with amp():
        tok, c_hat, eaux = m.enc(ids, doc, ei)
    cdoc = doc[:, ::4]
    mask = core.mask(cdoc)
    vq, targets = core.vocab.vq_loss(c_hat.to(fdt))                     # codebook-only gradient; counts hits
    c_leaf = c_hat.detach().requires_grad_()
    t_leaf = tok.detach().requires_grad_()
    wins = windows(K)
    S = None
    ce_pass, drops, lbs, zs, loads, idx_prev, div = [], [], [], [], [], None, []
    for wi, (p0, p1) in enumerate(wins):
        last = wi == len(wins) - 1
        with amp():
            X = core.W_in(c_leaf)
            if wi == 0:
                S = core.W_s(c_leaf)
        lw, us, xcache = 0.0, [], {}                                    # X~ branch of the gated reads: once per window
        for r in range(p0, p1 + 1):
            with amp():
                S, auxs = core.one_pass(X, S, mask, first=(r == 1), cache=xcache)
                u = core.readout(S)
                lc, cp = core.head(u.to(fdt), pred=(last and r == p1))
            ce = core.ncp_ce(lc.to(fdt), targets, cdoc)
            lb = torch.stack([x["lb"] for x in auxs]).mean(); z = torch.stack([x["z"] for x in auxs]).mean()
            lw = lw + (0.1 * ce + a.lam_lb * lb + a.lam_z * z) / K
            ce_pass.append(ce.detach()); lbs.append(lb.detach()); zs.append(z.detach())
            drops.append(torch.stack([x["dropped"] for x in auxs]).mean().detach())
            loads.append(torch.stack([x["load"] for x in auxs]).mean(0).detach())
            idx = auxs[1]["idx"][:, 0]                                  # layer 2, top-1 expert per chunk position
            if idx_prev is not None:
                div.append((idx != idx_prev).to(fdt).mean().detach())
            idx_prev = idx
            if last:
                us.append(u)
        if not last:
            (lw / accum).backward()
            S = S.detach()
            continue
        u_K = us[-1]
        cons = torch.stack([(u_K.detach().to(fdt) - ur.to(fdt)).pow(2).mean() for ur in us[:-1]]).mean() if len(us) > 1 \
            else torch.zeros((), device=ids.device)
        with amp():
            logits, daux = m.dec(t_leaf, u_K, cp, doc, m.enc.embed.weight)
        tgt = ids[:, 1:].masked_fill(~target_mask(ids, doc), -100)
        ntp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).to(fdt), tgt.reshape(-1), ignore_index=-100)
        if a.lam_h or a.lam_kl:
            with amp():
                l_h, l_kl = nextlat_losses(m, daux["h"], logits, ids, doc)
        else:
            l_h = l_kl = torch.zeros((), device=ids.device)
        final = lw + a.lam_cons * cons + ntp + vq + a.lam_h * l_h + a.lam_kl * l_kl
        (final / accum).backward()
    torch.autograd.backward([tok, c_hat], [t_leaf.grad, c_leaf.grad])      # one backward into the encoder
    ncp = torch.stack(ce_pass).mean()
    lb_m, z_m = torch.stack(lbs).mean(), torch.stack(zs).mean()
    total = ntp.detach() + 0.1 * ncp + a.lam_cons * cons.detach() + vq.detach() + a.lam_lb * lb_m + a.lam_z * z_m \
        + a.lam_h * l_h.detach() + a.lam_kl * l_kl.detach()
    load = torch.stack(loads).mean(0)
    ent = -(load.clamp_min(1e-9) * load.clamp_min(1e-9).log()).sum()
    return dict(loss=total, ntp=ntp.detach(), ncp=ncp, vq=vq.detach(), cons=cons.detach(), lb=lb_m, z=z_m,
                nl_h=l_h.detach(), nl_kl=l_kl.detach(), drop=torch.stack(drops).mean(), load_ent=ent,
                route_div=torch.stack(div).mean() if div else torch.zeros((), device=ids.device),
                ncp_pass=torch.stack(ce_pass), c_hat=c_hat.detach(), engram_alpha=eaux["engram_alpha"].to(fdt).mean().detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "run_m2_25M", "data")))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--micro_bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--lr_scale", type=float, default=1.0)
    ap.add_argument("--lr_min_frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--hours", type=float, default=48.0)          # training-time budget = cosine length
    ap.add_argument("--milestone_hours", type=float, default=24.0)  # full evaluation pause (0 = none)
    ap.add_argument("--est_total_steps", type=int, default=8000)   # dashboard progress before the speed is known
    ap.add_argument("--max_steps", type=int, default=0)           # hard stop (tests / smoke)
    ap.add_argument("--K_min", type=int, default=8)
    ap.add_argument("--K_max", type=int, default=12)
    ap.add_argument("--eval_R", type=int, default=6)
    ap.add_argument("--lam_cons", type=float, default=0.1)
    ap.add_argument("--lam_lb", type=float, default=0.01)
    ap.add_argument("--lam_z", type=float, default=0.001)
    ap.add_argument("--lam_h", type=float, default=1.0)
    ap.add_argument("--lam_kl", type=float, default=1.0)
    ap.add_argument("--amp", default="none")                      # none | bf16 | fp16 (autocast on matmuls; losses fp32)
    ap.add_argument("--eval_every_min", type=float, default=60.0)
    ap.add_argument("--eval_every_steps", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heartbeat", type=int, default=1)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--no_final_eval", action="store_true")
    ap.add_argument("--quick_eval_n", type=int, default=128)
    ap.add_argument("--resume", default="")
    ap.add_argument("--mem_frac_guard", type=float, default=0.90)
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--div_min_wall_min", type=float, default=30.0)
    ap.add_argument("--mem_restart_gib", type=float, default=24.0)   # V-011: footprint above this -> checkpoint + re-exec
    ap.add_argument("--mem_restart_growth_gib", type=float, default=12.0)   # ... and grown this much since this process started
    ap.add_argument("--debug_restart_step", type=int, default=0)     # testing the re-exec path only
    ap.add_argument("--dash_file", default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "run_m2_25M", "webapp", "data", "v2_live.json")))
    a = ap.parse_args()
    os.makedirs(a.run_dir, exist_ok=True)
    dev = torch.device(a.device)
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)

    m, ecfg, dcfg = build_model(small=a.small)
    m = m.to(dev)
    opt = torch.optim.AdamW(param_groups(m, 0.1), lr=a.lr, betas=(0.9, 0.95), eps=1e-8)
    st = dict(step=0, elapsed=0.0, best=float("inf"), cursor=0, milestone_done=False)
    if a.amp == "bf16":
        amp = lambda: torch.autocast(device_type=dev.type, dtype=torch.bfloat16)
    elif a.amp == "fp16":
        amp = lambda: torch.autocast(device_type=dev.type, dtype=torch.float16)
    else:
        import contextlib
        amp = contextlib.nullcontext

    def load(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        for k in ("step", "elapsed", "best", "cursor", "milestone_done"):
            st[k] = ck.get(k, st[k])
        for k, v in ck.get("args", {}).items():                        # the run's settings survive pause / resume
            if k not in ("run_dir", "resume", "device", "dash_file", "data_dir"):
                setattr(a, k, v)
        for k in ("lr_scale", "micro_bs", "accum", "hours", "warmup", "lr", "lr_min_frac", "milestone_hours"):
            if k in ck:
                setattr(a, k, ck[k])
        set_rng_state(ck["rng"])
        return ck

    if a.resume:
        load(a.resume)
        print(f"{now()} resumed from {a.resume}: step {st['step']}, elapsed {st['elapsed']/3600:.2f} h, cursor {st['cursor']}, "
              f"lr_scale {a.lr_scale}, micro_bs {a.micro_bs} x accum {a.accum}, milestone_done {st['milestone_done']}", flush=True)
    json.dump(dict(vars(a), params=param_table(m), time=now()), open(os.path.join(a.run_dir, "config.json"), "w"), indent=1)
    rec_max = torch.mps.recommended_max_memory() if dev.type == "mps" else 0

    # ---- codebook k-means++ init from the first real optimizer batch (before the first training forward), as v1
    if st["step"] == 0 and not a.resume:
        tok_rows = np.load(os.path.join(a.data_dir, "train", "tok.npy"), mmap_mode="r")
        doc_rows = np.load(os.path.join(a.data_dir, "train", "doc.npy"), mmap_mode="r")
        order = np.random.RandomState(a.seed).permutation(tok_rows.shape[0])
        canon = Canon(a.data_dir)
        m.eval(); cs = []
        with torch.no_grad():
            for i in range(a.accum):
                idx = order[i * a.micro_bs:(i + 1) * a.micro_bs]
                ids, doc, ei = make_batch(np.asarray(tok_rows[idx]), np.asarray(doc_rows[idx]), canon, ecfg)
                cs.append(m.enc(ids.to(dev), doc.to(dev), ei.to(dev))[1].float().cpu())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(a.seed + 12345)
            tmp = ConceptVocab(ecfg.dc, m.core.vocab.S, m.core.vocab.N)
            tmp.init_from(torch.cat(cs))
        m.core.vocab.e.data.copy_(tmp.e.data.to(dev))
        print(f"{now()} codebook k-means++ init from {sum(c.shape[0] * c.shape[1] for c in cs)} concepts", flush=True)
        del cs, tmp
    m.train()

    stream = ShardStream(a.data_dir, "train", ecfg, a.micro_bs, seed=a.seed, cursor=st["cursor"])
    ev = Evaluator(a.data_dir, ecfg, dev, batch_rows=4)
    tlog = open(os.path.join(a.run_dir, "train_log.jsonl"), "a")
    elog = open(os.path.join(a.run_dir, "log.jsonl"), "a")
    glog = open(os.path.join(a.run_dir, "GUARDS.jsonl"), "a")
    t_start = time.time() - st["elapsed"]
    last_eval_wall = time.time()
    tok_count, t_win, ema = 0, time.time(), None
    hist, eval_secs, last_c = [], [0.0], None
    KEYS = ("loss", "ntp", "ncp", "vq", "cons", "lb", "z", "nl_h", "nl_kl", "drop", "load_ent", "route_div", "engram_alpha")
    new_acc = lambda: dict({k: 0.0 for k in KEYS}, n=0, gn=0.0, gn_cb=0.0, K=0.0, ncp_pass=None)
    acc = new_acc()
    dash = []                                                      # history for the v2 dashboard tab
    params_other = [p for n, p in m.named_parameters() if n != "core.vocab.e"]

    def elapsed():
        return time.time() - t_start

    def do_eval(tag):
        m.eval(); t0 = time.time()
        res = ev.quick(m, R=a.eval_R, n_lang=a.quick_eval_n, n_probe=a.quick_eval_n)
        res.update(step=st["step"], tag=tag, wall_h=elapsed() / 3600, tokens=st["step"] * a.accum * a.micro_bs * 1024,
                   eval_sec=time.time() - t0, time=now(), lr_scale=a.lr_scale)
        is_best = res["val_lang_mean"] < st["best"]
        if is_best:
            st["best"] = res["val_lang_mean"]
        res["is_best"] = is_best
        elog.write(json.dumps(res) + "\n"); elog.flush()
        print(f"{now()} EVAL[{tag}] step {st['step']} val_lang {res['val_lang_mean']:.4f} "
              f"(ts {res['val_tinystories']['nats']:.3f} wk {res['val_wikitext']['nats']:.3f}) probes " +
              " ".join(f"{k[3:]}={v['exact']:.3f}" for k, v in res['probes'].items()) +
              f" swap_recall1={res['recall1_plan_swap']['exact']:.3f} codes_min={res['codebook']['codes_used_frac_min']:.2f}"
              f" VE={res['codebook']['var_explained']:.3f} ncp={res['codebook']['ncp_ce']:.3f} ({res['eval_sec']:.0f}s)", flush=True)
        if dev.type == "mps":
            torch.mps.empty_cache()
        m.train(); eval_secs[0] += time.time() - t0
        write_dash(eval_rec=res)
        return is_best

    def checkpoint(is_best):
        state = dict(model=m.state_dict(), opt=opt.state_dict(), step=st["step"], elapsed=elapsed(), best=st["best"],
                     cursor=stream.cursor, milestone_done=st["milestone_done"], lr_scale=a.lr_scale, micro_bs=a.micro_bs,
                     accum=a.accum, hours=a.hours, warmup=a.warmup, lr=a.lr, lr_min_frac=a.lr_min_frac,
                     milestone_hours=a.milestone_hours, rng=rng_state(), args=vars(a))
        save_ckpt(a.run_dir, state, st["step"], keep=a.keep, best=is_best)

    def guard_event(kind, **kw):
        rec = dict(kind=kind, step=st["step"], time=now(), wall_h=elapsed() / 3600, **kw)
        glog.write(json.dumps(rec) + "\n"); glog.flush()
        print(f"{now()} GUARD {rec}", flush=True)

    evals_dash = []

    def write_dash(train_rec=None, eval_rec=None):
        if not a.dash_file:
            return
        try:
            if train_rec:
                dash.append(train_rec)
                del dash[:-2000]
            if eval_rec:
                evals_dash.append(dict(step=eval_rec["step"], val=eval_rec["val_lang_mean"],
                                       recall=eval_rec["probes"].get("lr_recall1", {}).get("exact"),
                                       carry=eval_rec["probes"].get("lr_carry3", {}).get("exact"),
                                       multi=eval_rec["probes"].get("lr_multi4", {}).get("exact"),
                                       swap=eval_rec["recall1_plan_swap"]["exact"]))
            last = dash[-1] if dash else {}
            H = {k: [[r["step"], r[k]] for r in dash if r.get(k) is not None] for k in
                 ("loss", "ntp", "ncp", "cons", "vq", "nl_h", "nl_kl", "lb", "load_ent", "drop", "route_div", "tok_s")}
            H["total"] = H.pop("loss"); H["toks"] = H.pop("tok_s"); H["route_ent"] = H.pop("load_ent")
            for p_ in (1, 4, 6, 8):                                     # per-pass NCP series (should fall with the pass)
                H[f"ncp_p{p_}"] = [[r["step"], r["ncp_pass"][p_ - 1]] for r in dash if len(r.get("ncp_pass") or []) >= p_]
            H["ncp_pass_last"] = dash[-1].get("ncp_pass") if dash else None
            H["val"] = [[e["step"], e["val"]] for e in evals_dash]
            H["recall"] = [[e["step"], e["recall"]] for e in evals_dash if e["recall"] is not None]
            out = dict(phase=getattr(a, "phase", "main"), step=st["step"], time=now(), t=time.time(),
                       wall_h=elapsed() / 3600, hours=a.hours, lr=last.get("lr"), gn=last.get("gn_preclip"),
                       tok_s=last.get("tok_s"), loss=last.get("loss"), ntp=last.get("ntp"), history=H,
                       evals=evals_dash[-50:])
            with open(a.dash_file + ".tmp", "w") as f:
                json.dump(out, f)
            os.replace(a.dash_file + ".tmp", a.dash_file)
        except Exception as e:  # never stop training for the dashboard
            print(f"{now()} dash write failed: {e}", flush=True)

    def projection():                                              # "start: step S/N (cosine over N)" for the dashboard
        el = elapsed()
        if st["step"] >= 50 and el > 0:
            n_tot = int(st["step"] + (a.hours * 3600 - el) * st["step"] / el)
        else:
            n_tot = a.est_total_steps
        print(f"{now()} start: step {st['step']}/{n_tot} (cosine over {n_tot}) [steps projected at the current speed; "
              f"the schedule itself runs on training time]", flush=True)

    projection()
    print(f"{now()} config: step {st['step']}, budget {a.hours} h (cosine over training time), milestone {a.milestone_hours} h, "
          f"micro_bs {a.micro_bs} x accum {a.accum} x 1024, lr {a.lr} (scale {a.lr_scale}), warmup {a.warmup}, "
          f"K ~ U{{{a.K_min}..{a.K_max}}}, eval R {a.eval_R}, amp {a.amp}, NextLat {a.lam_h}/{a.lam_kl}", flush=True)
    stop_reason = None
    proc_start, mem0 = time.time(), [None]                           # V-011 memory guard: uptime and baseline footprint
    while True:
        if os.path.exists(os.path.join(a.run_dir, "PAUSE")):
            stop_reason = "pause"; break
        if a.max_steps and st["step"] >= a.max_steps:
            stop_reason = "max_steps"; break
        if elapsed() >= a.hours * 3600:
            stop_reason = "time_budget"; break
        if a.milestone_hours and not st["milestone_done"] and elapsed() >= a.milestone_hours * 3600:
            stop_reason = "milestone"; break
        K = random.randint(a.K_min, a.K_max)
        lr = lr_at(st["step"], elapsed(), a)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        sums = None
        for _ in range(a.accum):
            ids, doc, ei = stream.next()
            ids, doc, ei = ids.to(dev), doc.to(dev), ei.to(dev)
            out = micro_step(m, ids, doc, ei, K, a, a.accum, amp)
            last_c = out.pop("c_hat")
            vec = torch.stack([out[k].float() for k in KEYS])
            sums = vec / a.accum if sums is None else sums + vec / a.accum
            npass = out["ncp_pass"] / a.accum
            acc_np = npass if acc.get("_np_step") is None else acc["_np_step"] + npass
            acc["_np_step"] = acc_np
            tok_count += ids.numel()
        gn_cb = m.core.vocab.e.grad.detach().norm() if m.core.vocab.e.grad is not None else torch.zeros((), device=dev)
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        vals = sums.tolist()
        losses = dict(zip(KEYS, vals))
        np_step = acc.pop("_np_step").tolist()
        bad = not math.isfinite(losses["loss"]) or not torch.isfinite(gn)
        if bad:
            nf = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            print(f"{now()} NONFINITE at step {st['step']}: losses {losses}, gn {gn.item()}, K {K}, grads {nf[:8]} ({len(nf)})", flush=True)
        else:
            opt.step()
            st["step"] += 1
            if st["step"] % 100 == 0:
                r = restart_dead_now(m.core.vocab, last_c)
                r.update(step=st["step"]); tlog.write(json.dumps(dict(r, kind="restart")) + "\n"); tlog.flush()
            for k in KEYS:
                acc[k] += losses[k]
            acc["n"] += 1; acc["gn"] += gn.item(); acc["gn_cb"] += gn_cb.item(); acc["K"] += K
            acc["ncp_pass"] = np_step                                   # last step's per-pass NCP (K entries)
            ema = losses["ntp"] if ema is None else 0.98 * ema + 0.02 * losses["ntp"]
            if a.heartbeat:
                try:
                    hb = dict(step=st["step"], t=time.time(), time=now(), wall_h=elapsed() / 3600, lr=lr, R=K, K=K,
                              gn=float(gn), ema=ema, **losses)
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
            stream = ShardStream(a.data_dir, "train", ecfg, a.micro_bs, seed=a.seed, cursor=st["cursor"])
            hist.clear(); ema = None; acc = new_acc(); tok_count, t_win = 0, time.time()
            guard_event("reloaded", from_step=st["step"], lr_scale_new=a.lr_scale)
            continue
        if st["step"] % a.log_every == 0 and acc["n"]:
            dt = time.time() - t_win - eval_secs[0]; eval_secs[0] = 0.0
            n = acc["n"]
            rec = dict(step=st["step"], time=now(), wall_h=elapsed() / 3600, lr=lr, R_last=K, K_mean=acc["K"] / n,
                       **{k: acc[k] / n for k in KEYS}, gn_preclip=acc["gn"] / n, gn_codebook=acc["gn_cb"] / n,
                       ncp_pass=[round(x, 4) for x in acc["ncp_pass"]], tok_s=tok_count / max(dt, 1e-9),
                       epochs=stream.epochs(), ntp_ema=ema, swap_mb=swap_used_mb(), micro_bs=a.micro_bs, accum=a.accum,
                       progress=min(1.0, elapsed() / (a.hours * 3600)), footprint_gib=phys_footprint_gib())
            if dev.type == "mps":
                rec.update(mps_alloc_gib=torch.mps.current_allocated_memory() / 2**30,
                           mps_driver_gib=torch.mps.driver_allocated_memory() / 2**30)
            tlog.write(json.dumps(rec) + "\n"); tlog.flush()
            print(f"{now()} step {st['step']} loss {rec['loss']:.4f} ntp {rec['ntp']:.4f} ncp {rec['ncp']:.3f} vq {rec['vq']:.4f} "
                  f"cons {rec['cons']:.4f} nl_h {rec['nl_h']:.4f} nl_kl {rec['nl_kl']:.3f} lb {rec['lb']:.3f} drop {rec['drop']:.3f} "
                  f"div {rec['route_div']:.2f} gn {rec['gn_preclip']:.2f} lr {lr:.2e} {rec['tok_s']:.0f} tok/s", flush=True)
            write_dash(train_rec=rec)
            acc = new_acc(); tok_count, t_win = 0, time.time()
            if dev.type == "mps" and rec_max and torch.mps.current_allocated_memory() > a.mem_frac_guard * rec_max and a.micro_bs > 1:
                checkpoint(False)
                guard_event("memory", alloc_gib=torch.mps.current_allocated_memory() / 2**30, micro_bs_old=a.micro_bs, accum_old=a.accum)
                a.micro_bs //= 2; a.accum *= 2
                cur = stream.cursor; stream.close()
                stream = ShardStream(a.data_dir, "train", ecfg, a.micro_bs, seed=a.seed, cursor=cur)
                torch.mps.empty_cache()
        # ---- memory-leak guard (V-011): native allocations grow ~1-2 GiB/h under MPS; a fresh process frees them.
        # Checkpoint and re-exec the same command with --resume (same pid for the launcher, exact state, no crash count).
        if st["step"] % a.log_every == 0 and not bad:
            fp = phys_footprint_gib()
            if mem0[0] is None and fp > 0:
                mem0[0] = fp                                          # this process's baseline (first log step)
            forced = a.debug_restart_step and st["step"] == a.debug_restart_step and not os.environ.get("HRM_REEXEC")
            grown = mem0[0] is not None and fp - mem0[0] > a.mem_restart_growth_gib
            old = time.time() - proc_start > 20 * 60                  # never re-exec a young process (no loop)
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
        due = (st["step"] % a.eval_every_steps == 0) if a.eval_every_steps else (time.time() - last_eval_wall >= a.eval_every_min * 60)
        if due and st["step"] > 0:
            is_best = do_eval("hourly")
            checkpoint(is_best)
            last_eval_wall = time.time()
            projection()
    stream.close()
    if stop_reason in ("max_steps", "time_budget"):
        if a.no_final_eval:
            checkpoint(False)
        else:
            checkpoint(do_eval("final"))
    elif stop_reason == "milestone":
        st["milestone_done"] = True
        checkpoint(do_eval("milestone"))
    elif stop_reason == "pause":
        checkpoint(False)
    print(f"{now()} STOP ({stop_reason}) at step {st['step']}, {elapsed()/3600:.2f} h", flush=True)
    sys.exit({"max_steps": 0, "time_budget": 0, "pause": 4, "milestone": 5}.get(stop_reason, 3))


if __name__ == "__main__":
    main()
