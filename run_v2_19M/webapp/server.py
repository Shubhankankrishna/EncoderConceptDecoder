"""HRM THINKER-25M dashboard server (read-only, stdlib only; D-036).

    nohup nice -n 19 /opt/anaconda3/bin/python3 webapp/server.py > webapp/server.log 2>&1 &
    open http://127.0.0.1:8765

Never writes under runs/, never imports torch, never touches the GPU. Log files are read incrementally
(byte offset per file) and the assembled JSON is rebuilt only when a watched file changed.
"""
import json, os, re, subprocess, threading, time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("HRM_RUNS") or os.path.normpath(os.path.join(ROOT, "..", "runs"))   # HRM_RUNS: tests only
MAIN = os.path.join(RUNS, os.environ.get("HRM_MAIN", "main"))     # HRM_MAIN: the v2 dashboard instance uses v2_48h
CONT = os.path.join(RUNS, "cont")                              # continuation (D-041): active once init_from_main.pt exists
PORT = int(os.environ.get("PORT", 8765))
TOTAL_BUDGET_H = 24.0


def budget_h():
    """v2: the training-time budget from the run's config.json (48 h), else 24 h."""
    try:
        return float(json.load(open(os.path.join(MAIN, "config.json"))).get("hours") or TOTAL_BUDGET_H)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return TOTAL_BUDGET_H


class Tail:
    """Incremental JSONL / text reader: keeps a byte offset and the unterminated trailing fragment."""
    def __init__(self, path, jsonl=True, keep=None):
        self.path, self.jsonl, self.keep = path, jsonl, keep
        self.off, self.frag, self.rows, self.sig = 0, b"", [], None

    def update(self):
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return False
        sig = (st.st_size, st.st_mtime_ns, st.st_ino)
        if sig == self.sig:
            return False
        if self.sig and (st.st_ino != self.sig[2] or st.st_size < self.off):   # replaced / truncated
            self.off, self.frag, self.rows = 0, b"", []
        self.sig = sig
        with open(self.path, "rb") as f:
            f.seek(self.off)
            data = f.read()
        self.off += len(data)
        data = self.frag + data
        lines = data.split(b"\n")
        self.frag = lines.pop()                                   # partial last line (or b"")
        for ln in lines:
            s = ln.decode("utf-8", "replace").strip()
            if not s:
                continue
            if self.jsonl:
                try:
                    self.rows.append(json.loads(s))
                except json.JSONDecodeError:
                    pass
            else:
                self.rows.append(s)
        if self.keep and len(self.rows) > self.keep:
            self.rows = self.rows[-self.keep:]
        return True


tails = {
    "train": Tail(os.path.join(MAIN, "train_log.jsonl")),
    "evals": Tail(os.path.join(MAIN, "log.jsonl")),
    "guards": Tail(os.path.join(MAIN, "GUARDS.jsonl")),
    "stdout": Tail(os.path.join(MAIN, "stdout.log"), jsonl=False, keep=400),
    "chain": Tail(os.path.join(RUNS, "chain.log"), jsonl=False, keep=200),
    "launcher": Tail(os.path.join(MAIN, "launcher.log"), jsonl=False, keep=100),
}
cont_tails = {
    "train": Tail(os.path.join(CONT, "train_log.jsonl")),
    "evals": Tail(os.path.join(CONT, "log.jsonl")),
    "guards": Tail(os.path.join(CONT, "GUARDS.jsonl")),
    "stdout": Tail(os.path.join(CONT, "stdout.log"), jsonl=False, keep=400),
    "launcher": Tail(os.path.join(CONT, "launcher.log"), jsonl=False, keep=100),
}


def cont_active():
    return os.path.exists(os.path.join(CONT, "init_from_main.pt"))


def active_dir():
    return CONT if cont_active() else MAIN


def cont_sched():
    """Schedule of the continuation from its config.json (step_offset = global step of its step 0)."""
    try:
        c = json.load(open(os.path.join(CONT, "config.json")))
        return dict(offset=int(c.get("step_offset") or 4050), lr=c.get("lr"), warmup=c.get("warmup"), total=c.get("total_steps"),
                    warmup_start_frac=c.get("warmup_start_frac"), lr_min_frac=c.get("lr_min_frac"))
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return dict(offset=4050)


def update_all():
    ch = [k for k, t in tails.items() if t.update()]
    ch += ["c_" + k for k, t in cont_tails.items() if t.update()]
    return ch


def shifted(rows, off):
    """Continuation rows on the global step axis (train rows carry gstep; restart and eval rows are shifted here)."""
    out = []
    for r in rows:
        r = dict(r)
        if isinstance(r.get("step"), int):
            r["gstep"] = r.get("gstep", r["step"] + off); r["local_step"] = r["step"]; r["step"] = r["gstep"]; r["run"] = "cont"
        if r.get("gtokens"):
            r["tokens"] = r["gtokens"]
        out.append(r)
    return out


def shift_line_steps(lines, off):
    return [re.sub(r"(step |'step': )(\d+)", lambda m: m.group(1) + str(int(m.group(2)) + off), l) for l in lines]


screen_tails = {s: Tail(os.path.join(RUNS, "screens", s, "log.jsonl")) for s in ("S1", "S2", "S3")}
screen_train = {s: Tail(os.path.join(RUNS, "screens", s, "train_log.jsonl")) for s in ("S1", "S2", "S3")}

_lock = threading.Lock()
_cache = {"body": b"{}", "built": 0.0}
_sys = {"t": 0.0, "val": {}}


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return ""


def system_state():
    """Cheap OS probes, at most once every 20 s."""
    if time.time() - _sys["t"] < 20:
        return _sys["val"]
    batt = sh(["pmset", "-g", "batt"])
    m = re.search(r"(\d+)%;\s*([a-zA-Z ]+);", batt)
    swap = sh(["sysctl", "-n", "vm.swapusage"])
    sm = re.search(r"used = ([\d.]+)M", swap)
    A = active_dir()
    procs = sh(["pgrep", "-fl", "run_dir ../runs/" + os.path.basename(A)])
    guard = sh(["pgrep", "-f", "battery_guard2?.sh"]).strip() != ""
    evaluating = sh(["pgrep", "-f", "final_eval.p[y]"]).strip() != ""          # v2: 24 h / final evaluation running
    ckpts = []
    try:
        for f in sorted(os.listdir(A)):
            if (f.startswith("ckpt") or f == "init_from_main.pt") and f.endswith(".pt"):
                st = os.stat(os.path.join(A, f))
                ckpts.append(dict(name=f, inode=st.st_ino, links=st.st_nlink, size_gb=round(st.st_size / 2**30, 3),
                                  mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))))
    except FileNotFoundError:
        pass
    du = os.statvfs(RUNS)
    _sys["val"] = dict(
        battery_pct=int(m.group(1)) if m else None, battery_state=m.group(2).strip() if m else None,
        on_ac="AC Power" in batt, swap_used_mb=float(sm.group(1)) if sm else None,
        trainer_running=bool(procs.strip()), battery_guard=guard, pause_flag=os.path.exists(os.path.join(A, "PAUSE")),
        evaluating=evaluating, milestone_pending=os.path.exists(os.path.join(A, "MILESTONE_EVAL_PENDING")),
        launcher_alive=sh(["pgrep", "-f", "code/run_job.s[h]"]).strip() != "",
        active_run=os.path.basename(A), chain4_alive=bool(sh(["pgrep", "-f", "bash code/chain4.sh"]).strip()),
        checkpoints=ckpts, disk_free_gb=round(du.f_bavail * du.f_frsize / 2**30, 1), probed=time.strftime("%H:%M:%S"))
    _sys["t"] = time.time()
    return _sys["val"]


STEPS_FILE = os.environ.get("HRM_STEPS_FILE") or os.path.join(ROOT, "data", "live_steps.jsonl")   # server-owned heartbeat copy
_live = {"hist": [], "steps": [], "last_hb_step": None, "cur": {}, "batt": (0.0, None)}


def _load_steps():
    try:
        with open(STEPS_FILE) as f:
            _live["steps"] = [json.loads(l) for l in f if l.strip()][-6000:]
        if _live["steps"]:
            _live["last_hb_step"] = _live["steps"][-1]["step"]
    except FileNotFoundError:
        pass


def _sig(path):
    try:
        st = os.stat(path); return [st.st_size, st.st_mtime]
    except FileNotFoundError:
        return None


def sample_live():
    """Called every 5 s by the sampler thread: GPU, trainer process, swap, battery (30 s), signatures, heartbeat."""
    now = time.time()
    io = sh(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"])
    g = lambda k: (lambda m: int(m.group(1)) if m else None)(re.search(r'"%s"=(\d+)' % re.escape(k), io))
    gpu = dict(device=g("Device Utilization %"), renderer=g("Renderer Utilization %"),
               mem_gb=round(g("In use system memory") / 2**30, 2) if g("In use system memory") else None)
    A = active_dir()
    pid = sh(["pgrep", "-f", "run_dir ../runs/" + os.path.basename(A)]).split()
    proc = None
    if pid:
        ps = sh(["ps", "-o", "%cpu=,rss=,etime=", "-p", pid[0]]).split()
        if len(ps) >= 3:
            proc = dict(pid=int(pid[0]), cpu=float(ps[0]), rss_gb=round(int(ps[1]) / 2**20, 2), etime=ps[2])
    sm = re.search(r"used = ([\d.]+)M", sh(["sysctl", "-n", "vm.swapusage"]))
    if now - _live["batt"][0] > 30:
        b = sh(["pmset", "-g", "batt"]); m = re.search(r"(\d+)%;\s*([a-zA-Z ]+);", b)
        _live["batt"] = (now, dict(pct=int(m.group(1)) if m else None, state=m.group(2).strip() if m else None, ac="AC Power" in b))
    hb = None
    try:
        with open(os.path.join(A, "live.json")) as f:       # per-step heartbeat written by train.py (D-037)
            hb = json.load(f)
        if A == CONT and hb.get("step") is not None:
            hb = dict(hb, local_step=hb["step"], step=hb["step"] + cont_sched()["offset"], run="cont")
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    if hb and hb.get("step") is not None and hb["step"] != _live["last_hb_step"]:
        _live["last_hb_step"] = hb["step"]
        _live["steps"].append(hb); _live["steps"] = _live["steps"][-6000:]
        try:
            os.makedirs(os.path.dirname(STEPS_FILE), exist_ok=True)
            with open(STEPS_FILE, "a") as f:
                f.write(json.dumps(hb) + "\n")
        except OSError:
            pass
    _live["hist"].append([round(now), gpu["device"], proc["cpu"] if proc else None])
    _live["hist"] = _live["hist"][-720:]                        # 1 hour at 5 s
    _live["cur"] = dict(t=now, gpu=gpu, proc=proc, swap_gb=round(float(sm.group(1)) / 1024, 2) if sm else None,
                        battery=_live["batt"][1], pause_flag=os.path.exists(os.path.join(A, "PAUSE")), heartbeat=hb,
                        active_run=os.path.basename(A),
                        sig={**{k: _sig(t.path) for k, t in tails.items() if k in ("train", "evals", "stdout")},
                             **{"c_" + k: _sig(t.path) for k, t in cont_tails.items() if k in ("train", "evals", "stdout")},
                             "cont": _sig(os.path.join(CONT, "init_from_main.pt"))})


def sampler():
    while True:
        try:
            sample_live()
        except Exception as ex:                                  # never let the sampler die
            print("sampler error:", ex, flush=True)
        time.sleep(5)


def live_body():
    if cont_active():
        off = cont_sched()["offset"]
        tr = [r for r in shifted(cont_tails["train"].rows, off) if "loss" in r]
        so = cont_tails["stdout"].rows
    else:
        tr = [r for r in tails["train"].rows if "loss" in r]
        so = tails["stdout"].rows
    out = dict(_live["cur"], hist=_live["hist"], steps=_live["steps"][-400:], server_time=time.time())
    if tr:
        out["last_row"] = {k: tr[-1].get(k) for k in ("step", "time", "wall_h", "tok_s", "lr")}
        ds = [(b["wall_h"] - a["wall_h"]) * 3600 / (b["step"] - a["step"])
              for a, b in zip(tr[-7:-1], tr[-6:]) if b["step"] > a["step"] and b["wall_h"] > a["wall_h"]]
        if ds:
            ds.sort(); out["sec_per_step"] = ds[len(ds) // 2]
    ev_t = None
    for l in reversed(so):
        if "EVAL[hourly]" in l or "resumed from" in l or "start: step" in l:
            ev_t = l[:19]; break
    out["last_eval_anchor"] = ev_t
    out["last_line"] = so[-1] if so else ""
    return json.dumps(out, separators=(",", ":")).encode()


def slim_eval(e):
    cb = e.get("codebook", {})
    tel = e.get("telemetry", {})
    src = e.get("probes_256") or e.get("probes", {})          # screens' final evals also scored 256 items per probe
    pr = {k: {kk: vv for kk, vv in v.items() if kk != "n"} for k, v in src.items()}
    return dict(tag=e.get("tag"), step=e.get("step"), time=e.get("time"), wall_h=e.get("wall_h"), tokens=e.get("tokens"),
                val_mean=e.get("val_lang_mean"), val_ts=e.get("val_tinystories", {}).get("nats"),
                val_wk=e.get("val_wikitext", {}).get("nats"), bpb_ts=e.get("val_tinystories", {}).get("bpb"),
                bpb_wk=e.get("val_wikitext", {}).get("bpb"), probes=pr,
                swap_recall1=e.get("recall1_plan_swap", {}).get("exact"), codebook=cb,
                crc_mix=tel.get("crc_mix_e0_L_G"), engram_alpha=tel.get("engram_alpha_mean"),
                engram_wv_norm=tel.get("engram_Wv_norm"), tau=tel.get("tau"), eval_sec=e.get("eval_sec"),
                is_best=e.get("is_best"), mps_driver_gib=tel.get("mps_driver_gib"), run=e.get("run", "main"))


def run_status(train, stdout, sysv, off=0):
    last = stdout[-1] if stdout else ""
    stops = [l for l in stdout if " STOP (" in l]
    last_stop = stops[-1] if stops else None
    if sysv.get("trainer_running"):
        state = "running"
        if last.split(" ", 2)[-1].startswith("resumed") or ("start:" in last and "[steps projected" not in last):
            state = "starting"
    elif sysv.get("evaluating"):
        state = "evaluating"
    elif sysv.get("pause_flag") or (last_stop and "STOP (pause)" in last_stop):
        state = "paused"
    elif last_stop and ("max_steps" in last_stop or "time_budget" in last_stop):
        state = "finished"
    else:
        state = "stopped"
    total = None
    for l in reversed(stdout):
        m = re.search(r"start: step (\d+)/(\d+) \(cosine over (\d+)\)", l)
        if m:
            total = int(m.group(3)) + off; break
    rows = [r for r in train if "loss" in r]
    out = dict(state=state, last_line=last, total_steps=total, last_stop=last_stop, active_run="cont" if off else "main")
    if rows:
        r = rows[-1]
        out.update(step=r["step"], wall_h=r["wall_h"], time=r["time"], lr=r["lr"], tok_s=r["tok_s"],
                   tokens=r["step"] * r["micro_bs"] * r["accum"] * 1024, micro_bs=r["micro_bs"], accum=r["accum"])
        if off:
            out["schedule"] = cont_sched()
        # seconds per step from consecutive rows within one launch (wall_h is training time, pauses excluded)
        ds = [(b["wall_h"] - a["wall_h"]) * 3600 / (b["step"] - a["step"])
              for a, b in zip(rows[-7:-1], rows[-6:]) if b["step"] > a["step"] and b["wall_h"] > a["wall_h"]]
        if ds and total:
            ds.sort(); sps = ds[len(ds) // 2]
            rem = max(total - r["step"], 0) * sps
            out.update(sec_per_step=sps, eta_h=rem / 3600,
                       budget_left_h=budget_h() - r["wall_h"],
                       finish_by_budget=r["wall_h"] + rem / 3600 > budget_h())
        if r.get("progress") is not None:                        # v2: the schedule runs on training time
            out.update(progress=r["progress"], hours=budget_h(), time_left_h=max(budget_h() - r["wall_h"], 0.0))
    return out


def events(lines):
    """Resume / pause / guard / schedule events from the trainer output, for chart markers."""
    out = []
    for l in lines:
        m = re.search(r"^(\S+ \S+) resumed from .*?: step (\d+)", l)
        if m:
            out.append(dict(time=m.group(1), step=int(m.group(2)), kind="resume")); continue
        m = re.search(r"^(\S+ \S+) STOP \((\w+)\) at step (\d+)", l)
        if m:
            out.append(dict(time=m.group(1), step=int(m.group(3)), kind="stop:" + m.group(2))); continue
        if " GUARD " in l:
            m = re.search(r"'step': (\d+)", l)
            out.append(dict(time=l[:19], step=int(m.group(1)) if m else None, kind="guard"))
    return out


def screens():
    res = {}
    for s, t in screen_tails.items():
        t.update(); screen_train[s].update()
        fin = [e for e in t.rows if e.get("tag") == "final"]
        if fin:
            e = slim_eval(fin[-1])
            rs = [r for r in screen_train[s].rows if "restart_dead" in r]
            e["codes_min_seg_restart"] = rs[-1]["codes_used_min_seg"] if rs else None
            e["curve"] = [dict(step=r["step"], ntp=r["ntp"], ncp=r["ncp"], vq=r["vq"]) for r in screen_train[s].rows if "loss" in r]
            res[s] = e
    return res


CONT_PLAN = dict(steps=4050, lr=6e-4, warmup=300, lr_start=6e-5, lr_end=6e-5, hours=24)   # code/chain4.sh (D-041)


def plan_state(sysv, status):
    """v2 four stages from runs/chain.log: train 0 -> 24 h · 24 h evaluation · train 24 -> 48 h · final evaluation."""
    chain = tails["chain"].rows
    lastn = lambda pat: max([i for i, l in enumerate(chain) if pat in l] or [-1])
    st = status.get("state", "stopped")
    ms, ev_s, ev_d = lastn("v2 milestone"), lastn("v2 24 h evaluation start"), lastn("v2 24 h evaluation done")
    tdone, done = lastn("v2 training done"), lastn("v2 done")
    s = ["waiting"] * 4
    if ms < 0:
        s[0] = st
    else:
        s[0] = "done"
        if ev_d > ev_s:
            s[1] = "done"
        elif sysv.get("evaluating") or sysv.get("milestone_pending"):
            s[1] = "evaluating" if sysv.get("evaluating") else "pending"
        if s[1] == "done":
            s[2] = "done" if tdone >= 0 else st
            if tdone >= 0:
                s[3] = "done" if done > tdone else ("evaluating" if sysv.get("evaluating") else "pending")
    return dict(v2=True, stages=s, hours=budget_h(), milestone_h=budget_h() / 2, wall_h=status.get("wall_h") or 0.0,
                stage=1 + next((i for i, x in enumerate(s) if x != "done"), 3))


def build():
    changed = update_all()
    sysv = system_state()
    if not changed and _cache["built"] and time.time() - _cache["built"] < 20:
        return _cache["body"]
    tr, evs, gds = tails["train"].rows, tails["evals"].rows, tails["guards"].rows
    so, la, off = tails["stdout"].rows, tails["launcher"].rows, 0
    cont_ev = []
    if cont_active():                                          # one timeline: main, then the continuation at step + offset
        off = cont_sched()["offset"]
        ctr = shifted(cont_tails["train"].rows, off)
        tr = tr + ctr
        evs = evs + shifted(cont_tails["evals"].rows, off)
        gds = gds + shifted(cont_tails["guards"].rows, off)
        so, la = shift_line_steps(cont_tails["stdout"].rows, off), cont_tails["launcher"].rows
        cont_ev = [dict(time=(cont_tails["stdout"].rows[:1] or [""])[0][:19], step=off, kind="cont")]
        status = run_status(ctr, cont_tails["stdout"].rows, sysv, off=off)
    else:
        status = run_status(tr, so, sysv)
    body = dict(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        status=status,
        plan=plan_state(sysv, status),
        system=sysv,
        train=[{k: r.get(k) for k in ("step", "time", "wall_h", "lr", "R_last", "loss", "ntp", "ntp_ema", "ncp", "vq",
                                      "gn_preclip", "gn_codebook", "gn_without_codebook", "tok_s", "epochs", "swap_mb",
                                      "mps_alloc_gib", "mps_driver_gib", "micro_bs", "accum",
                                      "cons", "lb", "z", "nl_h", "nl_kl", "drop", "load_ent", "route_div", "K_mean",
                                      "ncp_pass", "engram_alpha", "progress", "footprint_gib")} for r in tr if "loss" in r],   # + v2 signals
        restarts=[{k: r.get(k) for k in ("step", "restart_dead", "codes_used_min_seg")} for r in tr if "restart_dead" in r],
        evals=[slim_eval(e) for e in evs],
        guards=gds[-50:],
        stdout=so[-40:],
        events=events(tails["stdout"].rows) + cont_ev + (events(so) if off else []),
        chain=tails["chain"].rows[-40:],
        launcher=la[-20:],
        screens=screens(),
        steps=_live["steps"][-6000:],
    )
    _cache["body"] = json.dumps(body, separators=(",", ":")).encode()
    _cache["built"] = time.time()
    return _cache["body"]


def final_body():
    """Final evaluations (D-044): full eval_final.json of the main run and, once it exists, of the continuation."""
    def jload(p):
        try:
            return json.load(open(p))
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    def tload(p):
        try:
            return open(p).read()
        except FileNotFoundError:
            return None
    out = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    for run, d in (("main", MAIN), ("cont", CONT)):
        e = jload(os.path.join(d, "eval_final.json"))            # final_eval writes the file part by part:
        out[run] = e if e and all(k in e for k in ("lang", "trunc", "items", "probes", "health")) else None   # only complete ones
        out[run + "_p5"] = tload(os.path.join(d, "p5_trained.txt"))
        out[run + "_best"] = jload(os.path.join(d, "eval_best.json"))
    out["screens_p5"] = {s_: tload(os.path.join(RUNS, "screens", s_, "p5_trained.txt")) for s_ in ("S1", "S2", "S3")}
    return json.dumps(out, separators=(",", ":")).encode()


# ------------------------------------------------------------------ Analysis tab (D-045): the page asks Claude (local CLI)
CLAUDE_BIN = os.path.expanduser("~/.local/bin/claude")
ANALYSIS_FILE = os.path.join(ROOT, "data", "analysis.json")
_an = {"running": False, "started": None, "error": None}


def _f(v, d=3):
    return "—" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def build_digest():
    """Plain-text digest of every page's data (architecture, screens, main run, final eval, gates, continuation)."""
    tel = json.loads(build())
    fin = json.loads(final_body())
    L = []
    try:
        A = json.load(open(os.path.join(ROOT, "data", "arch.json"))); T = A["table"]
        L.append(f"ARCHITECTURE: HRM Thinker, encoder (L local stream reach {A['reach_enc']} tokens, G global stream, 6 sliding-window + 2 full-attention "
                 f"layers, Engram hashed n-gram memory) -> concept module (every 4 tokens pooled into one concept; one shared attention layer looped R times, "
                 f"R~U(1..4) in training, R=4 at eval; predicts the next concept's product-quantised codes, 48x128 codebook) -> confined decoder "
                 f"(reach {A['reach_dec']} tokens, reads only its own plan). Compute params {T['compute_params']:,}, stored {T['stored_total']:,} "
                 f"(Engram table {T['engram_table']:,}). d=384, vocab 8,192, context 1,024. Loss = next-token + 0.1 x next-concept + VQ.")
    except Exception:
        pass
    L.append("HARDWARE/DATA: Apple M2 MacBook Air 16 GB (MPS, float32), 32,768 tokens per step, data mix: language 45% (TinyStories + WikiText-103), "
             "symbolic 22%, retrieval 13%, code 10%, synthetic long-range probes 10% (answers 128-420 tokens back, beyond the decoder's reach).")
    sc = tel.get("screens", {})
    if sc:
        L.append("SCREENS (265 steps each, 8.7M tokens): " + "; ".join(f"{k} val {_f(v.get('val_mean'))} recall1 {_f(v['probes'].get('lr_recall1', {}).get('exact'))}"
                 for k, v in sc.items()) + " (S1 confined decoder = design; S2 stack decoder control; S3 no recurrence R=1). Screen gate: confined within 0.02 of stack -> PASS.")
    ev = [e for e in tel["evals"] if e.get("run", "main") == "main"]
    L.append("MAIN RUN HOURLY EVALS (step: val_mean | TinyStories | WikiText | recall1 | recall1 plan-swapped | carry3 | multi4 | hop | bind | NCP gap H-NCP | VE):")
    for e in ev:
        pr = e["probes"]; cb = e["codebook"]
        L.append(f"  {e['step']}: {_f(e['val_mean'])} | {_f(e['val_ts'])} | {_f(e['val_wk'])} | {_f(pr.get('lr_recall1', {}).get('exact'))} | {_f(e.get('swap_recall1'))} | "
                 f"{_f(pr.get('lr_carry3', {}).get('exact'))} | {_f(pr.get('lr_multi4', {}).get('exact'))} | {_f(pr.get('lr_hop', {}).get('exact'))} | {_f(pr.get('lr_bind', {}).get('exact'))} | "
                 f"{_f(cb.get('code_entropy', 0) - cb.get('ncp_ce', 0), 2)} | {_f(cb.get('var_explained'))}")
    tr = [r for r in tel["train"] if r.get("tok_s")]
    main_tr = [r for r in tr if r["step"] <= 4050]
    if main_tr:
        tps = sorted(r["tok_s"] for r in main_tr); gns = [r["gn_preclip"] for r in main_tr if r["step"] > 300 and r.get("gn_preclip")]
        L.append(f"MAIN TRAINING: 4,050 steps (132.7M tokens, ~0.47 epoch), 21.95 h, cosine LR peak 6e-4 -> 6e-5; median {tps[len(tps)//2]:.0f} tok/s; "
                 f"grad norm {min(gns):.2f}-{max(gns):.2f} after warm-up; safety-guard events: {len(tel.get('guards', []))}; final train NTP (smoothed) {_f(main_tr[-1].get('ntp_ema'))}.")
    E = fin.get("main")
    if E:
        lg = E["lang"]; ti = E["items"]["test"]; oo = E["items"]["ood"]
        L.append("FINAL EVAL OF THE 24 h MODEL (step 4,050), test sets:")
        for nm in ("tinystories", "wikitext"):
            d = lg[nm]
            L.append(f"  language {nm} nats: intact {_f(d['intact']['nats'])} (bpb {_f(d['intact']['bpb'])}), predicted-concept off {_f(d['c_off']['nats'])}, loop-state off {_f(d['u_off']['nats'])}, "
                     f"concept module off {_f(d['core_off']['nats'])}, plan swapped {_f(d['plan_swap']['nats'])}; R sweep " + ", ".join(f"R{r} {_f(d['R'+str(r)]['nats'],4)}" for r in (1, 2, 3, 4, 6, 8)))
        for nm in ("tinystories", "wikitext"):
            t = E["trunc"][nm]; sm = t["summary"]
            L.append(f"  truncation {nm}: " + ", ".join(f"last {r} tokens {_f(t['r'+str(r)]['nats'])}" for r in (72, 128, 256, 512, 1024)) +
                     f"; plan swapped {_f(t['swap_r1024']['nats'])}; gain beyond 70 {_f(sm['gain_beyond_70'])}; rows with >=512 context {sm['rows_ctx_ge_512']}/{sm['rows']}")
        for split, S in (("test", ti), ("OOD (about 2x longer)", oo)):
            L.append(f"  probes {split} (512 items each, exact match): " + "; ".join(
                f"{k[3:]} intact {_f(S['intact'][k]['exact'])} / concept off {_f(S['core_off'][k]['exact'])} / swapped {_f(S['plan_swap'][k]['exact'])} / R1 {_f(S['R1'][k]['exact'])}"
                for k in ("lr_recall1", "lr_carry3", "lr_multi4", "lr_hop", "lr_bind")))
        L.append(f"  per-token partial credit test: carry3 {ti['intact']['lr_carry3']['by_offset']}, multi4 {ti['intact']['lr_multi4']['by_offset']}")
        fams = [k for k in oo["intact"] if not k.startswith(("lr_", "_"))]
        L.append(f"  synthetic tasks macro: test {_f(ti['intact']['_macro_synthetic'])}, OOD {_f(oo['intact']['_macro_synthetic'])}; per family test/OOD: " +
                 ", ".join(f"{f} {_f(ti['intact'].get(f, {}).get('exact'), 2)}/{_f(oo['intact'].get(f, {}).get('exact'), 2)}" for f in fams))
        P = E["probes"]
        L.append("  linear probes: concept->its 4 tokens " + ", ".join(f"{_f(v['acc'])} (baseline {_f(v['majority_baseline'])})" for v in P["G_E1_concept_to_token"].values()) +
                 "; loop state at question->answer token 1: " + ", ".join(f"{k.replace('u_query_to_answer_lr_', '')} {_f(v['tok0']['acc'])}" for k, v in P.items() if k.startswith("u_query")) +
                 "; hop bridge entity by loop pass " + ", ".join(f"{k} {_f(v['acc'])}" for k, v in P["lr_hop_bridge_by_pass"].items()))
        H = E["health"]; cb = H["codebook"]; ld = H["loop_dynamics_lr_hop"]
        L.append(f"  health: codebook = 48 dictionaries x 128 codes (6,144 codes); fraction of codes used in the least-used dictionary {cb['codes_used_frac_min']} (i.e. every code in every dictionary), mean per-dictionary perplexity {_f(cb['perplexity_mean'],1)} of 128, variance explained {_f(cb['var_explained'])}, next-concept loss {_f(cb['ncp_ce'])} vs code entropy {_f(cb['code_entropy'])}; "
                 f"concept mix embedding/local/global {H['telemetry']['crc_mix_e0_L_G']}; Engram gate {_f(H['telemetry']['engram_alpha_mean'])}; loop step size pass1->8 {_f(ld['step'][0],2)}->{_f(ld['step'][-1],2)}, "
                 f"angle between consecutive steps {_f(ld['consecutive_step_angle_deg'][0],2)}->{_f(ld['consecutive_step_angle_deg'][-1],2)} deg")
        g = E["generation"]["tinystories"]["greedy"][0]
        L.append(f"  sample (TinyStories, greedy): PROMPT: {g['prompt'][-160:]!r} CONTINUATION: {g['continuation'].split('<eos>')[0][:300]!r}")
        L.append(f"  P5 exact structural tests on trained weights: {(fin.get('main_p5') or '').count('PASS')} of 3 PASS (causality; decoder reach exactly 70 with concepts off; document isolation).")
        L.append("GATES (pre-registered): G-E0/D0 exact tests PASS; G-E1 concept->4 tokens >=0.99 NOT MET (0.63-0.70); G-E2 VE>=0.6 PASS; G-C1 codes>=95% PASS; G-C2 NCP<entropy PASS; "
                 "G-E3/G-D2 plan swap -> chance PASS but wrong-plan cost >> gain beyond 70 -> PARTLY; G-C3 R4 beats R1 on hop: 3 items of 512, z=0.33 -> WITHIN NOISE; G-D3 multi4>=0.5 NOT MET (0.08).")
    st = tel.get("status", {}); pl = tel.get("plan", {})
    ce = [e for e in tel["evals"] if e.get("run") == "cont"]
    ct = [r for r in tr if r["step"] > 4050]
    L.append(f"CONTINUATION (+24 h, started 2026-09-28 09:26 from the step-4,050 checkpoint, same data mix, unseen rows): stage {pl.get('stage')} state {pl.get('s2')}; "
             f"LR re-warm 6e-5 -> 6e-4 over 300 steps then the main run's cosine to 6e-5 over 4,050 steps (chosen by the owner after a schedule-aware loss fit on the main run predicted ~0.07 lower final validation than a 3e-4 peak; about 0.2 nats of the main run's late gains came from annealing); "
             f"schedule-aware loss fit predicts a temporary rise (+0.10-0.15, back below 2.754 after ~1,500 steps) and a final validation ~2.53 at step 8,100. "
             f"Now: global step {st.get('step')}/{st.get('total_steps')}; recent train rows: " + ", ".join(f"step {r['step']} NTP {_f(r['ntp'])} lr {r['lr']:.1e}" for r in ct[-4:]) +
             ("; continuation evals: " + ", ".join(f"step {e['step']} val {_f(e['val_mean'])} recall1 {_f(e['probes'].get('lr_recall1', {}).get('exact'))}" for e in ce) if ce else "; no continuation eval yet"))
    return "\n".join(L)


ANALYSIS_PROMPT = """You are analysing a small research language-model training run for its owner, who is not an ML expert.
Below is a digest of every number shown on their dashboard. Write a detailed but plain-language analysis in Markdown:

## In one paragraph — what we built and whether it worked
## What we achieved (the evidence) — bullets, each with the number that proves it
## What did not work or is still weak — honest, with numbers
## Surprises and interesting observations
## How it compares to what we expected (the pre-registered gates, explained simply)
## What the running +24 h continuation is testing and what to watch for
## Recommended next steps — concrete, ranked
## Glossary — one line each for the technical terms you used

Rules: use only numbers from the digest (never invent), say when something is within noise or uncertain, explain every
technical term the first time in plain words, keep sentences short, about 900–1,300 words. Do not use tools.

DIGEST:
"""


def run_analysis():
    try:
        digest = build_digest()
        import tempfile
        with tempfile.TemporaryDirectory() as td:                # no project files, no tools; prompt via stdin, not argv
            r = subprocess.run([CLAUDE_BIN, "-p", "--tools", "", "--output-format", "text", "--model", "opus", "--no-session-persistence"],
                               input=ANALYSIS_PROMPT + digest, capture_output=True, text=True, timeout=600, cwd=td)
        if r.returncode != 0 or not r.stdout.strip():
            raise RuntimeError((r.stderr or r.stdout or f"exit {r.returncode}").strip()[-400:])
        rec = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), model="Claude Opus (Claude Code CLI, headless, no tools)",
                   markdown=r.stdout.strip(), digest=digest, seconds=round(time.time() - _an["started"]))
        os.makedirs(os.path.dirname(ANALYSIS_FILE), exist_ok=True)
        json.dump(rec, open(ANALYSIS_FILE + ".tmp", "w")); os.replace(ANALYSIS_FILE + ".tmp", ANALYSIS_FILE)
        _an["error"] = None
    except Exception as ex:
        _an["error"] = str(ex)[:500]
    finally:
        _an["running"] = False


def analysis_body(start=False):
    if start and not _an["running"]:
        _an.update(running=True, started=time.time(), error=None)
        threading.Thread(target=run_analysis, daemon=True).start()
    try:
        rec = json.load(open(ANALYSIS_FILE))
    except (FileNotFoundError, json.JSONDecodeError):
        rec = None
    out = dict(analysis=rec, running=_an["running"], elapsed=round(time.time() - _an["started"]) if _an["running"] else None, error=_an["error"])
    if not rec:
        try:
            out["digest_preview"] = build_digest()
        except Exception as ex:
            out["digest_preview"] = f"(digest failed: {ex})"
    return json.dumps(out).encode()


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")          # the 8765 page embeds / polls this v2 instance
        if self.path.startswith("/api/") or self.path.endswith((".html", ".js", ".css", "/")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path.startswith("/api/live"):
            with _lock:
                update_all()
                body = live_body()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/api/analysis"):
            body = analysis_body(start="run=1" in self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/api/final"):
            body = final_body()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/api/telemetry"):
            with _lock:
                body = build()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/server.py"):
            self.send_error(404); return
        return super().do_GET()


if __name__ == "__main__":
    try:
        os.nice(19)
    except Exception:
        pass
    _load_steps()
    threading.Thread(target=sampler, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"dashboard on http://127.0.0.1:{PORT}  (runs: {RUNS})", flush=True)
    srv.serve_forever()
