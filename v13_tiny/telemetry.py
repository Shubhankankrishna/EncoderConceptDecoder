"""Telemetry for the 24-hour V13 test: the base run, the machine (power, sleep, GPU, memory), the evaluation panels,
the pre-flight checks, the post-base pipeline, the continual methods and the results, against the 24-hour plan.
Reads logs and system counters only (no torch, no GPU), so it runs next to training at background priority.

    python3 telemetry.py                     # terminal summary, once
    python3 telemetry.py --loop              # every 60 s: telemetry/dashboard.html (interactive, reloads itself),
                                             #   status.json, system.jsonl (machine history), alerts.log
    python3 telemetry.py --run runs/base_tf --name "baseline TF" --out telemetry_tf     # another run
Open telemetry/dashboard.html in a browser.
"""
import argparse
import datetime as dt
import json
import math
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMS = ['grid', 'prog', 'state', 'plan', 'text']
TARGET_SHARE = {'grid': 0.225, 'prog': 0.225, 'state': 0.225, 'plan': 0.225, 'text': 0.10}   # data.SHARES
CURRICULUM = [(0.0, (0.7, 0.3, 0.0)), (0.1, (0.3, 0.6, 0.1)), (0.3, (0.15, 0.65, 0.2))]   # data.curriculum
PANELS = ['base_test', 'base_dev', 'challenge', 'cl_test']
POST_STEPS = ['probe_v13', 'probe_seq', 'final_eval', 'diagnostics', 'cl_seq', 'cl_v13', 'cl_replay', 'cl_eval', 'report']
PLANNED_MIN = dict(probe_v13=6, probe_seq=3, final_eval=60, diagnostics=15, cl_seq=70, cl_v13=145, cl_replay=85,
                   cl_eval=60, report=2)
GROUPS = [('MPS probe', ['probe_v13', 'probe_seq']), ('Final eval + diagnostics', ['final_eval', 'diagnostics']),
          ('Continual: seq', ['cl_seq']), ('Continual: v13', ['cl_v13']), ('Continual: replay', ['cl_replay']),
          ('Continual matrix', ['cl_eval']), ('Tables + write-up', ['report'])]
BENCH_TOK_S = 3234                                                  # compiled B = 8 benchmark at launch
WRITEUP_MIN = 30


# ------------------------------------------------------------------------------------------------ small helpers
def sh(cmd, timeout=20):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ''


def jl(p):
    out = []
    if Path(p).exists():
        for line in open(p):
            try: out.append(json.loads(line))
            except Exception: pass
    return out


def jload(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None


def mtime(p):
    p = Path(p)
    return p.stat().st_mtime if p.exists() else None


def ts(s):
    return dt.datetime.strptime(s, '%Y-%m-%d %H:%M').timestamp()


def hm(t):
    return dt.datetime.fromtimestamp(t).strftime('%a %H:%M') if t else '—'


def clean(x):
    """JSON-safe copy: NaN and infinities become None (JSON.parse rejects them)."""
    if isinstance(x, float): return x if math.isfinite(x) else None
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    return x


def date_line(ln):
    """'Wed Oct 14 10:07:44 IST 2026 message' -> (epoch, message)."""
    m = re.match(r'^\w{3} (\w{3}) +(\d+) (\d\d:\d\d:\d\d) \S+ (\d{4}) (.*)$', ln)
    if not m: return None, ln
    t = dt.datetime.strptime(f'{m.group(1)} {m.group(2)} {m.group(4)} {m.group(3)}', '%b %d %Y %H:%M:%S').timestamp()
    return t, m.group(5)


# ------------------------------------------------------------------------------------------------ the machine
def machine():
    s = dict(t=time.time())
    m = re.search(r'(\d+)%', sh('memory_pressure | grep "free percentage"')); s['mem_free_pct'] = int(m.group(1)) if m else None
    m = re.search(r'used = ([\d.]+)M', sh('sysctl -n vm.swapusage')); s['swap_gb'] = round(float(m.group(1)) / 1024, 2) if m else None
    top = sh('top -l 2 -n 0 -s 1', timeout=30)                        # the second sample has valid CPU shares
    cpu = re.findall(r'CPU usage: ([\d.]+)% user, ([\d.]+)% sys, ([\d.]+)% idle', top)
    if cpu: s['cpu_busy'] = round(100 - float(cpu[-1][2]), 1)
    la = re.findall(r'Load Avg: ([\d.]+)', top); s['load1'] = float(la[-1]) if la else None
    pm = re.findall(r'PhysMem: ([\d.]+)([MG]) used \(([\d.]+)([MG]) wired(?:, ([\d.]+)([MG]) compressor)?\), ([\d.]+)([MG]) unused', top)
    if pm:
        g = lambda v, u: round(float(v) / (1024 if u == 'M' else 1), 2) if v else 0.0
        p = pm[-1]; s.update(mem_used_gb=g(p[0], p[1]), wired_gb=g(p[2], p[3]), compressor_gb=g(p[4], p[5]), unused_gb=g(p[6], p[7]))
    io = sh('ioreg -r -d 1 -w 0 -c IOAccelerator')
    m = re.search(r'"PerformanceStatistics" = \{([^}]*)\}', io)
    if m:
        st = {k: int(v) for k, v in re.findall(r'"([^"]+)"=(\d+)', m.group(1))}
        s['gpu_util'] = st.get('Device Utilization %'); s['gpu_renderer'] = st.get('Renderer Utilization %')
        if 'In use system memory' in st: s['gpu_inuse_gb'] = round(st['In use system memory'] / 2 ** 30, 2)
        if 'Alloc system memory' in st: s['gpu_alloc_gb'] = round(st['Alloc system memory'] / 2 ** 30, 2)
    b = sh('pmset -g batt')
    s['power'] = 'AC' if "'AC Power'" in b else ('battery' if "'Battery Power'" in b else 'unknown')
    m = re.search(r'(\d+)%;\s*([^;]+);\s*(.*?)\s+present', b)
    if m: s.update(battery_pct=int(m.group(1)), battery_state=m.group(2).strip(), battery_remaining=m.group(3).strip())
    a = sh('pmset -g assertions')
    m = re.search(r'^\s*PreventUserIdleSystemSleep\s+(\d+)', a, re.M); s['idle_sleep_blocked'] = (m.group(1) != '0') if m else None
    m = re.search(r'^\s*PreventSystemSleep\s+(\d+)', a, re.M); s['system_sleep_blocked'] = (m.group(1) != '0') if m else None
    s['caffeinate_pids'] = sorted({int(x) for x in re.findall(r'pid (\d+)\(caffeinate\)', a)})
    m = re.search(r'"AppleClamshellState" = (Yes|No)', sh('ioreg -r -k AppleClamshellState -d 4'))
    s['lid_closed'] = (m.group(1) == 'Yes') if m else None
    for k in ('sleeptime', 'waketime'):
        m = re.search(r'sec = (\d+)', sh(f'sysctl -n kern.{k}')); s['last_' + k[:-4]] = int(m.group(1)) if m else None
    m = re.search(r'\s(\d+)\s+\d+%\s', sh(f'df -k "{HERE}" | tail -1')); s['disk_free_gb'] = round(int(m.group(1)) / 2 ** 20, 1) if m else None
    th = sh('pmset -g therm'); s['thermal_warning'] = ('No thermal warning' not in th) if th else None
    return s


ROLES = [('train.py', 'base training'), ('run_base.sh', 'base launcher (auto-restart)'), ('post_base.sh', 'post-base pipeline'),
         ('build_panels.py', 'panel build'), ('final_eval.py', 'final evaluation'), ('diagnostics.py', 'diagnostics'),
         ('cl_eval.py', 'continual matrix'), ('cl.py', 'continual learning'), ('make_report.py', 'report tables'),
         ('telemetry.py', 'telemetry')]


def processes():
    rows = []
    for line in sh('ps -axo pid=,ppid=,pcpu=,rss=,etime=,command=').splitlines():
        p = line.split(None, 5)
        if len(p) < 6: continue
        try: rows.append(dict(pid=int(p[0]), ppid=int(p[1]), cpu=float(p[2]), rss=int(p[3]) // 1024, etime=p[4], cmd=p[5]))
        except ValueError: pass
    main, guards = [], []
    for r in rows:
        c = r['cmd']
        if 'caffeinate' in c.split(' ')[0]:
            guards.append(dict(pid=r['pid'], what=c[:90])); continue
        if 'multiprocessing' in c or 'shell-snapshots' in c or c.startswith('/usr/bin/time') or 'grep' in c: continue
        for key, role in ROLES:
            if re.search(r'(^|[ /])' + re.escape(key) + r'(\s|$)', c):
                kids = [k for k in rows if k['ppid'] == r['pid'] and 'multiprocessing' in k['cmd']]
                args = c[c.find(key) + len(key):].strip()[:70]
                main.append(dict(pid=r['pid'], role=role, args=args, cpu=round(r['cpu'] + sum(k['cpu'] for k in kids), 1),
                                 own_cpu=r['cpu'], workers_cpu=round(sum(k['cpu'] for k in kids), 1),
                                 workers_max=round(max([k['cpu'] for k in kids] or [0]), 1),
                                 rss_mb=r['rss'] + sum(k['rss'] for k in kids), workers=len(kids), etime=r['etime']))
                break
    return main, guards


# ------------------------------------------------------------------------------------------------ the base run
def base_run(run, hist):
    cfg = jload(run / 'config.json') or {}
    log = jl(run / 'log.jsonl')
    r = dict(dir=str(run.relative_to(HERE)), params=cfg.get('params'), hours=cfg.get('hours', 11.0), started=cfg.get('started'),
             B=cfg.get('B'), T=cfg.get('T'), seed=cfg.get('seed'), done=(run / 'DONE').exists(), done_at=mtime(run / 'DONE'),
             fallback=(run / 'DONE_FALLBACK').exists())
    r['start_ts'] = dt.datetime.strptime(cfg['started'], '%Y-%m-%d %H:%M:%S').timestamp() if cfg.get('started') else None
    out = (run / 'train.out').read_text(errors='replace') if (run / 'train.out').exists() else ''
    r['restarts'] = len(re.findall(r'^resumed at step', out, re.M))
    r['nonfinite_events'] = len(re.findall(r'non-finite', out))
    r['errors'] = [ln for ln in out.splitlines() if 'Traceback' in ln or re.search(r'\w+Error\b', ln)][-3:]
    r['out_tail'] = [ln[:220] for ln in out.splitlines() if ln.startswith(('step ', 'resumed', 'finished', 'params'))][-3:]
    r['last_ckpt'] = mtime(run / 'latest.pt')
    r['snaps'] = [dict(name=p.name, pct=int(p.name[5:8]), t=p.stat().st_mtime) for p in sorted(run.glob('snap_*.pt'))]
    ft = jload(run / 'family_tokens.json') or {}
    tot = sum(ft.values()) or 1
    r['family_tokens'] = ft; r['family_share'] = {f: ft.get(f, 0) / tot for f in FAMS}; r['target_share'] = TARGET_SHARE
    r['family_tokens_at'] = mtime(run / 'family_tokens.json')
    if not log: return r
    last = log[-1]; lm = mtime(run / 'log.jsonl')
    r.update(last=last, log_mtime=lm, age_s=int(time.time() - lm), pct=round(100 * last['h'] / r['hours'], 1))
    ratio = None
    if r['start_ts'] and last['h'] > 0.05: ratio = (lm - r['start_ts']) / (last['h'] * 3600)
    pairs = [(x['base_log_mtime'], x['base_h']) for x in hist if x.get('base_h') is not None and x.get('base_log_mtime')]
    if pairs:                                                      # wall / training time over the last ~45 min
        c = min(pairs, key=lambda p: abs(p[0] - (lm - 2700)))
        if lm - c[0] > 600 and last['h'] - c[1] > 0.05: ratio = (lm - c[0]) / ((last['h'] - c[1]) * 3600)
    r['wall_ratio'] = round(ratio, 4) if ratio else None
    remaining = max(0.0, r['hours'] - last['h']) * 3600
    r['eta'] = r['done_at'] if r['done'] else lm + remaining * (ratio or 1.0)
    recent = log[-20:]
    r['tok_s_recent'] = int(sum(x['tok_s'] for x in recent) / len(recent))
    r['s_step_recent'] = round(sum(x['s_step'] for x in recent) / len(recent), 3)
    r['tok_s_overall'] = int(last['tokens'] / max(1.0, lm - r['start_ts'])) if r['start_ts'] else None
    r['tokens_projected'] = int(last['tokens'] + (0 if r['done'] else remaining * (ratio or 1.0) * r['tok_s_recent']))
    fam = {}
    for f in FAMS:
        v = [x for x in log if x['fam'] == f]
        if v:
            med = sorted(x['ntp'] for x in v[-3:])[len(v[-3:]) // 2]
            fam[f] = dict(n=len(v), last=v[-1]['ntp'], med3=med, step=v[-1]['step'], ncp_acc=v[-1]['ncp_acc'])
        else:
            fam[f] = dict(n=0)
    r['by_family'] = fam
    frac = last['frac']; stage = max(i for i, (f0, _) in enumerate(CURRICULUM) if frac >= f0)
    nxt = CURRICULUM[stage + 1][0] if stage + 1 < len(CURRICULUM) else None
    r['curriculum'] = dict(stage=stage, mix=CURRICULUM[stage][1], frac=frac, next_frac=nxt,
                           next_eta=(lm + (nxt * r['hours'] - last['h']) * 3600 * (ratio or 1.0)) if nxt else None)
    keys = ['step', 'h', 'fam', 'ntp', 'ncp', 'ncp_acc', 'tok_s', 's_step', 'gn', 'ppl', 'lr', 'lam', 'mem_gb', 'tau', 'beta', 'skipped']
    r['series'] = [{k: x.get(k) for k in keys} for x in log]
    return r


# ------------------------------------------------------------------------------------------------ panels, checks, pipeline
def panels():
    built = {}
    txt = (HERE / 'data/build_panels.out').read_text(errors='replace') if (HERE / 'data/build_panels.out').exists() else ''
    for name, n, sec in re.findall(r'panel (\w+): (\d+) items in (\d+)s', txt):
        built[name] = (int(n), int(sec))
    running = bool(sh('pgrep -f "python3 build_panels.py"').strip())
    out, current, prev_at = [], None, None
    for p in PANELS:
        f = HERE / 'data/panels' / f'{p}.pkl'; done = f.exists()
        st = 'built' if done else ('building' if running and current is None else 'waiting')
        n, sec = built.get(p, (None, None))
        if st == 'building':
            current = p; sec = int(time.time() - prev_at) if prev_at else None   # time since the previous panel landed
        out.append(dict(name=p, state=st, items=n, seconds=sec, at=mtime(f)))
        prev_at = mtime(f) or prev_at
    return dict(list=out, builder_running=running)


def preflight():
    q = HERE / 'runs/tmp_eval'
    def state(done, out, pat):
        if done: return 'pass'
        txt = Path(out).read_text(errors='replace') if Path(out).exists() else ''
        if 'Traceback' in txt: return 'fail'
        if pat and sh(f'pgrep -f "{pat}"').strip(): return 'running'
        return 'pending'
    checks = [dict(name='G0 correctness tests (float64)', state='pass', detail='reach, causality, isolation, GDN-2 equivalence, gradients')]
    for m in ('v13', 'seq', 'replay'):
        ok = (HERE / f'runs/tmp_cl_cpu/{m}/log.jsonl').exists()
        checks.append(dict(name=f'continual {m}: CPU test', state='pass' if ok else 'pending', detail='B = 1 on CPU'))
    checks.append(dict(name='diagnostics: quick test', detail='CPU, latest checkpoint',
                       state=state((q / 'diag/diagnostics.json').exists(), q / 'diag.out', 'diagnostics.py --ckpt runs/base/latest.pt')))
    fo = q / 'final.out'
    checks.append(dict(name='final evaluation: quick test', detail='CPU, 1 item per cell, with the snapshot curve',
                       state=state(fo.exists() and 'final eval done' in fo.read_text(errors='replace'), fo, 'final_eval.py  --ckpt runs/base/latest.pt|final_eval.py --ckpt runs/base/latest.pt')))
    co = q / 'cleval.out'
    checks.append(dict(name='continual matrix: quick test', detail='CPU, CPU-test states',
                       state=state(co.exists() and 'paired differences' in co.read_text(errors='replace'), co, 'cl_eval.py    --cl runs/tmp_cl_cpu|cl_eval.py --cl runs/tmp_cl_cpu')))
    checks.append(dict(name='report tables: quick test', detail='make_report on the quick outputs',
                       state=state((q / 'REPORT_TABLES_test.md').exists(), q / 'report.out', None)))
    pv = [jl(HERE / f'runs/post/probe/{m}/log.jsonl') for m in ('v13', 'seq')]
    checks.append(dict(name='MPS probe at B = 8', detail='first step after training', state='pass' if all(pv) else 'pending'))
    return checks


def pipeline():
    p = HERE / 'runs/post/status.log'
    lines = p.read_text().splitlines() if p.exists() else []
    steps = {k: dict(name=k, state='pending', start=None, end=None) for k in POST_STEPS}
    notes = []
    for ln in lines:
        t, msg = date_line(ln)
        m = re.match(r'start (\w+)$', msg)
        if m and m.group(1) in steps:
            steps[m.group(1)].update(state='running', start=t); continue
        m = re.match(r'end (\w+) exit (\d+)$', msg)
        if m and m.group(1) in steps:
            steps[m.group(1)].update(state='ok' if m.group(2) == '0' else f'failed ({m.group(2)})', end=t); continue
        notes.append(dict(t=t, msg=msg))
    armed = bool(sh('pgrep -f "post_base.sh"').strip())
    return dict(steps=[steps[k] for k in POST_STEPS], notes=notes[-8:], armed=armed,
                all_done=(HERE / 'runs/post/ALL_DONE').exists(), budget=jload(HERE / 'runs/post/probe/budget.json'))


def continual():
    meth = {m: jl(HERE / 'runs/cl' / m / 'log.jsonl') for m in ('seq', 'v13', 'replay')}
    probe = {m: jl(HERE / 'runs/post/probe' / m / 'log.jsonl') for m in ('v13', 'seq')}
    return dict(methods={k: v for k, v in meth.items() if v}, probe={k: v for k, v in probe.items() if v})


def results():
    r = {}
    ev = jload(HERE / 'runs/eval/base/results.json')
    if ev: r['base'] = ev
    dg = jload(HERE / 'runs/eval/diag/diagnostics.json')
    if dg: r['diag'] = dg
    cl = jload(HERE / 'runs/eval/cl/cl_results.json')
    if cl:
        r['cl'] = {m: v['metrics'] for m, v in cl.items() if isinstance(v, dict) and 'metrics' in v}
        r['cl_paired'] = cl.get('paired')
    return r


# ------------------------------------------------------------------------------------------------ the 24-hour plan
def plan(base, post, a):
    now = time.time()
    req, ddl, cld = ts(a.request), ts(a.deadline), ts(a.cl_deadline)
    segs = []
    bstart = base.get('start_ts') or req
    segs.append(dict(name='Build, tests, launch', start=req, end=bstart, state='done'))
    bend = base.get('eta') or (bstart + base['hours'] * 3600)
    segs.append(dict(name='Base training (11 h)', start=bstart, end=bend, state='done' if base.get('done') else 'running',
                     pct=base.get('pct')))
    plan_min = dict(PLANNED_MIN)
    b = post.get('budget')
    if b:                                                          # the probe's timings, as cl_budget.py uses them
        n = b['steps']
        plan_min['cl_seq'] = (6 * n * b['t_seq'] + 6 * 60 + 300) / 60
        plan_min['cl_replay'] = (6 * n * 1.25 * b['t_seq'] + 6 * 60 + 600) / 60
        plan_min['cl_v13'] = (6 * n * b['t_v13'] + 6 * 150 + 300) / 60
    steps = {s['name']: s for s in post['steps']}
    t = max(bend, now) if not base.get('done') else bend
    for name, keys in GROUPS:
        a0, a1, states = None, t, []
        for k in keys:
            s = steps[k]
            if s['start'] and s['end']: x0, x1 = s['start'], s['end']
            elif s['start']: x0, x1 = s['start'], max(now, s['start'] + plan_min[k] * 60)
            else: x0, x1 = a1, a1 + plan_min[k] * 60
            a0 = x0 if a0 is None else a0; a1 = x1; states.append(s['state'])
        if name == 'Tables + write-up': a1 += WRITEUP_MIN * 60
        st = ('failed' if any(x.startswith('failed') for x in states) else 'done' if all(x == 'ok' for x in states)
              else 'running' if any(x in ('running', 'ok') for x in states) else 'planned')
        segs.append(dict(name=name, start=a0, end=a1, state=st))
        t = a1
    cl_end = next(sg['end'] for sg in segs if sg['name'] == 'Continual: replay')
    return dict(t0=req, t1=ddl, now=now, segments=segs, markers=[dict(t=cld, label='continual deadline 04:00'),
                dict(t=ddl, label='24-h deadline')], predicted_end=t, slack_s=ddl - t, cl_end=cl_end, cl_slack_s=cld - cl_end)


# ------------------------------------------------------------------------------------------------ alerts
def alerts(d):
    A = []
    add = lambda level, key, text: A.append(dict(level=level, key=key, text=text))   # key: stable id for alerts.log
    s, b, post, pl = d['machine'], d['base'], d['post'], d['plan']
    run_active = not b.get('done') or (post['armed'] and not post['all_done'])
    if s.get('power') == 'battery':
        add('critical', 'battery', f"On battery ({s.get('battery_pct')}%, {s.get('battery_state')}, {s.get('battery_remaining')}). "
                        "Plug in the charger: on battery, caffeinate cannot hold off system sleep, and the run stops when the battery runs out.")
    if s.get('lid_closed'): add('critical', 'lid', 'The lid is closed: the Mac sleeps despite caffeinate unless an external display is attached.')
    if b.get('start_ts') and s.get('last_sleep') and s['last_sleep'] > b['start_ts']:
        add('critical', 'slept', f"The Mac slept at {hm(s['last_sleep'])} (woke {hm(s.get('last_wake'))}) after the run started: "
                        "a training step that spanned the sleep counts it as training time.")
    if run_active and s.get('idle_sleep_blocked') is False: add('critical', 'no_sleep_guard', 'Nothing is blocking idle sleep (no caffeinate assertion).')
    procs = {p['role'] for p in d['procs']}
    if not b.get('done') and 'base training' not in procs: add('critical', 'train_dead', 'The base training process is not running.')
    if not b.get('done') and b.get('age_s', 0) > 900: add('critical', 'log_stale', f"No training log line for {b['age_s'] // 60} min.")
    if b.get('last', {}).get('skipped'): add('critical', 'skipped', f"{b['last']['skipped']} updates skipped for non-finite gradients.")
    if b.get('errors'): add('critical', 'train_error', 'Error text in train.out: ' + b['errors'][-1][:120])
    if b.get('restarts'): add('warn', 'restarts', f"Training restarted {b['restarts']} time(s) (resumed from latest.pt).")
    if not b.get('done') and b.get('s_step_recent', 0) > 4.5: add('warn', 'slow', f"Training slow: {b['s_step_recent']} s/step over the last 20 logs (≈3.0 expected).")
    tr = next((p for p in d['procs'] if p['role'] == 'base training'), None)
    if tr and tr['workers'] and tr['workers_cpu'] > 85 * tr['workers']:
        add('warn', 'data_saturated', f"Training's {tr['workers']} data workers are saturated ({tr['workers_cpu']:.0f}% CPU in total): training may be waiting on data.")
    hist = d['history'][-5:]
    gu = [h['gpu_util'] for h in hist if h.get('gpu_util') is not None]
    if not b.get('done') and len(gu) >= 3 and sum(gu) / len(gu) < 60:
        add('warn', 'gpu_low', f"GPU utilisation averaged {sum(gu) / len(gu):.0f}% over the last {len(gu)} samples: training may be waiting on data or CPU.")
    if (s.get('swap_gb') or 0) > 6: add('critical', 'swap', f"Swap {s['swap_gb']} GB: the machine is paging.")
    elif (s.get('swap_gb') or 0) > 4: add('warn', 'swap', f"Swap {s['swap_gb']} GB.")
    if s.get('mem_free_pct') is not None and s['mem_free_pct'] < 10: add('critical', 'memory', f"Memory free {s['mem_free_pct']}%.")
    elif s.get('mem_free_pct') is not None and s['mem_free_pct'] < 15: add('warn', 'memory', f"Memory free {s['mem_free_pct']}%.")
    if s.get('disk_free_gb') is not None and s['disk_free_gb'] < 5: add('critical', 'disk', f"Disk free {s['disk_free_gb']} GB.")
    if s.get('thermal_warning'): add('warn', 'thermal', 'A thermal warning is recorded (pmset).')
    for st in post['steps']:
        if st['state'].startswith('failed'): add('critical', 'post_failed_' + st['name'], f"Post-base step {st['name']} {st['state']}: see runs/post/{st['name']}.out")
    if b.get('done') and not post['armed'] and not post['all_done']: add('critical', 'post_not_running', 'Base run finished but the post-base pipeline is not running.')
    if not b.get('done') and not post['armed']: add('warn', 'post_not_armed', 'The post-base pipeline is not armed (post_base.sh not running).')
    for c in d['preflight']:
        if c['state'] == 'fail': add('warn', 'preflight_' + c['name'], f"Pre-flight check failed: {c['name']}")
    if pl['slack_s'] < 0: add('critical', 'slack', f"Predicted finish {hm(pl['predicted_end'])} is after the 24-hour deadline.")
    elif pl['slack_s'] < 3600: add('warn', 'slack', f"Only {pl['slack_s'] / 60:.0f} min of slack before the 24-hour deadline.")
    order = {'critical': 0, 'warn': 1, 'info': 2}
    return sorted(A, key=lambda x: order[x['level']])


# ------------------------------------------------------------------------------------------------ assemble
def collect(run, out, a):
    hist = jl(out / 'system.jsonl')
    s = machine(); procs, guards = processes()
    b = base_run(run, hist)
    tick = {k: v for k, v in s.items() if not isinstance(v, (list, dict))}
    tick.update(base_step=b.get('last', {}).get('step'), base_h=b.get('last', {}).get('h'), base_log_mtime=b.get('log_mtime'),
                tok_s=b.get('last', {}).get('tok_s'))
    hist.append(tick)
    post = pipeline()
    d = dict(generated=time.time(), generated_str=time.strftime('%Y-%m-%d %H:%M:%S'), name=a.name, every=a.every,
             machine=s, procs=procs, guards=guards, base=b, panels=panels(), preflight=preflight(), post=post,
             continual=continual(), results=results(), bench_tok_s=BENCH_TOK_S)
    stride = max(1, len(hist) // 720)
    d['history'] = hist[::stride] if stride > 1 else hist
    d['plan'] = plan(b, post, a)
    d['alerts'] = alerts(d)
    return d, tick


def terminal(d):
    b, s, p, pl = d['base'], d['machine'], d['post'], d['plan']
    print(f"[{d['generated_str']}] {d['name']} ({b['dir']})")
    if b.get('last'):
        last = b['last']
        print(f"  base   step {last['step']:,}  {last['h']:.2f}/{b['hours']:.0f} h ({b['pct']}%)  ends ~{hm(b['eta'])}  "
              f"{b['tok_s_recent']:,} tok/s  {b['s_step_recent']} s/step  {last['tokens'] / 1e6:.1f}M tokens "
              f"(→ {b['tokens_projected'] / 1e6:.0f}M)  skipped {last.get('skipped', 0)}  restarts {b['restarts']}")
        print('  loss   ' + '  '.join(f"{f} {v['med3']:.3f}" if v.get('n') else f'{f} —' for f, v in b['by_family'].items()) +
              f"   NCP acc {last['ncp_acc']:.2f}  code ppl {last['ppl']:.1f}")
    print(f"  power  {s.get('power')} {s.get('battery_pct')}% {s.get('battery_state', '')}  lid {'closed' if s.get('lid_closed') else 'open'}  "
          f"idle-sleep blocked {s.get('idle_sleep_blocked')}")
    print(f"  machine GPU {s.get('gpu_util')}% ({s.get('gpu_inuse_gb')} GB)  CPU {s.get('cpu_busy')}%  free {s.get('mem_free_pct')}%  "
          f"swap {s.get('swap_gb')} GB  disk {s.get('disk_free_gb')} GB")
    print('  panels ' + ' '.join(f"{x['name']}:{x['state']}" for x in d['panels']['list']))
    print('  checks ' + ' '.join(f"{c['name'].split(':')[0]}:{c['state']}" for c in d['preflight'][1:]))
    print(f"  post   {'armed' if p['armed'] else 'NOT RUNNING'}  " + ' '.join(f"{x['name']}:{x['state']}" for x in p['steps'] if x['state'] != 'pending'))
    print(f"  plan   predicted finish {hm(pl['predicted_end'])}, slack {pl['slack_s'] / 3600:+.1f} h; continual ends ~{hm(pl['cl_end'])} "
          f"(deadline 04:00, slack {pl['cl_slack_s'] / 3600:+.1f} h)")
    for x in d['alerts']: print(f"  {x['level'].upper():8s} {x['text']}")
    if not d['alerts']: print('  no alerts')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='runs/base'); ap.add_argument('--name', default='V13-tiny-v2 · seed 1301')
    ap.add_argument('--loop', action='store_true'); ap.add_argument('--every', type=int, default=60)
    ap.add_argument('--out', default='telemetry')
    ap.add_argument('--request', default='2026-10-14 07:30'); ap.add_argument('--deadline', default='2026-10-15 07:30')
    ap.add_argument('--cl-deadline', default='2026-10-15 04:00')
    a = ap.parse_args()
    run = HERE / a.run; out = HERE / a.out; out.mkdir(exist_ok=True)
    active = {}
    while True:
        tpl = (HERE / 'telemetry_page.html').read_text()             # re-read: page edits apply without a restart
        d, tick = collect(run, out, a)
        with open(out / 'system.jsonl', 'a') as f: f.write(json.dumps(clean(tick)) + '\n')
        blob = json.dumps(clean(d), separators=(',', ':')).replace('</', '<\\/')
        tmp = out / 'dashboard.tmp'; tmp.write_text(tpl.replace('/*__DATA__*/null', blob)); tmp.replace(out / 'dashboard.html')
        slim = dict(d); slim['base'] = {k: v for k, v in d['base'].items() if k != 'series'}; slim.pop('history')
        (out / 'status.json').write_text(json.dumps(clean(slim), indent=1))
        now_alerts = {x['key']: x for x in d['alerts']}
        with open(out / ('alerts.log' if a.loop else 'alerts_once.log'), 'a') as f:    # the log belongs to the loop
            for k, x in now_alerts.items():
                if k not in active: f.write(f"{d['generated_str']} {x['level'].upper()} [{k}] {x['text']}\n")
            for k in set(active) - set(now_alerts): f.write(f"{d['generated_str']} RESOLVED [{k}]\n")
        active = now_alerts
        if not a.loop:
            terminal(d); break
        time.sleep(max(5, a.every - (time.time() - d['generated'])))


if __name__ == '__main__':
    main()
