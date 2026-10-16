"""Phase-2 driver: for each base run, GT controllers, then the metacontroller α sweep, then evaluation.

  python -B phase2.py RUN [RUN ...]

Order (revised 12:15, advisor #13): stage 1 for every run first (GT controllers, then evaluate), then the α
jobs of all runs interleaved by priority 0.1, 0.3, 0, 1, 0.05, 0.17, 0.5, then one 6000-step sensitivity run per
base (α 0.3). Parallelism: as given while any train.py runs, else 2. Cut (pre-registered): a base with
< 4 α points started by 17:30 keeps only {0, 0.1, 0.3, 1}. If the 6000-step run moves pair F1 by > 0.1 vs its
2000-step twin, the whole sweep is labelled budget-limited (paper: 64 000 steps × 512; ours 2 000 × 64).
Original per-run order (two GPU jobs at a time, each memory-capped):
  1. evaluate.py (dev NLL per family, probes, competence, concept-off usage)
  2. gtctrl.py at each control point (TF: mid = e_3 and L5 = e_5; ECD: concept, dec_in)
  3. meta.py, alpha in {0, 0.05, 0.1, 0.17, 0.3, 0.5, 1}, seed 1, at the primary control point
     (TF: chosen from step 2 by the 11:58 rule; ECD: concept); ECD also at dec_in for alpha in {0.1, 0.3}

Pre-registered RL gate (written 10:12, before any Phase-2 result): internal RL on the 3-pair novel task
0-1-2-3-4-5 runs for a (base, control point) only if (a) short_success_gt - short_success_base >= 0.03 from gtctrl (floor added 14:05, advisor #16), and
(b) at least one alpha reaches switch F1 >= 0.8 against pair switches on dev_meta. Otherwise RL is not run and
the result is reported as a finding about base competence/controllability at this scale.
"""
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV = dict(PYTORCH_MPS_HIGH_WATERMARK_RATIO='0.55', PYTORCH_MPS_LOW_WATERMARK_RATIO='0.5')
ALPHAS = (0.0, 0.05, 0.1, 0.17, 0.3, 0.5, 1.0)


PRIORITY = (0.1, 0.3, 0.0, 1.0, 0.05, 0.17, 0.5)     # α order (advisor #13): informative points first
CUT = '17:30'                                          # pre-registered: if a base has < 4 α points by 17:30,
CUT_SET = (0.0, 0.1, 0.3, 1.0)                         # its remaining sweep is restricted to {0, 0.1, 0.3, 1}


def is_ecd(run):
    return json.loads((ROOT / 'runs' / run / 'manifest.json').read_text())['cfg']['model'] == 'ECD'


def stage1(run):
    return [['gtctrl.py', run, p] for p in (('concept', 'dec_in') if is_ecd(run) else ('-', 'L5'))] + [['evaluate.py', run]]


FLOOR = 0.03   # added 14:05 (advisor #16), after mid (margin 3/256) and before L5 was seen


def tf_point(run):
    """TF point (pre-registered 11:58; noise floor added 14:05 before L5 was seen): whichever of mid (e_3) / L5
    (e_5) has 3-pair GT success above base by >= 0.03, the larger margin if both; ties and neither -> mid."""
    best, margin = 'mid', 0.0
    for p in ('mid', 'L5'):
        try: g = json.loads((ROOT / 'runs' / run / f'gtctrl_{p}.json').read_text())
        except Exception: continue
        m = g.get('short_success_gt', 0) - g.get('short_success_base', 0)
        if m < FLOOR: m = 0.0                         # n = 256: margins under 0.03 (~8 episodes) are noise
        if m > margin: best, margin = p, m
    print(time.strftime('%H:%M'), run, 'TF control point for the α sweep:', best, 'margin', round(margin, 3), flush=True)
    (ROOT / 'runs' / run / 'phase2_choice.json').write_text(json.dumps(dict(point=best, margin=margin)))
    return best


def stage2(runs):
    """α jobs from all runs interleaved by priority; ECD dec_in points (0.3 then 0.1) after the concept sweep's
    first two; one 6000-step sensitivity run per base last (TF: chosen point; ECD: concept), at α = 0.3."""
    per = {}
    for r in runs:
        if is_ecd(r):
            pt = 'concept'
            jobs = [(a, ['meta.py', r, str(a), '1', '2000', pt]) for a in PRIORITY]
            jobs = jobs[:2] + [(a, ['meta.py', r, str(a), '1', '2000', 'dec_in']) for a in (0.3, 0.1)] + jobs[2:]
        else:
            pt = tf_point(r); arg = pt if pt != 'mid' else '-'
            jobs = [(a, ['meta.py', r, str(a), '1', '2000', arg]) for a in PRIORITY]
            pt = arg
        per[r] = (jobs, ['meta.py', r, '0.3', '1', '6000', pt])
    out = []
    for i in range(max(len(v[0]) for v in per.values())):
        for r in runs:
            if i < len(per[r][0]): out.append((r, per[r][0][i]))
    tail = [(r, (None, per[r][1])) for r in runs]
    return out, tail


def parallel_now(base):
    """2 only when no training process is running (never three GPU processes; advisor #14)."""
    busy = subprocess.run(['pgrep', '-f', '^[^ ]*[Pp]ython[0-9.]* -B train.py'], capture_output=True).stdout.strip()
    return base if busy else 2


def output_of(j):
    """Result file a job writes (jobs whose result exists are skipped on restart)."""
    s, run = j[0], j[1]; d = ROOT / 'runs' / run
    if s == 'evaluate.py': return d / 'eval_final.json'
    if s == 'gtctrl.py': return d / f"gtctrl_{'mid' if j[2] == '-' else j[2]}.json"
    if s == 'meta.py':
        pt = j[5] if len(j) > 5 else '-'; pt = ('mid' if pt == '-' else pt)
        a = float(j[2]); n = int(j[4]) if len(j) > 4 else 2000
        return d / 'meta' / (f"{pt}_a{a}_s{j[3]}" + ('' if n == 2000 else f'_n{n}') + '.json')
    return None


PAT = '^[^ ]*[Pp]ython[0-9.]* -B (gtctrl|evaluate|meta|meta_xm|internal_rl)\\.py'


def running():
    """Command lines of Phase-2 jobs currently running (from any driver instance): restart-safe accounting."""
    out = subprocess.run(['pgrep', '-fl', PAT], capture_output=True, text=True).stdout.strip().splitlines()
    return [' '.join(l.split()[1:]) for l in out if l.strip()]


def run_jobs(jobs, env, parallel, logd, keep=lambda item: True):
    def cmd(it): return it[1][1] if isinstance(it, tuple) else it
    jobs = [it for it in jobs if not ((o := output_of(cmd(it))) and o.exists())]
    queue = list(jobs)
    while queue or running():
        live = running()
        while queue and len(live) < parallel_now(parallel):
            item = queue.pop(0); j = cmd(item)
            if (o := output_of(j)) and o.exists():
                print(time.strftime('%H:%M'), 'done already', ' '.join(j), flush=True); continue
            if any(' '.join(j) in l for l in live):
                print(time.strftime('%H:%M'), 'already running', ' '.join(j), flush=True); continue
            if not keep(item): print(time.strftime('%H:%M'), 'cut', ' '.join(j), flush=True); continue
            name = '_'.join(x.replace('.py', '') for x in j)
            subprocess.Popen([sys.executable, '-B'] + j, cwd=ROOT, env=env,
                             stdout=open(logd / f'{name}.log', 'w'), stderr=subprocess.STDOUT)
            print(time.strftime('%H:%M'), 'start', ' '.join(j), flush=True)
            time.sleep(3); live = running()
        time.sleep(10)


def main(runs, parallel=2):
    import os
    env = dict(os.environ, **ENV)
    logd = ROOT / 'runs' / 'phase2_logs'; logd.mkdir(exist_ok=True)
    run_jobs([j for r in runs for j in stage1(r)], env, parallel, logd)        # all GT controllers first (cheap, Fig. 4)
    alpha_jobs, tail = stage2(runs)
    done = {r: 0 for r in runs}

    def keep(item):
        r, (a, j) = item
        if a is None: return True
        if time.strftime('%H:%M') >= CUT and done[r] < 4 and a not in CUT_SET: return False
        done[r] += 1; return True
    run_jobs(alpha_jobs + tail, env, parallel, logd, keep)
    print(time.strftime('%H:%M'), 'phase2 done', runs, flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    par = 2
    if a and a[0].startswith('--parallel='): par = int(a[0].split('=')[1]); a = a[1:]
    main(a, par)
