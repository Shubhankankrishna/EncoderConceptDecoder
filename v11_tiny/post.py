"""After-training pipeline (never more than 3 GPU processes together with the trainer launcher):
  1. evaluate each finished run in a free slot;
  2. when the launcher's 12 jobs are done, train K1SC_s1 (diagnostic: K1 with separate clip groups) and
     C0_s3 (third C0 seed), then evaluate them;
  3. idle 5 min, solo throughput smokes for all six arms (thermal state recorded before/after);
  4. idle 5 min, the 10M benchmark (solo);
  5. grad_share, training_summary, compare.

  nohup python3 -B -m v11_tiny.post > v11_tiny/runs/post.log 2>&1 &
"""
import json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent; RUNS = ROOT / 'runs'
ARMS = ('K1', 'X8', 'K1NB', 'NOAUX', 'ORD', 'C0'); SLOTS = 3
NAMES = [f'{a}_s{s}' for a in ARMS for s in (1, 2)]
EXTRA = ['K1SC_s1', 'C0_s3', 'ORDE_s1']   # K1SC = K1 with separate clip groups (diagnostic, seed 1); C0 third seed


def therm():
    try: return subprocess.run(['pmset', '-g', 'therm'], capture_output=True, text=True).stdout.strip().replace('\n', ' | ')
    except Exception as e: return repr(e)


def active_trainers():
    if (RUNS / 'launch.done').exists(): return 0
    p = RUNS / 'launch_status.json'
    st = json.loads(p.read_text()) if p.exists() else {}
    started = len(st); finished = sum('finished' in v for v in st.values())
    return (started - finished) if len(NAMES) - started == 0 else SLOTS


def py(*args, log):
    return subprocess.Popen([sys.executable, '-B', '-m', *args], stdout=open(log, 'w'), stderr=subprocess.STDOUT, cwd=ROOT.parent)


def train_cmd(name):
    arm, seed = name.split('_s')
    flags = ('--separate-clip',) if arm == 'K1SC' else ()
    return ('v11_tiny.train', '--arm', 'K1' if arm == 'K1SC' else arm, '--seed', seed, '--device', 'mps', '--foreach',
            '--threads', '2', '--max-steps', '6000', '--run-dir', str(RUNS / name), *flags)


evals, extra = {}, {}
while True:
    for k in [k for k, p in evals.items() if p.poll() is not None]:
        print(k, 'eval rc', evals.pop(k).returncode, time.strftime('%H:%M:%S'), flush=True)
    for k in [k for k, p in extra.items() if p.poll() is not None]:
        print(k, 'train rc', extra.pop(k).returncode, time.strftime('%H:%M:%S'), flush=True)
    free = SLOTS - active_trainers() - len(evals) - len(extra)
    if True:   # extras start as soon as the launcher has no pending jobs and a slot is free
        for n in EXTRA:
            if free > 0 and not (RUNS / n).exists() and n not in extra:
                extra[n] = py(*train_cmd(n), log=RUNS / f'{n}.log'); free -= 1
                print(n, 'train started', time.strftime('%H:%M:%S'), flush=True)
    todo = [n for n in NAMES + EXTRA if (RUNS / n / 'final.pt').exists() and not (RUNS / n / 'evaluation.json').exists()
            and n not in evals]
    for n in todo[:max(0, free)]:
        evals[n] = py('v11_tiny.evaluate', '--run-dir', str(RUNS / n), log=RUNS / f'{n}_eval.log')
        print(n, 'eval started', time.strftime('%H:%M:%S'), flush=True)
    if all((RUNS / n / 'evaluation.json').exists() for n in NAMES + EXTRA) and not evals and not extra: break
    time.sleep(20)
print('evaluations done', time.strftime('%H:%M:%S'), flush=True)
thermal = dict(before_idle=therm()); time.sleep(300); thermal['before_solo'] = therm()
solo = RUNS / 'solo'; solo.mkdir(exist_ok=True)
for arm in ARMS:
    if (solo / arm).exists(): continue
    p = py('v11_tiny.train', '--arm', arm, '--seed', '1', '--device', 'mps', '--foreach', '--threads', '2', '--max-steps', '300',
           '--eval-every', '100000', '--run-dir', str(solo / arm), log=solo / f'{arm}.log'); p.wait()
    print('solo', arm, p.returncode, time.strftime('%H:%M:%S'), flush=True)
thermal['after_solo'] = therm(); time.sleep(300); thermal['before_bench'] = therm()
p = py('v11_tiny.bench_10m', log=RUNS / 'bench_10m.log'); p.wait(); print('bench rc', p.returncode, time.strftime('%H:%M:%S'), flush=True)
thermal['after_bench'] = therm(); (RUNS / 'thermal.json').write_text(json.dumps(thermal, indent=1))
for mod in ('v11_tiny.grad_share', 'v11_tiny.training_summary', 'v11_tiny.compare'):
    p = py(mod, log=RUNS / f'{mod.split(".")[-1]}.log'); p.wait(); print(mod, p.returncode, flush=True)
(RUNS / 'post.done').write_text(time.strftime('%H:%M:%S'))
print('POST DONE', flush=True)
