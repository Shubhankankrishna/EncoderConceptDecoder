"""Replacement after-training pipeline. Slot use is counted from live processes (pgrep), so a restart can
never oversubscribe the GPU (max 3). Queue:
  trainers NOAUXSC_s1, K1NBSC_s1 (NOAUX / K1NB with --separate-clip; seed 1);
  evaluations for every run with final.pt and no evaluation.json;
  then idle 3 min -> solo throughput set -> idle 3 min -> 10M bench -> offsets_extra -> grad_share
  -> training_summary -> compare.

  nohup python3 -B -m v11_tiny.post2 > v11_tiny/runs/post2.log 2>&1 &
"""
import json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent; RUNS = ROOT / 'runs'; SLOTS = 3
TRAIN = [('NOAUXSC_s1', 'NOAUX'), ('K1NBSC_s1', 'K1NB')]
ARMS = ('K1', 'X8', 'K1NB', 'NOAUX', 'ORD', 'C0')


def live():
    out = subprocess.run(['pgrep', '-f', r'v11_tiny\.(train|evaluate|bench_10m|offsets_extra)'], capture_output=True, text=True).stdout
    return len([l for l in out.split() if l.strip()])


def therm():
    return subprocess.run(['pmset', '-g', 'therm'], capture_output=True, text=True).stdout.strip().replace('\n', ' | ')


def py(*args, log):
    return subprocess.Popen([sys.executable, '-B', '-m', *args], stdout=open(log, 'w'), stderr=subprocess.STDOUT, cwd=ROOT.parent)


procs = {}
def runs_with_final():
    return [p.name for p in RUNS.iterdir() if p.is_dir() and p.name != 'solo' and (p / 'final.pt').exists()]

while True:
    for k in [k for k, p in procs.items() if p.poll() is not None]:
        print(k, 'rc', procs.pop(k).returncode, time.strftime('%H:%M:%S'), flush=True)
    for name, arm in TRAIN:
        if not (RUNS / name).exists() and live() < SLOTS:
            procs[name + ':train'] = py('v11_tiny.train', '--arm', arm, '--seed', '1', '--device', 'mps', '--foreach', '--threads', '2',
                                        '--max-steps', '6000', '--separate-clip', '--run-dir', str(RUNS / name), log=RUNS / f'{name}.log')
            print(name, 'train started', time.strftime('%H:%M:%S'), flush=True); time.sleep(3)
    running_evals = subprocess.run(['pgrep', '-fl', r'v11_tiny\.evaluate'], capture_output=True, text=True).stdout
    for n in sorted(runs_with_final()):
        if (RUNS / n / 'evaluation.json').exists() or n in running_evals or n + ':eval' in procs: continue
        if live() >= SLOTS: break
        procs[n + ':eval'] = py('v11_tiny.evaluate', '--run-dir', str(RUNS / n), log=RUNS / f'{n}_eval.log')
        print(n, 'eval started', time.strftime('%H:%M:%S'), flush=True); time.sleep(3)
    trained = all((RUNS / n / 'final.pt').exists() for n, _ in TRAIN)
    if trained and live() == 0 and all((RUNS / n / 'evaluation.json').exists() for n in runs_with_final()): break
    time.sleep(20)
print('evaluations done', time.strftime('%H:%M:%S'), flush=True)
thermal = dict(before_idle=therm()); time.sleep(180); thermal['before_solo'] = therm()
solo = RUNS / 'solo'; solo.mkdir(exist_ok=True)
for arm in ARMS:
    if (solo / arm).exists(): continue
    p = py('v11_tiny.train', '--arm', arm, '--seed', '1', '--device', 'mps', '--foreach', '--threads', '2', '--max-steps', '300',
           '--eval-every', '100000', '--run-dir', str(solo / arm), log=solo / f'{arm}.log'); p.wait()
    print('solo', arm, p.returncode, time.strftime('%H:%M:%S'), flush=True)
thermal['after_solo'] = therm(); time.sleep(180); thermal['before_bench'] = therm()
p = py('v11_tiny.bench_10m', log=RUNS / 'bench_10m.log'); p.wait(); print('bench rc', p.returncode, time.strftime('%H:%M:%S'), flush=True)
thermal['after_bench'] = therm(); (RUNS / 'thermal.json').write_text(json.dumps(thermal, indent=1))
for mod in ('v11_tiny.offsets_extra', 'v11_tiny.grad_share', 'v11_tiny.training_summary', 'v11_tiny.compare'):
    p = py(mod, log=RUNS / f'{mod.split(".")[-1]}.log'); p.wait(); print(mod, p.returncode, time.strftime('%H:%M:%S'), flush=True)
(RUNS / 'post.done').write_text(time.strftime('%H:%M:%S'))
print('POST DONE', flush=True)
