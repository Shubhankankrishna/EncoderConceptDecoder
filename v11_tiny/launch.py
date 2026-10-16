"""Run the 12 objective-test trainings on Metal, 3 at a time, longest first (dynamic queue).

  nohup python3 -B -m v11_tiny.launch > v11_tiny/runs/launch.log 2>&1 &
"""
import json, subprocess, sys, threading, time
from pathlib import Path
RUNS = Path(__file__).resolve().parent / 'runs'
ORDER = ['K1', 'X8', 'K1NB', 'NOAUX', 'ORD', 'C0']        # longest solo step time first (smokes)
JOBS = [(arm, seed) for arm in ORDER for seed in (1, 2)]
SLOTS = 3
lock = threading.Lock(); status = {}


def worker():
    while True:
        with lock:
            if not JOBS: return
            arm, seed = JOBS.pop(0)
        name = f'{arm}_s{seed}'; start = time.time()
        status[name] = dict(started=start)
        with open(RUNS / f'{name}.log', 'w') as log:
            code = subprocess.call([sys.executable, '-B', '-m', 'v11_tiny.train', '--arm', arm, '--seed', str(seed),
                                    '--device', 'mps', '--foreach', '--threads', '2', '--max-steps', '6000',
                                    '--run-dir', str(RUNS / name)], stdout=log, stderr=subprocess.STDOUT)
        status[name].update(finished=time.time(), seconds=time.time() - start, returncode=code)
        print(name, 'rc', code, 'seconds', round(time.time() - start), flush=True)
        (RUNS / 'launch_status.json').write_text(json.dumps(status, indent=1))


threads = [threading.Thread(target=worker) for _ in range(SLOTS)]
for t in threads: t.start(); time.sleep(5)
for t in threads: t.join()
(RUNS / 'launch.done').write_text(json.dumps(status, indent=1))
print('ALL DONE', flush=True)
