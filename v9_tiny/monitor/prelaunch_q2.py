"""One-off chain that finishes the I21 restart of Q2, unattended:
   1. wait for the scratch S6 pilot (production shape, edited s6.py); it must finish and write its estimate;
   2. check that the Q2 contract tests passed under the current lock (they were rerun after the edit);
   3. run the Q2 smoke test under that lock (the launcher refuses without it);
   4. launch the Q2 block with code/launch_block.py.
It stops at the first failure and writes an alert; nothing after a failed step runs. After step 4 the watchdog (watch.py) keeps Q2 going."""
import json, os, subprocess, sys, time
from pathlib import Path
V9 = Path(__file__).resolve().parents[1]; CODE, TESTS, SPEC, MON = V9 / 'code', V9 / 'tests', V9 / 'spec', V9 / 'monitor'
SCRATCH = Path(sys.argv[1]); PILOT_PID = int(sys.argv[2])
LOG = MON / 'prelaunch_q2.log'


def log(m):
    with LOG.open('a') as f:
        f.write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + '  ' + m + '\n')


def alive(pid):
    try:
        os.kill(pid, 0); return True
    except Exception:
        return False


def fail(msg):
    log('STOPPED: ' + msg); (MON / 'ALERT.txt').write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + '\nQ2 pre-launch chain stopped: ' + msg + '\n')
    subprocess.run(['osascript', '-e', 'display notification "Q2 pre-launch stopped: see v9_tiny/monitor/prelaunch_q2.log" with title "V9 queue needs attention"'], capture_output=True)
    sys.exit(1)


log(f'chain started; waiting for the scratch pilot (pid {PILOT_PID})')
while alive(PILOT_PID):
    time.sleep(20)
pilot = SCRATCH / 'pilot_scratch.json'
if not pilot.exists():
    tail = (SCRATCH / 'pilot.log').read_text()[-800:] if (SCRATCH / 'pilot.log').exists() else ''
    fail('the scratch pilot did not finish (no pilot_scratch.json). Tail: ' + tail.replace('\n', ' | ')[-500:])
est = json.loads(pilot.read_text()).get('estimate')
if not est:
    fail('scratch pilot output has no estimate')
log(f'scratch pilot passed: estimate q2_total_h={est.get("q2_total_h"):.1f}, per_lsl_phase_s={est.get("per_lsl_phase_s"):.0f}, per_other_phase_s={est.get("per_other_phase_s"):.0f}')
lock = json.loads((SPEC / 'LOCK_Q2.json').read_text())['lock_sha256']; c = json.loads((TESTS / 'q2_contracts.json').read_text())
if c.get('status') != 'passed' or c.get('lock_sha256') != lock:
    fail('q2_contracts.json has not passed under the current LOCK_Q2')
log('contracts passed under the current lock; running the Q2 smoke test (about an hour or more)')
t0 = time.time(); r = subprocess.run([sys.executable, str(CODE / 'q2_smoke.py')], cwd=V9, capture_output=True, text=True, timeout=4 * 3600)
log(f'smoke exit {r.returncode} after {int(time.time() - t0)} s: {(r.stdout + r.stderr)[-300:]!r}')
s = json.loads((TESTS / 'q2_smoke.json').read_text())
if r.returncode != 0 or s.get('status') != 'passed' or s.get('lock_sha256') != lock:
    fail('q2_smoke did not pass under the current lock. See tests/q2_smoke.json')
log('smoke passed; launching Q2')
r = subprocess.run([sys.executable, str(CODE / 'launch_block.py'), 'Q2'], cwd=V9, capture_output=True, text=True, timeout=120)
log(f'launch_block exit {r.returncode}: {(r.stdout + r.stderr).strip()[-300:]!r}')
if r.returncode != 0 and 'already running' not in (r.stdout + r.stderr):
    fail('launch_block refused: ' + (r.stdout + r.stderr).strip()[-300:])
log('DONE: Q2 is launched. The watchdog takes over.')
