"""V9 queue watchdog.

Every 15 minutes it checks the block drivers (Q2, Q3, Q4, Q5, Q6, QF) and keeps the queue moving:

  * a driver that died (crash, SIGKILL, reboot) while its block is running or waiting is relaunched with code/launch_block.py,
    which refuses unless the block's lock and its contract and smoke tests still match; a block whose training child is still
    alive is never relaunched (that would start a second copy of the same run);
  * a block whose status is 'failed' is resumed the same way, at most MAX_ATTEMPTS times per stage with RETRY_GAP between tries,
    then it is left alone and an alert is raised (a deterministic failure would otherwise loop forever);
  * a block that was stopped on purpose ('stopped_by_user', i.e. the driver received SIGTERM or SIGINT) is NOT resumed;
  * stalls are reported: a training stage whose logged step has not moved for NOPROGRESS_CYCLES checks, or a status file that
    stopped refreshing while a child is running. Nothing is killed;
  * swap pressure, low disk space and gaps in its own schedule (the Mac slept) are logged.

Resuming is safe because the drivers are resumable: completed stages are verified and skipped, and an interrupted training
stage restarts from its checkpoint (queue_base.Queue.train). The watchdog changes no code, lock, seed or setting.

Everything it writes is under v9_tiny/monitor/ (never under runs/):  status.json  watch.log  alerts.log  ALERT.txt  state.json  watch.pid
Stop it with:  touch v9_tiny/monitor/STOP   (or kill the pid in watch.pid).

    python3 watch.py --once        one check, print the summary, relaunch nothing unless --act is given
    python3 watch.py --once --act  one check with actions
    python3 watch.py --detach      start the 15-minute loop in the background (new session, sleep prevented)
"""
import argparse, calendar, json, os, re, shutil, signal, subprocess, sys, time
from pathlib import Path

V9 = Path(__file__).resolve().parents[1]
RUNS, CODE, TESTS, SPEC, MON = V9 / 'runs', V9 / 'code', V9 / 'tests', V9 / 'spec', V9 / 'monitor'
BLOCKS = ('Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'QF')            # run order on the one GPU (I13); Q1 is finished
INTERVAL = int(os.environ.get('V9_WATCH_INTERVAL', 900))
MAX_ATTEMPTS = 3
RETRY_GAP = 600
STALL_S = 1500
NOPROGRESS_CYCLES = 4
SWAP_WARN = 0.92
DISK_WARN_GB = 10
ALERT_DEBOUNCE = 3 * 3600
PROJECT_SCRIPT = re.compile(re.escape(str(CODE)) + r'/(?!launch_block|launch\.py)\S+\.py')


def utc(t=None):
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t if t is not None else time.time()))


def local(t=None):
    return time.strftime('%d %b %H:%M', time.localtime(t if t is not None else time.time()))


def epoch(s):
    try:
        return calendar.timegm(time.strptime(s, '%Y-%m-%dT%H:%M:%SZ'))
    except Exception:
        return None


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def write_json(p, obj):
    p = Path(p); tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_text(json.dumps(obj, indent=1, default=str)); os.replace(tmp, p)


def log(msg):
    MON.mkdir(exist_ok=True); lp = MON / 'watch.log'
    if lp.exists() and lp.stat().st_size > 5_000_000:
        lp.replace(MON / 'watch.log.1')
    with lp.open('a') as f:
        f.write(f'{utc()}  {msg}\n')


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0); return True
    except Exception:
        return False


def command_of(pid):
    r = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True)
    return r.stdout.strip()


def project_procs():
    """pid -> command line of every running process that executes a script from v9_tiny/code (drivers, trainers, measurers)."""
    out = {}
    r = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        line = line.strip(); pid, _, cmd = line.partition(' ')
        if PROJECT_SCRIPT.search(cmd) and 'watch.py' not in cmd and pid.isdigit() and int(pid) != os.getpid():
            out[int(pid)] = cmd.strip()
    return out


def swap_info():
    try:
        s = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True, timeout=3).stdout
        m = {k: float(v) for k, v in re.findall(r'(total|used|free) = ([\d.]+)M', s)}
        return dict(total_mb=m.get('total'), used_mb=m.get('used'))
    except Exception:
        return {}


def notify(title, text):
    try:
        subprocess.run(['osascript', '-e', f'display notification {json.dumps(text[:230])} with title {json.dumps(title)}'], capture_output=True, timeout=10)
    except Exception:
        pass


class Watch:
    def __init__(self, act):
        self.act = act; self.state = read_json(MON / 'state.json', {}) or {}
        self.state.setdefault('attempts', {}); self.state.setdefault('alerted', {}); self.state.setdefault('progress', {})
        self.events = []; self.level = 'OK'

    def raise_level(self, lvl):
        order = ['OK', 'INFO', 'WARN', 'ALERT']
        if order.index(lvl) > order.index(self.level):
            self.level = lvl

    def note(self, lvl, msg, alert_key=None):
        self.raise_level(lvl); self.events.append(f'{lvl}: {msg}'); log(f'{lvl}  {msg}')
        if lvl == 'ALERT' and alert_key:
            last = self.state['alerted'].get(alert_key, 0)
            if time.time() - last > ALERT_DEBOUNCE:
                self.state['alerted'][alert_key] = time.time()
                with (MON / 'alerts.log').open('a') as f:
                    f.write(f'{utc()}  {msg}\n')
                (MON / 'ALERT.txt').write_text(f'{utc()}\n{msg}\n')
                notify('V9 queue needs attention', msg)

    # ------------------------------------------------------------------ resume
    def relaunch(self, block, reason, stage):
        key = f'{block}:{stage}'; a = self.state['attempts'].get(key, {'n': 0, 't': 0})
        if a['n'] >= MAX_ATTEMPTS:
            self.note('ALERT', f'{block} stage {stage}: {a["n"]} resume attempts did not get past it. Not retrying again. Look at runs/{stage}/process.log and runs/failure_{block}.json.', alert_key=f'giveup:{key}')
            return False
        if time.time() - a['t'] < RETRY_GAP:
            self.note('INFO', f'{block}: waiting {RETRY_GAP // 60} min between resume attempts'); return False
        lj = read_json(RUNS / f'launch_{block}.json', {}); lt = epoch(lj.get('utc', '')) or 0
        if lj.get('pid') and alive(lj['pid']) and time.time() - lt < 180:
            self.note('INFO', f'{block}: launched {int(time.time() - lt)} s ago, giving it time to start'); return False
        if not self.act:
            self.note('INFO', f'{block}: would resume now ({reason}); dry run'); return False
        r = subprocess.run([sys.executable, str(CODE / 'launch_block.py'), block], cwd=V9, capture_output=True, text=True, timeout=120)
        text = (r.stdout + r.stderr).strip()
        if r.returncode != 0:
            if re.search(r'files changed since LOCK|has not passed under the current lock|already running', text):
                self.note('INFO', f'{block}: cannot be launched yet ({text.splitlines()[-1][:160]}). This does not count as a resume attempt.')
            else:
                self.state['attempts'][key] = dict(n=a['n'] + 1, t=time.time())
                self.note('ALERT', f'{block}: launch_block failed: {text[-300:]}', alert_key=f'launchfail:{key}')
            return False
        self.state['attempts'][key] = dict(n=a['n'] + 1, t=time.time())
        self.note('WARN', f'{block}: RESUMED ({reason}; attempt {a["n"] + 1} of {MAX_ATTEMPTS} for stage {stage}). {text[-120:]}')
        notify('V9 queue resumed', f'{block} restarted after: {reason}')
        return True

    def failure_reason(self, block, stage):
        tail = ''
        try:
            tail = (RUNS / stage / 'process.log').read_text()[-6000:]
        except Exception:
            pass
        if re.search(r'out of memory|MPS backend out of memory', tail, re.I):
            return 'MPS out of memory'
        if re.search(r'Killed|SIGKILL|terminated', tail):
            return 'process killed'
        err = (read_json(RUNS / f'failure_{block}.json', {}) or {}).get('error', '')
        return (err or 'error')[:140]

    # ------------------------------------------------------------------ one block
    def check_block(self, block, procs):
        st = read_json(RUNS / f'status_{block}.json')
        if not st:
            self.note('INFO', f'{block}: no status file yet'); return None
        s, stage = st.get('status'), st.get('stage')
        for k in [k for k in self.state['attempts'] if k.startswith(block + ':') and k != f'{block}:{stage}']:
            del self.state['attempts'][k]          # the block got past the stage it used to fail in
        row = dict(status=s, stage=stage, driver_alive=alive(st.get('driver_pid')), updated_utc=st.get('updated_utc'))
        if s == 'completed':
            return row
        child = st.get('child_pid'); child_alive = bool(child and alive(child) and PROJECT_SCRIPT.search(command_of(child)))
        row['child_alive'] = child_alive
        live = st.get('live') or {}
        if live.get('step'):
            row['progress'] = f'{live["step"]}/{live.get("planned_steps")}'
            if live.get('elapsed_s') and live['step']:
                row['eta_min'] = round((live['planned_steps'] - live['step']) * live['elapsed_s'] / live['step'] / 60)
        if s == 'stopped_by_user':
            self.note('INFO', f'{block}: was stopped on purpose (SIGTERM/SIGINT). Not resuming it.'); return row
        if s == 'failed':
            why = self.failure_reason(block, stage); row['failure'] = why
            self.note('WARN', f'{block}: stage {stage} FAILED ({why})')
            self.relaunch(block, f'{stage} failed: {why}', stage); return row
        if s in ('running', 'waiting'):
            if not row['driver_alive']:
                if child_alive:
                    self.note('WARN', f'{block}: driver is dead but its training child {child} is still running. Not relaunching (it would start a duplicate); will check again.'); return row
                self.note('WARN', f'{block}: driver {st.get("driver_pid")} is gone while status was {s} ({stage})')
                self.relaunch(block, f'driver died during {stage}', stage); return row
            age = time.time() - (epoch(st.get('updated_utc', '')) or time.time())
            if s == 'running' and child_alive and age > STALL_S:
                self.note('ALERT', f'{block}: status file not refreshed for {int(age / 60)} min while {stage} is running (driver may be hung).', alert_key=f'stale:{block}')
            # training progress between checks
            if s == 'running' and child_alive and live.get('step') is not None:
                p = self.state['progress'].get(block, {}); mark = (stage, live.get('step'), live.get('elapsed_s'))
                if p.get('mark') == list(mark):
                    p['same'] = p.get('same', 0) + 1
                else:
                    p = dict(mark=list(mark), same=0)
                self.state['progress'][block] = p
                if p['same'] >= NOPROGRESS_CYCLES:
                    self.note('ALERT', f'{block}: {stage} has not logged a new step for {p["same"] * INTERVAL // 60} min (step {live.get("step")}).', alert_key=f'noprog:{block}:{stage}')
        return row

    # ------------------------------------------------------------------ whole cycle
    def cycle(self):
        t0 = time.time(); last = self.state.get('last_cycle')
        if last and t0 - last > 2 * INTERVAL + 120:
            self.note('WARN', f'monitor was not running for {int((t0 - last) / 60)} min (the Mac slept, or the monitor was stopped)')
        self.state['last_cycle'] = t0
        procs = project_procs(); rows = {}
        for b in BLOCKS:
            rows[b] = self.check_block(b, procs)
        active = next((b for b in BLOCKS if rows.get(b) and rows[b].get('status') != 'completed'), None)
        sw = swap_info()
        if sw.get('total_mb') and sw['used_mb'] / sw['total_mb'] > SWAP_WARN:
            self.note('WARN', f'swap is {sw["used_mb"]:.0f} of {sw["total_mb"]:.0f} MB used (close other apps; MPS memory errors get likelier)')
        free_gb = shutil.disk_usage(V9).free / 1e9
        if free_gb < DISK_WARN_GB:
            self.note('ALERT', f'only {free_gb:.1f} GB of disk left', alert_key='disk')
        a = rows.get(active) if active else None
        if active is None:
            head = 'all blocks completed'
        elif a:
            head = f'{active} {a["status"]} · {a["stage"]}' + (f' · step {a["progress"]}' if a.get('progress') else '') + (f' · about {a["eta_min"]} min left in this stage' if a.get('eta_min') is not None else '')
        else:
            head = f'{active}: no status yet'
        summary = f'[{self.level}] {head}'
        snap = dict(checked_utc=utc(t0), checked_local=local(t0), level=self.level, summary=summary, active_block=active, blocks=rows, events=self.events[-12:],
                    swap=sw, disk_free_gb=round(free_gb, 1), project_processes=len(procs), next_check_local=local(t0 + INTERVAL), interval_s=INTERVAL, acting=self.act,
                    attempts=self.state['attempts'])
        write_json(MON / 'status.json', snap); write_json(MON / 'state.json', self.state)
        log(f'CHECK  {summary}  swap {sw.get("used_mb", 0):.0f}/{sw.get("total_mb", 0):.0f} MB  disk {free_gb:.0f} GB')
        return snap


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--once', action='store_true'); ap.add_argument('--act', action='store_true'); ap.add_argument('--detach', action='store_true')
    ap.add_argument('--loop', action='store_true', help=argparse.SUPPRESS); a = ap.parse_args()
    MON.mkdir(exist_ok=True)
    if a.detach:
        pf = read_json(MON / 'watch.pid', {})
        if pf.get('pid') and alive(pf['pid']) and 'watch.py' in command_of(pf['pid']):
            print(f'monitor already running (pid {pf["pid"]})'); return
        (MON / 'STOP').unlink(missing_ok=True)
        lg = open(MON / 'watch.out', 'a')
        p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--loop', '--act'], cwd=V9, stdout=lg, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        subprocess.Popen(['caffeinate', '-is', '-w', str(p.pid)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        write_json(MON / 'watch.pid', dict(pid=p.pid, started_utc=utc())); print(json.dumps(dict(monitor_pid=p.pid, interval_s=INTERVAL))); return
    if a.once:
        snap = Watch(a.act).cycle(); print(json.dumps(dict(summary=snap['summary'], events=snap['events'], swap=snap['swap']), indent=1)); return
    # loop mode
    write_json(MON / 'watch.pid', dict(pid=os.getpid(), started_utc=utc())); stop = {'v': False}
    for sg in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sg, lambda *_: stop.update(v=True))
    log(f'MONITOR STARTED pid {os.getpid()} interval {INTERVAL}s act={a.act}')
    while not stop['v'] and not (MON / 'STOP').exists():
        try:
            Watch(a.act).cycle()
        except Exception as e:
            import traceback; log('MONITOR ERROR ' + traceback.format_exc().replace('\n', ' | ')[:1500])
        t_end = time.time() + INTERVAL
        while time.time() < t_end and not stop['v'] and not (MON / 'STOP').exists():
            time.sleep(5)
    log('MONITOR STOPPED')


if __name__ == '__main__':
    main()
