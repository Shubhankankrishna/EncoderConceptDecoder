#!/usr/bin/env python3
"""V14 Observatory. Standard-library, read-only experiment collector and LAN server.

Never imports the model, torch, or pickle, or opens checkpoint contents. Runtime
history belongs to this service; experiment files are only read.
"""
import argparse
import csv
import datetime as dt
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import sqlite3
import statistics
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IST = ZoneInfo('Asia/Kolkata')
FAMILIES = ['grid', 'prog', 'state', 'plan', 'text']
STAGES = [('final_eval', 'Base evaluation'), ('diagnostics', 'Concept diagnostics'),
          ('era', 'Target alignment / era refit'), ('cl_choice', 'Choose continual method'),
          ('cl_seq', 'Sequential learning'), ('cl_v14', 'V14 continual learning'),
          ('cl_replay', 'Replay learning'), ('cl_eval', 'Continual evaluation'), ('report', 'Result tables')]
ARTIFACTS = {'evaluation': 'runs/eval/base/results.json', 'diagnostics': 'runs/eval/diag/diagnostics_v14.json',
             'continual': 'runs/eval/cl/cl_results.json', 'era': 'runs/base/era.json',
             'choice': 'runs/cl/choice.json', 'configuration': 'runs/base/config.json',
             'checkpoint_choice': 'runs/base/ckpt_choice.json', 'protocol': 'REPORT.md',
             'report': 'REPORT_TABLES.md', 'deviations': 'DEVIATIONS.md'}
LOGS = {'training': 'runs/base/train.out', 'launch': 'runs/base/launch.log',
        'restarts': 'runs/base/restarts.log', 'pipeline': 'runs/post/status.log',
        **{k: 'runs/post/' + k + '.out' for k, _ in STAGES}}


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def command(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=4).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ''


def stamp(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def tail(path, limit=48000):
    try:
        with path.open('rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - limit))
            data = f.read(limit).decode('utf-8', errors='replace')
            return data.split('\n', 1)[-1] if size > limit else data
    except OSError:
        return ''


class JsonReader:
    """Retain the previous complete document during a non-atomic writer update."""
    def __init__(self):
        self.cache = {}
        self.pending = set()

    def read(self, path):
        key = str(path)
        try:
            st = path.stat()
        except FileNotFoundError:
            self.cache.pop(key, None)
            self.pending.discard(key)
            return None
        version = (st.st_mtime_ns, st.st_size)
        old = self.cache.get(key)
        if old and old[0] == version:
            return old[1]
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError):
            self.pending.add(key)
            return old[1] if old else None
        self.pending.discard(key)
        self.cache[key] = (version, value)
        return value


class JsonlReader:
    """Incremental complete-line reader, with step rollback and replacement support."""
    def __init__(self, path):
        self.path = path
        self.inode = None
        self.offset = 0
        self.pending = b''
        self.rows = []
        self.invalid = 0

    def read(self):
        try:
            st = self.path.stat()
        except FileNotFoundError:
            self.rows, self.pending, self.offset, self.inode = [], b'', 0, None
            return []
        if st.st_ino != self.inode or st.st_size < self.offset:
            self.rows, self.pending, self.offset, self.invalid = [], b'', 0, 0
            self.inode = st.st_ino
        with self.path.open('rb') as f:
            f.seek(self.offset)
            data = self.pending + f.read()
            self.offset = f.tell()
        lines = data.split(b'\n')
        self.pending = lines.pop()
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('not an object')
            except (ValueError, UnicodeDecodeError):
                self.invalid += 1
                continue
            # A restart can roll back to an earlier checkpoint. Discard the old
            # branch's later observations instead of showing impossible progress.
            if isinstance(row.get('step'), (int, float)):
                while self.rows and self.rows[-1].get('step', -1) >= row['step']:
                    self.rows.pop()
            self.rows.append(row)
        return self.rows


def median(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
    return statistics.median(vals) if vals else None


def sample(rows, count=1000):
    if len(rows) <= count:
        return rows
    return [rows[round(i * (len(rows) - 1) / (count - 1))] for i in range(count)]


def parse_date(text):
    m = re.match(r'^\w{3} (\w{3}) +(\d+) (\d\d:\d\d:\d\d) \S+ (\d{4}) (.*)$', text)
    if not m:
        return None, text
    try:
        when = dt.datetime.strptime(' '.join([m[1], m[2], m[4], m[3]]), '%b %d %Y %H:%M:%S')
        return when.replace(tzinfo=IST).timestamp(), m[5]
    except ValueError:
        return None, text


def pipeline(root, active):
    stages = {k: dict(id=k, name=name, state='pending', start=None, end=None, exit_code=None) for k, name in STAGES}
    events = []
    for line in tail(root / 'runs/post/status.log', 200000).splitlines():
        t, message = parse_date(line)
        events.append(dict(time=t, message=message))
        bits = message.split()
        if len(bits) >= 2 and bits[1] in stages:
            s = stages[bits[1]]
            if bits[0] == 'start':
                s.update(state='running', start=t, end=None, exit_code=None)
            elif bits[0] == 'end' and len(bits) >= 4:
                try:
                    code = int(bits[3])
                except ValueError:
                    continue
                s.update(state='complete' if code == 0 else 'failed', end=t, exit_code=code)
    for s in stages.values():
        if s['state'] == 'running' and not active:
            s['state'] = 'interrupted'
    return dict(stages=list(stages.values()), events=events[-40:],
                finished=(root / 'runs/post/ALL_DONE').exists(), armed=active)


def processes(root):
    rows = []
    names = {'train.py': 'Base training', 'run_base_v14.sh': 'Training supervisor',
             'post_base_v14.sh': 'Pipeline supervisor', 'final_eval.py': 'Base evaluation',
             'diagnostics_v14.py': 'Concept diagnostics', 'era_v14.py': 'Era refit',
             'cl.py': 'Continual learning', 'cl_eval.py': 'Continual evaluation',
             'make_report.py': 'Report', 'cl_choice.py': 'Continual selection'}
    for line in command(['/bin/ps', '-axo', 'pid=,ppid=,pcpu=,rss=,etime=,command=']).splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6:
            continue
        try:
            args = shlex.split(fields[5])
        except ValueError:
            continue
        # Only executable/script positions, never text inside a shell command.
        if not args or ('python' not in Path(args[0]).name and Path(args[0]).name not in ('zsh', 'bash', 'sh')):
            continue
        script = next((a for a in args[1:3] if Path(a).name in names), None)
        if script is None:
            continue
        pid = int(fields[0])
        cwd = command(['/usr/sbin/lsof', '-a', '-p', str(pid), '-d', 'cwd', '-Fn'])
        if 'n' + str(root.resolve()) not in cwd.splitlines():
            continue
        role = names[Path(script).name]
        if '--quick' in args or '--smoke' in args:
            role = 'Auxiliary test · ' + role.lower()
        rows.append(dict(pid=pid, role=role, script=Path(script).name,
                         auxiliary=('--quick' in args or '--smoke' in args),
                         cpu=float(fields[2]), memory_mb=int(fields[3]) / 1024, elapsed=fields[4]))
    return rows


def machine(root):
    info = dict(time=time.time(), disk_free_gb=shutil.disk_usage(root).free / 2**30)
    batt = command(['/usr/bin/pmset', '-g', 'batt'])
    info['power'] = 'AC power' if "'AC Power'" in batt else 'Battery' if "'Battery Power'" in batt else 'Unavailable'
    match = re.search(r'(\d+)%', batt)
    info['battery_pct'] = int(match[1]) if match else None
    swap = re.search(r'used = ([\d.]+)M', command(['/usr/sbin/sysctl', '-n', 'vm.swapusage']))
    info['swap_gb'] = float(swap[1]) / 1024 if swap else None
    pressure = re.search(r'free percentage: (\d+)%', command(['/usr/bin/memory_pressure']))
    info['memory_free_pct'] = int(pressure[1]) if pressure else None
    assertions = command(['/usr/bin/pmset', '-g', 'assertions'])
    sleep = re.search(r'PreventUserIdleSystemSleep\s+(\d+)', assertions)
    info['sleep_prevented'] = bool(int(sleep[1])) if sleep else None
    io = command(['/usr/sbin/ioreg', '-r', '-d', '1', '-w', '0', '-c', 'IOAccelerator'])
    gpu = re.search(r'"Device Utilization %"=(\d+)', io)
    info['gpu_pct'] = int(gpu[1]) if gpu else None
    info['load1'] = os.getloadavg()[0]
    ip = command(['/usr/sbin/ipconfig', 'getifaddr', 'en0']).strip()
    if not ip:
        ip = command(['/usr/sbin/ipconfig', 'getifaddr', 'en1']).strip()
    info['lan_ip'] = ip or None
    return info


class Engine:
    def __init__(self, root=ROOT, runtime=None, every=10, port=8714):
        self.root = root.resolve()
        self.runtime = runtime or HERE / 'runtime'
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.every, self.port = every, port
        self.reader = JsonReader()
        self.train = JsonlReader(self.root / 'runs/base/log.jsonl')
        self.cl = {m: JsonlReader(self.root / ('runs/cl/' + m + '/log.jsonl')) for m in ('seq', 'v14', 'replay')}
        self.lock = threading.Lock()
        self.state = None
        self.error = None
        self.machine = {}
        self.procs = []
        self.last_system = 0
        self.stop = threading.Event()

    def collect(self, system=True):
        now = time.time()
        if system and now - self.last_system >= 30:
            self.machine, self.procs = machine(self.root), processes(self.root)
            self.last_system = now
        artifacts = {k: self.reader.read(self.root / v) for k, v in ARTIFACTS.items() if v.endswith('.json')}
        cfg = artifacts['configuration'] or {}
        rows = self.train.read()
        last = rows[-1] if rows else {}
        recent = [r for r in rows if r.get('step', 0) > 60][-20:]
        run = self.root / 'runs/base'
        done, fallback = (run / 'DONE').exists(), (run / 'DONE_FALLBACK').exists()
        log_time = stamp(run / 'log.jsonl')
        age = max(0, now - log_time) if log_time else None
        scripts = {p['script'] for p in self.procs if not p.get('auxiliary')}
        train_active = 'train.py' in scripts
        post = pipeline(self.root, 'post_base_v14.sh' in scripts)
        hours = cfg.get('hours', 11)
        elapsed = last.get('h', 0)
        stale_after = max(300, (median(recent, 's_step') or 3) * 50 * 3)
        status = ('complete' if done else 'fallback' if fallback else
                  'stale' if train_active and age is not None and age > stale_after else
                  'training' if train_active else 'interrupted' if rows else 'waiting')
        eta = log_time + max(0, hours - elapsed) * 3600 if status == 'training' and log_time else None
        try:
            start = dt.datetime.strptime(cfg['started'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=IST).timestamp()
        except (KeyError, ValueError):
            start = None
        families = {}
        for fam in FAMILIES:
            records = [r for r in rows if r.get('fam') == fam]
            families[fam] = dict(latest=records[-1] if records else None,
                                 median_ntp=median(records[-5:], 'ntp'), observations=len(records),
                                 series=sample(records, 400))
        restart_text = tail(run / 'restarts.log')
        restarts = len(re.findall(r'restart \d+', restart_text))
        ft = self.reader.read(run / 'family_tokens.json') or {}
        base = dict(status=status, done=done, fallback=fallback, last=last, config=cfg,
                    logged_at=log_time, age_s=age, stale_after_s=stale_after, started_at=start,
                    progress=min(1, elapsed / hours) if hours else 0, eta=eta,
                    deadline=start + 24 * 3600 if start else None, restarts=restarts,
                    throughput=median(recent, 'tok_s'), step_seconds=median(recent, 's_step'),
                    average_tok_s=last.get('tokens', 0) / (elapsed * 3600) if elapsed else None,
                    peak_memory_gb=max((r.get('mem_gb', 0) for r in rows), default=None),
                    family_tokens=ft, family_tokens_at=stamp(run / 'family_tokens.json'),
                    checkpoint_at=stamp(run / 'latest.pt'), families=families,
                    snapshots=[dict(name=p.name, percent=int(p.stem.split('_')[1]), time=stamp(p)) for p in sorted(run.glob('snap_*.pt')) if p.stem.split('_')[-1].isdigit()],
                    series=sample(rows), records=len(rows), invalid_lines=self.train.invalid,
                    cap=next((r for r in reversed(rows) if r.get('cap')), None))
        alerts = []
        def alert(level, title, detail):
            alerts.append(dict(level=level, title=title, detail=detail))
        if status in ('stale', 'interrupted'):
            alert('critical', 'Training needs attention', 'No matching training process is running.' if status == 'interrupted' else 'The process is alive, but training metrics have stopped updating.')
        if fallback:
            alert('warning', 'Fallback checkpoint', 'The pipeline is using a fallback checkpoint; the planned base run did not finish normally.')
        if restarts:
            alert('info', str(restarts) + ' historical restarts', 'Recorded by the launcher. Current status is checked separately; inspect the restart log for context.')
        if last.get('skipped', 0):
            alert('warning', 'Skipped updates', str(last['skipped']) + ' updates were skipped for non-finite gradients.')
        if not post['armed'] and not post['finished']:
            alert('warning', 'Post-training supervisor unavailable', 'The queued evaluation pipeline is not currently observed running.')
        for stage in post['stages']:
            if stage['state'] in ('failed', 'interrupted'):
                alert('critical', stage['name'] + ' ' + stage['state'], 'Inspect this stage’s log. Later stages may still run after a failure.')
        if self.machine.get('power') == 'Battery':
            alert('warning', 'Running on battery', 'Connect power to sustain the experiment.')
        if self.machine.get('disk_free_gb', 100) < 5:
            alert('critical', 'Low disk space', 'Less than 5 GB remains for checkpoints and evaluation outputs.')
        if (self.machine.get('swap_gb') or 0) > 4:
            alert('warning', 'Memory pressure', 'Swap exceeds 4 GB; inspect throughput for slowdowns.')
        if self.reader.pending:
            alert('info', 'A result file is being updated', 'Showing the last complete version until the writer finishes.')
        if self.train.invalid:
            alert('warning', 'Malformed log records', str(self.train.invalid) + ' complete lines could not be parsed and were excluded.')
        files = [dict(id=k, name=Path(v).name, path=v, updated=stamp(self.root / v),
                      available=(self.root / v).is_file()) for k, v in ARTIFACTS.items()]
        current_stage = next((s['name'] for s in post['stages'] if s['state'] == 'running'), None)
        phase = 'Base training' if status in ('training', 'stale') else current_stage or (
            'Finished with issues' if post['finished'] and any(s['state'] != 'complete' for s in post['stages']) else
            'Experiment complete' if post['finished'] else 'Waiting for next stage' if done or fallback else 'Training interrupted' if rows else 'Waiting for training')
        return clean(dict(generated=now, interval=self.every, phase=phase, base=base, pipeline=post,
                          alerts=alerts, machine=self.machine, processes=self.procs, artifacts=artifacts, files=files,
                          continual={m: r.read() for m, r in self.cl.items()},
                          connection=dict(local='http://localhost:' + str(self.port),
                              lan=('http://' + self.machine['lan_ip'] + ':' + str(self.port)) if self.machine.get('lan_ip') else None),
                          provenance='Live batch metrics are provisional. Final verdicts come from saved evaluation artifacts. One seed; seed-unreplicated.'))

    def run(self):
        db = sqlite3.connect(str(self.runtime / 'history.sqlite3'))
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE IF NOT EXISTS samples (time REAL PRIMARY KEY, payload TEXT NOT NULL)')
        last_saved = 0
        while not self.stop.is_set():
            start = time.monotonic()
            try:
                state = self.collect()
                if state['generated'] - last_saved >= 30:
                    record = dict(state['machine'], time=state['generated'], step=state['base']['last'].get('step'),
                                  throughput=state['base']['throughput'])
                    db.execute('INSERT OR REPLACE INTO samples VALUES (?, ?)', (record['time'], json.dumps(record)))
                    db.execute('DELETE FROM samples WHERE time < ?', (state['generated'] - 7 * 86400,))
                    db.commit()
                    last_saved = state['generated']
                history = [json.loads(r[0]) for r in db.execute('SELECT payload FROM samples ORDER BY time DESC LIMIT 720')]
                state['history'] = list(reversed(history))
                with self.lock:
                    self.state, self.error = state, None
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
                print('Collector error:', repr(exc), flush=True)
            self.stop.wait(max(1, self.every - (time.monotonic() - start)))
        db.close()

    def snapshot(self):
        with self.lock:
            return dict(self.state, collector_error=self.error) if self.state else None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if args and str(args[1] if len(args) > 1 else '') not in ('200', '304'):
            super().log_message(fmt, *args)

    def send(self, body, content_type, status=200, attachment=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        if attachment:
            self.send_header('Content-Disposition', 'attachment; filename="' + attachment + '"')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        engine = self.server.engine
        if url.path in ('/api/state', '/api/export'):
            state = engine.snapshot()
            self.send(json.dumps(state or {'error': 'Collector is starting'}), 'application/json', 200 if state else 503,
                      'v14-snapshot.json' if url.path == '/api/export' else None)
        elif url.path == '/api/log':
            name = query.get('name', ['training'])[0]
            if name not in LOGS:
                return self.send('Unknown log', 'text/plain', 404)
            p = engine.root / LOGS[name]
            self.send(json.dumps(dict(name=name, path=LOGS[name], updated=stamp(p), exists=p.is_file(), text=tail(p))), 'application/json')
        elif url.path == '/api/artifact':
            key = query.get('id', [''])[0]
            if key not in ARTIFACTS:
                return self.send('Unknown artifact', 'text/plain', 404)
            p = engine.root / ARTIFACTS[key]
            try:
                body = p.read_bytes()
            except OSError:
                return self.send('Not available yet', 'text/plain', 404)
            self.send(body, 'application/octet-stream', attachment=p.name)
        elif url.path == '/api/training.csv':
            state = engine.snapshot()
            if not state:
                return self.send('Starting', 'text/plain', 503)
            # Export the same restart-aware observations independently of the collector.
            rows = JsonlReader(engine.root / 'runs/base/log.jsonl').read()
            fields = ['step', 'h', 'fam', 'ntp', 'ncp', 'tok_s', 's_step', 'wait_s', 'lr', 'lam', 'mu', 'gn', 'mem_gb', 'tokens', 'skipped', 'same4', 'sig_a', 'sig_b', 'sig_bu', 'lam_scale']
            out = io.StringIO()
            writer = csv.DictWriter(out, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(clean(rows))
            self.send(out.getvalue(), 'text/csv', attachment='v14-training.csv')
        elif url.path in ('/', '/index.html', '/app.js', '/style.css'):
            path = HERE / 'static' / ('index.html' if url.path == '/' else url.path[1:])
            mime = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8'}[path.suffix]
            self.send(path.read_bytes(), mime)
        elif url.path == '/favicon.ico':
            self.send(b'', 'image/x-icon', 204)
        else:
            self.send('Not found', 'text/plain', 404)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8714)
    parser.add_argument('--every', type=int, default=10)
    args = parser.parse_args()
    os.nice(10)
    engine = Engine(every=max(5, args.every), port=args.port)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.engine = engine
    thread = threading.Thread(target=engine.run, daemon=True)
    thread.start()
    def stop(*_):
        engine.stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print('V14 Observatory listening on ' + args.host + ':' + str(args.port), flush=True)
    try:
        server.serve_forever()
    finally:
        engine.stop.set()
        thread.join(timeout=20)
        server.server_close()


if __name__ == '__main__':
    main()
