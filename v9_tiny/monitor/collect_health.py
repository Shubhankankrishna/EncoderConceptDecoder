"""Read queue/process/lock health and append a durable record. Never restart jobs."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

V9 = Path(__file__).resolve().parents[1]
MON = V9 / 'monitor'


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def files_hash(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def command(args):
    return subprocess.check_output(args, text=True).strip()


def collect(action, notification):
    procs = {}
    for line in command(['ps', '-axo', 'pid=,ppid=,state=,pcpu=,etime=,command=']).splitlines():
        p = line.strip().split(None, 5)
        if len(p) == 6 and p[0].isdigit():
            procs[int(p[0])] = dict(ppid=int(p[1]), state=p[2], cpu=p[3], elapsed=p[4], command=p[5])
    blocks = {}
    for path in sorted((V9 / 'runs').glob('status*.json')):
        d = read(path)
        if not d:
            blocks[path.stem] = dict(error='unreadable status')
            continue
        b = d['block']
        pid, child = d.get('driver_pid'), d.get('child_pid')
        live = d.get('live') or {}
        blocks[b] = dict(status=d['status'], stage=d['stage'], updated_utc=d.get('updated_utc'),
                         driver_alive=pid in procs and f'/code/drive_{b.lower()}.py' in procs[pid]['command'],
                         child_alive=child in procs and procs[child]['ppid'] == pid,
                         driver=procs.get(pid), child=procs.get(child), live=live)
    frozen = files_hash(list((V9 / 'frozen/code').glob('*.py')))
    locks, completed_locks = {}, {}
    for path in sorted((V9 / 'spec').glob('LOCK_*.json')):
        if path.stem == 'LOCK_Q1':
            continue  # Q1 has completed and uses a separate lock format.
        d = read(path)
        checks = dict(code=files_hash([V9 / 'code' / n for n in d['files']]) == d['files_sha256'],
                      frozen=frozen == d['frozen_implementation_sha256'])
        for label, filename, key in (('manifest', 'manifest.json', 'data_manifest_sha256'),
                                     ('exclusions', 'evaluation_symbolic_hashes.txt', 'exclusion_list_sha256')):
            checks[label] = hashlib.sha256((V9 / 'frozen/data' / filename).read_bytes()).hexdigest() == d[key]
        identity = {k: d[k] for k in ('block', 'frozen_implementation_sha256', 'files_sha256',
                                     'data_manifest_sha256', 'exclusion_list_sha256')}
        checks['lock_identity'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() == d['lock_sha256']
        tests = {}
        for kind in ('contracts', 'smoke'):
            t = read(V9 / 'tests' / f'{d["block"].lower()}_{kind}.json', {})
            tests[kind] = t.get('status') == 'passed' and t.get('lock_sha256') == d['lock_sha256']
        entry = dict(inputs=checks, tests_current=tests, lock_sha256=d['lock_sha256'])
        if blocks.get(d['block'], {}).get('status') == 'completed':
            # A completed block keeps its historical source lock when a shared
            # implementation is subsequently repaired for pending work.
            completed_locks[d['block']] = entry
        else:
            locks[d['block']] = entry
    watch = read(MON / 'watch.pid', {})
    monitor = read(MON / 'status.json', {})
    val = read(MON / 'q4_validation.json', {})
    i22 = read(MON / 'i22_validation.json', {})
    probe = read(MON / 'i22_probe.json', {})
    q5_memory = read(MON / 'q5_memory_recovery.json', {})
    issues, samples = [], 0
    for path in sorted((V9 / 'runs').glob('*/log.jsonl')):
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                row = json.loads(line)
            except ValueError:
                issues.append(dict(path=str(path), line=line_number, error='invalid JSON; verify concurrent append'))
                continue
            samples += 1
            for key, value in row.items():
                if isinstance(value, float) and not math.isfinite(value):
                    issues.append(dict(path=str(path), line=line_number, key=key, error='nonfinite'))
    record = dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), blocks=blocks,
                  pending_locks=locks, completed_locks=completed_locks, watchdog_alive=watch.get('pid') in procs
                  and '/monitor/watch.py' in procs[watch['pid']]['command'],
                  watchdog_acting='--act' in procs.get(watch.get('pid'), {}).get('command', ''),
                  watchdog_snapshot=monitor, q4_validation=val,
                  validation_process=procs.get(val.get('pid')), validation_child=procs.get(val.get('child_pid')),
                  i22_validation=i22, i22_validation_process=procs.get(i22.get('pid')),
                  i22_validation_child=procs.get(i22.get('child_pid')),
                  i22_probe=probe, i22_probe_process=procs.get(probe.get('pid')),
                  q5_memory_recovery=q5_memory,
                  q5_memory_recovery_process=procs.get(q5_memory.get('pid')),
                  q5_memory_validation_child=procs.get(q5_memory.get('child_pid')),
                  q5_memory_probe_process=procs.get(q5_memory.get('probe_pid')),
                  disk_free_gb=round(shutil.disk_usage(V9).free / 1e9, 1),
                  swap=command(['sysctl', '-n', 'vm.swapusage']), power=command(['pmset', '-g', 'batt']),
                  current_training_log_samples=samples, current_training_nonfinite_values=issues,
                  action=action, notification=notification)
    tmp = MON / 'codex_health.json.tmp'
    tmp.write_text(json.dumps(record, indent=2))
    os.replace(tmp, MON / 'codex_health.json')
    with (MON / 'codex_checks.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')
    return record


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--action', default='Health check; no recovery action performed.')
    ap.add_argument('--notification', default='No new actionable change; stay quiet.')
    args = ap.parse_args()
    d = collect(args.action, args.notification)
    print(json.dumps(dict(checked_utc=d['checked_utc'], blocks=d['blocks'], locks=d['pending_locks'],
                          watchdog_alive=d['watchdog_alive'], watchdog_acting=d['watchdog_acting'],
                          validation=d['q4_validation'], nonfinite=d['current_training_nonfinite_values'],
                          i22_validation={k: d['i22_validation'].get(k) for k in ('status', 'stage', 'pid', 'child_pid', 'updated_utc', 'resumed', 'error')},
                          i22_probe={k: d['i22_probe'].get(k) for k in ('status', 'batch', 'pid', 'updated_utc')},
                          q5_memory_recovery={k: d['q5_memory_recovery'].get(k) for k in ('status', 'stage', 'pid', 'child_pid', 'probe_pid', 'updated_utc', 'completed_stages', 'error')},
                          disk_free_gb=d['disk_free_gb'], swap=d['swap'], power=d['power'])))
