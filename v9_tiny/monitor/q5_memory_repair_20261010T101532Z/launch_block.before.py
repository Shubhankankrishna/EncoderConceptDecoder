"""Start a Q2+ block driver detached (new session) with idle-sleep prevention. Refuses to start unless the block's
contracts and smoke passed under the current lock, and refuses a duplicate of the same block."""
import json, subprocess, sys, time
import v9paths
from v9paths import V9, RUNS, CODE, TESTS, SPEC, read_json, write_json
from queue_base import lock_identity
import importlib


def main(block):
    b = block.lower(); mod = importlib.import_module(f'drive_{b}')
    files = getattr(mod, f'{block}_FILES'); lock = read_json(SPEC / f'LOCK_{block}.json'); now = lock_identity(block, files)
    if now['lock_sha256'] != lock['lock_sha256']:
        raise SystemExit(f'{block}: files changed since LOCK_{block}; rerun its contracts')
    for t in (f'{b}_contracts.json', f'{b}_smoke.json'):
        j = read_json(TESTS / t)
        if j.get('status') != 'passed' or j.get('lock_sha256') != lock['lock_sha256']:
            raise SystemExit(f'{block}: {t} has not passed under the current lock')
    st = RUNS / f'status_{block}.json'
    if st.exists() and read_json(st)['status'] in ('running', 'waiting'):
        pid = read_json(st).get('driver_pid')
        if pid and subprocess.run(['kill', '-0', str(pid)], capture_output=True).returncode == 0:
            raise SystemExit(f'{block} driver already running (pid {pid})')
    log = open(RUNS / f'driver_{block}.log', 'a')
    p = subprocess.Popen([sys.executable, str(CODE / f'drive_{b}.py')], cwd=V9, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    c = subprocess.Popen(['caffeinate', '-is', '-w', str(p.pid)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    write_json(RUNS / f'launch_{block}.json', dict(block=block, pid=p.pid, caffeinate_pid=c.pid, lock_sha256=lock['lock_sha256'], utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print(json.dumps(dict(block=block, pid=p.pid)))


if __name__ == '__main__':
    main(sys.argv[1])
