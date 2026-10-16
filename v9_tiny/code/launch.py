"""Start a V9 block driver detached from this shell (new session), with macOS idle-sleep prevention while it runs."""
import json, subprocess, sys, time
from pathlib import Path
import v9paths
from v9paths import V9, RUNS, CODE, read_json, write_json

def main(script):
    status = RUNS / 'status.json'
    if status.exists() and read_json(status)['status'] == 'running':
        pid = read_json(status)['driver_pid']
        if subprocess.run(['kill', '-0', str(pid)], capture_output=True).returncode == 0:
            raise SystemExit(f'a driver is already running (pid {pid}); not launching a duplicate')
    log = open(RUNS / 'driver.log', 'a')
    p = subprocess.Popen([sys.executable, str(CODE / script)], cwd=V9, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    c = subprocess.Popen(['caffeinate', '-is', '-w', str(p.pid)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    write_json(RUNS / f'launch_{script.replace(".py", "")}.json', dict(script=script, pid=p.pid, caffeinate_pid=c.pid, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print(json.dumps(dict(pid=p.pid, caffeinate=c.pid)))

if __name__ == '__main__':
    main(sys.argv[1])
