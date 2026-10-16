#!/usr/bin/env python3
"""Start, stop or inspect only the V14 Observatory service."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / 'runtime'
PID = RUNTIME / 'server.pid'


def running():
    try:
        pid = int(PID.read_text())
        cmd = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout
        return pid if str(HERE / 'server.py') in cmd else None
    except (OSError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'stop', 'status'])
    args = parser.parse_args()
    RUNTIME.mkdir(exist_ok=True)
    with (RUNTIME / 'control.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pid = running()
        if args.action == 'stop':
            if pid:
                os.kill(pid, signal.SIGTERM)
                for _ in range(100):
                    if not running():
                        break
                    time.sleep(.1)
                if running():
                    print('Stop requested; collector is finishing its current sample.')
                    return
                PID.unlink(missing_ok=True)
                print('V14 Observatory stopped. Training is unaffected.')
            else:
                print('V14 Observatory is not running.')
            return
        if args.action == 'start' and not pid:
            with (RUNTIME / 'server.log').open('ab') as log:
                child = subprocess.Popen([sys.executable, str(HERE / 'server.py')], cwd=HERE,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
            pid = child.pid
            PID.write_text(str(pid))
        if not pid:
            print('V14 Observatory is not running.')
            return
        print('V14 Observatory process:', pid)
        for _ in range(45 if args.action == 'start' else 1):
            try:
                with urlopen('http://127.0.0.1:8714/api/state', timeout=2) as response:
                    state = json.load(response)
                print('Local:  ' + state['connection']['local'])
                print('Wi-Fi:  ' + (state['connection']['lan'] or 'unavailable'))
                print('Stage:  ' + state['phase'])
                return
            except Exception:
                if not running():
                    print('Server exited. Inspect ' + str(RUNTIME / 'server.log'))
                    sys.exit(1)
                time.sleep(1)
        print('Process is running; data is not ready. Inspect ' + str(RUNTIME / 'server.log'))


if __name__ == '__main__':
    main()
