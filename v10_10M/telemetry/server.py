"""Read-only local API and independent incremental training-log collector."""
import argparse
import csv
import fcntl
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import threading
import time
from urllib.parse import parse_qs, urlparse
import psutil
from . import SCHEMA_VERSION
from .store import Store, atomic_json, encode, strict_json, METRIC_AVAILABILITY

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT/'telemetry/data'
WEB = Path(__file__).resolve().parent/'web'
WEB_ASSETS = {'/': ('index.html', 'text/html'), '/index.html': ('index.html', 'text/html'),
              '/style.css': ('style.css', 'text/css'), '/core.mjs': ('core.mjs', 'text/javascript'),
              '/chart.mjs': ('chart.mjs', 'text/javascript'), '/app.mjs': ('app.mjs', 'text/javascript')}


class Monitor:
    def __init__(self, run):
        self.run = Path(run).resolve()
        self.process = None
        self.last_cpu_pid = None

    def sample(self, include_host=False):
        now = time.time()
        path = self.run/'status.json'
        status = strict_json(path.read_bytes()) if path.exists() else {}
        health = dict(trainer_alive=False, trainer_pid=status.get('pid'),
                      status_age_seconds=max(0,now-path.stat().st_mtime) if path.exists() else None)
        pid = status.get('pid')
        try:
            proc = psutil.Process(pid) if pid else None
            if proc:
                cmd = proc.cmdline()
                # Match executable purpose AND this exact run, not just a possibly reused PID.
                valid = any(x in ('v10_10M.supervise','v10_10M.train') for x in cmd) and str(self.run) in cmd
                health['trainer_alive'] = valid and proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE
                if health['trainer_alive']:
                    if self.process is None or self.process.pid!=pid or self.process.create_time()!=proc.create_time():
                        self.process=proc;self.last_cpu_pid=None
                    health['process_started_unix']=proc.create_time()
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            self.process=None
        ck=self.run/'latest.pt'
        health['checkpoint_age_seconds']=max(0,now-ck.stat().st_mtime) if ck.exists() else None
        health['checkpoint_bytes']=ck.stat().st_size if ck.exists() else None
        host=None
        if include_host:
            vm=psutil.virtual_memory();swap=psutil.swap_memory();disk=psutil.disk_usage(self.run)
            host=dict(system_available_gib=vm.available/2**30,system_memory_percent=vm.percent,
                      swap_used_gib=swap.used/2**30,disk_free_gib=disk.free/2**30,
                      collector_rss_mib=psutil.Process().memory_info().rss/2**20,
                      trainer_alive=health['trainer_alive'])
            if self.process and health['trainer_alive']:
                try:
                    cpu=self.process.cpu_percent(interval=None)
                    host.update(trainer_rss_gib=self.process.memory_info().rss/2**30,
                                trainer_threads=self.process.num_threads())
                    if self.last_cpu_pid==pid:host['trainer_cpu_percent']=cpu
                    self.last_cpu_pid=pid
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        return health,host


class Collector:
    def __init__(self, store, run, data, poll=5, host_interval=30):
        self.store=store;self.run=Path(run).resolve();self.data=Path(data)
        self.monitor=Monitor(run);self.poll=poll;self.host_interval=host_interval
        self.stopping=threading.Event();self.last_host=0;self.last_success=None;self.error=None
        self.started=time.time()

    def once(self):
        host_due=time.monotonic()-self.last_host>=self.host_interval
        health,host=self.monitor.sample(host_due)
        if host_due:self.last_host=time.monotonic()
        run_id=self.store.ingest(self.run,health,host)
        atomic_json(self.data/'overview.json',self.store.overview(run_id))
        atomic_json(self.data/'metrics_catalog.json',dict(schema_version=SCHEMA_VERSION,metrics=self.store.catalog(run_id)))
        self.last_success=time.time();self.error=None

    def loop(self):
        while not self.stopping.is_set():
            try:self.once()
            except Exception as exc:
                self.error=f'{type(exc).__name__}: {exc}'
                print(encode(dict(at=time.time(),collector_error=self.error)),flush=True)
            self.stopping.wait(self.poll)

    def health(self):
        return dict(schema_version=SCHEMA_VERSION,status='ok' if self.last_success and not self.error else 'degraded',
                    pid=os.getpid(),started_unix=self.started,last_success_unix=self.last_success,
                    last_error=self.error,poll_seconds=self.poll,database=str(self.store.path))


def make_handler(store, collector):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):pass

        def respond(self, code, payload=None, mime='application/json', filename=None, raw=None):
            if raw is None:raw=encode(payload).encode()
            self.send_response(code)
            self.send_header('Content-Type',mime+'; charset=utf-8')
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(raw)))
            self.send_header('X-Content-Type-Options','nosniff')
            if filename:self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
            origin=self.headers.get('Origin')
            if origin:
                parsed=urlparse(origin)
                if parsed.scheme in ('http','https') and parsed.hostname in ('localhost','127.0.0.1','::1'):
                    self.send_header('Access-Control-Allow-Origin',origin)
                    self.send_header('Vary','Origin')
            self.end_headers()
            try:self.wfile.write(raw)
            except (BrokenPipeError,ConnectionResetError):pass

        def do_OPTIONS(self):
            self.respond(200,dict(methods=['GET']))

        def do_GET(self):
            try:
                parsed=urlparse(self.path);q=parse_qs(parsed.query)
                if parsed.path in WEB_ASSETS:
                    name, mime=WEB_ASSETS[parsed.path]
                    raw=(WEB/name).read_bytes()
                    self.send_response(200)
                    self.send_header('Content-Type',mime+'; charset=utf-8')
                    self.send_header('Content-Length',str(len(raw)))
                    self.send_header('Cache-Control','no-store')
                    self.send_header('X-Content-Type-Options','nosniff')
                    self.end_headers()
                    try:self.wfile.write(raw)
                    except (BrokenPipeError,ConnectionResetError):pass
                    return
                route=parsed.path.removeprefix('/api/v1/')
                if not parsed.path.startswith('/api/v1/'):return self.respond(404,dict(error='Use /api/v1/health or /api/v1/runs'))
                if route=='health':return self.respond(200,collector.health())
                if route=='schema':return self.respond(200,dict(schema_version=SCHEMA_VERSION,metric_availability=METRIC_AVAILABILITY,
                    cursor='Integer event id, not training step; ingest order can differ from wall time.',
                    timestamps='Unix seconds UTC. Training time is reconstructed from manifest start + logged elapsed.',
                    routes=['health','runs','overview','catalog','series','events','evaluations','artifacts','issues','download'],read_only=True))
                runs=store.runs()
                if route=='runs':return self.respond(200,dict(schema_version=SCHEMA_VERSION,runs=runs))
                run_id=q.get('run_id',[collector.run.name])[0]
                if run_id not in [r['id'] for r in runs]:return self.respond(404,dict(error='Unknown run_id'))
                if route=='download':
                    format=q.get('format',['snapshot'])[0]
                    if format not in ('snapshot','csv'):raise ValueError('format must be snapshot or csv')
                    name=q.get('name',[''])[0]
                    if format=='csv' and not name:raise ValueError('CSV download requires name')
                    with store.connect() as db:
                        db.execute('BEGIN')
                        watermark=db.execute('SELECT COALESCE(MAX(id),0) FROM events WHERE run_id=?',(run_id,)).fetchone()[0]
                        if format=='csv':
                            out=io.StringIO(newline='');writer=csv.writer(out)
                            writer.writerow(['event_id','step','unix_seconds','value'])
                            writer.writerows(tuple(r) for r in db.execute('SELECT event_id,step,wall_time,value FROM metrics WHERE run_id=? AND name=? AND event_id<=? ORDER BY event_id',(run_id,name,watermark)))
                            raw=out.getvalue().encode()
                        else:
                            rows=[dict(r) for r in db.execute('SELECT id,kind,source,step,wall_time,time_quality,ingested_at,payload FROM events WHERE run_id=? AND id<=? ORDER BY id',(run_id,watermark))]
                            for row in rows:row['payload']=strict_json(row['payload'])
                    if format=='csv':return self.respond(200,mime='text/csv',filename='v10-metric.csv',raw=raw)
                    return self.respond(200,dict(schema_version=SCHEMA_VERSION,run_id=run_id,exported_at=time.time(),event_watermark=watermark,
                        scope='All events through this consistent database watermark; overview and catalog are independent completion-time snapshots.',
                        overview=store.overview(run_id),catalog=store.catalog(run_id),events=rows),filename='v10-telemetry-snapshot.json')
                limit=int(q.get('limit',['1000'])[0]);after=int(q.get('after_id',['0'])[0])
                if not 1<=limit<=5000 or after<0:raise ValueError('limit must be 1..5000; after_id must be >=0')
                if route=='overview':return self.respond(200,store.overview(run_id))
                if route=='catalog':payload=dict(metrics=store.catalog(run_id))
                elif route=='artifacts':payload=dict(artifacts=store.artifacts(run_id))
                elif route=='issues':payload=dict(issues=store.issues(run_id))
                elif route in ('events','evaluations'):
                    kind='evaluation' if route=='evaluations' else q.get('kind',[None])[0]
                    if kind not in (None,'training','evaluation','host','probe','audit','probe_evaluation'):raise ValueError('Unknown event kind')
                    rows=store.events(run_id,kind,after,limit+1)
                    more=len(rows)>limit;rows=rows[:limit]
                    payload=dict(events=rows,next_after_id=rows[-1]['id'] if rows else after,has_more=more)
                elif route=='series':
                    name=q.get('name',[''])[0]
                    if not name:raise ValueError('series requires name')
                    rows=store.series(run_id,name,after,limit+1);more=len(rows)>limit;rows=rows[:limit]
                    payload=dict(name=name,points=rows,next_after_id=rows[-1]['event_id'] if rows else after,has_more=more)
                else:return self.respond(404,dict(error='Unknown endpoint'))
                self.respond(200,dict(schema_version=SCHEMA_VERSION,run_id=run_id,**payload))
            except (ValueError,TypeError) as exc:self.respond(400,dict(error=str(exc)))
            except Exception as exc:
                print(encode(dict(api_error=f'{type(exc).__name__}: {exc}')),flush=True)
                self.respond(500,dict(error='Telemetry read failed; inspect collector log'))
    return Handler


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default=str(ROOT/'runs/pretrain_24h'))
    p.add_argument('--data-dir',default=str(DEFAULT_DATA));p.add_argument('--port',type=int,default=8766)
    p.add_argument('--poll',type=float,default=5);p.add_argument('--host-interval',type=float,default=30)
    args=p.parse_args()
    if args.poll<1 or args.host_interval<5:raise SystemExit('Minimum poll 1s and host interval 5s')
    data=Path(args.data_dir).resolve();data.mkdir(parents=True,exist_ok=True)
    with (data/'server.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Telemetry collector already running for this data directory')
        store=Store(data/'telemetry.sqlite3');collector=Collector(store,args.run_dir,data,args.poll,args.host_interval)
        server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(store,collector))
        server.timeout=.5
        def stop(*_):collector.stopping.set()
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        collector.once()
        atomic_json(data/'server.json',dict(pid=os.getpid(),url=f'http://127.0.0.1:{args.port}/api/v1',
                    run_dir=str(collector.run),started_unix=collector.started))
        thread=threading.Thread(target=collector.loop,daemon=True);thread.start()
        print(encode(dict(event='ready',**collector.health())),flush=True)
        try:
            while not collector.stopping.is_set():server.handle_request()
        finally:
            collector.stopping.set();thread.join(timeout=15);server.server_close()


if __name__=='__main__':main()
