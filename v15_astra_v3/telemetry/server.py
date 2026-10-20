"""Loopback-only HTTP API and dashboard, with one inexpensive collector thread."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import threading
import time
from urllib.parse import parse_qs, unquote, urlsplit

from .engine import Engine, clean

HERE = Path(__file__).resolve().parent
STATIC = {'/': ('index.html','text/html; charset=utf-8'),
          '/index.html': ('index.html','text/html; charset=utf-8'),
          '/style.css': ('style.css','text/css; charset=utf-8'),
          '/app.js': ('app.js','text/javascript; charset=utf-8'),
          '/view.js': ('view.js','text/javascript; charset=utf-8')}


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, engine, engines=None):
        self.engine = engine
        self.engines = engines or {'current':engine}
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = 'AstraTelemetry/1'

    def log_message(self, fmt, *args):
        # Five-second polling should not grow the daemon log forever.
        if args and str(args[1] if len(args)>1 else '').startswith(('4','5')):
            super().log_message(fmt, *args)

    def respond(self, code, payload, content_type='application/json; charset=utf-8', *, download=None):
        body = payload if isinstance(payload, bytes) else payload.encode('utf-8') if isinstance(payload, str) else json.dumps(clean(payload),allow_nan=False).encode()
        self.send_response(code)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        if download:
            self.send_header('Content-Disposition',f'attachment; filename="{download}"')
        self.end_headers()
        if self.command != 'HEAD':
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    @staticmethod
    def integer(query, name, default, low, high):
        if name not in query:
            return default
        if len(query[name]) != 1:
            raise ValueError(f'{name} must occur once')
        value = int(query[name][0])
        if not low <= value <= high:
            raise ValueError(f'{name} must be between {low} and {high}')
        return value

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        self.respond(405,dict(error='This dashboard only supports reading telemetry.'))

    do_PUT = do_POST
    do_DELETE = do_POST
    do_PATCH = do_POST

    def do_GET(self):
        host = self.headers.get('Host','').split(':')[0].lower()
        if host not in ('127.0.0.1','localhost'):
            self.respond(403,dict(error='Local dashboard host required'))
            return
        url = urlsplit(self.path)
        path = unquote(url.path)
        query = parse_qs(url.query,keep_blank_values=True)
        engine = self.server.engine
        try:
            if 'arm' in query:
                if len(query['arm']) != 1 or query['arm'][0] not in self.server.engines:
                    raise ValueError('Unknown experiment arm')
                engine = self.server.engines[query['arm'][0]]
            if path in STATIC:
                name,mime = STATIC[path]
                self.respond(200,(HERE/'web'/name).read_bytes(),mime)
            elif path == '/api/runs':
                self.respond(200,dict(runs=[dict(id=name,name=entry.run.name) for name,entry in self.server.engines.items()]))
            elif path == '/api/snapshot':
                self.respond(200,engine.snapshot())
            elif path == '/api/health':
                snap = engine.snapshot()
                ready = snap['collector']['fresh'] and snap['collector']['error'] is None
                self.respond(200 if ready else 503,dict(ok=ready,**snap['collector']))
            elif path == '/api/history':
                self.respond(200,dict(rows=engine.history(self.integer(query,'limit',1200,2,5000),self.integer(query,'after_step',-1,-1,10**9))))
            elif path == '/api/evaluations':
                self.respond(200,dict(evaluations=engine.evaluations()))
            elif path == '/api/samples':
                keys=query.get('evaluation',[])
                if len(keys)!=1:
                    raise ValueError('Choose one evaluation')
                self.respond(200,engine.samples(keys[0],self.integer(query,'offset',0,0,10**6),self.integer(query,'limit',1,1,20)))
            elif path == '/api/events':
                self.respond(200,dict(events=engine.events(self.integer(query,'limit',80,1,500))))
            elif path == '/api/resources':
                self.respond(200,dict(rows=engine.resource_history(self.integer(query,'limit',180,2,1000))))
            elif path == '/api/export.csv':
                self.respond(200,engine.export_csv(),'text/csv; charset=utf-8',download='astra-training-metrics.csv')
            else:
                self.respond(404,dict(error='Not found'))
        except (ValueError, TypeError) as error:
            self.respond(400,dict(error=str(error)))
        except KeyError:
            self.respond(404,dict(error='Evaluation not found'))
        except Exception as error:
            self.log_error('%s',repr(error))
            self.respond(500,dict(error='Telemetry request failed; check the local service log'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=HERE.parent/'runs/20261018_24h_raw_gram')
    parser.add_argument('--database',type=Path,default=HERE/'runtime/telemetry.sqlite3')
    parser.add_argument('--comparison-root',type=Path)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--interval',type=float,default=5)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if not 0<=args.port<=65535 or not 2<=args.interval<=60:
        parser.error('Port must be valid and interval must be between 2 and 60 seconds')
    if args.comparison_root:
        args.run=args.comparison_root/'30m'
    if not (args.run/'plan.json').is_file():
        parser.error('Run must contain an existing plan.json')
    try:
        os.nice(10)
    except (OSError, AttributeError):
        pass
    engine=Engine(args.run,args.database,interval=args.interval)
    engines={'30m':engine,'10m':Engine(args.comparison_root/'10m',
        args.database.with_name(args.database.stem+'_10m.sqlite3'),interval=args.interval)} if args.comparison_root else {'current':engine}
    for entry in engines.values():entry.poll()
    if args.once:
        print(json.dumps(engine.snapshot(),indent=2))
        for entry in engines.values():entry.close()
        return
    server=Server(('127.0.0.1',args.port),engine,engines)
    stop=threading.Event()
    def collect():
        while not stop.wait(args.interval):
            for entry in engines.values():entry.poll()
    thread=threading.Thread(target=collect,name='astra-collector',daemon=True)
    thread.start()
    def shutdown(*_):
        threading.Thread(target=server.shutdown,daemon=True).start()
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,shutdown)
    url=f'http://127.0.0.1:{server.server_port}'
    state=dict(pid=os.getpid(),url=url,run=str(engine.run),database=str(engine.database),interval_seconds=args.interval,started=time.time())
    state_file=engine.database.parent/'service.json'
    state_file.write_text(json.dumps(state,indent=2))
    print(json.dumps(state),flush=True)
    try:
        server.serve_forever(poll_interval=.5)
    finally:
        stop.set();thread.join(timeout=10);server.server_close()
        for entry in engines.values():entry.close()


if __name__=='__main__':
    main()
