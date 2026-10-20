"""Lightweight loopback-only telemetry, no tensor/checkpoint loading."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import threading
import time
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parent


def read(path,default=None):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def snapshot(run):
    plan=read(run/'plan.json',{})
    status=read(run/'status.json',{})
    stages=[]
    for s in plan.get('stages',[])+[{'name':'renderer'}]:
        p=run/s['name'];evaluation=read(p/'latest_eval.json')
        state=read(p/'status.json',{'status':'queued','step':0})
        if evaluation:
            evaluation={k:v for k,v in evaluation.items() if k!='paired'}
            if 'strategies' in evaluation:
                evaluation['strategies']={z:{k:v for k,v in r.items() if k!='rollouts'}|
                    {'rollouts':{mode:{k:v for k,v in info.items() if k!='episodes'} for mode,info in r['rollouts'].items()}}
                    for z,r in evaluation['strategies'].items()}
        stages.append({'name':s['name'],'state':state,'evaluation':evaluation})
    import psutil
    resources={'available_gb':psutil.virtual_memory().available/1024**3,'cpu_percent':psutil.cpu_percent(interval=None)}
    return {'timestamp':time.time(),'plan':{k:v for k,v in plan.items() if k!='sources'},'status':status,
            'stages':stages,'gates':read(run/'gates.json',{}),'resources':resources,
            'quality':read(run/'renderer/quality_gates.json',{}),'final_available':(run/'final_results.json').exists()}


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*_):pass
    def do_GET(self):
        if self.headers.get('Host','').split(':')[0] not in ('127.0.0.1','localhost'):
            self.send_error(403);return
        path=urlsplit(self.path).path
        if path=='/api/snapshot':body=json.dumps(snapshot(self.server.run),allow_nan=False).encode();mime='application/json'
        elif path in ('/','/index.html','/app.js','/style.css'):
            filename='index.html' if path=='/' else path[1:]
            body=(ROOT/'web'/filename).read_bytes();mime={'index.html':'text/html','app.js':'application/javascript','style.css':'text/css'}[filename]
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',mime+'; charset=utf-8')
        self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=ROOT/'runs/20261019_lm1b_grounded');ap.add_argument('--port',type=int,default=8767)
    args=ap.parse_args();server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.run=args.run
    def shutdown(*_):threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,shutdown);signal.signal(signal.SIGINT,shutdown)
    (ROOT/'reports/dashboard_process.json').write_text(json.dumps({'pid':os.getpid(),'port':args.port}))
    server.serve_forever()


if __name__=='__main__':main()
