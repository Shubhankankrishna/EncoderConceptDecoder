"""Isolated UI failure-mode fixture. Never loads torch or writes to a training run.
Run from the project root: python -m v10_10M.telemetry.web.qa_server
Change telemetry/data/qa_scenario.txt to zero, empty, offline, or stale.
"""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import json,time
from ..server import WEB,WEB_ASSETS,DEFAULT_DATA
from ..store import METRIC_AVAILABILITY
SCENARIO=DEFAULT_DATA/'qa_scenario.txt'
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        route=self.path.split('?',1)[0]
        if route in WEB_ASSETS:
            name,mime=WEB_ASSETS[route];self.send_response(200);self.send_header('Content-Type',mime);self.end_headers();self.wfile.write((WEB/name).read_bytes());return
        scenario=SCENARIO.read_text().strip() if SCENARIO.exists() else 'zero'
        if scenario=='offline':self.send_response(503);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(b'{"error":"QA simulated outage"}');return
        now=time.time();run=dict(id='qa_fixture',manifest=dict(start_unix=now-60,deadline_unix=now,training=dict(hours=24,checkpoint_seconds=600),parameters=dict(dense=5465890,engram=20000320),model=dict(width=128),device='fixture',precision='float32'),status=dict(status='complete',step=0,elapsed_hours=0,tokens=0,loss={}),health=dict(trainer_alive=False,checkpoint_age_seconds=None),path='QA fixture; no training process')
        event=dict(id=1,kind='evaluation',source='QA synthetic zero trials',step=0,wall_time=now-30,time_quality='fixture_clock',payload=dict(split='dev',families={},generation=dict(math=dict(n=0,correct=0,examples=[]),sudoku_easy=dict(n=2,correct=0,examples=[dict(response='<img src=x onerror="alert(1)"> & <script>bad()</script>',reason='QA escaped model text',correct=False,eos=False)]))))
        if route.endswith('/runs'):payload=dict(runs=[] if scenario=='empty' else [run])
        elif route.endswith('/overview'):payload=dict(run=run,latest_evaluation=dict(event_id=1,**event),alerts=[],ingestion_issues=[],metric_availability=METRIC_AVAILABILITY,artifacts=[])
        elif route.endswith('/health'):payload=dict(status='ok',poll_seconds=5,last_success_unix=now-(120 if scenario=='stale' else 0))
        elif route.endswith('/catalog'):payload=dict(metrics=[])
        elif route.endswith('/events'):payload=dict(events=[] if 'after_id=1' in self.path else [event],next_after_id=1,has_more=False)
        else:payload={}
        raw=json.dumps(payload).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',8792),Handler).serve_forever()
