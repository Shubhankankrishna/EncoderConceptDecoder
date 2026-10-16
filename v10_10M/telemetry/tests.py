"""Failure-mode and API contract tests; no model or active-run mutation."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
from .store import Store, strict_json, evaluation_metrics
from .server import Collector, make_handler


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.run=self.root/'project/runs/run';self.run.mkdir(parents=True);self.data=self.root/'data';self.data.mkdir()
        (self.run/'manifest.json').write_text(json.dumps(dict(start_unix=1000)))
        (self.run/'status.json').write_text(json.dumps(dict(status='running',pid=99999999,step=20)))
        self.log=self.run/'train.jsonl';self.store=Store(self.data/'db.sqlite3')

    def tearDown(self):self.temp.cleanup()

    def record(self,step=20):
        return json.dumps(dict(step=step,elapsed_hours=step/3600,loss=dict(total=2,token=1),
                               dense_grad_norm=2,learning_rate=.001))+'\n'

    def ingest(self):self.store.ingest(self.run,dict(trainer_alive=False,status_age_seconds=0))

    def test_partial_tail_restart_duplicate_step_and_nonfinite(self):
        first=self.record();second=self.record(40)
        self.log.write_text(first+second[:19]);self.ingest()
        self.assertEqual(len(self.store.events('run')),1)
        with self.log.open('a') as f:f.write(second[19:])
        self.store=Store(self.data/'db.sqlite3');self.ingest();self.ingest()
        self.assertEqual(len(self.store.events('run')),2)
        with self.log.open('a') as f:
            f.write(self.record(20));f.write('{"step":60,"elapsed_hours":NaN}\n');f.write(self.record(80))
        self.ingest()
        self.assertEqual([x['step'] for x in self.store.events('run')],[20,40,20,80])
        self.assertEqual(len(self.store.issues('run')),1)
        self.assertEqual(self.store.series('run','train.gradient.dense_preclip_norm')[0]['value'],2)
        self.assertAlmostEqual(self.store.series('run','derived.dense_clip_scale')[0]['value'],.5,places=5)
        self.assertEqual(self.store.events('run')[0]['wall_time'],1020)

    def test_truncation_and_rotation_idempotence(self):
        self.log.write_text(self.record()+self.record(40));self.ingest()
        self.log.write_text(self.record());self.ingest();self.ingest()
        self.assertEqual(len(self.store.events('run')),3)
        with self.log.open('a') as f:f.write(self.record(60))
        self.ingest();self.ingest();self.assertEqual(len(self.store.events('run')),4)
        self.log.rename(self.run/'old.log');self.log.write_text(self.record(80))
        self.ingest();self.ingest();self.assertEqual(len(self.store.events('run')),5)

    def test_evaluation_rewrite_zero_trials_and_research_alerts(self):
        payload=dict(families=dict(math=dict(nll=1,concept_ablation_delta_nll=0,code_counts=[0,2,0,2])),
                     generation=dict(math=dict(n=2,correct=0),sudoku_hard=dict(n=0,correct=0)))
        path=self.run/'evaluation_0000040.json';path.write_text(json.dumps(payload));self.ingest();self.ingest()
        self.assertEqual(len(self.store.events('run','evaluation')),1)
        metrics=evaluation_metrics(payload)
        self.assertNotIn('eval.sudoku_hard.exact_accuracy',metrics)
        self.assertEqual(metrics['eval.math.effective_codes'],2)
        self.assertGreater(metrics['eval.math.accuracy_ci95_high'],.5)
        self.assertIn('trainer_missing',[a['code'] for a in self.store.overview('run')['alerts']])
        payload['generation']['math']['correct']=1;path.write_text(json.dumps(payload));self.ingest()
        self.assertEqual(len(self.store.events('run','evaluation')),2)
        self.assertEqual(self.store.latest_evaluation('run')['payload']['generation']['math']['correct'],1)

    def test_http_pagination_errors_cors_and_health(self):
        self.log.write_text(self.record()+self.record(40))
        collector=Collector(self.store,self.run,self.data);collector.once()
        self.assertFalse(self.store.overview('run')['run']['health']['trainer_alive'])
        server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(self.store,collector))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}/api/v1/'
        try:
            def get(route):
                with urlopen(base+route,timeout=3) as r:return json.load(r)
            first=get('events?kind=training&limit=1');self.assertTrue(first['has_more'])
            second=get(f'events?kind=training&limit=1&after_id={first["next_after_id"]}')
            self.assertFalse(second['has_more']);self.assertEqual(second['events'][0]['step'],40)
            self.assertEqual(get('health')['status'],'ok')
            for path,mime in [('', 'text/html'),('style.css','text/css'),('app.mjs','text/javascript'),('core.mjs','text/javascript'),('chart.mjs','text/javascript')]:
                with urlopen(base.removesuffix('api/v1/')+path,timeout=3) as r:
                    self.assertEqual(r.status,200)
                    self.assertTrue(r.headers['Content-Type'].startswith(mime))
                    self.assertEqual(r.headers['X-Content-Type-Options'],'nosniff')
            for path in ['server.py','data/telemetry.sqlite3','../store.py']:
                with self.assertRaises(HTTPError) as err:urlopen(base.removesuffix('api/v1/')+path,timeout=3)
                self.assertEqual(err.exception.code,404)
            self.assertEqual(len(get('series?name=train.loss.token')['points']),2)
            snapshot=get('download?format=snapshot')
            self.assertEqual(len([e for e in snapshot['events'] if e['kind']=='training']),2)
            self.assertEqual(max(e['id'] for e in snapshot['events']),snapshot['event_watermark'])
            with urlopen(base+'download?format=csv&name=train.loss.token') as r:
                self.assertIn('attachment',r.headers['Content-Disposition'])
                self.assertEqual(r.headers['Content-Type'],'text/csv; charset=utf-8')
                self.assertIn('event_id,step,unix_seconds,value',r.read().decode())
            for route,code in [('overview?run_id=missing',404),('series',400),('events?limit=0',400),('events?after_id=no',400),('download?format=bad',400),('download?format=csv',400),('bad',404)]:
                with self.assertRaises(HTTPError) as err:get(route)
                self.assertEqual(err.exception.code,code)
            req=Request(base+'health',headers={'Origin':'http://localhost:3000'})
            with urlopen(req) as r:self.assertEqual(r.headers.get('Access-Control-Allow-Origin'),'http://localhost:3000')
            req=Request(base+'health',headers={'Origin':'https://example.com'})
            with urlopen(req) as r:self.assertIsNone(r.headers.get('Access-Control-Allow-Origin'))
        finally:server.shutdown();server.server_close();thread.join()

    def test_offline_probe_namespace_and_export(self):
        from .export import export
        audit=self.root/'project/audit';audit.mkdir()
        probe=dict(run_id='run',step=10,families=dict(math=dict(token_nll=.5,gradient_probes=dict(token=dict(components=dict(core=.01))))))
        (audit/'checkpoint_probe.json').write_text(json.dumps(probe))
        (audit/'findings.json').write_text(json.dumps(dict(run_id='run',findings=[dict(code='data_defect',severity='warning',message='Example')])) )
        self.log.write_text(self.record());self.ingest();self.ingest()
        self.assertEqual(len(self.store.events('run','probe')),1)
        self.assertEqual(self.store.series('run','probe.math.gradient.token.components.core')[0]['value'],.01)
        self.assertEqual(len(self.store.series('run','train.gradient.dense_preclip_norm')),1)
        self.assertIn('data_defect',[x['code'] for x in self.store.overview('run')['alerts']])
        out=export(self.store,'run',self.root/'export')
        events=[json.loads(x) for x in (out/'events.jsonl').read_text().splitlines()]
        watermark=json.loads((out/'export.json').read_text())['event_watermark']
        self.assertEqual(len(events),3);self.assertEqual(max(e['id'] for e in events),watermark)

    def test_malformed_records_are_quarantined(self):
        self.log.write_text('{"step":1,"elapsed_hours":1e999}\n'+self.record())
        (self.run/'evaluation_0000001.json').write_text('{"generation":{"math":{"n":1,"correct":3}}}')
        self.ingest()
        self.assertEqual(len(self.store.events('run','training')),1)
        self.assertEqual(len(self.store.events('run','evaluation')),0)
        self.assertEqual(len(self.store.issues('run')),2)


if __name__=='__main__':unittest.main()
