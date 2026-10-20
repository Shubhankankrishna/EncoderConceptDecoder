import hashlib
from http.client import HTTPConnection
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from ..engine import Engine, generation_stats
from ..server import Server


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.run=self.root/'runs/pilot'
        self.gram=self.run/'gram'
        self.gram.mkdir(parents=True)
        self.now=[10000.0]
        self.host=dict(available=True,ram_total_bytes=16*1024**3,ram_available_bytes=4*1024**3,
                       disk_free_bytes=10*1024**3,ac_power=True,trainer=dict(alive=True),supervisor=dict(alive=True))
        self.write('plan.json',dict(start=9000,deadline=95400,training_deadline=94800,phase='P1',data=str(self.root/'artifacts/raw')))
        self.write('status.json',dict(status='training'))
        self.write('gram/manifest.json',dict(arm='code_gram',eval_every=1800,ledger=dict(trainable=10000333),config={}))
        self.engine=Engine(self.run,self.root/'telemetry/db.sqlite3',clock=lambda:self.now[0],system_provider=lambda:self.host)

    def tearDown(self):
        self.engine.close()
        self.temp.cleanup()

    def write(self,name,value):
        (self.run/name).write_text(json.dumps(value))

    def metric(self,step=1000,**extra):
        return dict(step=step,loss=7.,nll=7.,timestamp=self.now[0]-10,update_seconds=.5,kl_sum=.0001,**extra)

    def append(self,row):
        with (self.gram/'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row)+'\n')

    def evaluation(self,name='eval_00001000.json',step=1000,nll=6.5):
        self.write('gram/'+name,dict(step=step,timestamp=self.now[0],split='valid',nll_per_code=nll,seconds=2,
            records=[dict(index=1,nll_per_code=nll,free_generated_text='cup, cup. cup cup cup cup okay',reference_pretokenized_text='<script>alert(1)</script>',observed_history='history')]))

    def test_incremental_partial_line_and_restart_cursor(self):
        self.append(self.metric())
        line=json.dumps(self.metric(1050)).encode()
        with (self.gram/'metrics.jsonl').open('ab') as handle:handle.write(line[:15])
        self.engine.poll()
        self.assertEqual(len(self.engine.history()),1)
        self.engine.close()
        self.engine=Engine(self.run,self.root/'telemetry/db.sqlite3',clock=lambda:self.now[0],system_provider=lambda:self.host)
        with (self.gram/'metrics.jsonl').open('ab') as handle:handle.write(line[15:]+b'\n')
        self.engine.poll();self.engine.poll()
        self.assertEqual([r['step'] for r in self.engine.history()],[1000,1050])

    def test_rotation_and_truncation_deduplicate(self):
        for i in range(5):self.append(self.metric(1000+i*50))
        self.engine.poll()
        path=self.gram/'metrics.jsonl';path.rename(self.gram/'metrics.old')
        self.append(self.metric(1200));self.append(self.metric(1250));self.engine.poll()
        path.write_text(json.dumps(self.metric(1300))+'\n');self.engine.poll()
        self.assertEqual(len(self.engine.history()),7)
        self.assertTrue(any('rotated' in e['title'] for e in self.engine.events()))

    def test_malformed_and_nonfinite_rows_are_json_safe(self):
        with (self.gram/'metrics.jsonl').open('w') as handle:handle.write('broken\n[]\n')
        self.append(self.metric(loss_override=0))
        bad=self.metric(1050);bad['loss']=float('nan');self.append(bad)
        self.engine.poll()
        self.assertEqual(len(self.engine.history()),2)
        self.assertIsNone(self.engine.history()[-1]['loss'])
        json.dumps(self.engine.snapshot(),allow_nan=False)
        self.assertTrue(any(e['level']=='critical' for e in self.engine.events()))
        self.assertIn('numerics',{a['id'] for a in self.engine.snapshot()['alerts']})

    def test_metric_downsampling_includes_first_and_latest(self):
        for i in range(20):self.append(self.metric(i))
        self.engine.poll()
        self.assertEqual(len(self.engine.history(5)),5)
        self.assertEqual(self.engine.history(5)[0]['step'],0)
        self.assertEqual(self.engine.history(5)[-1]['step'],19)
        self.assertEqual(len(self.engine.history(after_step=16)),3)

    def test_validation_improvement_repetition_and_kl_alerts(self):
        for i in range(6):self.append(self.metric(1000+i*50))
        self.evaluation('eval_00000001.json',1,9)
        self.now[0]+=20;self.evaluation(nll=6)
        self.engine.poll()
        snap=self.engine.snapshot()
        self.assertAlmostEqual(snap['validation']['relative_nll_reduction'],1/3)
        self.assertEqual(snap['validation']['best']['step'],1000)
        self.assertTrue({'repetition','kl'}.issubset({a['id'] for a in snap['alerts']}))
        sample=self.engine.samples('eval_00001000.json')
        self.assertEqual(sample['records'][0]['reference_pretokenized_text'],'<script>alert(1)</script>')
        self.assertGreater(sample['records'][0]['repetition']['dominant_word_fraction'],.5)
        self.assertEqual(len(self.engine.evaluations()),2)

    def test_dead_and_stale_process_states_and_terminal_precedence(self):
        self.append(self.metric());self.engine.poll()
        self.now[0]+=200
        self.assertEqual(self.engine.snapshot()['run']['status'],'stale')
        self.host['trainer']['alive']=False;self.engine.poll()
        self.assertEqual(self.engine.snapshot()['run']['status'],'recovering')
        self.host['supervisor']['alive']=False;self.engine.poll()
        self.assertEqual(self.engine.snapshot()['run']['status'],'offline')
        self.write('status.json',dict(status='complete'));self.now[0]+=90000;self.engine.poll()
        snap=self.engine.snapshot()
        self.assertEqual(snap['run']['status'],'complete')
        self.assertEqual(snap['run']['progress_fraction'],1)
        self.assertEqual(snap['run']['remaining_seconds'],0)

    def test_input_files_never_change(self):
        self.append(self.metric());self.evaluation()
        (self.gram/'latest.pt').write_bytes(b'Not a real checkpoint. Must never be deserialized.')
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.run.rglob('*') if p.is_file()}
        self.engine.poll();self.engine.snapshot();self.engine.export_csv();self.engine.samples('eval_00001000.json')
        after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.run.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        self.assertEqual(self.engine.snapshot()['checkpoints'][0]['name'],'latest.pt')

    def test_storage_cannot_be_in_run_or_reused_for_another_run(self):
        with self.assertRaisesRegex(ValueError,'outside'):Engine(self.run,self.run/'telemetry.sqlite3')
        with self.assertRaisesRegex(ValueError,'different'):Engine(self.root/'other',self.root/'telemetry/db.sqlite3')

    def test_no_data_and_malformed_optional_files(self):
        (self.gram/'status.json').write_text('{')
        self.engine.poll()
        snap=self.engine.snapshot()
        self.assertEqual(snap['collector']['metric_rows'],0)
        self.assertIsNone(snap['validation']['latest'])
        self.assertEqual(self.engine.history(),[])

    def test_collector_error_is_reported_then_recovers(self):
        def broken():raise RuntimeError('fixture failure')
        self.engine.system_provider=broken;self.engine.poll()
        self.assertIn('fixture failure',self.engine.snapshot()['collector']['error'])
        self.engine.system_provider=lambda:self.host;self.engine.poll()
        self.assertIsNone(self.engine.snapshot()['collector']['error'])

    def test_historical_evaluation_events_use_saved_time_not_ingestion_time(self):
        self.append(dict(event='evaluation',step=1,nll_per_code=9,seconds=2))
        self.evaluation();self.engine.poll()
        self.assertEqual(len(self.engine.events()),1)
        self.assertEqual(self.engine.events()[0]['timestamp'],self.now[0])
        self.now[0]+=100;self.engine.poll()
        self.assertEqual(self.engine.events()[0]['timestamp'],self.now[0]-100)

    def test_initial_step_zero_is_not_labeled_final(self):
        self.evaluation('eval_00000000.json',step=0,nll=9.)
        self.engine.poll()
        self.assertIn('Step 0',self.engine.events()[0]['detail'])

    def test_untrained_samples_are_flagged_until_checkpoint_refresh(self):
        self.append(self.metric(2706))
        self.evaluation('eval_00000000.json',step=0,nll=9.)
        self.engine.poll();snapshot=self.engine.snapshot()
        self.assertTrue(snapshot['validation']['initial_checkpoint_only'])
        self.assertEqual(snapshot['validation']['training_steps_since_evaluation'],2706)
        self.assertIn('evaluation_initial',{a['id'] for a in snapshot['alerts']})
        self.now[0]+=10;self.evaluation('eval_00002706.json',step=2706,nll=7.2)
        self.engine.poll();snapshot=self.engine.snapshot()
        self.assertFalse(snapshot['validation']['initial_checkpoint_only'])
        self.assertEqual(snapshot['validation']['training_steps_since_evaluation'],0)
        self.assertNotIn('evaluation_initial',{a['id'] for a in snapshot['alerts']})

    def test_long_poll_interval_does_not_false_alarm(self):
        self.engine.interval=60;self.engine.poll();self.now[0]+=50
        self.assertTrue(self.engine.snapshot()['collector']['fresh'])

    def test_resource_and_missing_checkpoint_alerts(self):
        self.host.update(ram_available_bytes=1024**3,disk_free_bytes=2*1024**3,ac_power=False)
        self.append(self.metric());self.engine.poll()
        self.assertTrue({'memory','disk','power','checkpoint'}.issubset({a['id'] for a in self.engine.snapshot()['alerts']}))

    def test_final_test_is_displayed_without_changing_validation_selection(self):
        self.evaluation(nll=6.5)
        self.write('final_test.json',dict(split='test',nll_per_code=5.9,records=[]))
        self.engine.poll();snapshot=self.engine.snapshot()
        self.assertEqual(snapshot['validation']['best']['nll_per_code'],6.5)
        self.assertEqual(len(snapshot['evaluations']),2)
        self.assertEqual(self.engine.samples('final_test.json')['total'],0)

    def test_import_does_not_load_torch(self):
        root=Path(__file__).resolve().parents[3]
        result=subprocess.run([sys.executable,'-B','-c',"import sys; import v15_astra_v3.telemetry.server; assert 'torch' not in sys.modules"],cwd=root,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_waiting_model_turn_is_not_reported_as_a_dead_trainer(self):
        self.host['trainer']['alive']=False
        self.write('status.json',dict(status='waiting_turn'))
        self.append(self.metric());self.engine.poll();self.now[0]+=1900
        snapshot=self.engine.snapshot()
        self.assertEqual(snapshot['run']['status'],'waiting_turn')
        self.assertNotIn('process',{a['id'] for a in snapshot['alerts']})
        self.assertNotIn('stale',{a['id'] for a in snapshot['alerts']})

    def test_repetition_ignores_punctuation_and_case(self):
        self.assertEqual(generation_stats('Cup, CUP. cup cup?')['dominant_word_fraction'],1)
        self.assertEqual(generation_stats('')['words'],0)

    def test_native32_diagnostics_survive_evaluation_ingestion(self):
        self.write('gram/eval_native.json',dict(split='valid',step=1,nll_per_code=8,records=[],
            context_relevance=dict(mean_shuffled_minus_true_nll=.12),
            repetition=dict(mean_repeated_trigram_fraction=.25),
            end_to_end_content_tokens_per_second=24.,
            boundary_continuity=dict(same_word_seams=2,seam_count=12)))
        self.engine.poll();evaluation=self.engine.evaluations()[0]
        self.assertEqual(evaluation['context_nll_delta'],.12)
        self.assertEqual(evaluation['repeated_trigram_fraction'],.25)
        self.assertEqual(evaluation['end_to_end_content_tokens_per_second'],24.)
        self.assertEqual(evaluation['boundary_continuity']['seam_count'],12)

    def test_nested_arm_reads_preparation_from_the_project_tree(self):
        data=self.root/'artifacts/raw';data.mkdir(parents=True)
        (data/'preparation.json').write_text(json.dumps(dict(encoded_blocks=40,total_blocks=100)))
        arm=self.run/'30m';arm.mkdir()
        (arm/'plan.json').write_text(json.dumps(dict(data=str(data),target_ids=32,context_ids=256)))
        nested=Engine(arm,self.root/'telemetry/nested.sqlite3',clock=lambda:self.now[0],system_provider=lambda:self.host)
        try:
            nested.poll();snapshot=nested.snapshot()
            self.assertEqual(snapshot['dataset']['preparation']['encoded_blocks'],40)
            self.assertEqual(snapshot['dataset']['context_ids'],256)
        finally:nested.close()


class HTTPTests(unittest.TestCase):
    write = EngineTests.write
    append = EngineTests.append
    metric = EngineTests.metric
    evaluation = EngineTests.evaluation

    def setUp(self):
        EngineTests.setUp(self);self.append(self.metric());self.evaluation();self.engine.poll()
        self.server=Server(('127.0.0.1',0),self.engine)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()

    def tearDown(self):
        self.server.shutdown();self.thread.join();self.server.server_close();EngineTests.tearDown(self)

    def get(self,path,method='GET',headers=None):
        connection=HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
        connection.request(method,path,headers=headers or {})
        response=connection.getresponse();body=response.read();out=(response.status,dict(response.getheaders()),body)
        connection.close();return out

    def test_http_contract_exports_and_static_assets(self):
        for path in ['/','/style.css','/app.js','/view.js','/api/snapshot','/api/history','/api/health','/api/evaluations','/api/events','/api/resources']:
            status,headers,body=self.get(path);self.assertEqual(status,200,path);self.assertTrue(body)
            self.assertEqual(headers['X-Content-Type-Options'],'nosniff')
        status,headers,body=self.get('/api/export.csv');self.assertEqual(status,200)
        self.assertIn(b'step,timestamp,loss',body);self.assertIn('attachment',headers['Content-Disposition'])
        self.assertEqual(self.get('/api/snapshot','HEAD')[2],b'')

    def test_rejects_mutation_invalid_queries_and_path_traversal(self):
        self.assertEqual(self.get('/api/snapshot','POST')[0],405)
        for path in ['/api/history?limit=1','/api/history?limit=x','/api/history?limit=2&limit=3','/api/samples','/api/samples?evaluation=eval_00001000.json&offset=-1']:
            self.assertEqual(self.get(path)[0],400,path)
        for path in ['/%2e%2e/engine.py','/runtime/telemetry.sqlite3','/api/samples?evaluation=../../plan.json']:
            self.assertEqual(self.get(path)[0],404,path)
        self.assertEqual(self.get('/api/snapshot',headers={'Host':'attacker.example'})[0],403)

    def test_arm_selector_routes_and_rejects_unknown_arms(self):
        self.server.engines={'30m':self.engine,'10m':self.engine}
        status,_,body=self.get('/api/runs')
        self.assertEqual(status,200)
        self.assertEqual([r['id'] for r in json.loads(body)['runs']],['30m','10m'])
        self.assertEqual(self.get('/api/snapshot?arm=10m')[0],200)
        self.assertEqual(self.get('/api/snapshot?arm=missing')[0],400)
        self.assertEqual(self.get('/api/snapshot?arm=10m&arm=30m')[0],400)


if __name__=='__main__':unittest.main(verbosity=2)
