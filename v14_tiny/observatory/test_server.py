"""Fixture-only telemetry checks. Never start training or touch experiment files."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from server import Engine, JsonlReader, JsonReader, pipeline, processes, clean


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'runs/base').mkdir(parents=True)
        (self.root / 'runs/post').mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, path, value):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
        return p

    def test_partial_append_and_restart_rollback(self):
        p = self.root / 'log.jsonl'
        p.write_text('{"step":1}\n{"step":100}\n{"step":200}\n{"step":')
        reader = JsonlReader(p)
        self.assertEqual([r['step'] for r in reader.read()], [1, 100, 200])
        with p.open('a') as f:
            f.write('50}\ninvalid\n{"step":75}\n')
        self.assertEqual([r['step'] for r in reader.read()], [1, 50, 75])
        self.assertEqual(reader.invalid, 1)
        p.write_text('{"step":2}\n')
        self.assertEqual(reader.read(), [{'step': 2}])

    def test_non_atomic_json_retains_last_good(self):
        p = self.write('result.json', {'value': 1})
        reader = JsonReader()
        self.assertEqual(reader.read(p), {'value': 1})
        p.write_text('{')
        self.assertEqual(reader.read(p), {'value': 1})
        self.assertIn(str(p), reader.pending)
        p.write_text('{"value": 2}')
        self.assertEqual(reader.read(p), {'value': 2})
        self.assertFalse(reader.pending)
        p.unlink()
        self.assertIsNone(reader.read(p))

    def test_pipeline_failure_survives_all_done(self):
        (self.root / 'runs/post/status.log').write_text('Thu Oct 15 16:00:00 IST 2026 start final_eval\nThu Oct 15 16:10:00 IST 2026 end final_eval exit 1\nThu Oct 15 16:10:01 IST 2026 start diagnostics\n')
        (self.root / 'runs/post/ALL_DONE').touch()
        post = pipeline(self.root, False)
        self.assertEqual(post['stages'][0]['state'], 'failed')
        self.assertEqual(post['stages'][1]['state'], 'interrupted')
        self.assertTrue(post['finished'])
        self.assertEqual(post['stages'][0]['exit_code'], 1)

    def test_missing_logs_are_pending_not_zero_scores(self):
        state = Engine(self.root, self.root / 'monitor').collect(system=False)
        self.assertEqual(state['base']['status'], 'waiting')
        self.assertIsNone(state['artifacts']['evaluation'])
        self.assertIsNone(state['base']['throughput'])
        self.assertIsNone(state['base']['eta'])

    def test_active_done_and_fallback_states(self):
        self.write('runs/base/config.json', {'hours': 11, 'started': '2026-10-15 14:07:37'})
        (self.root / 'runs/base/log.jsonl').write_text(json.dumps({'step': 100, 'h': 1, 'tokens': 1000, 'tok_s': 42, 'fam': 'text', 'ntp': 3})+'\n')
        engine = Engine(self.root, self.root / 'monitor')
        engine.procs = [{'script': 'train.py'}, {'script': 'post_base_v14.sh'}]
        state = engine.collect(system=False)
        self.assertEqual(state['base']['status'], 'training')
        self.assertAlmostEqual(state['base']['progress'], 1/11)
        self.assertIsNotNone(state['base']['eta'])
        self.assertEqual(state['base']['families']['text']['median_ntp'], 3)
        engine.procs = []
        self.assertEqual(engine.collect(system=False)['base']['status'], 'interrupted')
        (self.root / 'runs/base/DONE_FALLBACK').touch()
        self.assertEqual(engine.collect(system=False)['base']['status'], 'fallback')
        (self.root / 'runs/base/DONE').touch()
        self.assertEqual(engine.collect(system=False)['base']['status'], 'complete')

    def test_partial_final_results_and_per_family_verdicts(self):
        self.write('runs/eval/base/results.json', {'reasoning': .1234, 'text_nll': 2.1})
        self.write('runs/eval/diag/diagnostics_v14.json', {'V': {'V1': {'passes': False}, 'V3': {'per_family': {'grid': {'passes': True}}}}, 'per_ckpt': {'snap_010': {}}})
        self.write('runs/eval/cl/cl_results.json', {'seq': {'A': {'base': {'text_nll': 2}}}})
        state = Engine(self.root, self.root / 'monitor').collect(system=False)
        self.assertEqual(state['artifacts']['evaluation']['reasoning'], .1234)
        self.assertFalse(state['artifacts']['diagnostics']['V']['V1']['passes'])
        self.assertNotIn('passes', state['artifacts']['diagnostics']['V']['V3'])
        self.assertNotIn('metrics', state['artifacts']['continual']['seq'])

    def test_processes_require_matching_working_directory(self):
        output = '10 1 20.0 1024 01:00 /usr/bin/python3 train.py --run runs/base\n20 1 30.0 2048 01:00 /usr/bin/python3 final_eval.py\n30 1 0.0 1024 01:00 /bin/zsh -c "echo train.py"\n'
        def fake(args):
            if args[0] == '/bin/ps':
                return output
            return 'p10\nfcwd\nn' + str(self.root.resolve()) + '\n' if args[3] == '10' else 'n/another/experiment\n'
        with patch('server.command', side_effect=fake):
            rows = processes(self.root)
        self.assertEqual([r['pid'] for r in rows], [10])

    def test_nonfinite_values_are_json_safe(self):
        self.assertEqual(clean({'x': float('nan'), 'y': [float('inf'), 3]}), {'x': None, 'y': [None, 3]})

    def test_background_collector_persists_and_serves_snapshot(self):
        engine = Engine(self.root, self.root / 'monitor')
        with patch('server.machine', return_value={'time': time.time(), 'swap_gb': 0}), patch('server.processes', return_value=[]):
            worker = threading.Thread(target=engine.run)
            worker.start()
            for _ in range(100):
                if engine.snapshot() or engine.error:
                    break
                time.sleep(.01)
            engine.stop.set()
            worker.join(2)
        self.assertIsNone(engine.error)
        self.assertEqual(len(engine.snapshot()['history']), 1)
        self.assertTrue((engine.runtime / 'history.sqlite3').exists())


if __name__ == '__main__':
    unittest.main()
