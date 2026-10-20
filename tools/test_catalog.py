"""Regression checks for publication exclusions and saved-run interpretation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import catalog


class CatalogTests(unittest.TestCase):
    def test_bulk_bodies_are_local_but_manifest_and_tokenizer_are_publishable(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'graphs.json'
            p.write_text('{}')
            self.assertIn('bulk dataset', catalog.exclusion('v15_astra_ncm_v4/data/graphs.json', p))
            p = Path(directory) / 'manifest.json'
            p.write_text('{}')
            self.assertIsNone(catalog.exclusion('v15_astra_ncm_v4/data/manifest.json', p))
            p = Path(directory) / 'tokenizer.json'
            p.write_text('{}')
            self.assertIsNone(catalog.exclusion('v15_astra_m2/artifacts/codec/tokenizer.json', p))

    def test_upstream_and_live_runtime_stay_local(self):
        p = Path('upstream.py')
        self.assertIn('upstream checkout', catalog.exclusion('v15_astra_m2/references/ncm/models/common.py', p))
        self.assertTrue(catalog.exclusion('v15_astra_v3/telemetry/runtime/service.json', p).startswith('live telemetry/'))

    def test_completed_stopped_and_slice_statuses_and_plan_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for name, status in [('r2_seed42', 'complete'), ('retired_seed43', 'stopped'),
                                 ('partial_seed44', 'slice_complete'), ('replaced_seed45', 'stop_requested')]:
                folder = root / 'astra' / 'runs' / name
                folder.mkdir(parents=True)
                (folder / 'status.json').write_text(json.dumps({'status': status}))
                (folder / 'plan.json').write_text(json.dumps({'seed': int(name[-2:])}))
                records.extend({'path': str(p.relative_to(root))} for p in folder.iterdir())
            with patch.object(catalog, 'ROOT', root):
                runs = catalog.discover_runs(records, [{'id': 'astra', 'paths': ['astra']}])
            actual = {Path(r['id']).name: (r['status'], r['seed']) for r in runs}
            self.assertEqual(actual, {'r2_seed42': ('completed', 42),
                                      'retired_seed43': ('interrupted', 43),
                                      'partial_seed44': ('partial-slice-completed', 44),
                                      'replaced_seed45': ('interrupted', 45)})


if __name__ == '__main__':
    unittest.main()
