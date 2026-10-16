"""Export a K = 1 H-NCP checkpoint as the exact frozen-V8 System checkpoint, for the frozen evaluator (replays)."""
import argparse
import v9paths
from runtime import atomic_save
import hncp

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
    hs, ck = hncp.load(a.source, 'cpu'); s = hs.as_system()
    atomic_save(a.out, dict(metadata=s.metadata(), model=s.state_dict(), config=ck.get('config', {}), extra=dict(exported_from=a.source)))
    print('exported', a.out)
