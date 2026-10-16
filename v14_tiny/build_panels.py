"""Build and cache every evaluation panel once, from the frozen generators (CPU only)."""
import os, sys, time
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import evaluate as E

if __name__ == '__main__':
    t0 = time.time()
    test, dev, ch = E.base_cells()
    E.build_panel('base_test', test)
    E.build_panel('base_dev', dev)
    nw = int(os.environ.get('PANEL_WORKERS', 4))                 # run at background priority: one per efficiency core
    E.build_panel('challenge', ch, workers=nw)
    E.build_panel('cl_test', E.cl_cells(24, 'test'), workers=nw)
    print(f'all panels in {(time.time() - t0) / 60:.1f} min', flush=True)
