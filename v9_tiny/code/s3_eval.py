"""S3 transplant check (draft §7.2 item 1, contract 2): u = (1-lam) u_T + lam u_S, A = (1-lam) port_T(u_T) + lam A_S through
System.plans, with the student's code-free path; full-recompute engine (cached generation bypasses System.plans)."""
import argparse, copy
import numpy as np
import torch
import v9paths
from v9paths import SMOKE, write_json
from model import System, freeze
from runtime import DEVICE, load_system
import hncp, s6

LAMS = (0., .3, .6, 1.)


class LamSystem(System):
    lam = 1.

    def plans(self, c, cd, z=None, lam=1.0):
        return super().plans(c, cd, z, self.lam)


def transplant(hs, lam):
    s = LamSystem(copy.deepcopy(hs.f)); s.core = copy.deepcopy(hs.core); s.lam = lam; freeze(s); return s.to(DEVICE).eval()


def curve(source, teacher_path, out):
    hs, _ = hncp.load(source, DEVICE); teacher, _ = load_system(teacher_path, DEVICE); freeze(teacher); teacher.eval()
    n_std = 64 if SMOKE else 1024; n_txt = 32 if SMOKE else 256; rows = {}
    for lam in LAMS:
        s = transplant(hs, lam)
        rows[str(lam)] = dict(standard=s6.symbolic(s, 'dev_standard', n_std)['rate'], lookup=s6.symbolic(s, 'dev_lookup', min(n_std, 512))['rate'],
                              text_edu=s6.text(s, 'base', 'edu', n_txt)['nll'], text_ts=s6.text(s, 'base', 'ts', n_txt)['nll'])
    r = dict(source=str(source), teacher=str(teacher_path), lams=list(LAMS), rows=rows,
             note='student side = code-free (u0, A0); lam = 0 reproduces the teacher exactly (contract 2)')
    write_json(out, r); return r


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--teacher', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
    torch.set_num_threads(4); curve(a.source, a.teacher, a.out)
