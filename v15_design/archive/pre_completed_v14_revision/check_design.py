"""Small, CPU-only checks of v15's specified math and accounting.

No torch import, checkpoint loading, model training, process control, or device work.
These are not implementation tests of the proposed architecture.
"""
from pathlib import Path
import hashlib
import json
import math
import random
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
rng = random.Random(15016)
checks = {}


def record(name, condition, detail):
    checks[name] = {"passed": bool(condition), "detail": detail}
    if not condition:
        raise AssertionError(name)


def dot(x, y):
    return sum(a*b for a, b in zip(x, y))


def matvec(a, x):
    return [dot(row, x) for row in a]


def solve(a, b):
    """Tiny pivoted Gaussian elimination, used only for independent ridge checks."""
    n = len(a)
    x = [list(row) + [v] for row, v in zip(a, b)]
    for j in range(n):
        p = max(range(j, n), key=lambda i: abs(x[i][j]))
        x[j], x[p] = x[p], x[j]
        s = x[j][j]
        if abs(s) < 1e-12:
            raise ValueError("singular toy system")
        x[j] = [v/s for v in x[j]]
        for i in range(n):
            if i != j:
                f = x[i][j]
                x[i] = [u-f*v for u, v in zip(x[i], x[j])]
    return [row[-1] for row in x]


# Counts derived from explicit matrix shapes; no v14/v15 modules are imported.
d, ha, gm, dk = 128, 16, 4, 4
write = d + ha*(d+1) + gm*dk*(ha+1) + gm*(ha+1)
read = write + gm
addresses = 13*(read+write)
coarse = d + (d+1)*d + (d+1)*32
fine = (d+1)*64 + 4*64 + 32*64 + (64+1)*32
bridge = 4*d*d + d + d + 4*d + 2*d + 1 + (d+2+4+1)
removed = 128 + 512 + 65920 + 16384 + 8192
student = 6807755 - removed - 12*d + addresses + coarse + fine + bridge
codec_block = 4*64*64 + 3*64*128 + 2*64
codec = 2048*64 + 2*16*64 + 3*codec_block + 64*64 + 64 + 64
ordinary = 2048*192 + 14*(4*192*192 + 3*192*544 + 2*192) + 192
accounting = dict(student=student, removed=removed, address_sites=addresses,
                  prior=coarse+fine, bridge=bridge, codec_trainable=codec,
                  codec_codebooks=2*32*64, ordinary_transformer=ordinary)
record("parameter_accounting", accounting == dict(student=6881071, removed=91136,
       address_sites=65884, prior=33408, bridge=66696, codec_trainable=260608,
       codec_codebooks=4096, ordinary_transformer=6849984), accounting)

B, T, groups = 2, 1024, 256
memory_mib = B*groups*128*4*4/2**20
inverse_mib = B*groups*4*4*4*4/2**20
joint_mib = B*groups*32*32*4/2**20
record("component_memory", memory_mib == 1 and inverse_mib == .125 and joint_mib == 2,
       {"M_MiB": memory_mib, "P_MiB": inverse_mib, "joint_MiB": joint_mib,
        "naive_13_M_states_MiB": 13*memory_mib, "not_total_peak_memory": True})

# Check the release/forecast age rule independently of any neural implementation.
for t in range(128):
    g = (t+1)//4-1
    if g >= 0:
        e = 4*g+3
        assert e <= t and 0 <= t-e < 4
        assert e+1 <= t+1 <= e+4
        assert (t+1)//4-1 < (t+1)//4  # incomplete next group cannot be read
for g in range(1, 16):
    e = 4*g+3
    previous_future_end = (4*(g-1)+3)+16
    assert previous_future_end > e
record("release_and_copy_baseline", True,
       "Release at row 4g+3; forecasts cover the next four predictions; previous future labels are unavailable.")

# Target-window masks are label-side only: count valid slots at document endings.
mask_examples = []
for doc_len in [4, 5, 7, 8, 15, 16, 17, 20, 31]:
    for e in range(3, doc_len, 4):
        valid = [e+1+i < doc_len for i in range(16)]
        counts = [sum(valid[4*h:4*h+4]) for h in range(4)]
        assert sum(counts) == min(16, doc_len-e-1)
        assert all(0 <= n <= 4 for n in counts)
        mask_examples.append([doc_len, e, counts])
record("partial_target_counts", True, {"cases": len(mask_examples), "examples": mask_examples[:6]})

# A conditional joint can retain paired modes that independent marginals destroy.
eps = 1e-6
j = [[.5*(1-eps), .5*eps], [.5*eps, .5*(1-eps)]]
cross_conditional = j[0][1]+j[1][0]
cross_independent = .5
record("correlated_alternatives", cross_conditional < 1e-5 and cross_independent == .5,
       {"conditional_cross_pair_mass": cross_conditional,
        "independent_marginal_cross_pair_mass": cross_independent})

root = [rng.random()+.01 for _ in range(32)]
root = [x/sum(root) for x in root]
joint = []
for c, p in enumerate(root):
    r = [rng.random()+.01 for _ in range(32)]
    s = sum(r)
    joint.extend((p*v/s, c, i) for i, v in enumerate(r))
selected = sorted(joint, key=lambda v: (-v[0], v[1], v[2]))[:4]
mass = sum(x[0] for x in selected)
tail = 1-mass
record("joint_top_four_and_tail", abs(sum(p for p, _, _ in joint)-1) < 1e-12
       and 0 <= tail <= 1 and abs(mass+tail-1) < 1e-12,
       {"retained_mass": mass, "tail_mass": tail,
        "selection": [[c, r] for _, c, r in selected]})

# Scalar projection identity verifies the nonexpansive homogeneous delta transition.
max_norm_ratio = 0.
for _ in range(200):
    k = [rng.uniform(-1, 1) for _ in range(4)]
    norm = math.sqrt(dot(k, k))
    k = [v/max(norm, 1e-6) for v in k]
    beta = rng.random()
    row = [rng.uniform(-1, 1) for _ in range(4)]
    updated = [v-beta*dot(row, k)*key for v, key in zip(row, k)]
    ratio = math.sqrt(dot(updated, updated)/dot(row, row))
    max_norm_ratio = max(max_norm_ratio, ratio)
record("delta_homogeneous_transition", max_norm_ratio <= 1+1e-12,
       {"max_norm_ratio": max_norm_ratio,
        "scope": "Fixed addresses, zero incoming value; not a full network stability test"})

# RLS variant: fixed lambda=1, alpha=1, omega in (0,1), as in this proposal.
K, D = 4, 3
P = [[float(i == j) for j in range(K)] for i in range(K)]
A = [row[:] for row in P]
M = [[0.]*K for _ in range(D)]
cross = [[0.]*K for _ in range(D)]
max_err, min_denom = 0., float("inf")
for _ in range(32):
    key = [rng.uniform(-1, 1) for _ in range(K)]
    norm = math.sqrt(dot(key, key))
    key = [v/max(norm, 1e-6) for v in key]
    value = [rng.uniform(-2, 2) for _ in range(D)]
    omega = rng.uniform(.05, .95)
    kw, vw = [math.sqrt(omega)*v for v in key], [math.sqrt(omega)*v for v in value]
    pk = matvec(P, kw)
    denom = 1+dot(kw, pk)
    min_denom = min(min_denom, denom)
    gain = [v/denom for v in pk]
    error = [v-pred for v, pred in zip(vw, matvec(M, kw))]
    M = [[v+error[i]*gain[j] for j, v in enumerate(row)] for i, row in enumerate(M)]
    kp = [sum(kw[i]*P[i][j] for i in range(K)) for j in range(K)]
    P = [[v-gain[i]*kp[j] for j, v in enumerate(row)] for i, row in enumerate(P)]
    A = [[v+kw[i]*kw[j] for j, v in enumerate(row)] for i, row in enumerate(A)]
    cross = [[v+vw[i]*kw[j] for j, v in enumerate(row)] for i, row in enumerate(cross)]
    direct = [solve(A, row) for row in cross]
    max_err = max(max_err, max(abs(a-b) for mr, dr in zip(M, direct) for a, b in zip(mr, dr)))
record("rls_equals_direct_ridge", max_err < 1e-11 and min_denom > 0,
       {"steps": 32, "max_abs_error": max_err, "minimum_denominator": min_denom})

spec = (ROOT/'v15_Architecture.md').read_text()
record("document_completeness", all(s in spec for s in ["## 16.", "V15-Δ", "V15-RLS",
       "not implemented", "future-validity", "11 device-hours", "6,881,071", "R17."]),
       {"lines": len(spec.splitlines()), "words": len(spec.split()),
        "spec_sha256": hashlib.sha256(spec.encode()).hexdigest()})

source_paths = ["v14_Architecture.md", "v14_NOTES.md", "Dataset_v13.md", "v14_tiny/model.py",
                "v14_tiny/train.py", "v14_tiny/v14.py", "v14_tiny/tests_g0.py",
                "v14_tiny/REPORT.md", "v14_tiny/DEVIATIONS.md", "v14_tiny/runs/base/config.json",
                "paper_reviews/next_concept_modeling_first_principles.md",
                "paper_reviews/depth_memory_first_principles.md"]
source_paths += ["/Users/shubhankankrishna/Desktop/RL/Papers Direct/Next-Concept Modeling.pdf",
                 "/Users/shubhankankrishna/Desktop/RL/Papers Direct/DEPTH MEMORY.pdf"]
manifest = []
for name in source_paths:
    p = Path(name) if name.startswith('/') else ROOT/name
    b = p.read_bytes()
    manifest.append({"path": str(p), "size_bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()})
last = {}
for line in (ROOT/'v14_tiny/runs/base/log.jsonl').open():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if 'step' in row:
        last = row
status = {}
for name in ['runs/base/DONE','runs/eval/base/results.json','runs/eval/diag/diagnostics_v14.json',
             'runs/post/ALL_DONE']:
    p = ROOT/'v14_tiny'/name
    status[name] = {"exists": p.exists(), "bytes": p.stat().st_size if p.exists() else None}
now = datetime.now(ZoneInfo('Asia/Kolkata')).isoformat()
evidence = {"captured_at_IST": now, "initial_design_evidence_time": "2026-10-16T01:16:59+05:30",
            "sources": manifest, "v14_status_at_manifest_capture": status,
            "last_training_log_record": last,
            "initial_observation": "Base DONE; final_eval started 01:09; no final result or diagnostic artifacts at 01:16.",
            "limitations": "No v14 final-result interpretation; no v15 model implemented or trained.",
            "source_commits": {"ncm":"c70274273f489abc73832b2b98584a8483b6b203",
                               "depth_memory":"9a6865feeb43a153a4501cc9a6457e047ffef3c3"}}
report = {"checked_at_IST": now, "scope": "Analytical toy checks, formula counts, and specification presence only",
          "all_passed": all(c['passed'] for c in checks.values()), "checks": checks}
OUT.mkdir(exist_ok=True)
(OUT/'evidence_manifest.json').write_text(json.dumps(evidence, indent=2)+'\n')
(OUT/'review_checks.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
