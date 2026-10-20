"""CPU-only architectural diagnostics; reads train/validation, never test/OOD.

Run from the repository root with v15_astra_m2/.venv/bin/python.
Does not load neural checkpoints, touch MPS, or modify the running experiment.
"""
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v15_astra_m2.tasks import ACTIONS, step

OUT = Path(__file__).resolve().parent
DATA = ROOT / 'v15_astra_m2/artifacts/data'
RUN = ROOT / 'v15_astra_m2/runs/20261018_48h'
LEVELS = (1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entropy(probs):
    p = np.asarray(probs)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main():
    started = time.perf_counter()
    sources_before = {p.name: sha(p) for p in (ROOT/'v15_astra_m2').glob('*.py')}
    rows = dict(np.load(DATA/'samples.npz'))
    # Read only train/validation episode objects for analysis.
    episodes = {}
    for i, line in enumerate((DATA/'episodes.jsonl').open()):
        e = json.loads(line)
        if e['split'] in ('train', 'valid'):
            episodes[i] = e
    codes = np.load(DATA/'codes.npy', mmap_mode='r')[:7].copy()
    assert np.array_equal(codes, np.load(ROOT/'v15_astra_m2/artifacts/lm1b_action_codes.npy'))
    assert np.unique(codes, axis=0).shape == (7, 641)
    train = np.flatnonzero(rows['split'] == 0)
    valid = np.flatnonzero(rows['split'] == 1)
    fixed = np.random.default_rng(1518).choice(valid, 128, replace=False)
    train_counts = np.bincount(rows['action'][train], minlength=7)
    p = train_counts / train_counts.sum()
    variable = np.array([len(np.unique(codes[:, j])) > 1 for j in range(641)])
    levels = []
    offset = 0
    for n in LEVELS:
        lo, hi = offset, offset+n
        prefix_groups = defaultdict(list)
        for a in range(7):
            prefix_groups[tuple(codes[a, :lo])].append(a)
        # Minimum expected teacher-forced per-coordinate CE for a predictor
        # that sees only true preceding levels and no problem/context/state.
        cond_bits = 0.
        for aa in prefix_groups.values():
            mass = p[aa].sum()
            for j in range(lo, hi):
                symbols = defaultdict(float)
                for a in aa:
                    symbols[int(codes[a, j])] += p[a] / mass
                cond_bits += mass * entropy(list(symbols.values()))
        levels.append(dict(level=len(levels)+1, positions=n,
                           variable_positions=int(variable[lo:hi].sum()),
                           unique_action_prefixes=int(np.unique(codes[:, :hi], axis=0).shape[0]),
                           no_context_teacher_forced_minimum_sum_bits=cond_bits))
        offset = hi
    # Same diagnostic with known task-domain; domain is observable from inputs.
    domain_entropy = sum(p[aa].sum()*entropy(p[aa]/p[aa].sum()) for aa in ([0,1,2],[3,4,5,6]))

    def baseline_result(indices):
        records = []
        for idx in indices:
            e = episodes[int(rows['episode'][idx])]
            r = e['trajectory'][int(rows['transition'][idx])]
            env, s, goal = e['env'], r['current'], e['goal']
            legal = [(a, step(env, s, a)) for a in range(7)]
            legal = [(a, t) for a, t in legal if t is not None]
            # No BFS/distance-to-goal oracle enters the proposal. Evaluate all
            # legal one-step successors using a fixed geometric heuristic.
            def heuristic(t):
                if env['domain'] == 'grid':
                    n = env['n']
                    return abs(t % n - goal % n) + abs(t // n - goal // n)
                d = abs(t-goal)
                return min(d, env['m']-d)
            best = min(heuristic(t) for a,t in legal)
            tied = [a for a,t in legal if heuristic(t) == best]
            optimal = set(r['valid_actions'])
            records.append(dict(domain=env['domain'], group=e['group'],
                                remaining=r['remaining'],
                                fixed=float(tied[0] in optimal),
                                tie_expected=sum(a in optimal for a in tied)/len(tied),
                                uniform=len(optimal)/len(legal)))
        def aggregate(rr):
            return dict(n=len(rr), fixed_tie_optimal_rate=float(np.mean([r['fixed'] for r in rr])),
                        uniform_tie_expected_optimal_rate=float(np.mean([r['tie_expected'] for r in rr])),
                        uniform_legal_expected_optimal_rate=float(np.mean([r['uniform'] for r in rr])))
        ans=aggregate(records)
        ans['by_domain']={d:aggregate([r for r in records if r['domain']==d]) for d in ('grid','mod')}
        ans['by_remaining_steps']={name:aggregate(rr) for name,lo,hi in [('1',1,1),('2-4',2,4),('5-9',5,9),('10-20',10,20)] if (rr:=[r for r in records if lo<=r['remaining']<=hi])}
        ans['legality_rate']=1.
        return ans

    def alias_analysis(indices):
        groups = defaultdict(list)
        for idx in indices:
            if rows['state'][idx, 0] != 2:
                continue
            e = episodes[int(rows['episode'][idx])]
            r = e['trajectory'][int(rows['transition'][idx])]
            groups[int(rows['x'][idx])].append(set(r['valid_actions']))
        conflicting = [(x, opts) for x,opts in groups.items() if not set.intersection(*opts)]
        # Best possible deterministic text-only action on this empirical panel
        # (a diagnostic oracle bound, NOT a trained/evaluated baseline).
        best_correct = sum(max(sum(a in opts for opts in optsets) for a in range(3,7)) for optsets in groups.values())
        n = sum(len(opts) for opts in groups.values())
        return dict(grid_rows=n, unique_text_prompts=len(groups),
                    prompts_with_no_common_optimal_action=len(conflicting),
                    rows_in_those_prompts=sum(len(opts) for _,opts in conflicting),
                    empirical_text_only_maximum_optimal_rate=best_correct/n,
                    note='Same prompt/code can describe different walls. Bound applies only to this empirical panel; state-conditioned model is not subject to it.')

    hist = Counter(len(e['trajectory']) for e in episodes.values() if e['split']=='valid')
    evals = []
    for path in sorted((RUN/'gram').glob('eval_*.json')):
        e = json.loads(path.read_text())
        evals.append({k:e[k] for k in ('step','training_seconds','nll_per_code','exact_code_rate','code_accuracy','optimal_action_rate','legal_action_rate','by_domain','gram_diagnostics','optimal_action_group_ci')})
    latest = evals[-1]
    current = json.loads((RUN/'gram/status.json').read_text())
    manifest = json.loads((RUN/'gram/manifest.json').read_text())
    param_groups=defaultdict(int)
    for name, info in manifest['trainable_parameters'].items():
        param_groups[name.split('.')[0]] += info['numel']
    output=dict(
        timestamp_ist=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),
        scope='CPU-only train/validation structural review; no test/OOD scoring; no neural inference or training',
        dataset_hashes={p:sha(DATA/p) for p in ('manifest.json','samples.npz','codes.npy')},
        source_hashes=sources_before,
        action_codes=dict(actions=list(ACTIONS), train_action_counts=train_counts.tolist(),
                          action_entropy_bits=entropy(p), action_entropy_given_domain_bits=domain_entropy,
                          maximum_action_entropy_bits=math.log2(7),
                          variable_positions=int(variable.sum()),constant_positions=int((~variable).sum()),
                          unique_action_codes=7, levels=levels,
                          context_free_teacher_forced_optimal_nll_per_code=sum(l['no_context_teacher_forced_minimum_sum_bits'] for l in levels)*math.log(2)/641,
                          nominal_raw_hierarchy_bits=641*13,
                          median_pairwise_code_agreement=float(np.median([(codes[i]==codes[j]).mean() for i in range(7) for j in range(i)]))),
        baselines=dict(definition='Choose legal successor minimizing Manhattan distance (grid) or circular residue distance (mod); fixed ties use existing action order. Uniform tie expectation also reported. No exact shortest-path oracle used in choice.',
                       fixed_validation_128=baseline_result(fixed),full_validation=baseline_result(valid)),
        prompt_aliases=dict(train=alias_analysis(train),validation=alias_analysis(valid)),
        validation_episode_lengths=dict(episodes=sum(hist.values()),histogram=dict(sorted(hist.items())),
                                       require_at_least_10=sum(v for k,v in hist.items() if k>=10),
                                       require_at_least_20=sum(v for k,v in hist.items() if k>=20)),
        parameter_groups=dict(param_groups),
        latest_validation=latest,validation_history=evals,live_status=current,
        source_hashes_unchanged=sources_before=={p.name:sha(p) for p in (ROOT/'v15_astra_m2').glob('*.py')},
        seconds=time.perf_counter()-started)
    assert output['source_hashes_unchanged']
    (OUT/'diagnostics.json').write_text(json.dumps(output,indent=2))
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(ROOT/'v15_astra_m2/artifacts/codec/lm1b/bert-tokenizer/tokenizer.json'))
    collisions = []
    for a,b in [('Foo = 1','foo = 1'),('A is different from a.','a is different from A.'),('résumé','resume')]:
        x,y = tokenizer.encode(a),tokenizer.encode(b)
        collisions.append(dict(left=a,right=b,same_ids=x.ids==y.ids,left_tokens=x.tokens,right_tokens=y.tokens))
    (OUT/'pretokenizer_collisions.json').write_text(json.dumps(dict(
        scope='Pre-tokenizer only; deterministic NCM receives identical input IDs for each collision; no codec inference.',
        collisions=collisions),indent=2,ensure_ascii=False))
    print(json.dumps({k:output[k] for k in ('timestamp_ist','action_codes','baselines','prompt_aliases','validation_episode_lengths','parameter_groups','source_hashes_unchanged','seconds')},indent=2))


if __name__ == '__main__':
    main()
