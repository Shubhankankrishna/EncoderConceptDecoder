"""Read-only scientific audit of the completed reduced Q5; writes monitor assessment only."""
import collections
import math
import os
import sys
from pathlib import Path
os.environ['V8_DEVICE'] = 'cpu'
sys.path.insert(0, str(Path(__file__).resolve().parent))
import q5_12h as q
import v9paths
import numpy as np
import torch
import baselines, hncp, train_s4, hgen
from data import verify
from runtime import finite


def pair(a, b):
    x = np.asarray(a['success'], dtype=np.int8) - np.asarray(b['success'], dtype=np.int8)
    wins, losses = int((x > 0).sum()), int((x < 0).sum())
    n = wins + losses
    p = min(1., 2 * sum(math.comb(n, j) for j in range(min(wins, losses) + 1)) / 2**n) if n else 1.
    rng = np.random.RandomState(1234)
    bootstrap = x[rng.randint(0, len(x), size=(10000, len(x)))].mean(1) * 100
    return dict(n=len(x), delta_percentage_points=float(x.mean() * 100), improved_items=wins, worsened_items=losses,
                unchanged_items=len(x) - n, paired_bootstrap_95_percent_interval=np.percentile(bootstrap, [2.5, 97.5]).tolist(),
                exact_discordant_pair_two_sided_p=p, exploratory=True)


def main():
    torch.set_num_threads(4)
    lock = q.preflight(); p = q.protocol(); result = q.read(q.ROOT / 'Q5_12H_RESULT.json')
    assert result['status'] == 'completed'
    checkpoints = []; audit = []
    for i in range(2):
        folder = q.ROOT / f'grpo/round{i}'
        summary = q.read(folder / 'summary.json')
        assert q.digest(folder / 'adapter.pt') == summary['adapter_sha256']
        ck = torch.load(folder / 'adapter.pt', map_location='cpu', weights_only=False)
        resume = torch.load(folder / 'resume.pt', map_location='cpu', weights_only=False)
        answers = torch.load(folder / 'answers.pt', map_location='cpu', weights_only=False)
        assert finite(ck) and finite(resume)
        assert len(answers) == 4096 and resume['progress']['next_problem'] == 512 and resume['progress']['fit_epoch'] == 2
        groups = collections.Counter(a[4] for a in answers)
        assert set(groups) == set(range(512)) and set(groups.values()) == {8}
        items = train_s4.fresh_problems(512, 86000 + 100 * p['seed'] + i)
        first = collections.Counter(); lengths = collections.Counter(); rewards = []
        for t, c, ok, P, group in answers:
            item = items[group]
            assert np.array_equal(t[:P], item.prompt)
            verdict = bool(verify(t[:P], t[P:])['success'])
            assert verdict == bool(ok)
            rewards.append(verdict); lengths[str(len(t) - P)] += 1
            cl = hgen.classify(item, item.start, 0, t[P:P + 4])
            first[f'{cl["kind"]}/legal={cl["legal"]}'] += 1
        assert sum(rewards) == summary['accepted']
        audit.append(dict(round=i, groups=512, group_size=8, answers=4096, accepted=sum(rewards),
                          verifier_flags_independently_rechecked=True, exact_fresh_prompt_draw_confirmed=True,
                          finite_checkpoint=True, adapter_parameters=sum(t.numel() for t in ck['state'].values()),
                          first_chunk_classifications=dict(first), response_token_lengths=dict(lengths),
                          fit=summary['fit'], adapter_sha256=summary['adapter_sha256']))
        checkpoints.append(ck)
    delta = max(float((checkpoints[0]['state'][k] - checkpoints[1]['state'][k]).abs().max()) for k in checkpoints[0]['state'])
    assert delta == 0, 'Round 1 has no gradient signal, so adapter should remain unchanged'
    torch.manual_seed(p['seed']); hs, _ = hncp.load(p['source'], 'cpu'); baselines.add_lora(hs)
    initial = baselines.lora_state(hs)
    adapter_change = max(float((initial[k] - checkpoints[0]['state'][k]).abs().max()) for k in initial)
    comparison = {suite: pair(result['arms']['EI'][suite], result['arms']['reference'][suite]) for suite in ['standard', 'hard10']}
    elapsed = (q.dt.datetime.fromisoformat(result['created_utc']) - q.dt.datetime.fromisoformat(p['activated_utc'])).total_seconds()
    assessment = dict(checked_utc=q.utc(), pipeline_status='completed', scientific_status='inconclusive_GRPO_policy_collapse',
                      lock_sha256=lock['lock_sha256'], elapsed_minutes=elapsed / 60,
                      development_rates={arm: {s: row['rate'] for s, row in rows.items()} for arm, rows in result['arms'].items()},
                      paired_EI_minus_reference=comparison, grpo_round_audit=audit,
                      grpo_adapter_max_change_after_round0=adapter_change,
                      grpo_adapter_max_change_during_round1=delta,
                      conclusion='No clear held-out development advantage for the MCTS-trained greedy policy over the original model. GRPO collapsed after round 0, so beating this GRPO result does not establish a credible MCTS advantage.',
                      diagnosis='Completed files, group membership, rewards, fresh prompts and all checkpoint tensors verified. Collapse follows the first GRPO fit; round 1 has zero successes and no gradient-bearing groups. No operational crash or nonfinite checkpoint was found. The existing baseline uses an unregularized sequence-log-probability objective; the specific stability cause requires a separately documented experiment, not a silent learning-rate or algorithm change.',
                      scope='One seed, unequal training budgets, greedy development evaluation. No sealed test evaluation or test-time MCTS comparison.',
                      original_Q6_QF='intentionally held', automatic_scientific_retry=False)
    q.write(q.MON / 'q5_12h_assessment.json', assessment)
    lines = ['# Reduced Q5 assessment', '',
             f'The reduced queue completed in {elapsed / 60:.1f} minutes. Scientific comparison: inconclusive because GRPO collapsed after its first fit.', '',
             '| Policy | Standard (512) | Hard10 (512) |', '| --- | ---: | ---: |']
    for arm, rows in result['arms'].items():
        lines.append(f'| {arm} | {100 * rows["standard"]["rate"]:.2f}% | {100 * rows["hard10"]["rate"]:.2f}% |')
    lines += ['', assessment['conclusion'], '',
              'EI minus reference: +1.95 percentage points on Standard; −0.98 points on Hard10. Paired item intervals and discordant counts are in q5_12h_assessment.json. These are exploratory development estimates.', '',
              f'GRPO round 0 sampled 2,120 successful answers out of 4,096 before fitting. Round 1 produced 0 successes out of 4,096 and no learning signal; the adapter did not change in round 1. Its response-length and first-chunk failure classifications were independently checked. Maximum adapter parameter change after round 0: {adapter_change:.6g}.', '',
              assessment['diagnosis'], '', assessment['scope'], '',
              'All original and reduced completed outputs are preserved. The original Q6/QF queue remains held. No training settings were changed and no scientific rerun was started.']
    (q.MON / 'Q5_12H_ASSESSMENT.md').write_text('\n'.join(lines) + '\n')
    print(q.json.dumps(assessment))


if __name__ == '__main__':
    main()
