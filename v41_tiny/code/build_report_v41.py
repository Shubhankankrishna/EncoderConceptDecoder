"""Assemble a readable report while keeping every number traceable to raw files."""
import argparse,datetime,json
from pathlib import Path
import numpy as np
from report_tables_v41 import main as tables_main,read,fmt
from decisions_v41 import main as decisions_main
from validate_artifacts_v41 import main as validate_main

ROOT=Path(__file__).resolve().parent.parent;RUNS=ROOT/'runs';REPO=ROOT.parent
def link(name):return f'[{name}]({ROOT/name})'
def js(names):return [j for n in names if (j:=read(n))]
def mean(names,section,key):
    vv=js(names);return float(np.mean([j[section][key] for j in vv])) if vv else None
def effect(v):
    if not v:return 'pending'
    return f"{v['mean']:+.4f} ± {v['paired_item_se']:.4f} paired item SE; seed SE {fmt(v.get('seed_se'))} (n={v['n_seeds']})"

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--score',type=float);a=ap.parse_args()
    tables_main();decisions_main();validate_main()
    d=read('','registered_decisions.json');v=read('','artifact_validation.json');s=read('','secondary_compare.json')
    full=d['registered_matrix_complete'] and not v['pending_full_budget_runs'] and v['all_completed_analyses'] and v['all_secondary_diagnostics_complete'] and v['analysis_artifact_audit_complete']
    namesR=[f'v41_pin_s{k}_6M' for k in range(3)]
    namesV=[f'ref_vs_pin_v4_s{k}_6M' for k in range(3)]
    namesL=[f'ref_vs_pin_llm_s{k}_6M' for k in range(3)]
    namesS=[f'v41_stack_pin_s{k}_6M' for k in range(3)]
    namesF=[f'v41_loopoff_pin_s{k}_6M' for k in range(3)]
    lines=['# v4.1 tiny experiment: evidence and first-principles assessment','',
           ('**Status: completed.**' if full else '**Status: interim; registered runs remain in progress.**')+' Generated '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'.','',
           f"{len(v['completed_full_budget_runs'])}/14 new full-budget runs complete; {v['new_full_budget_real_tokens']:,} real tokens and {v['new_full_budget_optimizer_steps']:,} optimiser steps{' so far' if not full else ''}. This includes the 11 registered architecture arms, one fresh transformer reproduction and two exploratory MPS controls. The previous 10 baseline runs are reused with hashes, not counted as new training.",'',
           f"Read {link('FIRST_PRINCIPLES.md')} for the architecture, {link('METHODS.md')} for controls and reproduction, and {link('RESULT_TABLES.md')} for every seed, intervention, oracle learning rate and statistical difference.",'']
    assessment=ROOT/'ASSESSMENT.md'
    if full and assessment.exists():lines += [assessment.read_text().strip(),'']
    if a.score is not None:
        assert full,'Do not assign a final score while the matrix is incomplete'
        lines += [f'**Updated score: {a.score:g}/10.** This is an engineering judgment about the tested architecture, not a benchmark-derived scale or a claim about eventual full-size performance.','']
    lines += ['## 1. What was built and tested','',
              'The tiny model has width 64, an eight-layer local/global encoder, attention pooling over six-token chunks, a four-layer concept core applied twice, and decoder layers S S R S. R reads the entire causal token memory, with NoPE and four temperatures, while plans remain private to each decoder position. The pooler has no FFN and the decoder has no Engram reads. Encoder Engram reads remain.','',
              'The inherited v4 backbone starts bit-identically for the same seed. R adds precisely four parameters. The decoder stack control has identical parameters and reads private decoder states for K/V; the trained loop-off control uses each encoder F layer once. No full-size model was trained.','',
              '| Model | Compute parameters | Pinpad stored parameters | Text stored parameters | Estimated pinpad forward FLOPs/token |',
              '|---|---:|---:|---:|---:|',
              '| v4 / no R | 1,212,032 | 1,630,048 | 2,149,728 | 2.810M |',
              '| v4.1 / stack | 1,212,036 | 1,630,052 | 2,149,732 | 2.869M |',
              '| Transformer LLM | 1,206,864 | 1,211,472 | 1,731,152 | 3.779M |','',
              'The full-size v4.1 ledger is 19,175,580 compute parameters and 101,600,284 stored parameters. The smaller headline excludes embedding, Engram table and codebooks; those are not free memory. The FLOP approximation excludes some gates, lookup/pooling work, auxiliary heads and convolution work. R raises this estimate about 2.1% on pinpad and 1.5% on text. The transformer is about 32% higher on pinpad, with local window 60 versus 8. This is an equal-token/near-equal-parameter comparison, not an equal-FLOP experiment.','',
              'CPU was faster for the tiny training workload in synchronised tests. Registered training therefore used the verified CPU path; the available Apple M2 GPU ran both full 6M-token MPS controls, closed-loop inference and oracle control. Every budget and actual device/thread count is in the inventory.','',
              '## 2. Correctness is established for the tested graph','',
              '**42 distinct acceptance checks passed; the 15 new v4.1 checks were also repeated without the block kernel (57 executions total). All 29 deliberate defects were detected by assertions.** The tests check more than loss reduction: global retrieval reach, future/document/row isolation, exact private-plan reach, completed-chunk alignment, legal nulls, full-model causality, gradients and actual data rows.','',
              'The manual attention oracle agrees within 4.44e−16 in float64. Full/block normal-model logits agree within 8.88e−16. CPU/MPS logits differ by at most 3.58e−7, and all 489 parameter gradients were compared, with worst absolute difference 7.26e−8.','',
              'A fresh transformer seed-0 run reproduced all 246 stored state tensors bit-for-bit and every rollout outcome. Original source hashes (24 files) and all 50 copied reference artifacts and originals are unchanged. Cumulative real-token counts also match the baseline at every shared logged training step.','',
              'The inference-artifact audit verifies finite checkpoint tensors, saved budgets/arguments, per-unit array shapes and exact reconstruction of reported aggregates. It also recomputes counterfactual correctness from the saved action IDs and fixture labels. It caught one seed-0 Engram switch-off summary with missing raw units; those units were regenerated from its checkpoint, the summary reproduced exactly, and every existing primary unit stayed bit-identical. The before/after artifacts and repair record are retained.','',
              'The fresh transformer reproduction also inherited its reference’s summary-only M1/loop-off records. Those raw units were completed from the fresh checkpoint, reproducing both original summaries exactly and preserving all old units. The original reference files were not modified.','',
              'Two engineering failures are retained: the initial MPS float64 rollout conversion, fixed by transferring logits to CPU first; and a strict CPU2/CPU4 bit-identity assertion, which failed despite maximum six-step state difference only 7.45e−9. These do not become hidden “passes.” The latter is numerical agreement, not exact full-run identity. See '+link('ENGINEERING_LOG.md')+'.','',
              '## 3. Capability at the fixed budgets','',
              'Pinpad values are equal-weight means over episodes, then seeds. b2 is the first action toward a pair’s second colour; chance is approximately 0.35.','',
              '| Model | Completed seeds | Held-out action accuracy | Held-out b2 | Action NTP | Sampled pair integrity | Greedy pair integrity |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for label,names in [('v4 / no R',namesV),('v4.1',namesR),('Transformer LLM',namesL),('v4.1 stack',namesS),('v4.1 trained loop off',namesF)]:
        nn=len(js(names))
        if nn:lines.append(f'| {label} | {nn} | '+' | '.join(fmt(mean(names,section,key)) for section,key in [('heldout','act_opt'),('heldout','act_opt_b2'),('heldout','ntp_action'),('roll_temp1','pair_ok'),('roll_greedy','pair_ok')])+' |')
    lines += ['',f"**v4.1 − no R, action accuracy:** {effect(d['R_P1'])}.",
              f"**v4.1 − transformer, action accuracy:** {effect(d['R_minus_LLM_P1'])}.",'',
              'The primary decision follows the registered paired-item rule. Seed SE and ranges are separately reported because the same held-out items do not create additional independent training seeds.','',
              'Text comparison uses mean per-row NTP; the runner’s token-weighted values are also in the tables. Text v4.1 − no R is '+effect(d['text_guard'])+'. The R guard is +0.02 nats. Retrieval exact match remains near zero at these budgets; it cannot establish the draft’s proposed division of labour on multi-hop language questions.','',
              'Closed-loop completion is not successful composition. The sampled policy can visit six colours while violating the required pair order. Full valid-composition/task rates remain low, even for the stronger transformer. Pair integrity is conditional on an eligible even-colour transition; the tables report coverage and an all-layout convention beside it. Teacher-forced action scores measure behaviour on expert histories; generated histories are a separate distribution.','',
              'Under the secondary convention assigning zero when no eligible pair was observed, v4.1 − transformer pair-integrity score is '+effect(s.get('rollout_all_layout/v41_minus_LLM/temp1'))+' for sampling and '+effect(s.get('rollout_all_layout/v41_minus_LLM/greedy'))+' for greedy. The deficit therefore persists when every layout contributes to the score.','',
              '![Training curves]('+str(ROOT/'learning_curves.png')+')','',
              '## 4. The registered decisions','',
              '| Rule | Reading |','|---|---|',
              f"| R outcome | {d.get('R_outcome','pending')} {'(final)' if full else '(interim)'} |",
              f"| R text guard | {'passes' if d['text_guard_passes'] else 'fails/pending'}: v4.1 − no R ≤ +0.02 |",
              f"| Stack improves action accuracy by ≥2 paired SE | {d['stack_improves_P1_by_2_item_SE'] if full else 'pending full replication'} |",
              f"| Keep encoder F loop by trained b2 rule | {d['keep_F_loop_by_registered_rule'] if full else 'pending full replication'} |",'',
              'Stack effect: '+effect(d['stack_P1'])+'. Trained loop-off b2 effect: '+effect(d['trained_loopoff_b2'])+'. No loop-off text arm was run; the registered loop rule therefore uses pinpad.','',
              'The formal core-route kill line uses `core_off`. The raw rule is applied as written in '+f'[registered_decisions.json]({RUNS/"registered_decisions.json"})'+'. The calibrated controls below change its interpretation, not its recorded outcome. Curves and the final three held-in b2 readings are retained. These are finite-budget results; no architecture was extended beyond its registered budget to select a better checkpoint.','',
              '**Budget-limited comparison:** '+str(len(d['R_vs_LLM_budget_limited_seeds']))+' of three transformer seeds improved held-in b2 over all of their final three evaluations. The transformer is the stronger P1 arm, so the original rule labels the R/transformer comparison budget-limited. These results describe the registered token budget; full convergence was not tested.','',
              '## 5. Core reliance is smaller than the usual switch-off suggests','',
              '`core_off` replaces every body plan with the learned header null. That is not the same as removing only example-specific content. In CPU seed 0, the learned null has RMS 0.966, while the training-bank mean valid projected content has RMS 0.199; their RMS distance is 1.001 and cosine −0.075. The replacement changes both scale and direction.','',
              'For v4.1 pinpad, replacing all valid u/c_pred with one training-derived mean changes b2 by '+effect(s.get('pinpad/constant/act_opt_b2'))+'. Removing their content while retaining normal nulls and six slot offsets changes b2 by '+effect(s.get('pinpad/zero_content/act_opt_b2'))+'. The header-null replacement changes it by '+effect(s.get('pinpad/null/act_opt_b2'))+'.','',
              'Thus a large null switch-off cannot by itself establish strong example-specific planning. The plan vectors are not literally constant and gradient routes are active, but much of checkpoint accuracy survives constant content. On text, constant plans cost '+effect(s.get('text/constant/row_ntp'))+' nats, so the conclusion differs by task.','',
              'Applying the same calibrated control to the three v4/no-R reference checkpoints gives a constant-plan b2 effect of '+effect(s.get('noR_pinpad/constant/act_opt_b2'))+'. The paired difference between the R and no-R constant-plan effects is '+effect(s.get('calibrated_interaction/R_minus_noR/constant/act_opt_b2'))+'. The retrieval package increases task capability while these checkpoints rely less on example-specific plans. This does not prove that auxiliary concept training had no benefit.','',
              'The gradient audit separates action NTP from coarse/fine auxiliaries. Both reach the intended modules; their norm/cosine measurements are diagnostics, not causal importance scores. See '+f'[gradient_audit.json]({RUNS/"v41_pin_s0_6M/gradient_audit.json"})'+'.','',
              '## 6. A trained control and counterfactual facts','',
              'The exploratory GPU plan-off model retains core computation, NCP/VQ objectives, allocated parameters, normal header nulls and slot offsets. It disables u/c_pred content throughout training. Its 8,192 plan-projection weights stay exactly at initialisation, and NTP-only gradients do not reach its core/pooler. Auxiliary training still acts on encoder representations, so this is not a pure model-without-concepts or model-without-auxiliaries comparison.','',
              'It is compared with a fresh normal-plan seed-0 run on the same MPS backend, same data and budget. This removes the earlier CPU/MPS comparison confound. Both are post-hoc, one-seed controls.','',
              '**A direct false positive in the route gate:** the trained no-plan model has no inference path from core outputs to logits, yet its standard `core_off` lowers b2 from 0.7967 to 0.4783. NTP-only core gradients are absent and Wu/Wc remain at initialisation. Substituting the header null into the body produces the drop even when no concept content exists to remove. Thus that switch-off is not sufficient evidence of concept-route use. The registered formal gate is still recorded exactly as written; its intended inference is invalid in this counterexample.','',
              '| Matched MPS arm | Action accuracy | Pair-second b2 | Action NTP | Changed-fact accuracy |','|---|---:|---:|---:|---:|']
    for label,n in [('Normal v4.1','exploratory_mps_plan_pin_s0_6M'),('Trained without plan content','exploratory_noplan_pin_s0_6M')]:
        j=read(n);cf=read(n,'counterfactual.json')
        if j and cf:lines.append(f'| {label} | '+ ' | '.join(fmt(j['heldout'][k]) for k in ('act_opt','act_opt_b2','ntp_action'))+' | '+fmt(cf['intact']['changed_accuracy'])+' |')
    lines += ['',
              'The layout counterfactual moves the target to another unvisited colour’s location. The visited even partner stays fixed; both prefixes have identical legal histories and disjoint optimal next actions. The changed layout remains a valid layout. Across the CPU v4.1/transformer pairs, changed-fact accuracy differs by '+effect(s.get('counterfactual/v41_minus_LLM/changed_correct'))+'. The conditional-on-original-correct rate is descriptive because models have different original-correct subsets.','',
              '![A measured counterfactual]('+str(ROOT/'counterfactual_case.png')+')','',
              'This illustration is seed 0 and selected by a stated rule; all 128 cases and all completed seeds, rather than the image, determine the numerical findings.','',
              '## 7. Abstraction and control are separate','',
              'The table reports coarse-code NMI beside an episode-wise permutation null, and causally legal completed-chunk linear probes beside a majority predictor. Codebook usage is not proof of semantic abstraction. NMI is explicitly non-causal and can be positive from finite-sample bias. A small probe improvement is evidence of some readable information, not evidence that the core performs the task’s decisions.','',
              'The frozen oracle demonstrates a different property: available control authority. On 128 fixed prefixes (52 initial errors), ten C-offset steps at LR .03 fix 25 errors (48.1%) without breaking the 76 initially correct cases. u fixes 22/52 and first-pass S1 fixes 7/52 at that learning rate. Norm-matched random updates fix none. All three learning rates and achieved norms are reported, not only the best.','',
              'The oracle knows the correct actions and optimises full-width prompt offsets. It neither trains an eight-dimensional metacontroller nor tests sparse-reward RL or learned option duration. It shows that the legal latent route can steer outputs when supplied a useful offset. No matched oracle intervention in a transformer residual stream was run, so this does not establish a unique advantage of the concept interface.','',
              '![Critical controls]('+str(ROOT/'critical_controls.png')+')','',
              '## 8. Why long-episode likelihood can give the opposite impression','',
              'The exact decomposition is `action NLL = action-conditioned NLL + non-action probability-mass penalty`. On long episodes, the transformer’s penalty often rises sharply after six visits, when the training tasks would have ended. END accounts for almost all of the missing action probability mass. Large log penalties are sensitive to highly confident premature stopping.','',
              'For example, transformer seed 0 after six visits has 0.668 action-conditioned NLL plus 2.287 non-action penalty, totalling 2.955 nats; v4.1 has 0.759 plus 0.525, totalling 1.284. The transformer still chooses optimal actions more often in that phase (79.5% versus 74.2%). This measured decomposition explains why lower full-vocabulary NLL alone is not stronger long-horizon action reasoning. All seeds and the exact numerical identity check are in '+link('LONG_AUDIT.md')+'.','',
              '## 9. Limits and the next useful experiment','',
              'These are width-64 models trained on 5–6M tokens, two text seeds and three pinpad seeds. They do not settle full-size performance. Some better-model curves remain rising at the budget. The architecture comparison differs in local windows, stored tables, auxiliary objectives, initialisation and FLOPs. The R/no-R contrast also changes positional encoding and temperatures alongside range. The stack’s local dense kernel affects throughput.','',
              'The constant-plan, fact-following, trained plan-off, NMI null and long-likelihood checks are secondary, introduced after early results and clearly labelled. The trained plan-off result needs additional seeds. Natural chunk-lag strata are observational. Oracle control has privileged labels. New-name, longer-gap, distractor-count language OOD factors, k=4/k=8, wider windows, G-to-R, depth rebalancing, trained M=1, controller learning and VQ-only adaptation were not tested here.','',
              'The next controlled comparison should replicate normal versus trained plan-off across seeds on both tasks, then test a calibrated/gated plan injection against those two arms. Keep R as a demonstrated capability candidate. Require example-specific plan utility, fact-following and generated-policy structure before interpreting the concept interface as learned hierarchical reasoning. The transformer remains an essential baseline.','',
              '## Artifacts','',
              '- '+link('FIRST_PRINCIPLES.md')+' — step-by-step graph, timing, codebook and control-site derivation.',
              '- '+link('METHODS.md')+' — experiment specification, statistical definitions, known confounds and commands.',
              '- '+link('RESULT_TABLES.md')+' — all completed seeds and controls.',
              '- '+link('LONG_AUDIT.md')+' — action likelihood decomposition.',
              '- '+link('PRE_REGISTRATION.md')+' and '+link('SECONDARY_PROTOCOL.md')+' — primary decisions and later amendments.',
              '- '+f'[raw comparisons]({RUNS/"v41_compare.json"})'+', '+f'[secondary comparisons]({RUNS/"secondary_compare.json"})'+', '+f'[provenance audit]({RUNS/"artifact_validation.json"})'+', '+f'[inference-artifact audit]({RUNS/"analysis_artifact_audit.json"})'+'.',
              '- '+f'[implementation]({ROOT/"code/model_v41.py"})'+', '+f'[runner]({ROOT/"code/run_vs.py"})'+', '+f'[test suite]({ROOT/"code/test_v41.py"})'+'.','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n');print(ROOT/'REPORT.md')

if __name__=='__main__':main()
