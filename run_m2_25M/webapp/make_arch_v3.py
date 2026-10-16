"""Build the static proposed-v3 page data from checked architecture and review artifacts."""
from pathlib import Path
import json, shutil

WEB = Path(__file__).resolve().parent
ROOT = WEB.parents[1]
REVIEW = ROOT / 'v3_design_review_2026-09-29'
base = json.loads((WEB / 'data/arch.json').read_text())
checks = json.loads((REVIEW / 'design_checks.json').read_text())
snapshot = json.loads((REVIEW / 'snapshot.json').read_text())
depth = json.loads((REVIEW / 'checkpoint_001703_diagnostics.json').read_text())
assert checks['passed']
assert base['table'] == checks['params']
assert base['macs']['total'] == checks['analytical_macs_per_token']['2.5']['total']
assert depth['step'] == 1703

base['proposal'] = {
    'status': 'Design only; v3 has not been implemented or trained',
    'source': 'proposed_v3.md',
    'evidence_cutoff': snapshot['time'],
    'scratchpad': 'S0 = exact zeros; each pass rereads C and writes S',
    'ordinary_depth': {'R2': 0.75, 'R4': 0.25},
    'hard_families_R4': ['lr_hop', 'pointer_chase', 'graph_walk'],
    'inference_modes': [2, 4],
    'backpropagation': 'Full through every executed pass',
    'loss': 'L_token + lambda_answer(s) L_answer + 0.1 L_NCP(final) + L_VQ',
    'answer_ramp': 'lambda_answer(s) = 0.1 min(1, s/300)',
    'optimizer': 'AdamW (0.9, 0.95), eps 1e-8, matrix decay 0.1, global clip 1',
    'schedule': '300-step warm-up to 6e-4, one cosine to 6e-5 over 4,050 steps',
    'slots_per_step': 2 * 16 * 1024,
    'slot_budget': 4050 * 2 * 16 * 1024,
    'macs_by_depth': checks['analytical_macs_per_token'],
    'comparison_arms': [
        ['A', 'historical-style v1 reproduction', 'legacy masks', 'uniform R1–4', 'no answer term'],
        ['B', 'corrected v1 control', 'corrected masks', 'uniform R1–4', 'no answer term'],
        ['C', 'short-depth candidate', 'corrected masks', 'R2/R4', 'no answer term'],
        ['D', 'proposed v3', 'corrected masks', 'R2/R4', 'ramp to 0.1'],
    ],
}
v1 = snapshot['runs']['v1']
v2 = snapshot['runs']['v2']
base['measured'] = {
    'label': 'Historical evidence only; no v3 measurement',
    'v1_final_test_tinystories': v1['final']['lang']['tinystories']['intact']['nats'],
    'v1_final_test_wikitext': v1['final']['lang']['wikitext']['intact']['nats'],
    'v1_recall_exact': v1['final']['items']['test']['intact']['lr_recall1']['exact'],
    'v1_final_R1_tinystories': v1['final']['lang']['tinystories']['R1']['nats'],
    'v1_final_R4_tinystories': v1['final']['lang']['tinystories']['R4']['nats'],
    'v1_equal_time_step': 2559,
    'v1_equal_time_val_mean': next(x['val_lang_mean'] for x in v1['eval'] if x['step'] == 2559),
    'v1_equal_time_recall_exact': next(x['probes']['lr_recall1']['exact'] for x in v1['eval'] if x['step'] == 2559),
    'v2_eval_step': v2['eval'][-1]['step'],
    'v2_val_mean': v2['eval'][-1]['val_lang_mean'],
    'v2_recall_exact': v2['eval'][-1]['probes']['lr_recall1']['exact'],
    'v2_sample_R1_tinystories': depth['language_summary']['tinystories']['1'],
    'v2_sample_R6_tinystories': depth['language_summary']['tinystories']['6'],
    'v2_sample_R1_wikitext': depth['language_summary']['wikitext']['1'],
    'v2_sample_R6_wikitext': depth['language_summary']['wikitext']['6'],
}
out = WEB / 'data/arch_v3.json'
out.write_text(json.dumps(base, indent=2) + '\n')
shutil.copyfile(ROOT / 'proposed_v3.md', WEB / 'data/proposed_v3.md')
print('Wrote', out, 'and the full proposal')
