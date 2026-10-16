"""G3 (register §12.3, draft §7.3), per S3 arm and seed, dev only, against the promoted S2 model of that seed.
G2 rows of the S3 model are computed with register_s2.rows_for_seed against the same V9-K1 reference as in S2."""
import numpy as np
import register_s2 as R
from register_s2 import boot_mean, boot_text_diff, zone

ORDER = {'pass': 3, 'partial': 2, 'inconclusive': 2, 'reported': 2, 'fail': 1}


def deploy_text(meas):
    return meas['nll']['text']['argmax' if meas['k'] > 1 else 'codefree']


def g3(s3, s2, k1, s5_depth=1):
    """s3, s2: measure_s2 outputs (S3 arm, promoted S2 model); k1: that seed's V9-K1 measure (G2 reference)."""
    r3 = R.rows_for_seed(s3, k1) if s3['k'] > 1 else None
    r2 = R.rows_for_seed(s2, k1) if s2['k'] > 1 else None
    out = {}
    std = boot_mean(np.array(s3['greedy']['standard']['success'], float) - np.array(s2['greedy']['standard']['success'], float))
    txt = boot_text_diff(deploy_text(s3), deploy_text(s2))
    out['standard_vs_S2'] = dict(measure=std, zone='pass' if std['interval'][0] >= -.02 else 'fail')
    out['text_vs_S2'] = dict(measure=txt, zone='pass' if txt['interval'][1] <= .01 else 'fail')
    out['hard10_vs_S2'] = boot_mean(np.array(s3['greedy']['hard10']['success'], float) - np.array(s2['greedy']['hard10']['success'], float))
    out['code_free_drift'] = s3['drift'].get('code_free_drift')
    if r3 is None:
        out['authority'] = 'not applicable (V9-K1: no codes; plain continued-pretraining check, G2.2(d) consequence)'
        out['G3'] = 'pass' if out['standard_vs_S2']['zone'] == out['text_vs_S2']['zone'] == 'pass' else 'fail'
        return out
    s3s = r3['G2.1b']['sensitivity']['point']; s2s = r2['G2.1b']['sensitivity']['point']
    out['G2.1b'] = dict(s3=s3s, s2=s2s, ratio=s3s / max(s2s, 1e-9), zone_s3=r3['G2.1b']['zone'], ok=r3['G2.1b']['zone'] == 'pass' and s3s >= .8 * s2s)
    out['G2.2'] = dict(zone=r3['G2.2']['zone'], ok=r3['G2.2']['zone'] == 'pass')
    keep = {}
    for H in sorted({1, s5_depth}):
        keep[f'G2.3_H{H}'] = (r3['G2.3_H'][H], r2['G2.3_H'][H], ORDER[r3['G2.3_H'][H]] >= ORDER[r2['G2.3_H'][H]])
    for row in ('G2.4_branching', 'G2.4_coverage', 'G2.4_pred_exec'):
        keep[row] = (r3[row]['zone'], r2[row]['zone'], ORDER[r3[row]['zone']] >= ORDER[r2[row]['zone']])
    out['transitions_no_worse'] = {k: dict(s3=v[0], s2=v[1], ok=v[2]) for k, v in keep.items()}
    out['G3'] = 'pass' if (out['G2.1b']['ok'] and out['G2.2']['ok'] and all(v[2] for v in keep.values())
                           and out['standard_vs_S2']['zone'] == 'pass' and out['text_vs_S2']['zone'] == 'pass') else 'fail'
    return out


def decide(per_arm):
    """per_arm: {arm: [g3 per seed]}. Seed rule (§12.3 rule 4): an arm passes G3 only if both seeds pass; a split takes the
    status quo (S2 forward). If several arms pass, S3a-dec is preferred (the predicted safe arm), then S3b."""
    status = {arm: ('pass' if all(g['G3'] == 'pass' for g in gs) else 'seed-dependent' if any(g['G3'] == 'pass' for g in gs) else 'fail') for arm, gs in per_arm.items()}
    winner = next((a for a in ('dec', 'joint', 'enc') if status.get(a) == 'pass'), None)
    h11 = None
    if 'dec' in per_arm and 'joint' in per_arm and all('G2.1b' in g for g in per_arm['dec'] + per_arm['joint']):
        h11 = dict(dec_keeps=[g['G2.1b']['ratio'] >= .8 for g in per_arm['dec']], joint_drop=[1 - g['G2.1b']['ratio'] for g in per_arm['joint']],
                   joint_loses_20pct=[g['G2.1b']['ratio'] <= .8 for g in per_arm['joint']])
    return dict(status=status, replaces_S2=winner, H11_descriptive=h11, note='S2 model goes forward if no arm passes; S3 then reported as a negative')
