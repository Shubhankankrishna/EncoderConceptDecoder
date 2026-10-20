"""Read-only telemetry extension; original registered dashboard source is untouched."""
import v15_astra_ncm_v4.dashboard as original

DEFAULT_RUN = original.ROOT / 'runs/20261019_renderer_prefix_followup'

base_snapshot = original.snapshot


def snapshot(run):
    data = base_snapshot(run)
    final = original.read(run / 'final_results.json', {})
    data['final_summary'] = [{
        'arm': name, 'selected_step': arm['selected_step'],
        'test': arm['evaluation']['test_strategy0']['rollouts']['greedy']['success_rate'],
        'ood': arm['evaluation']['ood_strategy0']['rollouts']['greedy']['success_rate']}
        for name, arm in final.get('models', {}).items()]
    diagnosis = original.read(original.ROOT / 'reports/renderer_diagnosis.json', {})
    data['diagnosis'] = {k: v for k, v in diagnosis.items() if k != 'cases'}
    followup = {'plan': original.read(DEFAULT_RUN / 'plan.json', {}),
        'status': original.read(DEFAULT_RUN / 'status.json', {'status': 'preparing'}), 'arms': []}
    followup['plan'].pop('sources', None)
    for name in ('control', 'on_policy'):
        evaluation = original.read(DEFAULT_RUN / name / 'latest_eval.json')
        if evaluation: evaluation = {**evaluation, 'samples': evaluation.get('samples', [])[:4]}
        followup['arms'].append({'name': name,
            'state': original.read(DEFAULT_RUN / name / 'status.json', {'status': 'queued'}),
            'evaluation': evaluation})
    final_followup = original.read(DEFAULT_RUN / 'final_results.json', {})
    followup['final_summary'] = {name: {k: v for k, v in result.items() if k != 'samples'}
        for name, result in final_followup.items() if name in ('parent', 'control', 'on_policy')}
    data['followup'] = followup
    return data


if __name__ == '__main__':
    original.snapshot = snapshot
    original.main()
