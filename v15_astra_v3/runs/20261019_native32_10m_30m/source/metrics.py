"""Episode/action estimands and paired cluster intervals with explicit units."""
import numpy as np


def cluster_interval(values,groups,*,estimand='example',seed=1518,draws=2000):
    values=np.asarray(values,dtype=float);groups=np.asarray(groups)
    if len(values)!=len(groups) or not len(values) or not np.isfinite(values).all():
        raise ValueError('Nonempty paired finite observations required')
    keys,idx=np.unique(groups,return_inverse=True)
    sums=np.bincount(idx,weights=values);counts=np.bincount(idx)
    if estimand not in ('example','group'):raise ValueError('Unknown estimand')
    if draws<2:raise ValueError('At least two bootstrap draws')
    means=sums/counts;rng=np.random.default_rng(seed);samples=[]
    for _ in range(draws):
        sample=rng.integers(len(keys),size=len(keys))
        samples.append(sums[sample].sum()/counts[sample].sum() if estimand=='example' else means[sample].mean())
    point=values.mean() if estimand=='example' else means.mean()
    return dict(estimand='example_weighted' if estimand=='example' else 'equal_group_weighted',
                mean=float(point),ci95=np.quantile(samples,[.025,.975]).tolist(),
                examples=len(values),groups=len(keys),bootstrap_draws=draws,seed=seed,
                sufficient_independent_groups=len(keys)>=2)


def paired_difference(a,b,groups,**kwargs):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape!=b.shape:raise ValueError('Policies must be evaluated on identical examples')
    return cluster_interval(a-b,groups,**kwargs)


def length_bucket(n):
    if n<1:raise ValueError('Nonterminal decision requires positive shortest distance')
    return '1-4' if n<5 else '5-9' if n<10 else '10-19' if n<20 else '20+'


def episode_report(records,horizons=(5,10,20)):
    """Each record is one independent rollout; include start's required length."""
    result={}
    for name in ('all','1-4','5-9','10-19','20+'):
        panel=[r for r in records if name=='all' or length_bucket(r['minimum_required_steps'])==name]
        if not panel:
            result[name]=dict(episodes=0,solved_by_horizon=None);continue
        result[name]=dict(episodes=len(panel),solved_by_horizon={str(h):cluster_interval(
            [r['steps_to_success'] is not None and r['steps_to_success']<=h for r in panel],
            [r['group'] for r in panel]) for h in horizons},
            invalid_fraction=float(np.mean([r['invalid'] for r in panel])))
    return result


def code_diagnostics(target,prediction,canonical_actions):
    target=np.asarray(target);prediction=np.asarray(prediction);canonical_actions=np.asarray(canonical_actions)
    variable=np.any(canonical_actions!=canonical_actions[0],axis=0)
    return dict(exact_code_rate=float(np.all(target==prediction,axis=-1).mean()),
                all_position_accuracy=float((target==prediction).mean()),
                variable_position_accuracy=float((target[:,variable]==prediction[:,variable]).mean()) if variable.any() else None,
                variable_positions=int(variable.sum()),constant_positions=int((~variable).sum()),
                note='Code agreement is not action correctness or text perplexity')
