"""Cheap exact-data audits. No model, MPS, codec, or legacy data writes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from .tasks import group_split,distances,optimal_actions,step,observed
from .representation import serialize_observation,parse_observation,fixed_observed_blocks,tokenizer_collisions
from .baselines import bfs_plan
from .isolation import LEGACY,atomic_json,sha256


def validate_episodes(episodes,excluded_groups=()):
    groups={};ids=set();forbidden=set(excluded_groups);rows=0
    for e in episodes:
        env=e['env'];group=e['group'];split=e['split']
        if e['episode'] in ids:raise ValueError('Duplicate episode')
        ids.add(e['episode'])
        if group in forbidden:raise ValueError('Legacy topology reused')
        if split not in ('train','valid','test') or group_split(group)!=split:raise ValueError('Split rule changed')
        if group in groups and groups[group]!=split:raise ValueError('Family crosses splits')
        groups[group]=split
        if env['domain']=='grid':
            expected='grid:'+hashlib.sha256(json.dumps([env['n'],env['walls']]).encode()).hexdigest()
            if expected!=group:raise ValueError('Topology identity mismatch')
        if env.get('group')!=group:raise ValueError('Environment group mismatch')
        dist=distances(env,e['goal']);s=e['initial'];trace=e['trajectory']
        if not trace or dist.get(s)!=len(trace) or len(trace)!=e['minimum_required_steps']:
            raise ValueError('Required length must be shortest distance')
        for r in trace:
            if r['current']!=s or r['remaining']!=dist[s]:raise ValueError('State or distance mismatch')
            if r['text']!=serialize_observation(env,s,e['goal']):raise ValueError('Observation is incomplete')
            parse_observation(r['text'])
            if r['state']!=observed(env,s,e['goal'])[1]:raise ValueError('Typed state mismatch')
            opts=optimal_actions(env,s,dist)
            if sorted(r['valid_actions'])!=sorted(opts) or r['action'] not in opts:raise ValueError('Wrong action supervision')
            s=step(env,s,r['action'])
            if s!=r['next']:raise ValueError('Transition mismatch')
            rows+=1
        if s!=e['goal']:raise ValueError('Teacher plan does not reach goal')
        if len(bfs_plan(env,e['initial'],e['goal'])['actions'])!=len(trace):raise ValueError('Independent BFS mismatch')
    return dict(episodes=len(episodes),groups=len(groups),transitions=rows,
                splits=dict(Counter(groups.values())),exact_transition_checks='passed')


def audit_dataset(path):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    if sha256(path/'episodes.jsonl')!=manifest['episodes_sha']:raise ValueError('Dataset hash changed')
    episodes=[json.loads(s) for s in (path/'episodes.jsonl').read_text().splitlines()]
    legacy=json.loads((LEGACY/'artifacts/data/source_manifest.json').read_text())
    report=validate_episodes(episodes,legacy['group_splits'])
    # Loading just the local tokenizer is cheap and does not load NCM weights.
    from tokenizers import Tokenizer
    token_path=LEGACY/'artifacts/codec/lm1b/bert-tokenizer/tokenizer.json'
    tok=Tokenizer.from_file(str(token_path));lengths=[]
    for e in episodes:
        for r in e['trajectory']:
            ids=tok.encode(r['text']).ids;lengths.append(len(ids))
            fixed_observed_blocks(ids,blocks=manifest['required_observed_blocks'])
    report.update(episodes_sha=manifest['episodes_sha'],pretokenizer_sha=sha256(token_path),
                  max_observation_token_ids=max(lengths),capacity=manifest['required_observed_blocks']*128,
                  no_truncation=True,codec_reconstruction_tested=False,
                  tokenizer_collisions=tokenizer_collisions(lambda s:tok.encode(s).ids,
                      [('Foo','foo'),('résumé','resume')]),
                  recipes=dict(Counter(e['recipe'] for e in episodes)),
                  initial_heuristic_failures=sum(not e['initial_heuristic_optimal'] for e in episodes))
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('dataset',type=Path);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();result=audit_dataset(args.dataset);atomic_json(args.out,result);print(json.dumps(result,indent=2))
