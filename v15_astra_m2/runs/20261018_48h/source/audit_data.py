"""Independent reconstruction of raw dataset labels, grouping and current inputs."""
from collections import Counter,defaultdict
import json

import numpy as np

from .codec import ROOT,sha256
from .data import DATA,atomic_json
from .tasks import observed,step,distances,optimal_actions,group_split


def main():
    source=json.loads((DATA/'source_manifest.json').read_text())
    for name,digest in source['files'].items():assert sha256(DATA/name)==digest
    rows=dict(np.load(DATA/'samples.npz'));texts=json.loads((DATA/'texts.json').read_text())
    episodes=[json.loads(line) for line in (DATA/'episodes.jsonl').read_text().splitlines()]
    groups=defaultdict(set);lengths=defaultdict(Counter);domains=defaultdict(Counter)
    cache={};verified=0;multivalid=0
    for e in episodes:
        groups[e['split']].add(e['group']);lengths[e['split']][len(e['trajectory'])]+=1
        assert e['split']==('ood' if e['env']['ood'] else group_split(e['group']))
        key=(e['group'],e['goal'])
        if key not in cache:cache[key]=distances(e['env'],e['goal'])
        d=cache[key];state=e['initial']
        assert d[state]==len(e['trajectory'])
        for r in e['trajectory']:
            assert state==r['current']
            assert observed(e['env'],state,e['goal'])==(r['text'],r['state'])
            opts=optimal_actions(e['env'],state,d)
            assert r['valid_actions']==opts and r['action'] in opts
            state=step(e['env'],state,r['action']);assert state==r['next']
            verified+=1;multivalid+=len(opts)>1;domains[e['split']][e['env']['domain']]+=1
        assert state==e['goal']
    names=['train','valid','test','ood']
    for a in names:
        for b in names:
            if a!=b:assert not groups[a]&groups[b]
    for i in range(len(rows['x'])):
        e=episodes[int(rows['episode'][i])];r=e['trajectory'][int(rows['transition'][i])]
        assert rows['split'][i]==names.index(e['split'])
        assert texts[rows['x'][i]]==r['text'] and np.array_equal(rows['state'][i],r['state'])
        assert rows['y'][i]==rows['action'][i]==r['action']
    assert verified==source['records']==len(rows['x']) and rows['state'].max()<128
    atomic_json(ROOT/'reports/data_audit.json',dict(verified_transitions=verified,group_overlap=False,
        groups={s:len(v) for s,v in groups.items()},domains={s:dict(v) for s,v in domains.items()},
        minimum_required_steps={s:dict(v) for s,v in lengths.items()},
        multiple_optimal_actions=multivalid,state_min=int(rows['state'].min()),state_max=int(rows['state'].max()),
        exact_current_state_reconstruction=True,label_and_index_alignment=True,
        tests_used_for='integrity only; no model outcomes or checkpoint selection'))
    print('Verified',verified,'transitions; disjoint groups; exact current-state inputs and labels')


if __name__=='__main__':main()
