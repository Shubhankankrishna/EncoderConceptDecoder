"""Complete observed-state serialization and loss-aware tokenizer acceptance."""
import json
from .tasks import observed

SCHEMA='astra-observed-state-v3'


def serialize_observation(env,state,goal):
    if env['domain']=='grid':
        obj=dict(domain='grid',n=env['n'],walls=env['walls'],state=state,goal=goal)
    elif env['domain']=='mod':
        obj=dict(domain='mod',m=env['m'],a=env['a'],b=env['b'],state=state,goal=goal)
    else:raise ValueError('Unknown environment')
    return json.dumps(obj,sort_keys=True,separators=(',',':'))


def parse_observation(text):
    obj=json.loads(text);domain=obj.get('domain')
    fields={'domain','n','walls','state','goal'} if domain=='grid' else {'domain','m','a','b','state','goal'}
    if domain not in ('grid','mod') or set(obj)!=fields:raise ValueError('Unexpected observation fields')
    if domain=='grid':
        if not isinstance(obj['n'],int) or not 1<=obj['n']<=9:raise ValueError('Grid outside v3 state representation')
        if len(obj['walls'])!=obj['n']**2 or any(w not in (0,1) for w in obj['walls']):raise ValueError('Invalid walls')
        size=obj['n']**2
    else:
        size=obj['m']
        if any(not isinstance(obj[k],int) or not 1<=obj[k]<128 for k in ('m','a','b')):raise ValueError('Invalid modular rule')
    if any(not isinstance(obj[k],int) or not 0<=obj[k]<size for k in ('state','goal')):raise ValueError('Invalid state/goal')
    if domain=='grid' and any(obj['walls'][obj[k]] for k in ('state','goal')):raise ValueError('State on wall')
    state=obj.pop('state');goal=obj.pop('goal')
    return obj,state,goal


def fixed_observed_blocks(token_ids,*,block_size=128,blocks=4,pad_id=0):
    """Never silently truncate. Padding is observed-input padding, not target length."""
    if block_size<1 or blocks<1:raise ValueError('Positive block dimensions required')
    if len(token_ids)>block_size*blocks:
        raise ValueError(f'Observation has {len(token_ids)} IDs; capacity is {blocks*block_size}; increase versioned block count')
    ids=list(token_ids)+[pad_id]*(block_size*blocks-len(token_ids))
    return [ids[i:i+block_size] for i in range(0,len(ids),block_size)]


def tokenizer_collisions(encode,pairs):
    return [dict(left=a,right=b,same_ids=list(encode(a))==list(encode(b))) for a,b in pairs]


def require_representation_contract(*,context_source,complete_observation,codec_roundtrip_passed,
                                    case_sensitive_domain=False,case_preserving=False):
    if case_sensitive_domain and not case_preserving:raise ValueError('Tokenizer loses required case distinctions')
    if context_source in ('codes','codes_and_state') and not complete_observation:
        raise ValueError('v3 code-context experiments require complete observed-state text')
    if context_source=='codes' and not codec_roundtrip_passed:
        raise ValueError('Code-only symbolic experiment requires a domain fidelity acceptance audit')
