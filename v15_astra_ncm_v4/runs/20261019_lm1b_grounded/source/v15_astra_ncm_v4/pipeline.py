"""Planning-to-language integration: only verified learned plans reach the renderer."""
import torch

from .evaluate import predictor,rollout
from .graphs import step
from .model import Agent,AgentConfig
from .ncm import decoded_text
from .adapt import parse_path


def load_agent(checkpoint,device='mps'):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    agent=Agent(AgentConfig(**saved['config'])).to(device)
    agent.load_state_dict(saved['model'],strict=True);agent.eval()
    return agent,saved['step']


@torch.no_grad()
def solve_and_render(graph,agent,net,conditioner,inference,strategy=0,search=False,temperature=0.):
    device=next(agent.parameters()).device
    result=rollout(graph,predictor(agent,device),strategy,'mcts' if search else 'greedy',16)
    # No oracle repair/fallback can turn a neural failure into a reported success.
    if not result['success']:
        return {'planning':result,'rendered':False,'reason':'Planner did not produce a verified successful route'}
    for a,b in zip(result['path'],result['path'][1:]):step(graph,a,b)
    cond=conditioner([result['path']],[strategy],net.device)
    codes=net.generate_codes(cond,temperature=temperature)
    text=decoded_text(net.codec[0],codes,inference,False)[0]
    return {'planning':result,'rendered':True,'plain_text':text,'parsed_path':parse_path(text),
            'parsed_path_matches_selected_plan':parse_path(text)==result['path'],
            'canonical_answer':result['path'],'decoder_temperature':temperature}
