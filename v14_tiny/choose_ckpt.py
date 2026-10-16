"""Activation checkpointing for the V14-tiny base run, decided from two 60-step smoke tests on the idle machine
(DEVIATIONS.md item 7, fixed before either ran): checkpointing OFF if that smoke finished, its peak MPS driver memory
(steps >= 10) is at most 9.0 GB, and its median s/step (steps >= 10, the gradient-share step 3 excluded) is at least
3% below the checkpointed smoke's; otherwise ON. Prints 'on' or 'off' and writes runs/base/ckpt_choice.json."""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    p = HERE / 'runs' / name / 'smoke.jsonl'
    if not p.exists(): return None
    r = [json.loads(l) for l in open(p)]
    r = [x for x in r if x['step'] >= 10]
    if len(r) < 40: return None
    return dict(steps=len(r), median_s=st.median(x['s_step'] for x in r), peak_gb=max(x['mem_gb'] for x in r),
                median_tok_s=st.median(x['tok_s'] for x in r))


def main():
    off, on = read('smoke_off'), read('smoke_on')
    choice = 'on'
    if off is not None and off['peak_gb'] <= 9.0 and (on is None or off['median_s'] <= 0.97 * on['median_s']):
        choice = 'off'
    (HERE / 'runs' / 'base').mkdir(parents=True, exist_ok=True)
    (HERE / 'runs' / 'base' / 'ckpt_choice.json').write_text(json.dumps(dict(off=off, on=on, choice=choice), indent=1))
    print(choice)


if __name__ == '__main__':
    main()
