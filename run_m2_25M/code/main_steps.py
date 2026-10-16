"""Main-run cosine length from screen S1's measured speed (PROMPT 6.1 step 3; DECISIONS D-021):
total_steps = floor((24 h - 24 x eval_sec - 24 x ckpt_sec) / sec_per_step), sec_per_step from S1's steady-state tok/s."""
import json, sys, statistics
run = sys.argv[1] if len(sys.argv) > 1 else "../runs/screens/S1"
tl = [json.loads(l) for l in open(f"{run}/train_log.jsonl") if '"tok_s"' in l]
el = [json.loads(l) for l in open(f"{run}/log.jsonl")]
tok_s = statistics.median(r["tok_s"] for r in tl[1:]) if len(tl) > 1 else tl[0]["tok_s"]
sec_step = 32768 / tok_s
eval_sec = max(r["eval_sec"] for r in el if r["tag"] == "hourly") if any(r["tag"] == "hourly" for r in el) else max(r["eval_sec"] for r in el)
ckpt_sec = 15.0
steps = int((24 * 3600 - 24 * (eval_sec + ckpt_sec)) / sec_step)
out = dict(tok_s_median=tok_s, sec_per_step=sec_step, eval_sec=eval_sec, ckpt_sec_assumed=ckpt_sec, total_steps=steps,
           tokens=steps * 32768, eval_fraction=24 * eval_sec / 86400)
json.dump(out, open(f"{run}/../../main_steps.json", "w"), indent=1)
print(steps)
