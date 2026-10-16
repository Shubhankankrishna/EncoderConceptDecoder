"""Reference curves for the v3 live page: v1 (run_m2_25M/runs/main) and v2 (run_v2_19M/runs/v2_48h) evaluations and
training losses, read-only, written to run_m2_25M/webapp/data/ref_v1v2.json. Same data order in all runs (seed 0), so
step k is the same tokens: these are matched-data references.   Run: python3 make_ref_v1v2.py
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "run_m2_25M", "webapp", "data", "ref_v1v2.json")
RUNS = {"v1": os.path.join(ROOT, "run_m2_25M", "runs", "main"), "v2": os.path.join(ROOT, "run_v2_19M", "runs", "v2_48h")}


def rows(path):
    out = []
    if not os.path.exists(path):
        return out
    for l in open(path):
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


def ev_row(r):
    pr = r.get("probes", {})
    cb = r.get("codebook", {})
    tel = r.get("telemetry", {})
    mix = tel.get("crc_mix_e0_L_G")
    return dict(step=r["step"], wall_h=r.get("wall_h"), val=r.get("val_lang_mean"),
                ts=r.get("val_tinystories", {}).get("nats"), wk=r.get("val_wikitext", {}).get("nats"),
                **{k[3:]: v.get("exact") for k, v in pr.items()}, swap=r.get("recall1_plan_swap", {}).get("exact"),
                VE=cb.get("var_explained"), codes_min=cb.get("codes_used_frac_min"), ncp=cb.get("ncp_ce"),
                G_share=mix[2] if mix else None)


out = {}
for name, d in RUNS.items():
    ev = [ev_row(r) for r in rows(os.path.join(d, "log.jsonl")) if "val_lang_mean" in r]
    tr = [r for r in rows(os.path.join(d, "train_log.jsonl")) if "ntp" in r and "tok_s" in r]
    step = 4 if len(tr) > 800 else 1                                     # thin long logs
    out[name] = dict(evals=ev, train=[dict(step=r["step"], ntp=r.get("ntp"), ncp=r.get("ncp"), vq=r.get("vq"),
                                           tok_s=r.get("tok_s"), gn=r.get("gn_preclip", r.get("gn")), lr=r.get("lr"),
                                           wall_h=r.get("wall_h")) for r in tr[::step]])
out["v1"]["final"] = dict(step=4050, val=2.7542, ts=1.9807, wk=3.5277, recall1_test=0.9707, wall_h=21.953,
                          loop=dict(angles=[6.70, 1.25, 0.55], rho=0.99, note="eval_final.json loop_dynamics (8 passes)"))
out["v2"]["final"] = dict(step=3089, val=3.1074, ts=2.306, wk=3.908, recall1=0.594, recall1_test_24h=0.605,
                          note="stopped by the owner after a partial 24 h evaluation")
out["gates"] = dict(E1=dict(step=1792, pass_=0.25, fail=0.05, v1=0.820), E2_steps=[1217, 1792, 2376], E2_margin=[0.10, 0.20],
                    E4=[1290, 1200], v1_angle23=1.25, v1_rho=0.99)
json.dump(out, open(OUT, "w"))
print("wrote", OUT, {k: (len(v["evals"]), len(v["train"])) for k, v in out.items() if k in RUNS})
