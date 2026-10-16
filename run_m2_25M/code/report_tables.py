"""Generate the numeric tables for REPORT.md directly from run files (so every number traces to a file).
  python report_tables.py > ../runs/report_tables.md"""
import json, os, glob
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
PROBES = ["lr_recall1", "lr_carry3", "lr_multi4", "lr_hop", "lr_bind"]


def rd(p):
    return json.load(open(p)) if os.path.exists(p) else None


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def f(x, n=4):
    return "—" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def screens():
    print("## Screens (stopped at step 265 of a 280-step cosine; validation split)\n")
    print("| arm | step | val TinyStories | val WikiText | mean | " + " | ".join(PROBES) + " | recall1 plan swap | codes used (min seg, restart window) | VE |")
    print("|---|---|---|---|---|" + "---|" * len(PROBES) + "---|---|---|")
    for s in ("S1", "S2", "S3"):
        el = jl(f"{R}/screens/{s}/log.jsonl")
        fin = [e for e in el if e["tag"] == "final"]
        if not fin:
            continue
        e = fin[-1]
        pr = e.get("probes_256", e["probes"])
        rs = [r for r in jl(f"{R}/screens/{s}/train_log.jsonl") if "restart_dead" in r]
        print(f"| {s} | {e['step']} | {f(e['val_tinystories']['nats'])} | {f(e['val_wikitext']['nats'])} | {f(e['val_lang_mean'])} | "
              + " | ".join(f(pr[p]['exact'], 3) for p in PROBES) + f" | {f(e['recall1_plan_swap']['exact'], 3)} | "
              + f"{f(rs[-1]['codes_used_min_seg'], 3) if rs else '—'} | {f(e['codebook']['var_explained'], 3)} |")
    m = [e for e in jl(f"{R}/main/log.jsonl") if e["tag"] == "step265"]
    if m:
        e = m[-1]
        print(f"| main @265 (warm-up, not annealed) | 265 | {f(e['val_tinystories']['nats'])} | {f(e['val_wikitext']['nats'])} | {f(e['val_lang_mean'])} | "
              + " | ".join(f(e['probes'][p]['exact'], 3) for p in PROBES) + f" | {f(e['recall1_plan_swap']['exact'], 3)} | — | {f(e['codebook']['var_explained'], 3)} |")
    print()


def main_curve():
    el = jl(f"{R}/main/log.jsonl")
    if not el:
        return
    print("## Main run: hourly evaluations (validation)\n")
    print("| tag | step | wall h | tokens (M) | val mean | " + " | ".join(p[3:] for p in PROBES) + " | swap recall1 | codes min (eval) | VE | NCP CE | code H | eval s |")
    print("|---|---|---|---|---|" + "---|" * len(PROBES) + "---|---|---|---|---|---|")
    for e in el:
        print(f"| {e['tag']} | {e['step']} | {e['wall_h']:.2f} | {e['tokens']/1e6:.1f} | {f(e['val_lang_mean'])} | "
              + " | ".join(f(e['probes'][p]['exact'], 3) for p in PROBES)
              + f" | {f(e['recall1_plan_swap']['exact'], 3)} | {f(e['codebook']['codes_used_frac_min'], 2)} | {f(e['codebook']['var_explained'], 3)}"
              + f" | {f(e['codebook']['ncp_ce'], 3)} | {f(e['codebook']['code_entropy'], 3)} | {e['eval_sec']:.0f} |")
    print()


def final(name="eval_final.json"):
    d = rd(f"{R}/main/{name}")
    if not d:
        return
    print(f"## Final evaluation `{name}` (step {d['step']}, R = {d['eval_R']})\n")
    if "lang" in d:
        print("### Language (test), nats / bits per byte\n")
        keys = list(d["lang"]["tinystories"].keys())
        print("| condition | TinyStories nats | TinyStories bpb | WikiText nats | WikiText bpb |")
        print("|---|---|---|---|---|")
        for k in keys:
            a, b = d["lang"]["tinystories"][k], d["lang"]["wikitext"][k]
            print(f"| {k} | {f(a['nats'])} | {f(a['bpb'])} | {f(b['nats'])} | {f(b['bpb'])} |")
        print()
    if "trunc" in d:
        print("### Truncation (last 16 positions)\n")
        print("| set | r72 | r128 | r256 | r512 | r1024 | gain beyond 70 | plan-swap cost | rows ctx≥512 | gain (ctx≥512) | swap cost (ctx≥512) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for n, t in d["trunc"].items():
            s = t["summary"]; c = s["curve"]
            print(f"| {n} | " + " | ".join(f(c[f'r{r}']) for r in (72, 128, 256, 512, 1024))
                  + f" | {f(s['gain_beyond_70'])} | {f(s['swap_cost'])} | {s['rows_ctx_ge_512']} | {f(s['gain_beyond_70_ctx512'])} | {f(s['swap_cost_ctx512'])} |")
        print()
    if "items" in d:
        for split, r in d["items"].items():
            conds = list(r.keys())
            print(f"### Items, {split}: exact match\n")
            fams = [k for k in r["intact"] if not k.startswith("_")]
            print("| family | " + " | ".join(conds) + " |")
            print("|---|" + "---|" * len(conds))
            for fam in sorted(fams, key=lambda x: (x not in PROBES, x)):
                print(f"| {fam} | " + " | ".join(f(r[c].get(fam, {}).get('exact'), 3) if fam in r[c] else "—" for c in conds) + " |")
            print(f"| **macro (synthetic)** | " + " | ".join(f(r[c].get('_macro_synthetic'), 3) for c in conds) + " |")
            for fam in ("lr_carry3", "lr_multi4"):
                if "by_offset" in r["intact"].get(fam, {}):
                    print(f"\nper-offset accuracy {fam} (intact): {r['intact'][fam]['by_offset']}")
            if "lr_hop" in r["intact"]:
                print("\nlr_hop by depth (intact): " + ", ".join(f"{k}={v:.3f}" for k, v in r["intact"]["lr_hop"].items() if k.startswith("exact_h")))
                for c in conds:
                    if c.startswith("R"):
                        print(f"  {c}: " + ", ".join(f"{k}={v:.3f}" for k, v in r[c]["lr_hop"].items() if k.startswith("exact_h")))
            print()
    if "probes" in d:
        print("### Linear probes (held-out top-1)\n")
        for k, v in d["probes"].items():
            print(f"- {k}: " + ", ".join(f"{kk} {vv['acc']:.3f} (maj {vv['majority_baseline']:.3f}, classes {vv['classes']})" for kk, vv in v.items()))
        print()
    if "health" in d:
        print("### Codebook health and loop dynamics\n")
        print("```\n" + json.dumps(d["health"]["codebook"], indent=1) + "\n```")
        print("```\n" + json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, list) else v) for k, v in d["health"]["loop_dynamics_lr_hop"].items()}, indent=1) + "\n```\n")


if __name__ == "__main__":
    print("# Report tables (generated by code/report_tables.py from runs/)\n")
    screens(); main_curve(); final("eval_final.json"); final("eval_best.json")
