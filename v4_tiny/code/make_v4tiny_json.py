"""Compile every v4_tiny result into run_m2_25M/webapp/data/v4_tiny.json for the dashboard's "Tiny experiment" tab.
Stdlib only; reads v4_tiny/runs/* (logs, final.json, decomp*.json, recall_check.json, reads_check.json, test logs)."""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.normpath(os.path.join(HERE, "..", "runs"))
OUT = "/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/data/v4_tiny.json"
ARMS = ["k4_attnffn_s0", "k6_attnffn_s0", "k6_attnffn_p07", "k6_linear_p07", "k6_gconv_p07"]
LABEL = {"k4_attnffn_s0": "k4 · attention + FFN · p 0.5", "k6_attnffn_s0": "k6 · attention + FFN · p 0.5",
         "k6_attnffn_p07": "k6 · attention + FFN · p 0.7", "k6_linear_p07": "k6 · linear / plain conv (v3) · p 0.7",
         "k6_gconv_p07": "k6 · gated conv (owner) · p 0.7"}


def jl(p):
    out = []
    for line in open(p):
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def arm(name):
    d = os.path.join(RUNS, name)
    rows = jl(os.path.join(d, "log.jsonl"))
    meta = next(r for r in rows if r.get("kind") == "meta")
    train = [r for r in rows if r.get("kind") == "train"]
    evals = [dict(step=r["step"], real=r["real_tokens"], val=r["val_ntp"], probes=r["probes"], swap=r.get("swap"),
                  ppl0=r["codebook"]["coarse_ppl"], ve=r["codebook"]["ve"]) for r in rows if r.get("kind") == "eval"]
    fin = load(os.path.join(d, "final.json"))
    dec = load(os.path.join(d, "decomp_full.json")) or load(os.path.join(d, "decomp.json"))
    decp = load(os.path.join(d, "decomp_probe.json"))
    decomp = None
    if dec:
        so = dict(dec["switch_offs"])
        if decp:
            so.update({k: v for k, v in decp["switch_offs"].items() if k not in so})
        decomp = dict(switch_offs=so, write_shares=dec.get("write_shares"),
                      text_by_slot=dec.get("text_by_slot"), text_pos=dec.get("text_position_var_share"),
                      facts_by_slot=dec.get("facts_by_slot"), facts_pos=dec.get("facts_position_var_share"))
    final = None
    if fin:
        final = dict(val=fin["val_ntp"], probes=fin["probes"], interventions=fin["interventions"], codebook=fin["codebook"],
                     pool=fin.get("pool"), steps=fin["step"], real=fin["real_tokens"], wall=fin["wall"])
    return dict(name=name, label=LABEL[name], args=meta["args"], params=meta["params"],
                tok_s=train[-1]["tok_s"] if train else None, steps=train[-1]["step"] if train else None, evals=evals, final=final,
                decomp=decomp, recall=load(os.path.join(d, "recall_check.json")), reads=load(os.path.join(d, "reads_check.json")))


def passes(path):
    if not os.path.exists(path):
        return []
    return [l.strip()[5:] for l in open(path) if l.startswith("PASS ")]


def main():
    res = dict(source="v4_tiny/runs · REPORT.md · GO_NOGO.md (2026-10-01)", arms=[arm(a) for a in ARMS])
    res["tests"] = passes(os.path.join(RUNS, "tests_block60.log"))
    res["tests_noblock_pass"] = sum(1 for _ in passes(os.path.join(RUNS, "tests_noblock.log")))
    mut = [l.rstrip() for l in open(os.path.join(RUNS, "mutants.log")) if re.match(r"^(KILLED|SURVIVED)", l)]
    res["mutants"] = [dict(status=m.split("|")[0].strip(), name=m.split("|")[1].strip(), caught=m.split("|")[2].strip()[:160]) for m in mut]
    res["v3_heartbeats"] = dict(
        reference=[["13:52", 2907], ["13:57", 3000], ["14:01", 3042], ["14:06", 3010], ["14:10", 3028], ["14:15", 3177], ["14:19", 3229], ["14:23", 3171]],
        two_arms=[["14:30", 2474], ["14:36", 2294], ["14:42", 2333], ["14:48", 2246], ["14:54", 2249], ["15:01", 2121]],
        three_arms=[["15:06", 2472], ["15:13", 2074], ["15:20", 2088], ["15:26", 2101], ["15:34", 2075], ["15:41", 2108], ["15:48", 2077], ["15:54", 2172],
                    ["16:01", 1954], ["16:08", 1892], ["16:15", 1922], ["16:22", 1948], ["16:29", 1935], ["16:38", 1958], ["16:45", 1963], ["16:52", 2040],
                    ["16:58", 2160], ["17:18", 2036], ["17:25", 1871], ["17:32", 2173]],
        tail=[["17:41", 1920], ["17:48", 1979], ["17:57", 1563], ["18:05", 1728]], recovered=[["18:10", 2900], ["18:34", 3358]])
    res["advisor_consults"] = sum(1 for l in open(os.path.join(RUNS, "..", "ADVISOR_LOG.md")) if re.match(r"^\| \d+ \|", l))
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("wrote", OUT, len(res["arms"]), "arms,", len(res["tests"]), "tests,", len(res["mutants"]), "mutants,", res["advisor_consults"], "consults")


if __name__ == "__main__":
    main()
