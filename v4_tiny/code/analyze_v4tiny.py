"""Summarise every arm in v4_tiny/runs: eval curves (log.jsonl) and final evaluations (final.json).
python analyze_v4tiny.py [--md out.md]   (stdlib only; safe to run while arms train)"""
import glob
import json
import math
import os
import sys

RUNS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"))
FAM = ("lr_recall1", "lr_carry3", "lr_multi4")


def load(path):
    out = dict(meta=None, train=[], evals=[], restarts=[], final=None)
    for line in open(path):
        r = json.loads(line)
        k = r.get("kind")
        if k == "meta":
            out["meta"] = r
        elif k == "train":
            out["train"].append(r)
        elif k == "eval":
            out["evals"].append(r)
        elif k == "restart":
            out["restarts"].append(r)
    f = os.path.join(os.path.dirname(path), "final.json")
    if os.path.exists(f):
        out["final"] = json.load(open(f))
    return out


def se(p, n):
    return math.sqrt(max(p * (1 - p), 1e-9) / n)


def fmt(x, n=3):
    return "—" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def main():
    md = []
    arms = {}
    for p in sorted(glob.glob(os.path.join(RUNS, "*", "log.jsonl"))):
        arms[os.path.basename(os.path.dirname(p))] = load(p)
    md.append("## Eval curves (validation; probes = teacher-forced exact match, 128 items per family)\n")
    md.append("| arm | step | real tokens | val NTP | recall1 | carry3 | multi4 | recall1 swap | coarse ppl | VE |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, a in arms.items():
        for e in a["evals"]:
            pr = e["probes"]; cb = e["codebook"]
            md.append(f"| {name} | {e['step']} | {e['real_tokens']/1e6:.2f}M | {e['val_ntp']:.3f} | "
                      f"{pr.get('lr_recall1', 0):.3f} | {pr.get('lr_carry3', 0):.3f} | {pr.get('lr_multi4', 0):.3f} | "
                      f"{fmt(e.get('swap'))} | {cb['coarse_ppl']:.1f} | {cb['ve']:.3f} |")
    md.append("\n## Training throughput and last training terms\n")
    md.append("| arm | k | pool | loop | steps | real tok/s | ntp | CE coarse | CE fine | VQ coarse | VQ fine | gn |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, a in arms.items():
        if not a["train"]:
            continue
        t = a["train"][-1]; ar = a["meta"]["args"]
        md.append(f"| {name} | {ar['k']} | {ar['pool']} | {ar['enc_loop']} | {t['step']} | {t['tok_s']:.0f} | {t['ntp']:.3f} | "
                  f"{t['ce_coarse']:.2f} | {t['ce_fine']:.2f} | {t['vq_coarse']:.2f} | {t['vq_fine']:.2f} | {t['gn']:.2f} |")
    finals = {n: a["final"] for n, a in arms.items() if a["final"]}
    if finals:
        md.append("\n## Final evaluation (192 items per family)\n")
        cols = ["intact", "core_off", "plan_swap", "c_off", "M1", "loop_off", "pool_ffn_off", "dec_reads_off"]
        md.append("| arm | metric | " + " | ".join(cols) + " |")
        md.append("|---|---|" + "---|" * len(cols))
        for n, f in finals.items():
            nprobe = f["args"]["n_probe_final"]
            rows = {"val NTP": [], "recall1": [], "carry3": [], "multi4": []}
            for c in cols:
                r = dict(val_ntp=f["val_ntp"], probes=f["probes"]) if c == "intact" else f["interventions"].get(c)
                rows["val NTP"].append(fmt(r["val_ntp"]) if r else "—")
                for fam, key in (("lr_recall1", "recall1"), ("lr_carry3", "carry3"), ("lr_multi4", "multi4")):
                    rows[key].append(fmt(r["probes"].get(fam)) if r else "—")
            for key, vals in rows.items():
                md.append(f"| {n} | {key} | " + " | ".join(vals) + " |")
            md.append(f"| {n} | SE (p = 0.5) | ±{se(0.5, nprobe):.3f} |" + " |" * (len(cols) - 1))
        md.append("\n## Codebook and pooler (final)\n")
        md.append("| arm | coarse ppl / K | VE coarse | VE total | CE coarse (ln K) | CE fine | text: top-100 / uniform (per head) | "
                  "text: rest / uniform | facts: content / uniform | facts: template / uniform | entropy / ln k |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for n, f in finals.items():
            cb, pl = f["codebook"], f.get("pool")
            r2 = lambda v: "[" + ", ".join(f"{x:.2f}" for x in v) + "]" if v else "—"
            md.append(f"| {n} | {cb['coarse_ppl']:.1f} / {round(math.exp(cb['ln_K']))} | {cb['ve_coarse']:.3f} | {cb['ve']:.3f} | "
                      f"{cb['ce_coarse']:.2f} ({cb['ln_K']:.2f}) | {cb['ce_fine']:.2f} | "
                      + (" | ".join(r2(pl.get(k)) for k in ("text_top100_over_uniform", "text_rest_over_uniform",
                                                            "fact_content_over_uniform", "fact_template_over_uniform",
                                                            "entropy_over_log_k")) if pl else "linear pool | — | — | — | —")
                      + " |")
        md.append("\n## Pooler slot profiles (mean weight per slot on full chunks, per head; uniform = 1/k)\n")
        for n, f in finals.items():
            pl = f.get("pool")
            if pl:
                md.append(f"- **{n}**: " + "; ".join("h%d [%s]" % (h, ", ".join(f"{x:.2f}" for x in prof))
                                                    for h, prof in enumerate(pl["slot_profile"])))
    text = "\n".join(md)
    print(text)
    if "--md" in sys.argv:
        open(sys.argv[sys.argv.index("--md") + 1], "w").write(text + "\n")


if __name__ == "__main__":
    main()
