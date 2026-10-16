"""Learning curves (PROMPT 7.4) from runs/<run>/train_log.jsonl and log.jsonl -> runs/<run>/figs/*.png.
  python plots.py ../runs/main [../runs/screens/S1 ...]   (screens are overlaid on the language panel)"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROBES = ["lr_recall1", "lr_carry3", "lr_multi4", "lr_hop", "lr_bind"]


def read(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def main(run, others):
    figs = os.path.join(run, "figs"); os.makedirs(figs, exist_ok=True)
    tl = [r for r in read(os.path.join(run, "train_log.jsonl")) if "loss" in r]
    el = read(os.path.join(run, "log.jsonl"))
    rs = [r for r in read(os.path.join(run, "train_log.jsonl")) if "restart_dead" in r]
    tok = lambda r: r["step"] * r.get("accum", 16) * r.get("micro_bs", 2) * 1024 / 1e6
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    a = ax[0, 0]
    a.plot([tok(r) for r in tl], [r["ntp"] for r in tl], label="train NTP", lw=1)
    a.plot([r["tokens"] / 1e6 for r in el], [r["val_tinystories"]["nats"] for r in el], "o-", label="val TinyStories")
    a.plot([r["tokens"] / 1e6 for r in el], [r["val_wikitext"]["nats"] for r in el], "s-", label="val WikiText")
    for o in others:
        eo = read(os.path.join(o, "log.jsonl"))
        a.plot([r["tokens"] / 1e6 for r in eo], [r["val_lang_mean"] for r in eo], "x--", label=f"{os.path.basename(o)} val mean")
    a.set_xlabel("tokens (M)"); a.set_ylabel("nats"); a.set_title("language loss"); a.legend(fontsize=8); a.set_ylim(top=min(10, a.get_ylim()[1]))
    a = ax[0, 1]
    for f in PROBES:
        a.plot([r["tokens"] / 1e6 for r in el], [r["probes"][f]["exact"] for r in el], "o-", label=f)
    a.plot([r["tokens"] / 1e6 for r in el], [r["recall1_plan_swap"]["exact"] for r in el], "k--", label="recall1, plan swap")
    a.set_xlabel("tokens (M)"); a.set_ylabel("exact match (val, 128/family)"); a.set_title("long-range probes"); a.legend(fontsize=8)
    a = ax[0, 2]
    a.plot([r["tokens"] / 1e6 for r in el], [r["probes"]["lr_recall1"]["exact"] - r["recall1_plan_swap"]["exact"] for r in el], "o-")
    a.set_xlabel("tokens (M)"); a.set_title("plan-swap cost on lr_recall1 (exact drop)")
    a = ax[1, 0]
    a.plot([r["step"] for r in rs], [r["codes_used_min_seg"] for r in rs], "o-", label="min segment, restart window")
    a.plot([r["step"] for r in rs], [r["codes_used_mean"] for r in rs], ".-", label="mean, restart window")
    a.plot([r["step"] for r in el], [r["codebook"]["codes_used_frac_min"] for r in el], "s--", label="min segment, eval sample")
    a.plot([r["step"] for r in el], [r["codebook"]["var_explained"] for r in el], "^-", label="variance explained")
    a.set_xlabel("step"); a.set_title("codebook health"); a.legend(fontsize=8)
    a = ax[1, 1]
    a.plot([r["step"] for r in el], [r["codebook"]["ncp_ce"] for r in el], "o-", label="NCP CE (eval)")
    a.plot([r["step"] for r in el], [r["codebook"]["code_entropy"] for r in el], "s-", label="code-frequency entropy")
    a.plot([r["step"] for r in tl], [r["ncp"] for r in tl], lw=1, label="NCP CE (train)")
    a.set_xlabel("step"); a.set_title("NCP vs code entropy (G-C2)"); a.legend(fontsize=8)
    a = ax[1, 2]
    a.plot([r["step"] for r in tl], [r["gn_preclip"] for r in tl], lw=1, label="grad norm (pre-clip)")
    a2 = a.twinx(); a2.plot([r["step"] for r in tl], [r["tok_s"] for r in tl], "g", lw=1, label="tok/s")
    a.set_xlabel("step"); a.set_title("grad norm / throughput"); a.legend(loc="upper left", fontsize=8); a2.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "curves.png"), dpi=110)
    print("wrote", os.path.join(figs, "curves.png"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
