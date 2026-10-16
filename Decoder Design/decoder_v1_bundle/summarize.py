"""Collect every run in runs/ into one table (RESULTS_DECODER.txt is written from this output)."""
import json, glob, os

rows = []
for f in sorted(glob.glob("runs/*.json")):
    r = json.load(open(f)); name = os.path.basename(f)[:-5]
    hist = {h["step"]: h for h in r["log"]}
    fin = r["final"]
    key = "exact" if "exact" in fin["intact"] else "value_acc"
    curve = " ".join(f"{hist[s][key]:.2f}" if s in hist else "  - " for s in (100, 250, 500, 750, 1000, 1500))
    rows.append((name, key, curve, fin["intact"]["loss"], fin["intact"]["filler"], fin["intact"][key],
                 fin["core_off"][key], fin["u_off"][key], fin["c_off"][key], r["reach"]))
print(f"{'run':28s} {'metric':9s} {'@100  @250  @500  @750  @1000 @1500':36s} {'loss':>7s} {'filler':>7s} {'final':>6s} "
      f"{'coreOff':>7s} {'uOff':>6s} {'cOff':>6s} {'reach':>5s}")
for n, k, c, l, fl, a, co, uo, cf, re in rows:
    print(f"{n:28s} {k:9s} {c:36s} {l:7.4f} {fl:7.4f} {a:6.3f} {co:7.3f} {uo:6.3f} {cf:6.3f} {re:5d}")
