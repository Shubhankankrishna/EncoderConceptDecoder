"""Build the V9 Guide.

    python3 build.py

Inlines src/ into one self-contained page.
  guide_body.html   artifact body (no <!doctype>/<head>; the Artifact skeleton adds those)
  ../v9_guide.html  the same body in a full page, for opening from disk or serving locally
Chapter files are src/ch_*.html (sorted), widgets src/viz_*.js (sorted), the 3D module src/three_guide.js.
variants.js is read from ../src (the Atlas descriptor table) so an arm is defined once for both pages."""
import json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
S = HERE / "src"

RUNS = HERE.parent.parent / "v9_tiny" / "runs"


def _rows(name):
    f = RUNS / name / "log.jsonl"
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "text" in r and "event" not in r:
            out.append(r)
    return out


def _blocks(rows, keys, n):
    """mean of consecutive logged rows, n rows per point (rows are logged every 100 updates)"""
    pts = []
    for i in range(0, len(rows) - n + 1, n):
        chunk = rows[i:i + n]
        pts.append(dict(step=chunk[-1]["step"], **{k: round(sum(c.get(k, 0) for c in chunk) / n, 4) for k in keys if k in chunk[0]}))
    return pts


def collect():
    """Real numbers from the experiment's own files (read-only). Embedded as window.GUIDE_DATA."""
    d = dict(generated=time.strftime("%Y-%m-%d %H:%M", time.localtime()), s1={}, s2={})
    try:
        r = json.loads((RUNS / "S1_RESULTS.json").read_text())
        d["s1"]["table"] = r["table"]
        d["s1"]["tests"] = {k: dict(pooled=r[k]["pooled"], supported=r[k]["supported"], holm=r[k].get("holm_p"), per_seed=r[k]["per_seed"]) for k in ("H3", "H4", "H5")}
        d["s1"]["g1"] = json.loads((RUNS / "G1.json").read_text())["seeds"]
    except Exception as e:
        print("no S1 results:", e)
    d["s1"]["curves"] = {n: _blocks(_rows(f"S1_A_{n}"), ("text", "symbolic", "ncp"), 4) for n in ("delayed_s901", "delayed_s902", "alpha0_s901", "alpha0_s902", "flat_s901", "flat_s902")}
    for f in sorted(RUNS.glob("S2_*_dense_s91?")):
        name = f.name
        rows = _rows(name)
        if not rows:
            continue
        rec = dict(steps=rows[-1]["step"], curve=_blocks(rows, ("text", "symbolic", "ncp", "loss"), 6))
        if name.startswith("S2_K8"):
            rec["k8"] = _blocks(rows, ("selector_entropy", "hypothesis_pair_distance", "kdec_is_kstar", "d_min", "d_mean", "selector"), 6)
        ts = f / "train_summary.json"
        if ts.exists():
            try:
                rec["wall_s"] = json.loads(ts.read_text()).get("wall_s")
            except Exception:
                pass
        m = f / "measure.json"
        if m.exists():
            try:
                g = json.loads(m.read_text())
                rec["measure"] = {k: dict(rate=v["rate"], n=v["n"]) for k, v in g.get("greedy", {}).items() if k in ("standard", "hard10", "hard14")}
                rec["measure_s"] = g.get("seconds")
            except Exception:
                pass
        d["s2"][name] = rec
    return d

rd = lambda p: Path(p).read_text()
css = rd(S / "style.css")
chapters = "\n".join(rd(p) for p in sorted(S.glob("ch_*.html")))
viz = "\n".join(f"/* ---- {p.name} */\n" + rd(p) for p in sorted(S.glob("viz_*.js")))
core = rd(S / "core.js")
variants = rd(HERE.parent / "src" / "variants.js")
chrome_top = rd(S / "chrome_top.html")
chrome_bot = rd(S / "chrome_bottom.html")
t3 = S / "three_guide.js"
three = f'<script type="module">\n{rd(t3)}\n</script>\n' if t3.exists() else ""
fonts = ('<link rel="preconnect" href="https://fonts.googleapis.com">\n'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">\n')
data = '<script>window.GUIDE_DATA = ' + json.dumps(collect(), separators=(',', ':')) + ';</script>'
body = (f'<title>V9 Experiments Guide</title>\n{fonts}<style>\n{css}\n</style>\n{chrome_top}\n{chapters}\n{chrome_bot}\n'
        f'{data}\n<script>\n{variants}\n</script>\n<script>\n{core}\n</script>\n<script>\n{viz}\n</script>\n{three}')
(HERE / "guide_body.html").write_text(body)
(HERE.parent / "v9_guide.html").write_text('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>\n' + body + '</body></html>\n')
print("built", len(body), "bytes;", len(list(S.glob("ch_*.html"))), "chapter files,", len(list(S.glob("viz_*.js"))), "viz files")
