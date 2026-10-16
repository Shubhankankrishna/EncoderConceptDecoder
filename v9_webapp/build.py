"""Inline src/ into one self-contained page: index.html (opens from disk, or served by serve.py) and v9_atlas.html (artifact body, no skeleton).

    python build.py [--no-snapshot]

The page reads live telemetry from serve.py when it is reachable. Without a server it falls back to a labelled snapshot of the
experiment's files taken at build time (window.V9_SNAPSHOT), so a page opened from disk or published elsewhere still shows
real numbers, stamped with when they were taken."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
S = HERE / "src"
read = lambda n: (S / n).read_text()
css, page, variants, app, live = read("style.css"), read("page.html"), read("variants.js"), read("app.js"), read("live.js")
m3d = S / "model3d.js"
three = ""
if m3d.exists():
    three = ('<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",'
             '"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>\n'
             f'<script type="module">\n{m3d.read_text()}\n</script>')
else:
    page = page.replace('<div class="hud" id="hud3d">loading 3D…</div>',
        '<div class="no3d">The 3D scenes (model and search tree) are not built yet: src/model3d.js is still to be written.</div>')
snap = ""
if "--no-snapshot" not in sys.argv:
    sys.path.insert(0, str(HERE))
    try:
        import serve
        d = serve.collect()
        d["kind"] = "snapshot"
        snap = "<script>window.V9_SNAPSHOT = " + json.dumps(d, separators=(",", ":")).replace("</", "<\\/") + ";</script>\n"
    except Exception as e:                                   # a missing runs/ folder just means no snapshot
        print("no snapshot:", e)
body = (f'<title>V9 Atlas</title>\n<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">\n'
        f'<style>\n{css}\n</style>\n{page}\n<script>\n{variants}\n</script>\n<script>\n{app}\n</script>\n{snap}<script>\n{live}\n</script>\n{three}\n')
(HERE / "v9_atlas.html").write_text(body)
(HERE / "index.html").write_text('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>\n' + body + '</body></html>\n')
print("built", len(body), "bytes", "(with snapshot)" if snap else "(no snapshot)")
