"""Build the V13 Anatomy page: inline the data, logic, content and app into one HTML file.

    python3 v13_viz/build.py      ->  v13_viz/dist/v13_anatomy.html (published: no document skeleton, per the
                                      Artifact contract) and v13_viz/dist/local.html (with a doctype, to open here)
three.js is the only external script (jsdelivr, pinned 0.180.0, imported by absolute URL, no import map).
"""
import json
import re
from pathlib import Path

H = Path(__file__).resolve().parent


def strip_exports(s):
    return re.sub(r'^export (const|function|let|class) ', r'\1 ', s, flags=re.M)


def main():
    page = (H / 'src/page.html').read_text()
    data = {k: json.loads((H / f'data/{n}.json').read_text())
            for k, n in (('tiny', 'tiny_v2'), ('tiers', 'tiers'), ('tokens', 'tokens'), ('snapshot', 'snapshot'))}
    logic = strip_exports((H / 'src/logic.js').read_text())
    content = strip_exports((H / 'src/content.js').read_text())
    app = (H / 'src/app.js').read_text()
    app, n = re.subn(r"^import \{[^}]*\} from '\./(logic|content)\.js';\n", '', app, flags=re.M)
    assert n == 2, 'expected two local imports in app.js'
    three = re.search(r"^import \* as THREE from '(https://cdn\.jsdelivr\.net/npm/three@0\.180\.0/build/three\.webgpu\.js)';\n", app, flags=re.M)
    assert three, 'three.js must be imported by its pinned absolute URL'
    app = app.replace(three.group(0), '')
    module = three.group(0) + logic + '\n' + content + '\n' + app
    blob = json.dumps(data, separators=(',', ':')).replace('</', '<\\/')
    page = page.replace('<!--DATA-->', f'<script>window.V13_DATA = {blob};</script>')
    page = page.replace('<!--SCRIPT-->', '<script type="module">\n' + module.replace('</script', '<\\/script') + '\n</script>')
    out = H / 'dist'; out.mkdir(exist_ok=True)
    (out / 'v13_anatomy.html').write_text(page)
    title = re.search(r'<title>.*?</title>', page).group(0)
    local = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
             '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
             f'{title}\n</head>\n<body>\n{page.replace(title, "", 1)}\n</body>\n</html>\n')
    (out / 'local.html').write_text(local)
    (out / 'module_check.mjs').write_text(module)                # for `node --check` (syntax only)
    print(f'dist/v13_anatomy.html {len(page) / 1024:.0f} KB, dist/local.html {len(local) / 1024:.0f} KB')


if __name__ == '__main__':
    main()
