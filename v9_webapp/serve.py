"""V9 Atlas telemetry server (read-only, loopback only).

    python serve.py [--port 8765]

Serves the built page at / and one JSON endpoint, /api/telemetry, aggregated from the experiment's own files under
v9_tiny/. Nothing is written, no path comes from the request, and every source is a fixed pattern:

    runs/status*.json  runs/launch_*.json  runs/driver*.json  runs/G1.json  runs/queue.jsonl
    runs/<run>/log.jsonl  runs/<run>/config.json  runs/<run>/development_step_*.json  runs/<run>/train_summary.json
    runs/<run>/measure.json (greedy rates only)  runs/<run>_dev/summary.json  runs/<run>_trained_contracts/summary.json
    tests/q*_contracts.json  tests/q*_smoke.json     spec/LOCK_*.json

collect() is also what build.py bakes into the page as a labelled snapshot, so the page works with no server too.
Numbers shown from it are development diagnostics (training loss and the 2k-update dev checks), not sealed results.
"""
import argparse, json, os, re, subprocess, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
TINY = HERE.parent / "v9_tiny"
RUNS, TESTS, SPEC = TINY / "runs", TINY / "tests", TINY / "spec"
NAME = re.compile(r"^[A-Za-z0-9_.\-]+$")
CONFIG_KEYS = ("stage", "flat", "core", "kind", "mode", "steps", "batch", "context", "seed", "alpha", "lr", "beta", "constant_alpha", "no_distill",
               "always_open", "device", "precision", "active_parameters", "resident_parameters", "trainable_parameters", "created_utc")
SUITES = ("standard", "lookup", "compose", "hard10", "hard14")
DROP = re.compile(r"(_targets$|^unique_|^scored_targets$|^planned_steps$|_format_ce$)")      # bulky counters the page doesn't draw
_cache = {}


def _round(v):
    return round(v, 5) if isinstance(v, float) else v


def _read(path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _cached(path, fn):
    """Parse a file once per (mtime, size). A log that is being appended is re-parsed when it grows."""
    try:
        st = path.stat()
    except OSError:
        return None
    key = str(path)
    hit = _cache.get(key)
    if hit and hit[0] == (st.st_mtime_ns, st.st_size):
        return hit[1]
    val = fn(path)
    _cache[key] = ((st.st_mtime_ns, st.st_size), val)
    return val


def _utc(s):
    try:
        return time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
    except Exception:
        return None


def _series(path):
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue                      # a partially written last line
    if not rows:
        return dict(n=0)
    keys = []
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in keys and not DROP.search(k):
                keys.append(k)
    cols = {k: [_round(r.get(k)) for r in rows] for k in keys}
    cols["t"] = [_utc(r.get("utc", "")) for r in rows]
    last = rows[-1]
    return dict(n=len(rows), cols=cols, exposure=last.get("exposure"), memory=last.get("memory_bytes"), source_targets=last.get("source_targets"))


def _dev_checks(run):
    out = []
    for f in sorted(run.glob("development_step_*.json"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1])):
        d = _cached(f, lambda p: _read(p, {})) or {}
        sym = d.get("symbolic", {})
        out.append(dict(step=d.get("step"), **{s: dict(rate=sym[s].get("rate"), interval=sym[s].get("interval"), first_legal=sym[s].get("first_legal_rate"), n=sym[s].get("n"))
                                              for s in SUITES if s in sym}))
    return out


def _summary(path):
    d = _read(path, {})
    sym = d.get("symbolic", {})
    return dict(status=d.get("status"), competence_pass=d.get("competence_pass"), split=d.get("split"),
                symbolic={s: dict(rate=sym[s].get("rate"), interval=sym[s].get("interval"), n=sym[s].get("n"), first_legal=sym[s].get("first_legal_rate")) for s in SUITES if s in sym},
                text={k: dict(nll=v.get("nll"), interval=v.get("interval")) for k, v in (d.get("text") or {}).items() if isinstance(v, dict)},
                checks=len(d.get("checks", [])) if isinstance(d.get("checks"), list) else None,
                checks_passed=sum(1 for c in d.get("checks", []) if c.get("passed")) if isinstance(d.get("checks"), list) else None)


def _measure_summary(path):
    d = _read(path, {})
    g = d.get("greedy", {})
    return dict(seconds=d.get("seconds"), k=d.get("k"), core=d.get("core"), greedy={n: dict(rate=v.get("rate"), n=v.get("n")) for n, v in g.items() if isinstance(v, dict)})


def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _runs():
    out = {}
    if not RUNS.is_dir():
        return out
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or not NAME.match(d.name) or not (d / "log.jsonl").exists():
            continue
        cfg = _cached(d / "config.json", lambda p: _read(p, {})) or {}
        ts = _cached(d / "train_summary.json", lambda p: _read(p, {})) or {}
        live = _read(d / "live.json", {}) or {}
        rec = dict(config={k: cfg[k] for k in CONFIG_KEYS if k in cfg}, series=_cached(d / "log.jsonl", _series) or dict(n=0), dev_checks=_dev_checks(d),
                   completed=bool(ts), wall_s=ts.get("wall_s"), completed_steps=ts.get("completed_steps"),
                   last_update_utc=live.get("utc"), metadata={k: v for k, v in (cfg.get("metadata") or {}).items() if k in ("flat", "depth", "core", "adapter", "phases")})
        if (d / "measure.json").exists():
            rec["measure"] = _cached(d / "measure.json", _measure_summary)
        for tag, folder in (("dev_eval", "_dev"), ("trained_contracts", "_trained_contracts")):
            p = RUNS / (d.name + folder) / "summary.json"
            if p.exists():
                rec[tag] = _cached(p, _summary)
        out[d.name] = rec
    return out


def _queue_events(limit=80):
    p = RUNS / "queue.jsonl"
    if not p.exists():
        return []
    ev = []
    for line in p.read_text().splitlines()[-limit:]:
        try:
            r = json.loads(line)
        except Exception:
            continue
        ev.append({k: r.get(k) for k in ("utc", "stage", "event", "block", "returncode", "elapsed_s", "reason") if k in r})
    return ev


def _blocks():
    out = {}
    for f in sorted(RUNS.glob("status*.json")):
        d = _read(f, {})
        blk = d.get("block")
        if not blk:
            continue
        d["driver_alive"] = _alive(d["driver_pid"]) if d.get("driver_pid") else False
        d["child_alive"] = _alive(d["child_pid"]) if d.get("child_pid") else False
        out[blk] = {k: d.get(k) for k in ("block", "status", "stage", "started_utc", "updated_utc", "elapsed_s", "driver_alive", "child_alive", "live")}
    for f in sorted(RUNS.glob("launch_*.json")):
        d = _read(f, {})
        if d.get("block") in out:
            out[d["block"]]["launched_utc"] = d.get("utc")
    return out


def _tests():
    out = {}
    for f in sorted(TESTS.glob("q*.json")):
        if not NAME.match(f.name):
            continue
        d = _read(f, {})
        t = d.get("tests")
        out[f.stem] = dict(status=d.get("status"), n=len(t) if isinstance(t, list) else None, seconds=d.get("seconds"), lock=(d.get("lock_sha256") or "")[:10] or None,
                           utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(f.stat().st_mtime)))
    return out


def _locks():
    return {f.stem.replace("LOCK_", ""): dict(created_utc=(_read(f, {}) or {}).get("created_utc"), files=len((_read(f, {}) or {}).get("files", []))) for f in sorted(SPEC.glob("LOCK_*.json"))}


def _system():
    now = time.time()
    hit = _cache.get("system")
    if hit and now - hit[0] < 8:
        return hit[1]
    sw = None
    try:
        s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=2).stdout
        m = {k: float(v) for k, v in re.findall(r"(total|used|free) = ([\d.]+)M", s)}
        sw = dict(total_mb=m.get("total"), used_mb=m.get("used"))
    except Exception:
        pass
    try:
        load = [round(x, 2) for x in os.getloadavg()]
    except Exception:
        load = None
    val = dict(swap=sw, load=load, cpus=os.cpu_count())
    _cache["system"] = (now, val)
    return val


def collect():
    g1 = _read(RUNS / "G1.json")
    return dict(server_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), kind="live", blocks=_blocks(), runs=_runs(), g1=g1, events=_queue_events(), tests=_tests(),
                locks=_locks(), system=_system(),
                note="Development diagnostics: training loss and 2k-update dev checks. They are not the sealed evaluations and decide nothing.")


class Handler(BaseHTTPRequestHandler):
    server_version = "V9Atlas"

    def log_message(self, *a):
        pass

    def _host_ok(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost")              # blocks DNS-rebinding

    def _send(self, code, body, ctype, extra=None):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        origin = self.headers.get("Origin")
        if origin and (origin == "null" or re.match(r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$", origin)):
            self.send_header("Access-Control-Allow-Origin", origin)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, "forbidden", "text/plain")
        path = self.path.split("?")[0]
        if path in ("/", "/index.html", "/v9_atlas.html"):
            f = HERE / "index.html"
            if not f.exists():
                return self._send(404, "index.html is not built: run build.py", "text/plain")
            return self._send(200, f.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/telemetry":
            try:
                return self._send(200, json.dumps(collect(), separators=(",", ":")), "application/json")
            except Exception as e:                              # never take the page down over a half-written file
                return self._send(500, json.dumps(dict(error=str(e))), "application/json")
        return self._send(404, "not found", "text/plain")

    do_HEAD = do_GET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"V9 Atlas telemetry on http://127.0.0.1:{a.port}/  (read-only; Ctrl-C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
