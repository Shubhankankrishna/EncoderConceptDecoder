"""Read-only phone gateway for the dashboard (LAN).

Serves this webapp folder on 0.0.0.0:8780 so a phone on the same Wi-Fi can open it, and forwards the live endpoints of
the two existing localhost-only servers:  /api/*  -> 127.0.0.1:8765/api/*   and   /v3api/live -> 127.0.0.1:8766/api/live.
GET / HEAD only. Access needs the key in .phone_key: open  http://<mac-ip>:8780/?k=<key>  once; a cookie keeps it.
Never touches a training run. Stop with:  kill $(cat webapp/.phone_server.pid)
"""
import http.server, json, os, secrets, socketserver, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PHONE_PORT", "8780"))
KEY_FILE = os.path.join(HERE, ".phone_key")
if not os.path.exists(KEY_FILE):
    with open(KEY_FILE, "w") as f:
        f.write(secrets.token_urlsafe(6))
KEY = open(KEY_FILE).read().strip()
UPSTREAM = {"/api/": "http://127.0.0.1:8765", "/v3api/live": "http://127.0.0.1:8766/api/live"}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def log_message(self, *a):                                   # quiet: no per-request log spam
        pass

    def authorised(self):
        if f"hrmk={KEY}" in (self.headers.get("Cookie") or ""):
            return True, False
        q = self.path.split("?", 1)[1] if "?" in self.path else ""
        return (f"k={KEY}" in q.split("&")), True

    def deny(self):
        self._set_cookie = False
        body = b"<h3 style='font-family:sans-serif'>HRM dashboard: open the link with its key (?k=...)</h3>"
        self.send_response(403); self.send_header("Content-Type", "text/html"); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def proxy(self, url, trim_steps=None):
        try:
            with urllib.request.urlopen(url, timeout=8) as r:
                body = r.read(); ctype = r.headers.get("Content-Type", "application/json")
            if trim_steps:                                       # mobile data: the page draws only the last 600 steps
                try:
                    j = json.loads(body)
                    if isinstance(j.get("steps"), list):
                        j["steps"] = j["steps"][-trim_steps:]
                    body = json.dumps(j, separators=(",", ":")).encode()
                except ValueError:
                    pass
            self.send_response(200)
        except Exception as e:                                   # upstream down: the page degrades gracefully
            body = ('{"error": "%s"}' % type(e).__name__).encode(); ctype = "application/json"
            self.send_response(502)
        self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.end_headers()
        if self.command == "GET":
            self.wfile.write(body)

    def end_headers(self):
        if getattr(self, "_set_cookie", False):
            self.send_header("Set-Cookie", f"hrmk={KEY}; Path=/; Max-Age=2592000; SameSite=Strict; HttpOnly")
        p = self.path.split("?", 1)[0]
        if p.endswith((".html", ".js", ".css", ".json", "/")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        ok, via_query = self.authorised()
        if not ok:
            return self.deny()
        self._set_cookie = via_query
        p = self.path.split("?", 1)[0]
        if p.startswith("/v3api/live"):
            return self.proxy(UPSTREAM["/v3api/live"], trim_steps=600)
        if p.startswith("/api/"):
            return self.proxy(UPSTREAM["/api/"] + self.path)
        if os.path.basename(p).startswith(".") or p.endswith(".py"):   # never serve the key, pid or source
            return self.deny()
        return super().do_GET()

    def do_HEAD(self):
        ok, via_query = self.authorised()
        if not ok:
            return self.deny()
        self._set_cookie = via_query
        return super().do_HEAD()

    def do_POST(self):
        self.send_error(405)
    do_PUT = do_DELETE = do_PATCH = do_POST


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    with open(os.path.join(HERE, ".phone_server.pid"), "w") as f:
        f.write(str(os.getpid()))
    print(f"phone gateway on 0.0.0.0:{PORT} (key in {KEY_FILE})", flush=True)
    Server(("0.0.0.0", PORT), Handler).serve_forever()
