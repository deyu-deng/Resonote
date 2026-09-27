"""Resonote web backend (L9 UX) — zero new dependencies.

A tiny HTTP server that exposes the pipeline to a browser:

  GET  /            -> serves the premium front-end (web/index.html)
  GET  /api/demo    -> run the built-in sample melody, return artifacts
  POST /api/arrange -> upload raw file bytes (header X-Filename), run the
                       pipeline, persist artifacts, return JSON with URLs

The client sends the file body verbatim (not multipart) so we don't depend on
the removed ``cgi`` module. Artifacts are written to ``web/results/<id>/`` and
served as static files.

Run:  python web_server.py [--port 8000] [--host 127.0.0.1]
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from pipeline import run
from arrangement import is_llm_configured

_HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(_HERE, "web")
RESULTS_DIR = os.path.join(WEB_DIR, "results")

# delete result dirs older than this (avoid disk bloat from uploads)
_MAX_AGE_HOURS = 1


def _params_from(query: str) -> dict:
    qs = parse_qs(query)
    return {
        "style": qs.get("style", ["fingerstyle"])[0],
        "instruction": qs.get("instruction", [""])[0],
        "no_separate": qs.get("no_separate", ["0"])[0].lower() in ("1", "true", "yes"),
        "amt": qs.get("amt", ["auto"])[0],
        "llm": qs.get("llm", ["0"])[0].lower() in ("1", "true", "yes"),
    }


def _cleanup_results() -> None:
    if not os.path.isdir(RESULTS_DIR):
        return
    now = time.time()
    for name in os.listdir(RESULTS_DIR):
        p = os.path.join(RESULTS_DIR, name)
        if os.path.isdir(p) and now - os.path.getmtime(p) > _MAX_AGE_HOURS * 3600:
            shutil.rmtree(p, ignore_errors=True)


def arrange_request(body: bytes, filename: str, params: dict) -> dict:
    """Run the pipeline on ``body`` (or the built-in demo) and persist results.

    Pure and testable: no socket involved. Returns a JSON-serializable dict
    with artifact URLs (or ``{"ok": False, "error": ...}`` on failure).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _cleanup_results()

    rid = uuid.uuid4().hex[:12]
    out_dir = os.path.join(RESULTS_DIR, rid)
    os.makedirs(out_dir, exist_ok=True)

    input_path = None
    is_demo = (filename == "__demo__")
    if not is_demo:
        ext = os.path.splitext(filename)[1] or ".bin"
        tmp = tempfile.NamedTemporaryFile(
            suffix=ext, delete=False, prefix="resonote_in_")
        try:
            tmp.write(body)
        finally:
            tmp.close()
        input_path = tmp.name

    try:
        instruction = params.get("instruction", "")
        use_llm = bool(params.get("llm", False)) or (
            bool(instruction) and is_llm_configured())
        if params.get("llm", False) and not is_llm_configured():
            return {"ok": False,
                    "error": "LLM requested but RESONOTE_LLM_API_KEY is not set."}
        res = run(
            input_path,
            demo=is_demo,
            no_separate=params.get("no_separate", False),
            amt=params.get("amt", "auto"),
            style=params.get("style", "fingerstyle"),
            instruction=instruction,
            llm=use_llm,
            gp5_path=os.path.join(out_dir, "out.gp5"),
            midi_path=os.path.join(out_dir, "out.mid"),
            wav_path=os.path.join(out_dir, "out.wav"),
            html_path=os.path.join(out_dir, "preview.html"),
        )
    except Exception as exc:  # surface a usable message to the UI
        return {"ok": False, "error": str(exc)}
    finally:
        if input_path:
            try:
                os.unlink(input_path)
            except OSError:
                pass

    return {
        "ok": True,
        "ascii_tab": res.ascii_tab,
        "summary": res.summary,
        "tempo": res.tempo,
        "key": res.key,
        "role_counts": res.role_counts,
        "note_count": len(res.placed),
        "judge_backend": "llm" if use_llm else "rules",
        "gp5_url": f"/results/{rid}/out.gp5",
        "midi_url": f"/results/{rid}/out.mid",
        "wav_url": f"/results/{rid}/out.wav",
        "html_url": f"/results/{rid}/preview.html",
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def _send_json(self, obj: dict, status: int = 200) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/"):
            self._handle_api_get()
            return
        super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/arrange"):
            self._handle_api_post()
            return
        self.send_error(405)

    def _handle_api_get(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/demo":
            payload = arrange_request(b"", "__demo__", _params_from(parsed.query))
            self._send_json(payload, 200 if payload.get("ok") else 500)
        else:
            self.send_error(404)

    def _handle_api_post(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        filename = self.headers.get("X-Filename", "input.bin")
        payload = arrange_request(body, filename, _params_from(parsed.query))
        self._send_json(payload, 200 if payload.get("ok") else 500)

    def log_message(self, fmt, *args):  # quieter console
        return


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    os.makedirs(WEB_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"Resonote web running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping.")
        httpd.shutdown()


def main():
    import argparse
    from arrangement import load_env_file
    load_env_file()          # .env -> os.environ, so the LLM layer can engage

    ap = argparse.ArgumentParser(description="Resonote web server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
