#!/usr/bin/env python3
"""Generate a self-contained HTML tab viewer for a .gp5 file.

The .gp5 is base64-embedded into the page, so the output is a single file you
can open anywhere -- no Guitar Pro, no TuxGuitar, no Rosetta. Rendering is done
in the browser by alphaTab (https://alphatab.net), which is a completely
separate Guitar Pro implementation from the pyguitarpro we export with, so the
page doubles as an independent correctness check: if alphaTab shows the tab,
the file really is well-formed.

alphaTab itself is loaded from a CDN, so first view needs network access.
"""

import argparse
import base64

_TEMPLATE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>__TITLE__ - Resonote tab viewer</title>
<style>
:root { color-scheme: dark; }
body { background:#15151f; color:#e8e8f0; font-family:system-ui,sans-serif; margin:0; }
header { padding:1rem 1.5rem .5rem; }
h1 { font-size:1.1rem; margin:0 0 .2rem; }
.meta { color:#9fc3e8; font-size:.85rem; }
#status { padding:.4rem 1.5rem; font-size:.85rem; color:#f0c674; }
#viewer { padding:0 1.5rem 2rem; overflow-x:auto; background:#fff; border-radius:6px;
          margin:0 1.5rem 2rem; }
#viewer svg { max-width:none; display:block; transform-origin:top left; }
#drop { margin:0 1.5rem 1rem; font-size:.85rem; color:#8a8a9a; }
input[type=file] { color:#c9c9d8; }
#err { white-space:pre-wrap; color:#ff8080; padding:0 1.5rem; font-size:.8rem; }
.zoom { margin:0 1.5rem .6rem; font-size:.85rem; }
/* alphaTab draws every music symbol as a glyph of the Bravura font. The
   high-level AlphaTabApi injects this itself, but the low-level ScoreRenderer
   we use does NOT -- without it every note is blank. */
@font-face {
  font-family: 'alphaTab';
  src: url('https://cdn.jsdelivr.net/npm/@coderline/alphatab@1.8.4/dist/font/Bravura.woff2') format('woff2'),
       url('https://cdn.jsdelivr.net/npm/@coderline/alphatab@1.8.4/dist/font/Bravura.woff') format('woff'),
       url('https://cdn.jsdelivr.net/npm/@coderline/alphatab@1.8.4/dist/font/Bravura.otf') format('opentype');
  font-weight: normal; font-style: normal; font-display: block;
}
.at-surface * { cursor: default; vertical-align: top; overflow: visible; }
.at { font-family: 'alphaTab'; speak: none; font-style: normal; font-weight: normal;
      font-variant: normal; text-transform: none; line-height: 1;
      -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
      font-size: 34px; overflow: visible !important; }
</style></head>
<body>
<header><h1>__TITLE__</h1><div class="meta">__META__</div></header>
<div class="zoom">缩放：
  <select id="zoom">
    <option value="0.6">60%</option><option value="0.8">80%</option>
    <option value="1" selected>100%</option><option value="1.3">130%</option>
    <option value="1.6">160%</option><option value="2">200%</option>
  </select>
</div>
<div id="drop">换一个文件：<input type="file" id="pick" accept=".gp5,.gp4,.gp3,.gp,.gpx"></div>
<div id="status">loading alphaTab from CDN…</div>
<div id="err"></div>
<div id="viewer"></div>

<script type="module">
const B64 = "__B64__";

const status = document.getElementById('status');
const errBox = document.getElementById('err');
const viewer = document.getElementById('viewer');
function fail(msg) { errBox.textContent = String(msg); status.textContent = 'failed'; }
window.addEventListener('error', e => fail(e.message));

function bytesFromB64(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function renderScore(bytes, label) {
  status.textContent = 'rendering ' + label + ' …';
  viewer.innerHTML = '';
  const settings = new alphaTab.Settings();
  const score = alphaTab.importer.ScoreLoader.loadScoreFromBytes(bytes, settings);

  settings.core.engine = 'svg';
  const renderer = new alphaTab.rendering.ScoreRenderer(settings);
  renderer.width = Math.max(900, Math.floor(window.innerWidth * 0.92));

  const chunks = [];
  renderer.preRender.on(() => { chunks.length = 0; });
  // since alphaTab 1.3 layout and render are separate partial steps
  renderer.partialLayoutFinished.on(r => renderer.renderResult(r.id));
  renderer.partialRenderFinished.on(r => chunks.push(r.renderResult));
  renderer.renderFinished.on(() => {
    viewer.innerHTML = chunks.join('\n');
    status.textContent = score.title + ' · ' + score.tempo + ' BPM · '
      + score.masterBars.length + ' bars · ' + score.tracks.length + ' track(s)';
  });
  renderer.renderScore(score, score.tracks.map(t => t.index));
}

document.getElementById('zoom').addEventListener('change', ev => {
  const z = parseFloat(ev.target.value);
  for (const svg of viewer.querySelectorAll('svg')) {
    svg.style.transform = 'scale(' + z + ')';
  }
});

let alphaTab;
try {
  alphaTab = await import('https://cdn.jsdelivr.net/npm/@coderline/alphatab@1.8.4/dist/alphaTab.min.mjs');
  status.textContent = 'alphaTab loaded';
} catch (e) { fail('cannot load alphaTab from CDN (offline?):\n' + e.message); throw e; }

try {
  await renderScore(bytesFromB64(B64), 'embedded file');
} catch (e) { fail('render failed:\n' + (e && e.stack || e)); throw e; }

document.getElementById('pick').addEventListener('change', async ev => {
  const f = ev.target.files[0];
  if (!f) return;
  try { await renderScore(new Uint8Array(await f.arrayBuffer()), f.name); }
  catch (e) { fail('render failed:\n' + (e && e.stack || e)); }
});
</script>
</body></html>
"""


def make_viewer(gp5_path: str, out_html: str, title: str = "", meta: str = "") -> str:
    with open(gp5_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    html = (_TEMPLATE
            .replace("__B64__", b64)
            .replace("__TITLE__", title or gp5_path)
            .replace("__META__", meta))
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("gp5")
    ap.add_argument("out_html")
    ap.add_argument("--title", default="")
    ap.add_argument("--meta", default="")
    a = ap.parse_args()
    print(make_viewer(a.gp5, a.out_html, a.title, a.meta))
