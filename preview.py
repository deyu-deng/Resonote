"""Lightweight tab preview: ASCII six-line tab + standalone HTML."""

from models import PlacedNote

# display order: low string (6) at the bottom
_LABELS = {6: "E", 5: "A", 4: "D", 3: "G", 2: "B", 1: "e"}


def ascii_tab(placed):
    rows = {s: [] for s in range(1, 7)}
    for p in placed:
        for s in range(1, 7):
            rows[s].append(str(p.fret) if s == p.string else "-")
    lines = []
    for s in range(6, 0, -1):
        lines.append(_LABELS[s] + "|" + "|".join(rows[s]) + "|")
    return "\n".join(lines)


def html_preview_string(placed, title="Resonote preview") -> str:
    """Build the standalone HTML preview as a string (no file I/O).

    Used by the web layer so it can embed / serve the markup directly.
    """
    tab = ascii_tab(placed).replace("&", "&amp;").replace("<", "&lt;")
    rows = "".join(
        f"<tr><td>{p.pitch}</td><td>str {p.string}</td>"
        f"<td>fret {p.fret}</td><td>finger {p.finger or 'open'}</td>"
        f"<td>{p.pluck or '-'}</td></tr>"
        for p in placed
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>body{{background:#15151f;color:#e8e8f0;font-family:system-ui,sans-serif;margin:2rem}}
pre{{background:#0e0e16;padding:1rem;border-radius:8px;font-size:15px;line-height:1.5}}
table{{border-collapse:collapse;margin-top:1rem}}td,th{{border:1px solid #333;padding:.3rem .7rem;font-size:13px}}
th{{color:#9fc3e8}}</style></head>
<body><h2>{title}</h2>
<pre>{tab}</pre>
<table><tr><th>pitch</th><th>string</th><th>fret</th><th>finger</th><th>pluck</th></tr>{rows}</table>
</body></html>"""


def write_html_preview(placed, out_html, title="Resonote preview"):
    html = html_preview_string(placed, title)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html
