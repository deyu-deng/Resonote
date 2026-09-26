"""Lightweight tab preview: ASCII six-line tab + standalone HTML.

The tab is laid out on the same eighth-note grid the GP5 exporter uses, so
what you see here is what lands in the file: one column per eighth note, bar
lines every measure, and a new block every few measures. Without that grid the
old preview was just a list of notes in sequence — unreadable as a tab and
useless for checking whether the arrangement is actually playable.
"""

from typing import List, Optional, Sequence

from models import PlacedNote

# display order: low string (6) at the bottom
_LABELS = {6: "E", 5: "A", 4: "D", 3: "G", 2: "B", 1: "e"}
_ORDER = [6, 5, 4, 3, 2, 1]          # top to bottom
SLOTS_PER_MEASURE = 8                # 8 eighths == 4/4, same as the exporter
# 3 characters per eighth-note column: frets go up to 19, so a 2-wide column
# makes "112" ambiguous (11|2 or 1|12). One extra char removes the guesswork.
SLOT_WIDTH = 3


def _grid(placed: Sequence[PlacedNote], tempo: float):
    """Map notes onto the eighth-note grid; return (items, total_slots).

    Mirrors ``gp_export._to_grid``: slot length is an eighth note, and the
    timeline starts at the first sounding note.
    """
    if not placed:
        return [], 0
    anchor = min(p.onset for p in placed)
    slots_per_sec = tempo / 30.0      # eighth notes per second
    items = []
    total = 0
    for p in placed:
        s = max(0, int(round((p.onset - anchor) * slots_per_sec)))
        length = max(1, int(round(p.duration * slots_per_sec)))
        items.append((s, length, p))
        total = max(total, s + length)
    return items, total


def ascii_tab(placed: Sequence[PlacedNote],
              tempo: float = 120.0,
              slots_per_measure: int = SLOTS_PER_MEASURE,
              measures_per_line: int = 4) -> str:
    """Render a time-aligned ASCII tablature with bar lines.

    A fret number marks a plucked note; ``-`` is silence (in practice the
    previous chord keeps ringing, see ``gp_export``). Blocks wrap every
    ``measures_per_line`` measures so the tab stays readable in a terminal.
    """
    if not placed:
        return ""

    items, total = _grid(placed, tempo)
    if total == 0:
        return ""
    n_measures = (total + slots_per_measure - 1) // slots_per_measure
    total = n_measures * slots_per_measure

    # per string: one cell per slot
    cells = {s: ["--"] * total for s in _ORDER}
    for (start, length, p) in items:
        if p.string not in cells:
            continue
        g = min(start, total - 1)
        cell = str(p.fret)
        cells[p.string][g] = cell if len(cell) >= SLOT_WIDTH else cell.ljust(SLOT_WIDTH)

    def measure_text(row, m):
        lo = m * slots_per_measure
        return "".join(row[lo:lo + slots_per_measure])

    blocks: List[str] = []
    for first in range(0, n_measures, measures_per_line):
        last = min(first + measures_per_line, n_measures)
        measures = list(range(first, last))

        # bar-number ruler, aligned under the "E|" prefix
        ruler = "  " + "".join(
            f"{m + 1:<{slots_per_measure * SLOT_WIDTH + 1}}" for m in measures)
        blocks.append(ruler.rstrip())

        for s in _ORDER:
            row = cells[s]
            body = "|".join(measure_text(row, m) for m in measures)
            blocks.append(f"{_LABELS[s]}|{body}|")
        blocks.append("")

    return "\n".join(blocks).rstrip() + "\n"


def html_preview_string(placed: Sequence[PlacedNote],
                        title: str = "Resonote preview",
                        tempo: Optional[float] = None,
                        subtitle: str = "",
                        audio_src: Optional[str] = None) -> str:
    """Build the standalone HTML preview as a string (no file I/O).

    Used by the web layer so it can embed / serve the markup directly.
    """
    tab = ascii_tab(placed, tempo=tempo or 120.0)
    tab = tab.replace("&", "&amp;").replace("<", "&lt;")
    rows = "".join(
        f"<tr><td>{p.pitch}</td><td>str {p.string}</td>"
        f"<td>fret {p.fret}</td><td>finger {p.finger or 'open'}</td>"
        f"<td>{p.pluck or '-'}</td></tr>"
        for p in placed
    )
    meta = " &middot; ".join(x for x in [subtitle,
                                         f"{tempo:.0f} BPM" if tempo else ""]
                             if x)
    audio = (f'<audio controls src="{audio_src}"></audio>' if audio_src else "")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>
body{{background:#15151f;color:#e8e8f0;font-family:system-ui,sans-serif;margin:2rem}}
h2{{margin-bottom:.2rem}}
.meta{{color:#9fc3e8;font-size:14px;margin-bottom:1rem}}
pre{{background:#0e0e16;padding:1rem;border-radius:8px;font-size:14px;line-height:1.45;
    overflow-x:auto;white-space:pre}}
audio{{width:100%;margin:1rem 0}}
details{{margin-top:1rem}}
table{{border-collapse:collapse;margin-top:.6rem}}
td,th{{border:1px solid #333;padding:.3rem .7rem;font-size:13px}}
th{{color:#9fc3e8}}
</style></head>
<body><h2>{title}</h2>
<div class="meta">{meta}</div>
{audio}
<pre>{tab}</pre>
<details><summary>note list ({len(placed)})</summary>
<table><tr><th>pitch</th><th>string</th><th>fret</th><th>finger</th><th>pluck</th></tr>{rows}</table>
</details>
</body></html>"""


def write_html_preview(placed, out_html, title="Resonote preview", **kw):
    html = html_preview_string(placed, title, **kw)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html
