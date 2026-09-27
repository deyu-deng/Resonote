"""Tablature PDF export — pure Python, no dependencies.

Why this exists: every other preview we have either needs an external app
(Guitar Pro / TuxGuitar), a web font loaded from a CDN (alphaTab), or is plain
text. A PDF needs none of that: it is vector, zooms without pixelating, prints,
and opens on anything.

No reportlab / cairosvg / weasyprint — the PDF is written by hand. That keeps
the dependency surface at zero, which matters because this project already
carries a heavy ML stack. Text uses the PDF base-14 Helvetica, so no font
embedding is required either (which also means non-ASCII titles are stripped:
Helvetica cannot render them).
"""

import math
from typing import Dict, List, Optional, Sequence

from models import PlacedNote, tuning_labels

# display order: high E on top, low E at the bottom
_ORDER = [1, 2, 3, 4, 5, 6]
SLOTS_PER_MEASURE = 8

A4 = (595.0, 842.0)
LETTER = (612.0, 792.0)

_HELV = "/Helvetica"
_HELV_B = "/Helvetica-Bold"


def _ascii(text: str, fallback: str = "Resonote") -> str:
    """Helvetica is Latin-only; drop anything it cannot draw."""
    cleaned = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in (text or ""))
    cleaned = " ".join(cleaned.split())
    return cleaned or fallback


class _Pdf:
    """Minimal single-font PDF writer (enough for vector text + lines)."""

    def __init__(self, page_w: float, page_h: float):
        self.page_w, self.page_h = page_w, page_h
        self.pages: List[List[str]] = []
        self._cur: List[str] = []

    # --- page handling ----------------------------------------------------
    def new_page(self):
        if self._cur:
            self.pages.append(self._cur)
        self._cur = []

    # --- drawing primitives ----------------------------------------------
    def _op(self, s: str):
        self._cur.append(s)

    def line(self, x1, y1, x2, y2, w: float = 0.6):
        self._op(f"{w} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def text(self, x, y, s, size: float = 8.0, bold: bool = False,
             align: str = "left"):
        s = _ascii(s, "")
        if not s:
            return
        if align != "left":
            w = len(s) * size * 0.55
            x = x - w if align == "right" else x - w / 2.0
        f = _HELV_B if bold else _HELV
        esc = s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        self._op(f"BT {f} {size} Tf {x:.2f} {y:.2f} Td ({esc}) Tj ET")

    # --- output -----------------------------------------------------------
    def save(self, path: str) -> str:
        if self._cur:
            self.pages.append(self._cur)
            self._cur = []
        if not self.pages:
            self.pages.append([])

        objs: List[bytes] = []

        def add(body: bytes) -> int:
            objs.append(body)
            return len(objs)          # 1-based object number

        catalog_id = add(b"")          # placeholder, filled below
        pages_id = add(b"")
        font_id = add(
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
            b"/Encoding /WinAnsiEncoding >>")
        fontb_id = add(
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold "
            b"/Encoding /WinAnsiEncoding >>")

        page_ids = []
        for content in self.pages:
            stream = ("\n".join(content)).encode("latin-1", "replace")
            cid = add(b"<< /Length " + str(len(stream)).encode()
                      + b" >>\nstream\n" + stream + b"\nendstream")
            pid = add(
                b"<< /Type /Page /Parent " + str(pages_id).encode()
                + b" 0 R /MediaBox [0 0 "
                + f"{self.page_w:.0f} {self.page_h:.0f}".encode()
                + b"] /Resources << /Font << /Helvetica "
                + str(font_id).encode() + b" 0 R /Helvetica-Bold "
                + str(fontb_id).encode() + b" 0 R >> >> /Contents "
                + str(cid).encode() + b" 0 R >>")
            page_ids.append(pid)

        kids = b" ".join(str(i).encode() + b" 0 R" for i in page_ids)
        objs[catalog_id - 1] = (b"<< /Type /Catalog /Pages "
                                + str(pages_id).encode() + b" 0 R >>")
        objs[pages_id - 1] = (b"<< /Type /Pages /Kids [" + kids
                              + b"] /Count " + str(len(page_ids)).encode()
                              + b" >>")

        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for i, body in enumerate(objs, start=1):
            offsets.append(len(out))
            out += str(i).encode() + b" 0 obj\n" + body + b"\nendobj\n"

        xref_at = len(out)
        out += b"xref\n0 " + str(len(objs) + 1).encode() + b"\n"
        out += b"0000000000 65535 f \n"
        for off in offsets:
            out += f"{off:010d} 00000 n \n".encode()
        out += (b"trailer\n<< /Size " + str(len(objs) + 1).encode()
                + b" /Root " + str(catalog_id).encode() + b" 0 R >>\n"
                + b"startxref\n" + str(xref_at).encode() + b"\n%%EOF\n")

        with open(path, "wb") as f:
            f.write(bytes(out))
        return path


def _grid(placed: Sequence[PlacedNote], tempo: float, slots_per_measure: int):
    """Notes -> (slot index, PlacedNote) on the eighth-note grid."""
    if not placed:
        return [], 0
    anchor = min(p.onset for p in placed)
    slots_per_sec = tempo / 30.0
    items, total = [], 0
    for p in placed:
        s = max(0, int(round((p.onset - anchor) * slots_per_sec)))
        length = max(1, int(round(p.duration * slots_per_sec)))
        items.append((s, length, p))
        total = max(total, s + length)
    n_measures = max(1, math.ceil(total / slots_per_measure))
    return items, n_measures * slots_per_measure


def write_tab_pdf(placed: Sequence[PlacedNote],
                  out_path: str,
                  tempo: float = 120.0,
                  title: str = "Resonote",
                  slots_per_measure: int = SLOTS_PER_MEASURE,
                  slot_width: float = 11.0,
                  page_size: str = "A4",
                  landscape: bool = True,
                  tuning: Optional[Sequence[int]] = None,
                  subtitle: str = "") -> str:
    """Render `placed` as a printable guitar-tablature PDF.

    ``tuning`` (six MIDI pitches, string 1 first) only changes the printed
    string labels -- the frets themselves come from the notes. Pass it when the
    tab is not in standard, or the PDF will label a DADGAD tab as EADGBE.
    """
    pw, ph = A4 if page_size.upper() == "A4" else LETTER
    if landscape:
        pw, ph = ph, pw

    margin = 40.0
    gutter = 20.0                      # room for the E/A/D/G/B/e labels
    line_gap = 11.0                    # spacing between the 6 staff lines
    staff_h = line_gap * 5
    num_h = 14.0                       # measure-number row
    sys_gap = 26.0
    sys_h = staff_h + num_h + sys_gap

    usable_w = pw - 2 * margin - gutter
    measure_w = slots_per_measure * slot_width
    per_line = max(1, int(usable_w // measure_w))
    per_line = min(per_line, 8)        # more than 8 gets cramped

    pdf = _Pdf(pw, ph)
    pdf.new_page()
    y = ph - margin

    head = _ascii(title)
    pdf.text(margin, y - 12, head, size=14, bold=True)
    pdf.text(pw - margin, y - 12, f"{tempo:.0f} BPM", size=10, align="right")
    y -= 22
    if subtitle:
        pdf.text(margin, y, _ascii(subtitle), size=9)
        y -= 12
    else:
        y -= 4

    labels = tuning_labels(tuning)

    items, total = _grid(placed, tempo, slots_per_measure)
    n_measures = total // slots_per_measure if placed else 0

    cells: dict = {}
    for (s, length, p) in items:
        cells.setdefault(s, {}).setdefault(p.string, p.fret)

    def draw_system(first: int, last: int, top: float):
        n = last - first
        x0 = margin + gutter
        width = n * measure_w
        bottom = top - staff_h

        # staff lines
        for i in range(6):
            ly = top - i * line_gap
            pdf.line(x0, ly, x0 + width, ly, 0.5)
        # string labels
        for i, s in enumerate(_ORDER):
            pdf.text(margin + 4, top - i * line_gap + 3.0, labels[s], size=8)
        # bar lines
        for m in range(n + 1):
            bx = x0 + m * measure_w
            pdf.line(bx, top, bx, bottom, 1.0 if (m == 0 or m == n) else 0.5)
        # measure numbers
        for m in range(n):
            mx = x0 + m * measure_w + 2
            pdf.text(mx, top + 4, str(first + m + 1), size=7)
        # frets
        for m in range(n):
            base = first + m
            for slot in range(slots_per_measure):
                g = base * slots_per_measure + slot
                hit = cells.get(g)
                if not hit:
                    continue
                cx = x0 + m * measure_w + slot * slot_width + slot_width / 2.0
                for s, fret in hit.items():
                    row = _ORDER.index(s)
                    pdf.text(cx, top - row * line_gap + 3.0, str(fret),
                             size=8, align="center")
        return bottom

    if not n_measures:
        pdf.text(margin, y, "(no notes)", size=10)
        return pdf.save(out_path)

    for first in range(0, n_measures, per_line):
        last = min(first + per_line, n_measures)
        if y - sys_h < margin:
            pdf.new_page()
            y = ph - margin
        bottom = draw_system(first, last, y - num_h)
        y = bottom - sys_gap

    return pdf.save(out_path)
