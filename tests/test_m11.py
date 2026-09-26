"""M11 — dependency-free tablature PDF export.

The PDF is written by hand (no reportlab / cairosvg), so these tests are the
only thing standing between a typo and a corrupt file. They check the parts
that actually break: xref offsets, page count, and that the content stream
really contains one fret number per placed note.
"""

import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PlacedNote
from tab_pdf import write_tab_pdf


def pn(pitch, onset, dur, string=1, fret=0):
    return PlacedNote(pitch=pitch, onset=onset, duration=dur, string=string,
                      fret=fret, finger=0, pluck="", role="melody")


def _read(path):
    with open(path, "rb") as f:
        return f.read()


def _streams(data):
    return b"\n".join(re.findall(rb"stream\n(.*?)\nendstream", data, re.S))


class TestTabPdf(unittest.TestCase):
    def _export(self, placed, **kw):
        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        try:
            write_tab_pdf(placed, path, **kw)
            return _read(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_valid_header_and_trailer(self):
        data = self._export([pn(64, 0.0, 0.5)])
        self.assertTrue(data.startswith(b"%PDF-1.4"))
        self.assertTrue(data.rstrip().endswith(b"%%EOF"))

    def test_xref_offsets_are_correct(self):
        """Every xref entry must point at its own 'N 0 obj'."""
        data = self._export([pn(64, 0.0, 0.5), pn(67, 0.5, 0.5, 3, 7)])
        xref_at = int(re.search(rb"startxref\s+(\d+)", data).group(1))
        self.assertEqual(data[xref_at:xref_at + 4], b"xref")
        entries = re.findall(rb"(\d{10}) (\d{5}) ([nf])", data[xref_at:])
        self.assertGreater(len(entries), 3)
        for i, (off, _gen, kind) in enumerate(entries):
            if kind == b"f":
                continue
            expect = str(i).encode() + b" 0 obj"
            self.assertTrue(data[int(off):].startswith(expect),
                            f"xref entry {i} points at the wrong offset")

    def test_one_fret_number_per_note_plus_measure_numbers(self):
        # spans 6 eighth-note slots at 120 BPM -> a single (padded) measure
        placed = [pn(64, 0.0, 0.25, 1, 5), pn(67, 0.25, 0.25, 3, 7),
                  pn(64, 1.0, 0.25, 1, 3), pn(69, 1.25, 0.25, 2, 9)]
        data = self._export(placed, tempo=120.0)
        content = _streams(data).decode("latin-1")
        numbers = re.findall(r"\((\d+)\) Tj", content)
        self.assertEqual(len(numbers), len(placed) + 1,   # + 1 measure number
                         "each placed note should appear once, plus measure nums")

    def test_landscape_a4_media_box(self):
        data = self._export([pn(64, 0.0, 0.5)])
        boxes = re.findall(r"/MediaBox \[0 0 (\d+) (\d+)\]",
                           data.decode("latin-1"))
        self.assertTrue(boxes)
        w, h = int(boxes[0][0]), int(boxes[0][1])
        self.assertGreater(w, h, "default page should be landscape")

    def test_non_ascii_title_is_stripped(self):
        """Helvetica is Latin-only -- CJK titles must not reach the stream."""
        data = self._export([pn(64, 0.0, 0.5)], title="陶喆《无缘》")
        content = _streams(data).decode("latin-1")
        self.assertTrue(all(ord(c) < 128 for c in content),
                        "non-ASCII leaked into the PDF content stream")

    def test_empty_arrangement_still_writes_a_file(self):
        data = self._export([])
        self.assertTrue(data.startswith(b"%PDF-1.4"))

    def test_long_piece_paginates(self):
        # 400 eighth notes = 100 measures; at 8 measures per line that is
        # 13 systems, which cannot fit on one landscape page.
        placed = [pn(64 + (i % 5), i * 0.25, 0.2, 1, i % 12)
                  for i in range(400)]
        data = self._export(placed, tempo=120.0)
        pages = len(re.findall(rb"/Type /Page[^s]", data))
        self.assertGreaterEqual(pages, 2, "100 measures must spill onto page 2")


if __name__ == "__main__":
    unittest.main()
