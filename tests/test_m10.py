"""M10 — GP5 export robustness against the pyguitarpro rest bug.

``guitarpro`` (pyguitarpro) returns 0 for the length of a beat whose status is
``BeatStatus.empty``, so the reader's running ``start`` never advances and the
next beat reuses — and overwrites — the same slot. Any rest therefore collapses
and shifts everything after it earlier, which corrupts the file when read back.

The exporter works around this by carrying the previous sounding set across
silent slots (a chord keeps ringing, which is what a guitarist does anyway), so
no beat is ever completely empty and every measure still tiles to 3840 ticks.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import guitarpro as gp

from gp_export import to_gp5
from models import PlacedNote


def pn(pitch, onset, dur, string=1, fret=0):
    return PlacedNote(pitch=pitch, onset=onset, duration=dur, string=string,
                      fret=fret, finger=0, pluck="", role="melody")


def _export(placed, tempo=120.0):
    fd, path = tempfile.mkstemp(suffix=".gp5")
    os.close(fd)
    try:
        to_gp5(placed, path, tempo=tempo)
        return gp.parse(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


class TestNoEmptyBeats(unittest.TestCase):
    def test_trailing_silence_still_tiles(self):
        # one short note at the very start, then ~3.5 s of nothing
        song = _export([pn(64, 0.0, 0.5, 1, 0)])
        m = song.tracks[0].measures[0]
        beats = [b for v in m.voices for b in v.beats]
        self.assertTrue(beats)
        self.assertEqual(sum(b.duration.time for b in beats), m.header.length)
        self.assertTrue(all(b.notes for b in beats),
                        "rest beats cannot be read back by pyguitarpro")

    def test_interior_gap_has_no_empty_beat(self):
        placed = [pn(64, 0.0, 0.5, 1, 0),       # 0.0 - 0.5
                  pn(67, 1.5, 0.5, 1, 3)]       # gap 0.5 - 1.5
        song = _export(placed)
        m = song.tracks[0].measures[0]
        beats = [b for v in m.voices for b in v.beats]
        self.assertEqual(sum(b.duration.time for b in beats), m.header.length)
        self.assertTrue(all(b.notes for b in beats))

    def test_every_measure_tiles(self):
        # notes spread over ~3 measures with gaps between them
        placed = [pn(64 + i, i * 1.0, 0.4, 1, i) for i in range(6)]
        song = _export(placed)
        for mi, m in enumerate(song.tracks[0].measures):
            beats = [b for v in m.voices for b in v.beats]
            self.assertEqual(sum(b.duration.time for b in beats),
                             m.header.length, f"measure {mi} does not tile")
            self.assertTrue(all(b.notes for b in beats))

    def test_no_string_collision(self):
        # two notes on the same string at the same time must not both survive
        placed = [pn(64, 0.0, 0.5, 1, 0), pn(67, 0.0, 0.5, 1, 3)]
        song = _export(placed)
        for m in song.tracks[0].measures:
            for v in m.voices:
                for b in v.beats:
                    strings = [n.string for n in b.notes]
                    self.assertEqual(len(strings), len(set(strings)))


if __name__ == "__main__":
    unittest.main()
