"""End-to-end test: arrange a melody, export .gp5, read it back.

Run:  python tests/test_e2e.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import guitarpro as gp
from models import Note
from arrange import arrange, OPEN_MIDI
from gp_export import to_gp5
from preview import ascii_tab

OUT = "/tmp/resonote_test.gp5"

# C-major scale, on a steady eighth grid (0.5s @ 120bpm == quarter)
PITCHES = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60]


def test_roundtrip():
    notes = [Note(pitch=p, onset=i * 0.5, duration=0.5) for i, p in enumerate(PITCHES)]
    placed = arrange(notes)

    # 1. arrangement sanity
    assert len(placed) == len(notes), "lost notes in arrangement"
    for p in placed:
        assert 1 <= p.string <= 6, f"bad string {p.string}"
        assert 0 <= p.fret <= 19, f"bad fret {p.fret}"
        # the chosen position must actually sound the right pitch
        assert OPEN_MIDI[p.string] + p.fret == p.pitch, "position mismatch"
        assert 0 <= p.finger <= 4, f"bad finger {p.finger}"

    # 2. export
    to_gp5(placed, OUT, tempo=120)

    # 3. read back with pyguitarpro
    song = gp.parse(OUT)
    read_notes = [n for t in song.tracks for m in t.measures
                  for v in m.voices for b in v.beats for n in b.notes]
    assert len(read_notes) == len(notes), f"notes lost on save ({len(read_notes)}!={len(notes)})"

    # 4. verify fret/string survived the round trip
    bad = [n for n in read_notes if not (1 <= n.string <= 6 and 0 <= n.value <= 19)]
    assert not bad, "invalid note in saved file"

    print("OK: arranged", len(placed), "notes; .gp5 written and re-read cleanly.")
    print(ascii_tab(placed))


if __name__ == "__main__":
    test_roundtrip()
