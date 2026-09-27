"""M13 — importing ASCII text tablature.

This is the format you actually get off the internet, and it is a completely
different problem from Guitar Pro: a .txt tab encodes *where on the neck* to
put your fingers and nothing else. No note values, no time signature, no tempo,
no left-hand fingering. So the importer has to make one real assumption (every
bar is one 4/4 measure, its columns divide it evenly) and one real guess (the
tuning's octave + which way round the six lines are).

These tests pin down the parts that must be exact -- frets, strings, tuning,
bar timing, multi-digit frets, and ignoring all the prose that surrounds a real
tab -- and they pin down the guesses so a future change cannot silently alter
them.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from importers import (
    STANDARD_TUNING, _top_down, is_tab_file, load_tab, load_text,
    parse_text_tab,
)
from gp_export import to_gp5

STANDARD = [64, 59, 55, 50, 45, 40]
DADGAD = [62, 57, 55, 50, 45, 38]

# --- fixtures --------------------------------------------------------------

OPEN_E_SHAPE = (
    "e|---0---|\n"
    "B|---0---|\n"
    "G|---1---|\n"
    "D|---2---|\n"
    "A|---2---|\n"
    "E|---0---|\n"
)

TWO_BARS = (
    "e|---0---|-------|\n"
    "B|-------|-------|\n"
    "G|-------|-------|\n"
    "D|-------|-------|\n"
    "A|-------|-------|\n"
    "E|-------|---0---|\n"
)

WITH_PROSE = (
    "Song: Whatever\n"
    "Artist: Someone\n"
    "\n"
    "[Verse 1]\n"
    "   Am        G\n"
    "e|---0---|   (let ring)\n"
    "B|---1---|\n"
    "G|---2---|\n"
    "D|---2---|\n"
    "A|---0---|\n"
    "E|-------|\n"
    "\n"
    "la la la la la la\n"
)

CHORD_ROW_WITH_PIPES = (
    "   |Am   |G    |\n"
    "e|---0---|---3---|\n"
    "B|---1---|---0---|\n"
    "G|---2---|---0---|\n"
    "D|---2---|---0---|\n"
    "A|---0---|---2---|\n"
    "E|-------|---3---|\n"
)

DADGAD_TAB = (
    "Tuning: DADGAD\n"
    "\n"
    "D|-------|\n"
    "A|-------|\n"
    "G|-------|\n"
    "D|-------|\n"
    "A|-------|\n"
    "D|-0-----|\n"
)

LOW_E_FIRST = (
    "E|---0---|\n"
    "A|---2---|\n"
    "D|---2---|\n"
    "G|---1---|\n"
    "B|---0---|\n"
    "E|---0---|\n"
)


def _tmp_text(text: str, suffix: str = ".txt") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


class TestIsTabFile(unittest.TestCase):
    def test_text_tabs_are_tab_files(self):
        for name in ("a.txt", "a.TAB", "a.text", "x/y/b.txt"):
            self.assertTrue(is_tab_file(name), name)

    def test_gp_still_recognised(self):
        for name in ("a.gp5", "a.gp4", "a.gp3", "a.gtp"):
            self.assertTrue(is_tab_file(name), name)

    def test_rejects_audio_and_unsupported(self):
        for name in ("a.mp3", "a.wav", "a.mid", "a.pdf", "a.gpx", "a.gp"):
            self.assertFalse(is_tab_file(name), name)


class TestBasicParse(unittest.TestCase):
    def test_open_e_shape(self):
        s = parse_text_tab(OPEN_E_SHAPE)
        self.assertEqual(len(s.placed), 6)
        self.assertEqual(sorted(p.pitch for p in s.placed), [40, 47, 52, 56, 59, 64])
        self.assertEqual(s.tuning, STANDARD)
        self.assertEqual(s.measures, 1)

    def test_every_note_sits_on_its_own_string(self):
        s = parse_text_tab(OPEN_E_SHAPE)
        self.assertEqual(sorted(p.string for p in s.placed), [1, 2, 3, 4, 5, 6])

    def test_not_a_tab_at_all(self):
        with self.assertRaises(ValueError):
            parse_text_tab("hello world\njust some prose\n")

    def test_a_chord_row_is_not_read_as_a_tab_line(self):
        # '   |Am   |G    |' has pipes but no digits and no string label
        s = parse_text_tab(CHORD_ROW_WITH_PIPES)
        self.assertEqual(s.measures, 2)
        # bar 1 is an Am shape (E string silent -> 5 notes), bar 2 a G (6)
        self.assertEqual(len(s.placed), 11)


class TestFrets(unittest.TestCase):
    def test_two_digit_fret_is_one_note(self):
        s = parse_text_tab(
            "e|---12--|\nB|-------|\nG|-------|\n"
            "D|-------|\nA|-------|\nE|-------|\n")
        self.assertEqual(len(s.placed), 1)
        self.assertEqual(s.placed[0].fret, 12)
        self.assertEqual(s.placed[0].pitch, 64 + 12)

    def test_separate_digits_are_separate_notes(self):
        s = parse_text_tab(
            "e|-1-2---|\nB|-------|\nG|-------|\n"
            "D|-------|\nA|-------|\nE|-------|\n")
        self.assertEqual([p.fret for p in s.placed], [1, 2])

    def test_hammer_on_and_pull_off_are_still_two_notes(self):
        s = parse_text_tab(
            "e|-5h7---|\nB|-------|\nG|-------|\n"
            "D|-------|\nA|-------|\nE|-------|\n")
        self.assertEqual([p.fret for p in s.placed], [5, 7])

    def test_muted_string_has_no_pitch(self):
        s = parse_text_tab(
            "e|-0-x-0-|\nB|-------|\nG|-------|\n"
            "D|-------|\nA|-------|\nE|-------|\n")
        self.assertEqual([p.fret for p in s.placed], [0, 0])
        self.assertTrue(any("muted-string" in w for w in s.warnings))


class TestTiming(unittest.TestCase):
    """The one assumption in this importer: a bar is a 4/4 measure and its
    columns divide it evenly."""

    def _onsets(self, text, tempo=120.0):
        return [round(p.onset, 4) for p in
                parse_text_tab(text, tempo=tempo).placed]

    def test_column_inside_a_bar(self):
        # 7 columns, note in column 3 -> 3/7 of a 4-beat bar -> 12/7 beats
        # at 120 BPM one beat is 0.5 s, so 0.857143 s
        s = parse_text_tab(OPEN_E_SHAPE)
        self.assertAlmostEqual(s.placed[0].onset, (3 / 7) * 4 * 0.5, places=4)

    def test_second_bar_starts_after_the_first(self):
        s = parse_text_tab(TWO_BARS)
        first = min(p.onset for p in s.placed)
        second = max(p.onset for p in s.placed)
        self.assertAlmostEqual(second - first, 2.0, places=4)   # 4 beats @120

    def test_tempo_scales_time(self):
        slow = self._onsets(TWO_BARS, tempo=60.0)
        fast = self._onsets(TWO_BARS, tempo=120.0)
        for a, b in zip(slow, fast):
            self.assertAlmostEqual(a, b * 2.0, places=3)

    def test_two_systems_keep_running_time(self):
        double = OPEN_E_SHAPE + "\n" + OPEN_E_SHAPE
        s = parse_text_tab(double)
        self.assertEqual(s.measures, 2)
        onsets = sorted({round(p.onset, 4) for p in s.placed})
        self.assertEqual(len(onsets), 2)
        self.assertAlmostEqual(onsets[1] - onsets[0], 2.0, places=4)


class TestRinging(unittest.TestCase):
    def test_note_rings_until_the_next_note_on_that_string(self):
        s = parse_text_tab(
            "e|-0-0---|\nB|-------|\nG|-------|\n"
            "D|-------|\nA|-------|\nE|-------|\n")
        # 7 columns -> one column is 4/7 beat; the two notes are 2 apart
        self.assertAlmostEqual(s.placed[0].duration, (2 / 7) * 4 * 0.5, places=4)

    def test_ringing_is_capped_at_one_measure(self):
        s = parse_text_tab(TWO_BARS)          # nothing follows on either string
        for p in s.placed:
            self.assertAlmostEqual(p.duration, 2.0, places=4)   # 4 beats @120

    def test_a_note_never_lasts_less_than_its_own_column(self):
        dense = ("e|-0-0-0-0-0-0-0-|\nB|---------------|\nG|---------------|\n"
                 "D|---------------|\nA|---------------|\nE|---------------|\n")
        s = parse_text_tab(dense)
        col = (1 / 15) * 4 * 0.5
        for p in s.placed[:-1]:
            self.assertGreaterEqual(p.duration + 1e-9, col)


class TestProseIsIgnored(unittest.TestCase):
    def test_headers_lyrics_and_annotations(self):
        s = parse_text_tab(WITH_PROSE)
        self.assertEqual(s.title, "Whatever")
        self.assertEqual(s.artist, "Someone")
        self.assertEqual(s.measures, 1)
        self.assertEqual(len(s.placed), 5)          # the open Am shape

    def test_annotation_after_the_closing_bar_line(self):
        # 'e|---0---|   (let ring)' must not turn into a second bar
        tab = ("e|---0---|  (let ring)\nB|-------|\nG|-------|\n"
               "D|-------|\nA|-------|\nE|-------|\n")
        s = parse_text_tab(tab)
        self.assertEqual(s.measures, 1)

    def test_double_bar_line_does_not_add_a_measure(self):
        tab = "".join(f"{s}|-0-||\n" for s in "eBGDAE")
        s = parse_text_tab(tab)
        self.assertEqual(s.measures, 1)
        self.assertEqual(len(s.placed), 6)

    def test_stray_lines_are_reported_not_silently_dropped(self):
        s = parse_text_tab(OPEN_E_SHAPE + "\ne|---0---|\nB|---1---|\nG|---2---|\n")
        self.assertEqual(s.measures, 1)
        self.assertTrue(any("stray tab line" in w for w in s.warnings))

    def test_lyrics_between_systems_do_not_reset_time(self):
        s = parse_text_tab(OPEN_E_SHAPE + "\nsome words here\n" + OPEN_E_SHAPE)
        self.assertEqual(s.measures, 2)


class TestFileEncodings(unittest.TestCase):
    """Downloaded .txt tabs come from Windows: CRLF and a UTF-8 BOM."""

    def test_crlf(self):
        s = parse_text_tab(OPEN_E_SHAPE.replace("\n", "\r\n"))
        self.assertEqual(len(s.placed), 6)

    def test_bom_does_not_eat_the_first_system(self):
        # without stripping the BOM the first line stops matching, the system
        # drops to 5 lines and the whole tab is silently discarded
        s = parse_text_tab("\ufeff" + OPEN_E_SHAPE)
        self.assertEqual(len(s.placed), 6)
        self.assertEqual(s.measures, 1)

    def test_bom_on_disk(self):
        path = _tmp_text(OPEN_E_SHAPE)
        try:
            with open(path, "wb") as f:
                f.write(b"\xef\xbb\xbf" + OPEN_E_SHAPE.encode("utf-8"))
            self.assertEqual(len(load_text(path).placed), 6)
        finally:
            os.unlink(path)

    def test_crlf_on_disk(self):
        path = _tmp_text(OPEN_E_SHAPE)
        try:
            with open(path, "wb") as f:
                f.write(OPEN_E_SHAPE.replace("\n", "\r\n").encode("utf-8"))
            self.assertEqual(len(load_text(path).placed), 6)
        finally:
            os.unlink(path)


class TestTuning(unittest.TestCase):
    def test_standard_labels(self):
        self.assertEqual(parse_text_tab(OPEN_E_SHAPE).tuning, STANDARD)

    def test_dadgad_labels(self):
        s = parse_text_tab(DADGAD_TAB)
        self.assertEqual(s.tuning, DADGAD)
        self.assertIn("D-A-D-G-A-D", s.tuning_note())

    def test_dadgad_lowest_string_is_d2(self):
        s = parse_text_tab(DADGAD_TAB)
        self.assertEqual(len(s.placed), 1)
        self.assertEqual(s.placed[0].string, 6)
        self.assertEqual(s.placed[0].pitch, 38)      # D2

    def test_tuning_header_when_no_labels(self):
        tab = ("Tuning: D A D G A D\n\n" +
               "".join("|---0---|\n" for _ in range(6)))
        s = parse_text_tab(tab)
        self.assertEqual(s.tuning, DADGAD)

    def test_standard_header_resolves_to_standard(self):
        tab = ("Tuning: EADGBE\n\n" + "".join("|---0---|\n" for _ in range(6)))
        self.assertEqual(parse_text_tab(tab).tuning, STANDARD)

    def test_no_labels_at_all_warns_instead_of_guessing_silently(self):
        s = parse_text_tab("".join("|---0---|\n" for _ in range(6)))
        self.assertEqual(s.tuning, STANDARD)
        self.assertTrue(any("standard EADGBE" in w for w in s.warnings))

    def test_standard_tuning_reports_no_tuning_note(self):
        self.assertEqual(parse_text_tab(OPEN_E_SHAPE).tuning_note(), "")

    def test_drop_d(self):
        tab = ("E|-------|\nB|-------|\nG|-------|\n"
               "D|-------|\nA|-------|\nD|-0-----|\n")
        s = parse_text_tab(tab)
        self.assertEqual(s.tuning, [64, 59, 55, 50, 45, 38])


class TestOrientation(unittest.TestCase):
    """Both line orders are common; they must land on the same notes."""

    def test_high_e_first_is_detected(self):
        pcs = [4, 11, 7, 2, 9, 4]        # e B G D A E
        self.assertTrue(_top_down(pcs))

    def test_low_e_first_is_detected(self):
        pcs = [4, 9, 2, 7, 11, 4]        # E A D G B E
        self.assertFalse(_top_down(pcs))

    def test_unknown_labels_default_to_high_first(self):
        self.assertTrue(_top_down([None] * 6))

    def test_both_orders_give_the_same_chord(self):
        high = parse_text_tab(OPEN_E_SHAPE)
        low = parse_text_tab(LOW_E_FIRST)
        self.assertEqual(sorted(p.pitch for p in high.placed),
                         sorted(p.pitch for p in low.placed))
        self.assertEqual({p.string for p in high.placed} ==
                         {p.string for p in low.placed}, True)

    def test_dadgad_written_high_first_beats_the_name_spelling(self):
        # 'D A G D A D' read top-down is much closer to EADGBE than the same
        # letters read bottom-up, so it must not be flipped
        self.assertTrue(_top_down([2, 9, 7, 2, 9, 2]))


class TestLoadTabDispatch(unittest.TestCase):
    def test_txt_routes_to_the_text_parser(self):
        path = _tmp_text(OPEN_E_SHAPE)
        try:
            s = load_tab(path)
            self.assertEqual(len(s.placed), 6)
            self.assertEqual(s.title, os.path.splitext(os.path.basename(path))[0])
        finally:
            os.unlink(path)

    def test_tempo_override(self):
        path = _tmp_text(OPEN_E_SHAPE)
        try:
            self.assertEqual(load_tab(path, tempo=60).tempo, 60.0)
            self.assertEqual(load_tab(path).tempo, 120.0)
        finally:
            os.unlink(path)

    def test_unknown_extension(self):
        with self.assertRaises(ValueError):
            load_tab("song.mp3")


class TestTextToGp5RoundTrip(unittest.TestCase):
    """The exported .gp5 must carry the tab's own tuning -- writing a DADGAD
    tab with standard tuning would shift every note."""

    def _export(self, text, tempo=120.0, suffix=".txt"):
        src = _tmp_text(text, suffix)
        dst = tempfile.mktemp(suffix=".gp5")
        try:
            score = load_text(src, tempo=tempo)
            to_gp5(score.placed, dst, tempo=tempo, title="t",
                   tuning=score.tuning)
            return score, load_tab(dst)
        finally:
            if os.path.exists(src):
                os.unlink(src)
            if os.path.exists(dst):
                os.unlink(dst)

    def test_notes_and_fingering_survive(self):
        src, back = self._export(TWO_BARS)
        self.assertEqual(len(back.placed), len(src.placed))
        self.assertEqual([(p.string, p.fret) for p in src.placed],
                         [(p.string, p.fret) for p in back.placed])

    def test_tuning_survives(self):
        src, back = self._export(DADGAD_TAB)
        self.assertEqual(back.tuning, DADGAD)
        self.assertEqual(src.tuning, DADGAD)

    def test_standard_tuning_unchanged(self):
        _src, back = self._export(OPEN_E_SHAPE)
        self.assertEqual(back.tuning, STANDARD)

    def test_rejects_a_short_tuning(self):
        with self.assertRaises(ValueError):
            to_gp5([], tempfile.mktemp(suffix=".gp5"), tuning=[64, 59, 55])


if __name__ == "__main__":
    unittest.main()
