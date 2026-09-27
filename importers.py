"""Import existing tablature files into the internal model.

Why this is a separate path from the audio pipeline: a tab somebody already
made is *ground truth*. Running it through the arrangement engine would throw
away the fingering a human chose and replace it with our guess. So the importer
preserves ``string`` / ``fret`` exactly as written and skips L1-L7 entirely.

It also reads the source tuning, so a DADGAD or Drop-D tab comes in as DADGAD
rather than being silently reinterpreted as standard.

Supported inputs
  * Guitar Pro 3/4/5 (``.gp3``/``.gp4``/``.gp5``/``.gtp``) -- exact: real
    durations, ties, tunings and left-hand fingering.
  * ASCII text tablature (``.txt``/``.tab``) -- what you get off the internet.
    Carries no rhythm at all; see :func:`load_text` for how timing is derived.

Limitations (honest):
  * pyguitarpro reads GP3/GP4/GP5 only. GP6 (.gpx) and GP7/8 (.gp) are not
    readable here -- .gp is a ZIP+XML container and would need its own parser.
  * GP repeats are NOT expanded: you get the measures as written, so a song
    with repeat signs imports shorter than it sounds.
  * MusicXML is not supported yet.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from models import NOTE_NAMES, STANDARD_TUNING, PlacedNote

PLUCK_BY_STRING = {6: "p", 5: "p", 4: "i", 3: "i", 2: "m", 1: "a"}

GP_EXTENSIONS = (".gp5", ".gp4", ".gp3", ".gtp")
TEXT_EXTENSIONS = (".txt", ".tab", ".text")
TAB_EXTENSIONS = GP_EXTENSIONS + TEXT_EXTENSIONS


@dataclass
class ImportedScore:
    placed: List[PlacedNote] = field(default_factory=list)
    tempo: float = 120.0
    title: str = ""
    artist: str = ""
    track_name: str = ""
    tuning: List[int] = field(default_factory=list)   # MIDI pitches, string 1..6
    measures: int = 0
    source: str = ""
    warnings: List[str] = field(default_factory=list)

    def tuning_note(self) -> str:
        """'tuning: D-A-D-G-A-D', low string first (the way a tuning is
        normally spoken). Empty when the tab is in standard -- nothing to say."""
        if not self.tuning or self.tuning == STANDARD_TUNING:
            return ""
        return "tuning (low to high): " + _tuning_name(self.tuning)


def is_tab_file(path: str) -> bool:
    return str(path).lower().endswith(TAB_EXTENSIONS)


def load_tab(path: str, *, tempo: Optional[float] = None,
             track_index: int = 0) -> ImportedScore:
    """Dispatch on extension: Guitar Pro -> ``load_gp``, text -> ``load_text``.

    ``tempo`` is only meaningful for text tabs (a GP file has its own);
    ``track_index`` only for GP files (a text tab is one part).
    """
    low = str(path).lower()
    if low.endswith(TEXT_EXTENSIONS):
        return load_text(path, tempo=tempo or 120.0)
    if low.endswith(GP_EXTENSIONS):
        return load_gp(path, track_index=track_index)
    raise ValueError(f"{path}: not a tablature file "
                     f"(expected {'/'.join(TAB_EXTENSIONS)})")


def load_gp(path: str, track_index: int = 0) -> ImportedScore:
    """Read a Guitar Pro 3/4/5 file into PlacedNote list, keeping fingering."""
    try:
        import guitarpro as gp
    except ImportError as e:      # pragma: no cover - optional dependency
        raise RuntimeError(
            "Reading .gp files needs pyguitarpro.\n"
            "Install it with:  pip install resonote[gp]") from e

    song = gp.parse(path)
    if not song.tracks:
        raise ValueError(f"{path}: file contains no tracks")

    # prefer the requested track, else the first non-percussion one
    track = song.tracks[track_index] if track_index < len(song.tracks) else song.tracks[0]
    if track.isPercussionTrack:
        for t in song.tracks:
            if not t.isPercussionTrack:
                track = t
                break

    tempo = float(song.tempo or 120)
    ticks_per_sec = gp.Duration.quarterTime * tempo / 60.0

    # Collect beats and normalise the timeline: pyguitarpro hands measure 0 a
    # non-zero start tick, so anchoring on the first beat keeps onset 0 at 0 s.
    beats = []
    for m in track.measures:
        for v in m.voices:
            if not v or not v.beats:
                continue
            for b in v.beats:
                beats.append((b, v, m))
    if not beats:
        return ImportedScore(tempo=tempo, title=song.title or "",
                             warnings=["file contains no beats"])

    t0 = min(b.start for (b, _v, _m) in beats)

    placed: List[PlacedNote] = []
    last_on_string: dict = {}

    for (b, _v, _m) in sorted(beats, key=lambda x: x[0].start):
        onset = (b.start - t0) / ticks_per_sec
        dur = b.duration.time / ticks_per_sec

        for n in b.notes:
            # a tied note is a continuation -- extend the previous note on that
            # string instead of creating a second pluck
            if n.type == gp.NoteType.tie:
                prev = last_on_string.get(n.string)
                if prev is not None:
                    prev.duration += dur
                continue
            if n.type == gp.NoteType.dead:
                continue

            finger = 0
            try:
                lhf = n.effect.leftHandFinger
                if lhf is not None:
                    finger = int(lhf.value) if hasattr(lhf, "value") else int(lhf)
            except Exception:
                finger = 0

            p = PlacedNote(
                pitch=int(n.realValue),
                onset=round(onset, 6),
                duration=round(max(dur, 1e-3), 6),
                string=int(n.string),
                fret=int(n.value),
                finger=finger,
                pluck=PLUCK_BY_STRING.get(int(n.string), ""),
                role="melody",
            )
            placed.append(p)
            last_on_string[n.string] = p

    tuning: List[int] = []
    for s in track.strings:
        if 1 <= s.number <= 6:
            tuning.append(int(s.value))
    tuning.sort(reverse=True)          # string 1 (highest) first

    warnings: List[str] = []
    if len(song.tracks) > 1:
        warnings.append(
            f"file has {len(song.tracks)} tracks; imported track "
            f"'{track.name}' only")
    if any(m.header.isRepeatOpen or m.header.repeatClose > 0
           for m in track.measures):
        warnings.append("file uses repeat signs; they are NOT expanded")

    return ImportedScore(
        placed=placed,
        tempo=tempo,
        title=song.title or "",
        artist=song.artist or "",
        track_name=track.name or "",
        tuning=tuning,
        measures=len(track.measures),
        source=str(path),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# ASCII text tablature
# ---------------------------------------------------------------------------
#
# The honest problem: a text tab encodes *where on the neck* to put your
# fingers, and nothing else. There is no note value, no time signature, no
# tempo. What it does carry implicitly is bar lines, and people write one bar
# per measure. So the model we assume is:
#
#   * every stretch between two bar lines is one 4/4 measure;
#   * the columns inside that stretch divide the measure evenly, so a bar with
#     8 columns is eighths and a bar with 16 columns is sixteenths;
#   * a plucked note rings until the next note on the same string, capped at
#     one measure (a guitar string does not ring much longer than that).
#
# That assumption is stated back to the caller as a warning -- it is the one
# thing in this importer that is a guess rather than a read.

BEATS_PER_BAR = 4.0      # one bar line pair == one 4/4 measure
MAX_RING_BEATS = 4.0     # a plucked note never rings past the next bar

# characters that may appear inside a tab line; anything else ends the line.
# Note that no chord letter (A C D E F G) other than the bend/tap decorations
# is allowed, which is what keeps "   Am   G   C" rows from being read as tab.
_DECOR = "hpbrtsvxwq"
_TAB_CHARS = set("-|=.:;,0123456789 <>[]*^~/\\|\t" + _DECOR + _DECOR.upper())
_MUTE_CHARS = set("xX")

_TAB_LINE_RE = re.compile(r"^\s*(?P<label>[A-Ga-g][#b]?)?\s*[|:](?P<body>.*)$")
_PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _label_pc(label: str) -> Optional[int]:
    """'D#' -> 3, 'e' -> 4, '' -> None."""
    if not label:
        return None
    base = label[0].upper()
    if base not in _PITCH_CLASS:
        return None
    pc = _PITCH_CLASS[base]
    if label[1:] == "#":
        pc += 1
    elif label[1:] == "b":
        pc -= 1
    return pc % 12


def _pc_dist(a: Sequence[int], b: Sequence[int]) -> int:
    return sum(min(abs(x - y), 12 - abs(x - y)) for x, y in zip(a, b))


def _top_down(pcs: Sequence[Optional[int]]) -> bool:
    """True when the first of the six lines is the highest string.

    Both orders are common: Guitar Pro and most tab sites write high-E first
    (``e B G D A E``), plenty of others write low-E first (``E A D G B E``).
    Settled by whichever reading sits closer to standard tuning.
    """
    if len(pcs) != 6 or any(p is None for p in pcs):
        return True                     # unknown -> high-E-first convention
    std_hi = [p % 12 for p in STANDARD_TUNING]          # string 1..6
    top = [int(p) for p in pcs]
    bottom = list(reversed(top))
    # ties (e.g. an all-fourths tuning) fall through to high-E-first
    return not (_pc_dist(bottom, std_hi) < _pc_dist(top, std_hi))


def _resolve_tuning(pcs: Sequence[Optional[int]]) -> Optional[List[int]]:
    """Six pitch classes (first line first) -> MIDI pitches for string 1..6.

    Two things have to be guessed: which way round the six lines are, and what
    octave each string is in. Both are settled against standard tuning:
      * orientation  -- see :func:`_top_down`;
      * octave       -- nearest to the standard pitch for that string index,
        while keeping the six strings strictly descending.
    """
    if len(pcs) != 6 or any(p is None for p in pcs):
        return None

    top_down = [int(p) for p in pcs]
    if not _top_down(pcs):
        top_down = list(reversed(top_down))

    out: List[int] = []
    prev: Optional[int] = None
    for i, pc in enumerate(top_down):
        anchor = STANDARD_TUNING[i]
        best, bestd = None, None
        for midi in range(pc, 96, 12):
            if not (24 <= midi <= 88):
                continue
            if prev is not None and midi >= prev:
                continue
            d = abs(midi - anchor)
            if bestd is None or d < bestd:
                best, bestd = midi, d
        if best is None:
            return None
        out.append(best)
        prev = best
    return out


def _looks_like_tab(line: str) -> bool:
    """Is this one of the six string lines of a tab system?"""
    if "|" not in line and ":" not in line:
        return False
    m = _TAB_LINE_RE.match(line)
    if not m:
        return False
    if any(c.isdigit() for c in line):
        return True
    # a bar with no note at all still counts -- but only when the line names
    # its string, otherwise a decorative |------| rule would be read as tab
    return bool(m.group("label")) and ("---" in line or "===" in line)


def _cut(line: str) -> str:
    """Longest leading run of tab-safe characters. Kills trailing annotations
    like ``e|---0---|  (let ring)`` without losing the bar itself."""
    out = []
    for ch in line:
        if ch in _TAB_CHARS:
            out.append(ch)
        else:
            break
    return "".join(out)


def _split_bars(rest: str) -> List[str]:
    """``'|---0-|---5-|'`` -> ``['---0-', '---5-']`` (the content of each bar)."""
    parts = rest.split("|")[1:]
    if rest.rstrip().endswith("|"):
        parts = parts[:-1]
    return parts


def _systems(lines: Sequence[str]) -> Tuple[List[List[str]], List[str]]:
    """Group raw lines into 6-line tab systems."""
    runs, cur = [], []
    for raw in lines:
        if _looks_like_tab(raw):
            cur.append(raw)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)

    systems: List[List[str]] = []
    warnings: List[str] = []
    for run in runs:
        n = len(run)
        if n < 6:
            if n >= 3:
                warnings.append(
                    f"ignored {n} stray tab line(s) starting "
                    f"{run[0].strip()[:32]!r}")
            continue
        if n % 6:
            warnings.append(f"a tab block has {n} lines (not a multiple of 6); "
                            f"parsed {n // 6} system(s) from it")
        for k in range(0, n - n % 6, 6):
            systems.append(run[k:k + 6])
    return systems, warnings


def _system_bars(system: Sequence[str]) -> Tuple[List[str], List[List[str]]]:
    """One system -> (string labels, bars), each bar a list of 6 padded rows."""
    labels: List[str] = []
    rows: List[List[str]] = []
    for line in system:
        # the label is read from the raw line; the cut starts at the first bar
        # line, because the tuning letters themselves are not tab characters
        m = _TAB_LINE_RE.match(line)
        labels.append((m.group("label") if m and m.group("label") else "") or "")
        i = line.find("|")
        if i < 0:
            i = line.find(":")
        rows.append(_split_bars(_cut(line[i:])) if i >= 0 else [])

    n_bars = min((len(r) for r in rows), default=0)
    bars: List[List[str]] = []
    for b in range(n_bars):
        seg = [r[b] if b < len(r) else "" for r in rows]
        width = max((len(s) for s in seg), default=0)
        if width == 0:      # e.g. the empty stretch after a '||' double bar
            continue
        bars.append([s.ljust(width, "-") for s in seg])
    return labels, bars


def _frets(seg: str) -> List[Tuple[int, int]]:
    """Columns holding a fret number. ``'-12-'`` -> ``[(1, 12)]``; a two-digit
    fret occupies two columns, which is why the column index matters.
    ``fret == -1`` marks a muted string (x), which has no pitch."""
    out: List[Tuple[int, int]] = []
    i, n = 0, len(seg)
    while i < n:
        ch = seg[i]
        if ch.isdigit():
            j = i
            while j < n and seg[j].isdigit():
                j += 1
            out.append((i, int(seg[i:j])))
            i = j
        elif ch in _MUTE_CHARS:
            out.append((i, -1))
            i += 1
        else:
            i += 1
    return out


def _tuning_header(text: str) -> Optional[List[int]]:
    m = re.search(r"^\s*tuning\s*[:\-=]\s*(.+)$", text, re.I | re.M)
    if not m:
        return None
    toks = re.findall(r"[A-Ga-g][#b]?", m.group(1))
    if len(toks) < 6:
        return None
    return _resolve_tuning([_label_pc(t) for t in toks[:6]])


def _meta(text: str) -> Tuple[str, str]:
    def grab(*keys) -> str:
        for k in keys:
            m = re.search(rf"^\s*{k}\s*[:\-]\s*(.+)$", text, re.I | re.M)
            if m and m.group(1).strip():
                return m.group(1).strip()
        return ""
    return grab("title", "song", "songname"), grab("artist", "band", "author")


def parse_text_tab(text: str, *, source: str = "",
                   tempo: float = 120.0) -> ImportedScore:
    """Parse ASCII tablature text into PlacedNotes. See the header comment for
    how timing is derived, and :func:`load_text` for the file entry point."""
    # Windows editors save .txt tabs as CRLF with a UTF-8 BOM; the BOM would
    # sit in front of the first string label and cost us the whole first system
    text = text.lstrip("\ufeff").replace("\t", " ")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    systems, warnings = _systems(lines)
    if not systems:
        raise ValueError(
            "no 6-line tablature block found -- this does not look like an "
            "ASCII tab (expected six string lines like 'e|---0---|')")

    header_tuning = _tuning_header(text)
    tuning: Optional[List[int]] = None
    declared = False            # did any system actually name its strings?
    measures = 0
    widths: List[int] = []
    beat = 0.0
    events: List[Tuple[int, int, float, float]] = []   # string, fret, beat, col_dur

    for si, system in enumerate(systems):
        labels, bars = _system_bars(system)
        pcs = [_label_pc(l) for l in labels]
        top_down = _top_down(pcs)
        sys_tuning = _resolve_tuning(pcs)
        if sys_tuning is None:
            sys_tuning = header_tuning or list(STANDARD_TUNING)
        else:
            declared = True
        if tuning is None:
            tuning = sys_tuning
        elif sys_tuning != tuning:
            warnings.append(
                f"system {si + 1} declares a different tuning "
                f"({_tuning_name(sys_tuning)}); kept the first "
                f"({_tuning_name(tuning)})")

        for bar in bars:
            width = len(bar[0])
            if width == 0:
                continue
            widths.append(width)
            col_dur = BEATS_PER_BAR / width
            for li in range(6):
                string = li + 1 if top_down else 6 - li
                for (col, fret) in _frets(bar[li]):
                    events.append((string, fret, beat + col * col_dur, col_dur))
            beat += BEATS_PER_BAR
            measures += 1

    if tuning is None:
        tuning = list(STANDARD_TUNING)
    if not declared and header_tuning is None:
        warnings.append("no string labels found; assumed standard EADGBE")
    # a non-standard tuning is reported by ``tuning_note()`` in the summary,
    # not repeated here

    # a note rings until the next note on the same string, never past a bar
    by_string: Dict[int, List[int]] = {}
    for i, e in enumerate(events):
        by_string.setdefault(e[0], []).append(i)
    ring = [0.0] * len(events)
    for idxs in by_string.values():
        for k, i in enumerate(idxs):
            onset = events[i][2]
            nxt = events[idxs[k + 1]][2] if k + 1 < len(idxs) else None
            span = MAX_RING_BEATS if nxt is None else min(nxt - onset,
                                                          MAX_RING_BEATS)
            ring[i] = max(span, events[i][3])

    sec_per_beat = 60.0 / float(tempo or 120.0)
    placed: List[PlacedNote] = []
    muted = 0
    for i, (string, fret, onset, _d) in enumerate(events):
        if fret < 0:
            muted += 1
            continue
        placed.append(PlacedNote(
            pitch=tuning[string - 1] + fret,
            onset=round(onset * sec_per_beat, 6),
            duration=round(ring[i] * sec_per_beat, 6),
            string=string,
            fret=fret,
            finger=0,                       # text tabs never say which finger
            pluck=PLUCK_BY_STRING.get(string, ""),
            role="melody",
        ))
    placed.sort(key=lambda p: (p.onset, p.string))

    if muted:
        warnings.append(f"{muted} muted-string mark(s) (x) skipped: no pitch")
    if widths:
        lo, hi = min(widths), max(widths)
        note = (f"bars hold {lo}-{hi} columns" if lo != hi
                else f"bars hold {lo} columns")
        warnings.append(
            f"no rhythm in a text tab: every bar was read as one 4/4 measure "
            f"and its columns divided evenly ({note})")

    title, artist = _meta(text)
    return ImportedScore(
        placed=placed,
        tempo=float(tempo or 120.0),
        title=title,
        artist=artist,
        track_name="",
        tuning=tuning,
        measures=measures,
        source=source,
        warnings=warnings,
    )


def _tuning_name(tuning: Sequence[int]) -> str:
    """'D-A-D-G-A-D' -- low string first, the way a tuning is spoken."""
    return "-".join(NOTE_NAMES[p % 12] for p in reversed(tuning))


def load_text(path: str, tempo: float = 120.0) -> ImportedScore:
    """Read an ASCII tab (.txt/.tab) into PlacedNotes, keeping the fingering."""
    # utf-8-sig: a downloaded .txt very often carries a BOM
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        text = f.read()
    score = parse_text_tab(text, source=str(path), tempo=tempo)
    if not score.title:
        score.title = os.path.splitext(os.path.basename(str(path)))[0]
    return score
