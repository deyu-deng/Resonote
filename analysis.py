"""L4 — Music-theory analysis layer.

Turns a list of transcribed ``Note`` objects (optionally plus the raw audio)
into the structured music knowledge the L5 arrangement engine needs:

* a **chord timeline** (which chord sounds where)
* **beat grid + tempo**
* **key / tonality**
* **song sections** (verse / chorus / bridge boundaries)

Design follows the same discipline as ``separate.py`` / ``transcribe.py``:

* A heavy, high-accuracy backend (``madmom``) is loaded **lazily** and only
  used when audio is available and the package is installed.
* A pure-Python **rule-based backend** (no external deps) always works on the
  already-transcribed ``Note`` list, so the whole pipeline runs in a thin
  environment and is fully unit-testable.

The rule engine is deliberately honest: on a monophonic melody it produces
sparse / low-confidence chords (there simply is no harmony yet). On a real
mix, prefer the ``madmom`` audio path which hears the accompaniment directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from models import Note


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class Chord:
    """A chord segment on the timeline."""
    label: str           # e.g. "C", "Am", "G7", "Fmaj7", or "N" (no chord)
    root: int            # pitch class 0-11 (ignored when label == "N")
    quality: str         # "", "m", "7", "maj7", "m7", "dim", "aug", ...
    start: float         # seconds
    end: float           # seconds
    confidence: float = 1.0


@dataclass
class Section:
    """A song section (verse / chorus / bridge … boundaries)."""
    label: str           # "A", "B", "C", ... or "intro"/"outro" hints
    start: float         # seconds
    end: float           # seconds
    bars: int = 0


@dataclass
class AnalysisResult:
    chords: List[Chord] = field(default_factory=list)
    beats: List[float] = field(default_factory=list)
    tempo: float = 120.0
    key: str = ""                 # e.g. "C major" / "A minor"
    sections: List[Section] = field(default_factory=list)
    backend: str = "note-based"   # which backend produced this


# --------------------------------------------------------------------------- #
# Chord templates (pitch-class offsets from root)
# --------------------------------------------------------------------------- #
CHORD_TEMPLATES: dict[str, set[int]] = {
    "":      {0, 4, 7},       # major triad
    "m":     {0, 3, 7},       # minor triad
    "dim":   {0, 3, 6},
    "aug":   {0, 4, 8},
    "sus2":  {0, 2, 7},
    "sus4":  {0, 5, 7},
    "7":     {0, 4, 7, 10},
    "maj7":  {0, 4, 7, 11},
    "m7":    {0, 3, 7, 10},
    "m7b5":  {0, 3, 6, 10},
    "dim7":  {0, 3, 6, 9},
    "6":     {0, 4, 7, 9},
    "m6":    {0, 3, 7, 9},
    "9":     {0, 4, 7, 10, 14},
    "maj9":  {0, 4, 7, 11, 14},
    "m9":    {0, 3, 7, 10, 14},
}

# --------------------------------------------------------------------------- #
# Krumhansl-Schmuckler key profiles
# --------------------------------------------------------------------------- #
_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def _pc(midi: int) -> int:
    return midi % 12


def _name(pc: int) -> str:
    return PITCH_NAMES[pc % 12]


def _note_pcs(notes: Sequence[Note]) -> set[int]:
    return {_pc(n.pitch) for n in notes}


def _weighted_pc_hist(notes: Sequence[Note]) -> List[float]:
    hist = [0.0] * 12
    for n in notes:
        dur = max(n.duration, 0.05)
        hist[_pc(n.pitch)] += dur
    return hist


# --------------------------------------------------------------------------- #
# Rule-based chord estimation (works on Notes alone)
# --------------------------------------------------------------------------- #
def estimate_chord(pcs: set[int], bass_pc: Optional[int] = None,
                   allow_extensions: bool = True) -> Tuple[str, int, str, float]:
    """Pick the best chord label for a set of pitch classes.

    Returns ``(label, root_pc, quality, confidence)``. Confidence is the
    **count-based** coverage of the winning template (present notes / template
    size), so a 2-note "root+third" fragment still reads as a triad (0.67) while
    a lone note can never masquerade as a full chord — it falls back to an
    *implied* tone with an empty quality and low confidence.

    ``allow_extensions=False`` restricts the search to triads / sus / dim / aug
    (no 7th/9th). Use this for essentially monophonic input, where a single
    melodic line cannot reliably imply extended harmony — reporting "Dm9" from a
    one-note-at-a-time arpeggio would be misleading.
    """
    if not pcs:
        return ("N", 0, "", 0.0)

    if not allow_extensions:
        ext = {"7", "maj7", "m7", "m7b5", "dim7", "6", "m6", "9", "maj9", "m9"}
    else:
        ext = set()

    best = ("N", 0, "", 0.0)
    best_score = -1e9

    for root in range(12):
        for quality, tpl in CHORD_TEMPLATES.items():
            if quality in ext:
                continue
            abs_set = {(root + t) % 12 for t in tpl}
            present = pcs & abs_set
            coverage = len(present) / len(tpl)        # count-based
            if coverage < 0.5:                         # need >= half the chord
                continue
            extra = pcs - abs_set
            extra_frac = len(extra) / max(len(pcs), 1)
            score = coverage - 0.7 * extra_frac
            if bass_pc is not None and bass_pc == root:
                score += 0.25                          # bass note implies root
            if score > best_score:
                best_score = score
                best = (f"{_name(root)}{quality}", root, quality, coverage)

    # no quality cleared the threshold -> report an implied tone, not a chord
    if best[0] == "N":
        if bass_pc is not None and bass_pc in pcs:
            return (f"{_name(bass_pc)}", bass_pc, "", 0.35)
        dominant = max(pcs)
        return (f"{_name(dominant)}", dominant, "", 0.35)
    return best


def _max_simultaneous(notes: Sequence[Note]) -> int:
    """Max notes sounding at (roughly) the same instant — a polyphony proxy."""
    if not notes:
        return 0
    buckets: dict[int, int] = {}
    for n in notes:
        k = round(n.onset / 0.05)
        buckets[k] = buckets.get(k, 0) + 1
    return max(buckets.values())


# --------------------------------------------------------------------------- #
# Key detection (Krumhansl-Schmuckler)
# --------------------------------------------------------------------------- #
def detect_key(notes: Sequence[Note]) -> str:
    hist = _weighted_pc_hist(notes)
    mean = sum(hist) / 12.0
    centered = [h - mean for h in hist]

    def corr(profile: List[float], rotation: int) -> float:
        rot = profile[-rotation:] + profile[:-rotation] if rotation else profile
        return sum(c * r for c, r in zip(centered, [r - (sum(rot)/12.0) for r in rot]))

    best_key, best_score = "C major", -1e9
    for root in range(12):
        for mode, prof in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
            s = corr(prof, root)
            if s > best_score:
                best_score = s
                best_key = f"{_name(root)} {mode}"
    return best_key


# --------------------------------------------------------------------------- #
# Beat / tempo detection from note onsets (IOI histogram)
# --------------------------------------------------------------------------- #
def detect_beats(notes: Sequence[Note]) -> Tuple[List[float], float]:
    """Estimate a beat grid and tempo from note onsets.

    Uses an inter-onset-interval histogram: the dominant small interval is
    taken as the beat (or a subdivision). Good enough for already-quantized
    MIDI; for raw audio prefer the madmom backend.
    """
    onsets = sorted({round(n.onset, 4) for n in notes})
    if len(onsets) < 2:
        return ([n.onset for n in notes], 120.0)

    # IOIs between consecutive onsets
    iois = [b - a for a, b in zip(onsets, onsets[1:]) if b - a > 1e-3]
    if not iois:
        return (onsets, 120.0)

    # quantize IOIs to a histogram (resolution 20ms) to find the mode
    res = 0.02
    buckets: dict[int, float] = {}
    for d in iois:
        b = round(d / res)
        buckets[b] = buckets.get(b, 0.0) + (1.0 / len(iois))
    mode_bucket = max(buckets, key=lambda k: buckets[k])
    beat_int = max(mode_bucket * res, 1e-3)

    # a beat interval longer than ~1.2s is probably a half/whole note; halve it
    while beat_int > 1.2:
        beat_int /= 2.0

    tempo = 60.0 / beat_int
    # snap tempo to a musical range
    while tempo < 50:
        tempo *= 2
    while tempo > 240:
        tempo /= 2

    # build a grid starting from the first onset
    t0 = onsets[0]
    t_end = onsets[-1] + beat_int
    beats = []
    t = t0
    while t <= t_end + 1e-6:
        beats.append(round(t, 4))
        t += beat_int
    return (beats, round(tempo, 1))


# --------------------------------------------------------------------------- #
# Bar / chord segmentation helpers
# --------------------------------------------------------------------------- #
def _build_bars(beats: List[float], notes_end: float, beats_per_bar: int = 4) -> List[Tuple[float, float]]:
    if len(beats) < 2:
        return [(beats[0] if beats else 0.0, max(notes_end, 1.0))]
    bars: List[Tuple[float, float]] = []
    for i in range(0, len(beats) - 1, beats_per_bar):
        start = beats[i]
        end_idx = min(i + beats_per_bar, len(beats) - 1)
        end = beats[end_idx]
        if end <= start:
            continue
        bars.append((start, end))
    # extend the last bar to cover any trailing notes
    if bars and notes_end > bars[-1][1]:
        bars[-1] = (bars[-1][0], notes_end)
    if not bars:
        bars = [(beats[0], max(notes_end, beats[0] + 1.0))]
    return bars


def _chord_timeline(notes: Sequence[Note], bars: List[Tuple[float, float]],
                    allow_extensions: bool = True) -> List[Chord]:
    chords: List[Chord] = []
    prev_label = None
    for (b0, b1) in bars:
        bar_notes = [n for n in notes if b0 - 1e-6 <= n.onset < b1]
        if not bar_notes:
            # carry the previous chord across a silent bar
            if chords and chords[-1].end == prev_b1:
                chords[-1] = Chord(chords[-1].label, chords[-1].root,
                                   chords[-1].quality, chords[-1].start, b1,
                                   chords[-1].confidence)
            prev_label = None
            prev_b1 = b1
            continue
        pcs = _note_pcs(bar_notes)
        # bass = lowest pitch in the bar
        bass = min(bar_notes, key=lambda n: n.pitch)
        label, root, quality, conf = estimate_chord(
            pcs, _pc(bass.pitch), allow_extensions=allow_extensions)
        if label == prev_label:
            # extend current segment
            chords[-1] = Chord(label, root, quality, chords[-1].start, b1, conf)
        else:
            chords.append(Chord(label, root, quality, b0, b1, conf))
        prev_label = label
        prev_b1 = b1
    # merge zero-length / adjacent equal chords defensively
    return _merge_adjacent(chords)


def _merge_adjacent(chords: List[Chord]) -> List[Chord]:
    out: List[Chord] = []
    for c in chords:
        if out and out[-1].label == c.label and abs(out[-1].end - c.start) < 1e-3:
            out[-1] = Chord(out[-1].label, out[-1].root, out[-1].quality,
                            out[-1].start, c.end, max(out[-1].confidence, c.confidence))
        else:
            out.append(c)
    return out


# --------------------------------------------------------------------------- #
# Section detection (repetition-based, no ML)
# --------------------------------------------------------------------------- #
def _detect_sections(bar_labels: List[Optional[str]],
                     bars: List[Tuple[float, float]]) -> List[Section]:
    """Group bars into sections by rests + repeated chord patterns.

    A rest bar (``None``) is a hard boundary. Non-rest runs become candidate
    sections; identical runs are given the same letter (A, B, C…), which is
    exactly the verse/chorus distinction the L5 engine needs for density.
    """
    if not bars:
        return []

    # 1. split into raw runs separated by rests
    runs: List[List[int]] = []  # each run = list of bar indices
    cur: List[int] = []
    for i, lab in enumerate(bar_labels):
        if lab is None:
            if cur:
                runs.append(cur)
                cur = []
        else:
            cur.append(i)
    if cur:
        runs.append(cur)

    # 2. represent each run by its chord-label signature
    signatures = ["|".join(bar_labels[i] or "REST" for i in run) for run in runs]

    # 3. assign section letters; identical signatures share a letter
    seen: dict[str, str] = {}
    sections: List[Section] = []
    letter_idx = 0
    letters = "ABCDEFGHIJ"
    for run, sig in zip(runs, signatures):
        if sig not in seen:
            seen[sig] = letters[letter_idx] if letter_idx < len(letters) else f"S{letter_idx}"
            letter_idx += 1
        sec = seen[sig]
        b0 = bars[run[0]][0]
        b1 = bars[run[-1]][1]
        sections.append(Section(sec, b0, b1, len(run)))
    return sections


# --------------------------------------------------------------------------- #
# Backend selection + public API
# --------------------------------------------------------------------------- #
def madmom_available() -> bool:
    try:
        import madmom  # noqa: F401
        return True
    except Exception:
        return False


def analyze(notes: Sequence[Note],
            audio: Optional[object] = None,
            sr: Optional[int] = None,
            backend: str = "auto") -> AnalysisResult:
    """Run the L4 analysis.

    Parameters
    ----------
    notes : transcribed notes (from L3)
    audio : optional raw audio (numpy array) for the madmom backend
    sr    : sample rate of ``audio``
    backend : "auto" | "note-based" | "madmom"

    Returns an :class:`AnalysisResult`.
    """
    if backend == "auto":
        backend = "madmom" if (audio is not None and madmom_available()) else "note-based"

    if backend == "madmom":
        if not madmom_available():
            raise RuntimeError(
                "madmom backend requested but 'madmom' is not installed.\n"
                "Install it for audio-driven chord/beat analysis:\n"
                "  pip install madmom\n"
                "Or drop the backend (use 'auto' / 'note-based').")
        return _analyze_madmom(notes, audio, sr)

    # ----- note-based rule engine (always available) -----
    # A single melodic line can't imply extended harmony; cap chord complexity.
    polyphony = _max_simultaneous(notes)
    allow_ext = polyphony > 1

    beats, tempo = detect_beats(notes)
    bars = _build_bars(beats, max((n.onset + n.duration for n in notes), default=0.0))
    chords = _chord_timeline(notes, bars, allow_extensions=allow_ext)
    key = detect_key(notes)

    # per-bar chord labels for sectioning (None = rest/silent bar)
    bar_labels: List[Optional[str]] = []
    for (b0, b1) in bars:
        bar_notes = [n for n in notes if b0 - 1e-6 <= n.onset < b1]
        if not bar_notes:
            bar_labels.append(None)
        else:
            pcs = _note_pcs(bar_notes)
            bass = min(bar_notes, key=lambda n: n.pitch)
            label, _, _, _ = estimate_chord(pcs, _pc(bass.pitch), allow_extensions=allow_ext)
            bar_labels.append(label)
    sections = _detect_sections(bar_labels, bars)

    return AnalysisResult(chords=chords, beats=beats, tempo=tempo,
                          key=key, sections=sections, backend="note-based")


# --------------------------------------------------------------------------- #
# madmom backend (lazy, audio-driven — high accuracy)
# --------------------------------------------------------------------------- #
def _parse_madmom_chord(label: str) -> Tuple[str, int, str]:
    """Convert a madmom chord label like 'C:maj7' / 'A:min' / 'N' to ours."""
    if label in ("N", "X", ""):
        return ("N", 0, "")
    try:
        root_str, qual = label.split(":")
    except ValueError:
        root_str, qual = label, ""
    root = PITCH_NAMES.index(root_str) if root_str in PITCH_NAMES else 0
    qmap = {"maj": "", "min": "m", "maj7": "maj7", "min7": "m7",
            "7": "7", "dim": "dim", "aug": "aug", "sus4": "sus4", "sus2": "sus2",
            "maj6": "6", "min6": "m6", "hdim7": "m7b5", "dim7": "dim7",
            "min9": "m9", "maj9": "maj9", "9": "9"}
    quality = qmap.get(qual, qual)
    return (f"{root_str}{quality}", root, quality)


def _analyze_madmom(notes: Sequence[Note], audio, sr) -> AnalysisResult:
    if audio is None:
        raise RuntimeError("madmom backend requires raw audio (got None)")
    import numpy as np
    import madmom

    # --- chords from audio ---
    feat = madmom.features.chords.CNNChordFeatureProcessor()
    rec = madmom.features.chords.CRFChordRecognitionProcessor()
    raw_chords = rec(feat(audio))
    chords: List[Chord] = []
    for start, end, lab in raw_chords:
        label, root, quality = _parse_madmom_chord(lab)
        if chords and chords[-1].label == label:
            chords[-1].end = end
        else:
            chords.append(Chord(label, root, quality, float(start), float(end), 1.0))

    # --- beats / tempo from audio ---
    beat_proc = madmom.features.beats.BeatTrackingProcessor(fps=100)
    beats = list(beat_proc(madmom.features.beats.RNNBeatProcessor()(audio)))
    tempo_proc = madmom.features.tempo.TempoEstimationProcessor()
    tempo = float(tempo_proc(madmom.features.tempo.TempoEstimationProcessor()(audio))[0][0]) \
        if False else 120.0  # safe fallback; real tempo extracted below
    try:
        tmp = madmom.features.tempo.TempoEstimationProcessor()
        est = tmp(madmom.features.tempo.CombFilterBankProcessor()(np.asarray(audio)))
        if len(est):
            tempo = float(est[0][0])
    except Exception:
        pass

    key = detect_key(notes)  # key derived from notes (pitch content)

    # sections: reuse the rule engine's repetition detector on the madmom bars
    bars = _build_bars([float(b) for b in beats],
                       max((n.onset + n.duration for n in notes), default=0.0))
    bar_labels: List[Optional[str]] = []
    for (b0, b1) in bars:
        active = [c for c in chords if c.start < b1 and c.end > b0]
        bar_labels.append(active[-1].label if active else None)
    sections = _detect_sections(bar_labels, bars)

    return AnalysisResult(chords=chords, beats=[float(b) for b in beats],
                          tempo=round(tempo, 1), key=key, sections=sections,
                          backend="madmom")


# --------------------------------------------------------------------------- #
# Convenience formatting for CLI / debugging
# --------------------------------------------------------------------------- #
def format_summary(a: AnalysisResult) -> str:
    lines = [f"  backend : {a.backend}",
             f"  key     : {a.key or '(unknown)'}",
             f"  tempo   : {a.tempo:.1f} BPM",
             f"  beats   : {len(a.beats)}",
             f"  chords  : {len(a.chords)}"]
    if a.chords:
        head = ", ".join(f"{c.label}@{c.start:.1f}s" for c in a.chords[:8])
        tail = " …" if len(a.chords) > 8 else ""
        lines.append(f"           {head}{tail}")
    if a.sections:
        secs = " ".join(f"{s.label}[{s.start:.1f}-{s.end:.1f}s,{s.bars}b]"
                        for s in a.sections)
        lines.append(f"  sections: {secs}")
    return "\n".join(lines)
