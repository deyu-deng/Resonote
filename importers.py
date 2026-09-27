"""Import existing tablature files (Guitar Pro 3/4/5) into the internal model.

Why this is a separate path from the audio pipeline: a tab somebody already
made is *ground truth*. Running it through the arrangement engine would throw
away the fingering a human chose and replace it with our guess. So the importer
preserves ``string`` / ``fret`` exactly as written and skips L1-L7 entirely.

It also reads the track's tuning, so a DADGAD or Drop-D tab comes in as DADGAD
rather than being silently reinterpreted as standard.

Limitations (honest):
  * pyguitarpro reads GP3/GP4/GP5 only. GP6 (.gpx) and GP7/8 (.gp) are not
    readable here -- .gp is a ZIP+XML container and would need its own parser.
  * Repeats are NOT expanded: you get the measures as written, so a song with
    repeat signs imports shorter than it sounds.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from models import PlacedNote

PLUCK_BY_STRING = {6: "p", 5: "p", 4: "i", 3: "i", 2: "m", 1: "a"}
GP_EXTENSIONS = (".gp5", ".gp4", ".gp3", ".gtp")


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


def is_tab_file(path: str) -> bool:
    return str(path).lower().endswith(GP_EXTENSIONS)


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
