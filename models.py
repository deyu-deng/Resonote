"""Shared data models for the audio -> guitar-tab pipeline."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# string 1..6 as MIDI pitches, 1 = high E (the thinnest). Every layer that
# writes a tab needs this, so it lives here instead of being copied around.
STANDARD_TUNING = [64, 59, 55, 50, 45, 40]


def tuning_labels(tuning: Optional[Sequence[int]] = None) -> Dict[int, str]:
    """String number -> the label printed at the left of the staff.

    Below middle C stays uppercase (B, A, E), at or above it goes lowercase
    (e, d) -- the convention every tab site uses, and why standard tuning reads
    ``e B G D A E`` top to bottom.
    """
    pitches = list(tuning) if tuning else STANDARD_TUNING
    if len(pitches) != 6:
        pitches = STANDARD_TUNING
    return {i + 1: (NOTE_NAMES[p % 12].lower() if p >= 60
                    else NOTE_NAMES[p % 12])
            for i, p in enumerate(pitches)}


@dataclass
class Note:
    """A transcribed musical note (monophonic melody-friendly)."""
    pitch: int           # MIDI note number (0-127)
    onset: float         # start time, seconds
    duration: float      # length, seconds
    velocity: int = 100  # 0-127
    instrument: str = ""  # source label, e.g. "vocals"/"bass"/"other" (L2 separation)
    track_id: int = 0    # which stem/track this note came from (0 = single source)


@dataclass
class PlacedNote:
    """A note placed on the guitar fretboard."""
    pitch: int
    onset: float
    duration: float
    string: int          # 1-6, where 1 = high E (thinnest string)
    fret: int            # 0 = open string
    finger: int          # 0 = open, 1-4 = left-hand finger
    pluck: str = ""      # right-hand: p/i/m/a (informational)
    role: str = ""       # L5 role: "melody" / "bass" / "harmony" (UX + judgment)
