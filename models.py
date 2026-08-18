"""Shared data models for the audio -> guitar-tab pipeline."""

from dataclasses import dataclass


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
