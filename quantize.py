"""L4.5 - Beat-grid quantization.

Raw AMT output has float onsets/durations that drift off the musical grid.
Before fingering and export, snap every note to the beat grid derived from the
L4 ``AnalysisResult`` (tempo + first detected beat). This is what makes the
tab rhythmically clean and actually playable -- sloppy timing in, clean rhythm
out.

Both the input ``Note`` stream and the final ``PlacedNote`` list can be
quantized; the latter is a cheap idempotent safety net right before export.

Design discipline (same as the rest of the pipeline):
  * Pure Python, zero external deps -- fully unit-testable in a thin env.
  * Grid is anchored at the first detected beat (``analysis.beats[0]``) so the
    downbeat lands on a bar line. ``subdivision`` sets grid resolution
    (2 = eighth notes, 4 = sixteenth notes). Both onset AND note-end are
    snapped, so no note straddles a grid line.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from models import Note, PlacedNote
from analysis import AnalysisResult


def grid_step(analysis: Optional[AnalysisResult], subdivision: int = 2) -> float:
    """Seconds per grid step (one grid step = 1 / ``subdivision`` of a beat)."""
    tempo = analysis.tempo if analysis else 120.0
    beat = 60.0 / tempo
    return beat / max(1, subdivision)


def anchor_of(analysis: Optional[AnalysisResult]) -> float:
    """Grid origin: the first detected beat (song downbeat). Falls back to 0.0."""
    if analysis and analysis.beats:
        return float(analysis.beats[0])
    return 0.0


def _snap(value: float, anchor: float, gu: float) -> float:
    rel = value - anchor
    return anchor + round(rel / gu) * gu


def quantize_notes(notes: Sequence[Note],
                   analysis: Optional[AnalysisResult] = None,
                   subdivision: int = 2) -> List[Note]:
    """Snap note onsets/durations to the detected beat grid.

    Returns a NEW list; inputs are not mutated.
    """
    gu = grid_step(analysis, subdivision)
    a = anchor_of(analysis)
    out: List[Note] = []
    for n in notes:
        new_onset = round(_snap(n.onset, a, gu), 6)
        end = _snap(n.onset + n.duration, a, gu)
        new_dur = max(gu, round(end - new_onset, 6))
        out.append(Note(
            pitch=n.pitch,
            onset=new_onset,
            duration=new_dur,
            velocity=n.velocity,
            instrument=n.instrument,
            track_id=n.track_id,
        ))
    return out


def quantize_placed(placed: Sequence[PlacedNote],
                    analysis: Optional[AnalysisResult] = None,
                    subdivision: int = 2) -> List[PlacedNote]:
    """Idempotent safety-net: snap a fingered ``PlacedNote`` list's timing only
    (pitch / string / fret / finger are untouched)."""
    gu = grid_step(analysis, subdivision)
    a = anchor_of(analysis)
    out: List[PlacedNote] = []
    for p in placed:
        new_onset = round(_snap(p.onset, a, gu), 6)
        end = _snap(p.onset + p.duration, a, gu)
        new_dur = max(gu, round(end - new_onset, 6))
        out.append(PlacedNote(
            pitch=p.pitch,
            onset=new_onset,
            duration=new_dur,
            string=p.string,
            fret=p.fret,
            finger=p.finger,
            pluck=p.pluck,
            role=p.role,
        ))
    return out
