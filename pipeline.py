"""Single orchestration entry for the Resonote pipeline (L1 -> L8).

Both the CLI (:mod:`main`) and the web backend (:mod:`web_server`) call
:func:`run` so the pipeline logic lives in exactly one place. It returns an
in-memory :class:`PipelineResult` (gp5/wav/midi bytes, html string, ascii
tab, analysis) and optionally writes the artifacts to disk when paths are
given.
"""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

from models import Note, PlacedNote

from transcribe import (
    get_backend, transcribe_audio,
    yourmt3_available, basic_pitch_available,
)
from separate import get_separator, demucs_available
from analysis import analyze, format_summary
from quantize import quantize_notes, quantize_placed
from arrange import arrange
from gp_export import to_gp5
from midi_export import placed_to_midi, placed_to_wav
from preview import ascii_tab, html_preview_string


def is_midi(path: str) -> bool:
    return path.lower().endswith((".mid", ".midi"))


@dataclass
class PipelineResult:
    placed: List[PlacedNote]
    analysis: object                      # AnalysisResult (key/tempo/beats/chords)
    summary: str                          # human-readable analysis summary
    ascii_tab: str                        # six-line ASCII tablature
    html_preview: str                     # standalone HTML preview (string)
    gp5_bytes: bytes = b""
    wav_bytes: Optional[bytes] = None
    midi_bytes: Optional[bytes] = None
    tempo: float = 120.0
    key: str = ""
    role_counts: dict = field(default_factory=dict)


def _write_to_temp(write_fn) -> bytes:
    """Call ``write_fn(path)`` against a temp file and return its bytes."""
    fd, p = tempfile.mkstemp()
    os.close(fd)
    try:
        write_fn(p)
        with open(p, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(p)
        except OSError:
            pass


def run(input_path: Optional[str] = None, *,
        demo: bool = False,
        tempo: int = 120,
        no_separate: bool = False,
        amt: str = "auto",
        style: str = "fingerstyle",
        instruction: str = "",
        gp5_path: Optional[str] = None,
        midi_path: Optional[str] = None,
        wav_path: Optional[str] = None,
        html_path: Optional[str] = None,
        emit_midi: bool = True,
        emit_wav: bool = True) -> PipelineResult:
    """Run the full pipeline and return a :class:`PipelineResult`.

    Either ``demo=True`` or ``input_path`` must be supplied. When a ``*_path``
    argument is given the corresponding artifact is also written to disk.
    """
    if demo:
        from tests.make_sample import sample_notes
        notes: List[Note] = sample_notes()
    elif not input_path:
        raise ValueError("provide input_path or demo=True")
    elif is_midi(input_path):
        notes = get_backend("midi").transcribe(input_path)
    else:
        # Audio path: L2 (separate) -> L3 (transcribe)
        notes = transcribe_audio(
            input_path,
            use_separation=not no_separate,
            amt=amt,
        )

    # L4 — music-theory analysis (feeds the L5 arrangement engine)
    analysis = analyze(notes)

    # L4.5 — quantize note timing onto the detected beat grid
    qnotes = quantize_notes(notes, analysis, subdivision=2)

    # L5/L6 — arrange + finger (role assignment, voicing, fingering)
    placed = arrange(qnotes, analysis=analysis,
                     instructions=instruction, style=style)
    placed = quantize_placed(placed, analysis, subdivision=2)  # idempotent net

    anchor = analysis.beats[0] if analysis.beats else 0.0

    # --- exports (in-memory always; to disk when a path is given) ---
    if gp5_path:
        to_gp5(placed, gp5_path, tempo=analysis.tempo, anchor=anchor)
        gp5_bytes = open(gp5_path, "rb").read()
    else:
        gp5_bytes = _write_to_temp(
            lambda p: to_gp5(placed, p, tempo=analysis.tempo, anchor=anchor))

    wav_bytes = None
    if emit_wav:
        if wav_path:
            placed_to_wav(placed, wav_path, tempo=analysis.tempo)
            wav_bytes = open(wav_path, "rb").read()
        else:
            wav_bytes = _write_to_temp(
                lambda p: placed_to_wav(placed, p, tempo=analysis.tempo))

    midi_bytes = None
    if emit_midi:
        if midi_path:
            placed_to_midi(placed, midi_path, tempo=analysis.tempo)
            midi_bytes = open(midi_path, "rb").read()
        else:
            midi_bytes = _write_to_temp(
                lambda p: placed_to_midi(placed, p, tempo=analysis.tempo))

    html_str = html_preview_string(placed)
    if html_path:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_str)

    rc = Counter(p.role or "melody" for p in placed)
    return PipelineResult(
        placed=placed,
        analysis=analysis,
        summary=format_summary(analysis),
        ascii_tab=ascii_tab(placed),
        html_preview=html_str,
        gp5_bytes=gp5_bytes,
        wav_bytes=wav_bytes,
        midi_bytes=midi_bytes,
        tempo=float(analysis.tempo),
        key=getattr(analysis, "key", "") or "",
        role_counts=dict(rc),
    )
