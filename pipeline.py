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
        tempo: Optional[float] = None,
        no_separate: bool = False,
        amt: str = "auto",
        style: str = "fingerstyle",
        instruction: str = "",
        llm: bool = False,
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

    # the detected tempo wins unless the caller explicitly overrides it
    eff_tempo = float(tempo) if tempo else float(analysis.tempo)

    # L4.5 — quantize note timing onto the detected beat grid
    qnotes = quantize_notes(notes, analysis, subdivision=2)

    # L5/L6 — arrange + finger (role assignment, voicing, fingering)
    placed = arrange(qnotes, analysis=analysis,
                     instructions=instruction, style=style, llm=llm)
    placed = quantize_placed(placed, analysis, subdivision=2)  # idempotent net

    anchor = analysis.beats[0] if analysis.beats else 0.0

    # --- exports (in-memory always; to disk when a path is given) ---
    if gp5_path:
        to_gp5(placed, gp5_path, tempo=eff_tempo, anchor=anchor)
        gp5_bytes = open(gp5_path, "rb").read()
    else:
        gp5_bytes = _write_to_temp(
            lambda p: to_gp5(placed, p, tempo=eff_tempo, anchor=anchor))

    wav_bytes = None
    if emit_wav:
        if wav_path:
            placed_to_wav(placed, wav_path, tempo=eff_tempo)
            wav_bytes = open(wav_path, "rb").read()
        else:
            wav_bytes = _write_to_temp(
                lambda p: placed_to_wav(placed, p, tempo=eff_tempo))

    midi_bytes = None
    if emit_midi:
        if midi_path:
            placed_to_midi(placed, midi_path, tempo=eff_tempo)
            midi_bytes = open(midi_path, "rb").read()
        else:
            midi_bytes = _write_to_temp(
                lambda p: placed_to_midi(placed, p, tempo=eff_tempo))

    html_str = html_preview_string(
        placed,
        tempo=eff_tempo,
        subtitle=(getattr(analysis, "key", "") or ""),
        audio_src=(os.path.basename(wav_path) if wav_path else None))
    if html_path:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_str)

    rc = Counter(p.role or "melody" for p in placed)
    return PipelineResult(
        placed=placed,
        analysis=analysis,
        summary=format_summary(analysis),
        ascii_tab=ascii_tab(placed, tempo=eff_tempo),
        html_preview=html_str,
        gp5_bytes=gp5_bytes,
        wav_bytes=wav_bytes,
        midi_bytes=midi_bytes,
        tempo=eff_tempo,
        key=getattr(analysis, "key", "") or "",
        role_counts=dict(rc),
    )


def run_import(tab_path: str, *,
               gp5_path: Optional[str] = None,
               midi_path: Optional[str] = None,
               wav_path: Optional[str] = None,
               html_path: Optional[str] = None,
               pdf_path: Optional[str] = None,
               musicxml_path: Optional[str] = None,
               emit_midi: bool = True,
               emit_wav: bool = True,
               tempo: Optional[float] = None,
               track_index: int = 0) -> PipelineResult:
    """Import an existing tab file and re-export it -- no re-arrangement.

    A tab somebody already made is ground truth: we keep the ``string`` /
    ``fret`` they chose and the tuning of the source, and skip L1-L7.
    This is the path for "I bought / downloaded a tab, give me my own PDF".

    ``tempo`` overrides the tempo found in the file; a text tab has none, so
    it defaults to 120 BPM.
    """
    from importers import load_tab
    from tab_pdf import write_tab_pdf

    score = load_tab(tab_path, tempo=tempo, track_index=track_index)
    placed = score.placed
    if not placed:
        raise ValueError(f"{tab_path}: no notes could be imported")

    title = score.title or score.track_name or "Resonote"
    gp5_bytes = b""
    if gp5_path:
        to_gp5(placed, gp5_path, tempo=score.tempo, title=title,
               tuning=score.tuning)
        gp5_bytes = open(gp5_path, "rb").read()

    if pdf_path:
        write_tab_pdf(placed, pdf_path, tempo=score.tempo, title=title,
                      tuning=score.tuning, subtitle=score.tuning_note())

    if musicxml_path:
        from musicxml_export import write_musicxml
        write_musicxml(placed, musicxml_path, tempo=score.tempo,
                       title=title, tuning=score.tuning)

    wav_bytes = None
    if emit_wav:
        if wav_path:
            placed_to_wav(placed, wav_path, tempo=score.tempo)
            wav_bytes = open(wav_path, "rb").read()
        else:
            wav_bytes = _write_to_temp(
                lambda p: placed_to_wav(placed, p, tempo=score.tempo))

    midi_bytes = None
    if emit_midi:
        if midi_path:
            placed_to_midi(placed, midi_path, tempo=score.tempo)
            midi_bytes = open(midi_path, "rb").read()
        else:
            midi_bytes = _write_to_temp(
                lambda p: placed_to_midi(placed, p, tempo=score.tempo))

    html_str = html_preview_string(
        placed, tempo=score.tempo,
        subtitle=(score.title or "") + (f" · {score.track_name}"
                                        if score.track_name else ""),
        audio_src=(os.path.basename(wav_path) if wav_path else None),
        tuning=score.tuning)
    if html_path:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_str)

    tuning_note = score.tuning_note()

    summary_lines = [
        f"imported  : {os.path.basename(tab_path)}",
        f"track     : {score.track_name or '-'}",
        f"tempo     : {score.tempo:.1f} BPM",
        f"measures  : {score.measures}",
        f"notes     : {len(placed)}",
    ]
    if tuning_note:
        summary_lines.append(tuning_note)
    for w in score.warnings:
        summary_lines.append(f"! {w}")
    summary_lines.append("mode      : import (fingering preserved, no re-arrangement)")

    return PipelineResult(
        placed=placed,
        analysis=None,
        summary="\n".join(summary_lines),
        ascii_tab=ascii_tab(placed, tempo=score.tempo, tuning=score.tuning),
        html_preview=html_str,
        gp5_bytes=gp5_bytes,
        wav_bytes=wav_bytes,
        midi_bytes=midi_bytes,
        tempo=float(score.tempo),
        key="",
        role_counts={"melody": len(placed)},
    )
