"""Transcribe the separated stems (no torch in this process) -> multitrack MIDI.

demucs (torch) and basic_pitch (TF) cannot share a process on a small
machine — the run gets killed mid-prediction. So separation runs first and
writes stems to disk (experiments/dump_stems.py), and this script transcribes
them in a fresh process, writing one named track per stem. The role
assignment downstream keys off those names (vocals/bass/other).
"""
import sys
sys.path.insert(0, '/Users/ciel/Projects/Resonote')

import pretty_midi
from transcribe import get_amt_backend

STEMS = ["bass", "other", "vocals"]          # drums are skipped for guitar

backend = get_amt_backend("basic-pitch")
pm = pretty_midi.PrettyMIDI()
for tid, name in enumerate(STEMS):
    path = f"/tmp/stems/{name}.wav"
    print("transcribing", path, "...", flush=True)
    notes = backend.transcribe(path)
    print("  ", name, len(notes), "notes")
    instr = pretty_midi.Instrument(program=0, name=name)
    for n in notes:
        instr.notes.append(pretty_midi.Note(
            velocity=max(1, min(127, n.velocity)),
            pitch=n.pitch,
            start=float(n.onset),
            end=float(n.onset + n.duration)))
    pm.instruments.append(instr)

pm.write("/tmp/melody_stems.mid")
print("wrote /tmp/melody_stems.mid")
