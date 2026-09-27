"""M16 -- handing print work to a real engraver.

The built-in PDF writer draws lines and digits; MuseScore does spacing,
beaming and page breaks. These tests pin the integration, and in particular
the fact that MuseScore's exit code cannot be trusted: it tears down through a
crash reporter that prints an error even when every file was written, so
success is defined by the artifact appearing.

They must pass on a machine with no notation software installed, so nothing
here shells out for real.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engraving
from engraving import engrave, find_engraver
import midi_export
from midi_export import render_preview
from models import PlacedNote


def test_find_engraver_returns_nothing_or_a_real_path():
    exe = find_engraver()
    assert exe is None or os.path.exists(exe), exe


def test_engrave_is_falsey_without_an_engraver(monkeypatch):
    monkeypatch.setattr(engraving, "find_engraver", lambda: None)
    assert engrave("/tmp/in.gp5", "/tmp/out.pdf") is False


def test_success_is_judged_by_the_artifact_not_the_exit_code(monkeypatch, tmp_path):
    out = str(tmp_path / "score.pdf")

    def fake_run(cmd, **kw):
        # MuseScore 4.7.5 writes the file and then dies noisily on teardown
        open(out, "wb").write(b"%PDF-1.4 fake")
        return type("R", (), {"returncode": 1, "stdout": "", "stderr":
                              "libc++abi: mutex lock failed"})()

    monkeypatch.setattr(engraving, "find_engraver", lambda: "/usr/bin/mscore")
    monkeypatch.setattr(engraving.subprocess, "run", fake_run)
    assert engrave(str(tmp_path / "in.gp5"), out) is True


def test_a_missing_artifact_is_a_failure_even_if_the_exit_code_is_zero(
        monkeypatch, tmp_path):
    monkeypatch.setattr(engraving, "find_engraver", lambda: "/usr/bin/mscore")
    monkeypatch.setattr(
        engraving.subprocess, "run",
        lambda *a, **kw: type("R", (), {"returncode": 0})())
    assert engrave("in.gp5", str(tmp_path / "never-written.pdf")) is False


def test_empty_output_does_not_count(monkeypatch, tmp_path):
    out = str(tmp_path / "empty.pdf")
    monkeypatch.setattr(engraving, "find_engraver", lambda: "/usr/bin/mscore")
    monkeypatch.setattr(engraving.subprocess, "run",
                        lambda *a, **kw: (open(out, "wb").close(),
                                          type("R", (), {"returncode": 0})())[1])
    assert engrave("in.gp5", out) is False


def test_no_soundfont_flag_is_not_passed(monkeypatch):
    """`-s` is documented in the MuseScore handbook but this build rejects it
    and exits without producing anything, so passing it is a silent failure."""
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return type("R", (), {"returncode": 1})()

    monkeypatch.setattr(engraving, "find_engraver", lambda: "/usr/bin/mscore")
    monkeypatch.setattr(engraving.subprocess, "run", fake_run)
    engrave("in.gp5", "out.pdf")
    assert "-s" not in seen["cmd"]
    assert "-F" in seen["cmd"], "factory settings keep a first run from prompting"


def test_a_timeout_is_not_a_crash(monkeypatch):
    import subprocess as sp
    monkeypatch.setattr(engraving, "find_engraver", lambda: "/usr/bin/mscore")

    def raise_timeout(*a, **kw):
        raise sp.TimeoutExpired(cmd="mscore", timeout=1)

    monkeypatch.setattr(engraving.subprocess, "run", raise_timeout)
    assert engrave("in.gp5", "/tmp/nope.pdf") is False


def test_preview_prefers_fluidsynth_when_a_soundfont_exists(monkeypatch, tmp_path):
    """Karplus-Strong is a plucked-string model, not a guitar. When a real
    SoundFont renderer is present we must use it and say which one we used."""
    out = str(tmp_path / "p.wav")
    monkeypatch.setattr(midi_export, "find_fluidsynth", lambda: "/usr/bin/fluidsynth")
    monkeypatch.setattr(midi_export, "find_soundfont", lambda: "/usr/share/x.sf3")
    called = {}

    def fake_run(cmd, **kw):
        called["cmd"] = cmd
        # > the 44-byte header render_preview requires of a real WAV
        open(out, "wb").write(b"RIFF....WAVEfmt " + b"\0" * 64)
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(midi_export.subprocess, "run", fake_run)
    backend = render_preview(
        [PlacedNote(pitch=64, onset=0.0, duration=0.5, string=1, fret=0,
                    finger=0)], out, tempo=111.1)
    assert backend.startswith("fluidsynth"), backend
    assert "-g" in called["cmd"], "FluidSynth defaults to master gain 0.1"
    assert called["cmd"][called["cmd"].index("-g") + 1] == "1.0"


def test_preview_falls_back_to_the_built_in_synth_without_a_soundfont(
        monkeypatch, tmp_path):
    monkeypatch.setattr(midi_export, "find_fluidsynth", lambda: None)
    monkeypatch.setattr(midi_export, "find_soundfont", lambda: None)
    out = str(tmp_path / "p.wav")
    backend = render_preview(
        [PlacedNote(pitch=64, onset=0.0, duration=0.5, string=1, fret=0,
                    finger=0)], out)
    assert backend == "karplus-strong (built-in)"
    assert os.path.getsize(out) > 1000


def test_a_broken_renderer_never_costs_the_preview(monkeypatch, tmp_path):
    """fluidsynth can fail for a hundred reasons (bad bank, no audio device);
    the user should still get an audible file."""
    out = str(tmp_path / "p.wav")
    monkeypatch.setattr(midi_export, "find_fluidsynth", lambda: "/usr/bin/fluidsynth")
    monkeypatch.setattr(midi_export, "find_soundfont", lambda: "/bad/x.sf2")

    def boom(*a, **kw):
        raise OSError("no such device")

    monkeypatch.setattr(midi_export.subprocess, "run", boom)
    backend = render_preview(
        [PlacedNote(pitch=64, onset=0.0, duration=0.5, string=1, fret=0,
                    finger=0)], out)
    assert backend == "karplus-strong (built-in)"
    assert os.path.exists(out)
