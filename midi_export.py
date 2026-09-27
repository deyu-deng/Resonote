"""Audible preview export for the fingerstyle arrangement (L5.5).

Turns a list of :class:`PlacedNote` into something you can actually *hear* —
the whole point of "能上手弹" is that you can first confirm the arrangement
sounds right before you read the tab:

  * ``placed_to_midi`` — write a standard ``.mid`` via pretty_midi. Lightweight,
    lossless, re-openable in any DAW / Guitar Pro.
  * ``placed_to_wav``  — pure-numpy **Karplus-Strong** plucked-string synthesis
    mixed straight to 16-bit WAV. **Zero heavy dependencies** (only numpy +
    the stdlib ``wave`` module), so it runs even when pretty_midi / a DAW
    isn't available.

Pitch is recovered from the fretboard the same way the tab is built:
``midi = OPEN_MIDI[string] + fret`` (string 1 = high E .. string 6 = low E).
"""

from typing import List, Optional

import os
import shutil
import subprocess
import tempfile

import numpy as np

from models import PlacedNote
from arrange import OPEN_MIDI

# GM program 24 = "Acoustic Guitar (nylon)" — the natural fingerstyle voice.
NYLON_GUITAR = 24

# Make the melody sit on top, bass solid, harmony supportive (so the preview
# actually reads like a fingerstyle texture instead of a flat mush).
ROLE_VELOCITY = {
    "melody": 100,
    "bass": 92,
    "harmony": 74,
    "": 88,
}

_FREQ_CACHE = {}


def _midi_to_freq(midi: int) -> float:
    if midi not in _FREQ_CACHE:
        _FREQ_CACHE[midi] = 440.0 * (2.0 ** ((midi - 69) / 12.0))
    return _FREQ_CACHE[midi]


def placed_to_midi(placed: List[PlacedNote], out_path: str,
                   tempo: float = 120, program: int = NYLON_GUITAR) -> str:
    """Write ``placed`` to a ``.mid`` file using pretty_midi.

    Timing is absolute (seconds), so ``tempo`` is only used to stamp a tempo
    marker for DAW display — it does not affect note placement.
    """
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=program, name="Resonote")

    for p in placed:
        midi_pitch = OPEN_MIDI[p.string] + p.fret
        velocity = ROLE_VELOCITY.get(p.role, ROLE_VELOCITY[""])
        instrument.notes.append(pretty_midi.Note(
            velocity=velocity,
            pitch=midi_pitch,
            start=max(0.0, p.onset),
            end=max(0.0, p.onset) + max(0.05, p.duration),
        ))
    midi.instruments.append(instrument)
    midi.write(out_path)
    return out_path


def _ks_note(freq: float, sr: int, dur: float,
             sustain: float = 0.996) -> np.ndarray:
    """One Karplus-Strong plucked string, returned as a float waveform.

    ``sustain`` is the feedback factor (<1 adds extra decay so the string
    doesn't ring forever). The excitation is white noise; the standard
    averaging low-pass filter turns it into a natural plucked decay.
    """
    N = max(2, int(round(sr / freq)))
    buf = np.random.uniform(-1.0, 1.0, N)
    total = int(sr * dur) + N  # include the excitation tail
    out = np.zeros(total, dtype=np.float64)
    for i in range(total):
        cur = buf[i % N]
        out[i] = cur
        nxt = buf[(i + 1) % N]
        buf[i % N] = sustain * 0.5 * (cur + nxt)
    return out


def placed_to_wav(placed: List[PlacedNote], out_path: str,
                  tempo: float = 120, sr: int = 44100,
                  sustain: float = 0.996, release: float = 0.4) -> str:
    """Synthesize ``placed`` to a 16-bit mono WAV via Karplus-Strong.

    ``tempo`` is accepted for API symmetry but unused (synthesis is in
    absolute seconds). ``release`` pads each note with extra tail so decays
    don't get clipped; the master mix is peak-normalized to avoid clipping.
    """
    if not placed:
        # write a silent 1-second file so callers get a valid artifact
        silent = np.zeros(sr, dtype=np.int16)
        _write_wav(silent, sr, out_path)
        return out_path

    end_time = max(p.onset + p.duration for p in placed) + release
    master = np.zeros(int(round(end_time * sr)), dtype=np.float64)

    fade = max(1, int(0.005 * sr))  # 5 ms attack to kill the click

    for p in placed:
        midi_pitch = OPEN_MIDI[p.string] + p.fret
        freq = _midi_to_freq(midi_pitch)
        note_dur = p.duration + release
        wf = _ks_note(freq, sr, note_dur, sustain=sustain)
        # short fade-in to avoid the initial pluck click
        if len(wf) > fade:
            wf[:fade] *= np.linspace(0.0, 1.0, fade)
        amp = ROLE_VELOCITY.get(p.role, ROLE_VELOCITY[""]) / 127.0
        start = int(round(max(0.0, p.onset) * sr))
        end = start + len(wf)
        if end > len(master):
            master = np.resize(master, end)
        master[start:end] += wf * amp

    peak = float(np.max(np.abs(master))) if master.size else 0.0
    if peak > 0.0:
        master = master / peak * 0.9
    pcm = (master * 32767.0).clip(-32768, 32767).astype("<i2")
    _write_wav(pcm, sr, out_path)
    return out_path


def find_fluidsynth() -> Optional[str]:
    return shutil.which("fluidsynth")


# Searched in order. RESONOTE_SOUNDFONT wins so a user can point at their own
# bank without a code change; MuseScore's bundled MS Basic is the pragmatic
# default on a machine that has MuseScore, which is the same machine that can
# already engrave our PDFs.
_SOUNDFONT_GLOBS = [
    "~/Library/Component-Kits/com.mrbumpy409.GeneralUserGS/GeneralUser GS.sf2",
    "/Library/Audio/Sounds/Fampacks/*.sf2",
    "/Library/Audio/Sounds/*.sf2",
    "/opt/homebrew/share/fluid-synth/*.sf2",
    "/Applications/MuseScore 4.app/Contents/Resources/sound/*.sf3",
    "~/.local/share/soundfonts/*.sf{2,3}",
]


def find_soundfont() -> Optional[str]:
    import glob
    env = os.environ.get("RESONOTE_SOUNDFONT")
    if env and os.path.exists(env):
        return env
    for pattern in _SOUNDFONT_GLOBS:
        for hit in sorted(glob.glob(os.path.expanduser(pattern))):
            if os.path.getsize(hit) > 1 << 20:        # ignore stubs
                return hit
    return None


def render_preview(placed: List[PlacedNote], out_path: str,
                   tempo: float = 120, sr: int = 44100) -> str:
    """Write an audible preview and report which renderer produced it.

    FluidSynth + a real guitar SoundFont when both are installed (GM program 24,
    nylon string, which placed_to_midi already selects), otherwise the
    dependency-free Karplus-Strong synth below. Karplus-Strong is a plucked
    string model, not a guitar recording, so it is the fallback and not the
    default -- callers print what they got rather than implying a recording.
    """
    fluid, font = find_fluidsynth(), find_soundfont()
    if fluid and font and placed:
        try:
            with tempfile.TemporaryDirectory() as d:
                mid = os.path.join(d, "preview.mid")
                placed_to_midi(placed, mid, tempo=tempo)
                subprocess.run(
                    # -g 1.0 matters: FluidSynth's default master gain is 0.1,
                    # which renders a preview several times quieter than the
                    # same MIDI played in any normal player.
                    [fluid, "-ni", "-g", "1.0",
                     "-F", out_path, "-r", str(sr), font, mid],
                    capture_output=True, timeout=600, check=True)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 44:
                return f"fluidsynth ({os.path.basename(font)})"
        except Exception:
            pass            # any renderer trouble -> the built-in synth, no crash
    placed_to_wav(placed, out_path, tempo=tempo, sr=sr)
    return "karplus-strong (built-in)"


def _write_wav(pcm: np.ndarray, sr: int, out_path: str) -> None:
    import wave
    with wave.open(out_path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
