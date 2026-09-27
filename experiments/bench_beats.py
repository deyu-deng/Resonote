"""Benchmark beat tracking on audio whose beat grid is known by construction.

runs/2026-09-27/out.melody.wav is our own render of the Melody arrangement:
93 BPM, onsets snapped to an eighth grid. That makes it a file with a ground
truth, so a tracker can be scored rather than eyeballed -- the same trick the
P0 eval set will need, at zero cost.

Only a smoke benchmark: synthesized Karplus-Strong audio has no drums, which
makes it easier than a real mix. A difference here is a signal, not a verdict.
"""
import sys
import time

import numpy as np
import soundfile as sf

WAV = sys.argv[1] if len(sys.argv) > 1 else "runs/2026-09-27/out.melody.wav"
TRUE_BPM = 93.0

audio, sr = sf.read(WAV)
if audio.ndim > 1:
    audio = audio.mean(axis=1)
dur = len(audio) / sr
print(f"{WAV}: {dur:.1f}s @ {sr} Hz")

# the grid the arrangement was quantized to: a beat every quarter, from t=0
grid = np.arange(0, dur, 60.0 / TRUE_BPM)


def score(name, beats):
    beats = np.asarray(beats, dtype=float)
    if beats.size == 0:
        print(f"{name:22s} no beats")
        return
    # median distance from each detected beat to the true grid, in ms
    err = np.min(np.abs(beats[:, None] - grid[None, :]), axis=1) * 1000
    # and the tempo implied by the mean inter-beat interval
    ibi = np.diff(np.sort(beats))
    bpm = 60.0 / np.median(ibi) if ibi.size else float("nan")
    print(f"{name:22s} {beats.size:4d} beats | off-grid median "
          f"{np.median(err):6.0f} ms | implied tempo {bpm:5.1f} BPM "
          f"(true {TRUE_BPM:.0f})")


t0 = time.time()
import librosa
t, local = librosa.beat.beat_track(y=audio, sr=sr, units="time")
score("librosa beat_track", t)
print(f"{'':22s} ({time.time() - t0:.1f}s)")

t0 = time.time()
from beat_this.inference import File2Beats
tracker = File2Beats(checkpoint_path="final0", device="cpu")
downbeats, beats = tracker(WAV)
score("beat_this final0", beats)
print(f"{'':22s} ({time.time() - t0:.1f}s, incl. model load)")
if len(downbeats) > 1:
    first = float(downbeats[0])
    print(f"{'':22s} first downbeat at {first:.2f}s "
          f"({len(downbeats)} downbeats -> "
          f"{len(downbeats) * 4 / dur * 60:.1f} BPM if 4/4)")
