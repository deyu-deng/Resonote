"""Beat and downbeat detection from the waveform, for L4.

Our own detector (:func:`analysis.detect_beats`) reads the beat grid off the
inter-onset intervals of the notes we already transcribed. That is structurally
circular -- it inherits every timing error of the transcription, and it has no
concept of a downbeat, so bar 1 gets anchored to whichever note happened to
come first.

Measured on a real song (陶喆 Melody, fixtures/audio/taozhe-melody.mp3): the
pipeline reported 94 BPM while the audio's own onset-strength autocorrelation
peaks at 0.540 s (+0.42) and 1.080 s (+0.49), i.e. ~111 BPM, and is flat at
0.645 s (-0.05), which is where 94 BPM would have to show a peak. A wrong tempo
is not a cosmetic problem: it mis-frames every note value in the printed score.

beat_this (CPJKU, MIT, ISMIR 2024) is a beat/downbeat tracker with an explicit
CPU path, so when it is installed the grid comes from the audio and the notes
are quantized onto something independent of themselves.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

_MODEL = None            # the checkpoint is 77 MB; load it once per process


def available() -> bool:
    try:
        import beat_this.inference          # noqa: F401
        return True
    except Exception:
        return False


def detect(audio: np.ndarray, sr: int) -> Optional[Dict]:
    """Return {"beats", "downbeats", "tempo"} in seconds, or None if beat_this
    is not installed or the model refuses the audio."""
    global _MODEL
    if not available():
        return None
    try:
        from beat_this.inference import Audio2Beats
        if _MODEL is None:
            _MODEL = Audio2Beats(checkpoint_path="final0", device="cpu")
        # Postprocessor.__call__ returns (beat, downbeat) -- but the library's
        # own File2File unpacks them as (downbeats, beats), so copy that
        # example and the two streams are swapped. Unpacked from the source.
        beats, downbeats = _MODEL(np.asarray(audio, dtype=np.float32), sr)
        beats = np.asarray(beats, dtype=float).ravel()
        downbeats = np.asarray(downbeats, dtype=float).ravel()
        if beats.size < 2:
            return None
        period = float(np.median(np.diff(np.sort(beats))))
        if not (0.2 <= period <= 2.0):        # nonsense tempo, ignore it
            return None
        return {"beats": [float(b) for b in sorted(beats)],
                "downbeats": [float(d) for d in sorted(downbeats)],
                "tempo": 60.0 / period}
    except Exception as e:                    # a missing/broken model is not
        print(f"[beats] beat_this failed ({e}); falling back to note-based grid")
        return None
