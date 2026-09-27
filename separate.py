"""Source separation layer (L2).

Separates a full mix into instrument stems so downstream AMT (L3) can
transcribe each stem independently and tag notes with ``instrument`` /
``track_id``.

Backends
--------
* DemucsSeparator — wraps Meta's HTDemucs v4 (lazy torch/demucs import).
  GPU preferred; works on CPU but slow. Produces 4 stems:
  drums / bass / other / vocals.
* PassthroughSeparator — no separation; reads the file and returns a single
  ``mix`` stem. Used when demucs is unavailable or ``--no-separate`` is set.

Both return a list of :class:`Stem` objects.
"""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

# Stem names produced by HTDemucs v4 (order matters — matches model.sources).
DEMUCS_STEMS = ("drums", "bass", "other", "vocals")


@dataclass
class Stem:
    """A separated audio stem (mono, float32)."""
    name: str            # "vocals" / "bass" / "drums" / "other" / "mix"
    audio: "object"      # numpy.ndarray, shape (n_samples,), float32
    sr: int              # sample rate


# --------------------------------------------------------------------------- #
#  Audio I/O helpers (lazy — only needed on the audio path)                   #
# --------------------------------------------------------------------------- #

def read_audio(path: str):
    """Read any audio file → (mono float32 ndarray, sample_rate).

    Tries soundfile → scipy → librosa in order (all lazy).
    """
    import numpy as np

    # 1. soundfile (fastest, most formats)
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
        return np.ascontiguousarray(mono), sr
    except ImportError:
        pass
    except Exception:
        pass  # fall through to next backend

    # 2. scipy (wav only)
    try:
        from scipy.io import wavfile
        import numpy as np
        sr, data = wavfile.read(path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0
        mono = data.mean(axis=1) if data.ndim > 1 else data
        return np.ascontiguousarray(mono), int(sr)
    except ImportError:
        pass
    except Exception:
        pass

    # 3. librosa (handles everything via ffmpeg/audioread)
    try:
        import librosa
        mono, sr = librosa.load(path, sr=None, mono=True)
        return mono, int(sr)
    except ImportError:
        pass
    except Exception:
        pass

    raise RuntimeError(
        "No audio I/O backend available. Install one of:\n"
        "  pip install soundfile   (recommended)\n"
        "  pip install scipy       (wav only)\n"
        "  pip install librosa     (handles mp3/m4a/flac via ffmpeg)"
    )


def write_stem_wav(stem: Stem, path: str) -> str:
    """Write a Stem to a 16-bit PCM wav file (for Basic-Pitch / debugging)."""
    import numpy as np
    try:
        import soundfile as sf
        sf.write(path, stem.audio, stem.sr, subtype="PCM_16")
        return path
    except ImportError:
        pass
    try:
        from scipy.io import wavfile
        wavfile.write(path, stem.sr,
                      (stem.audio * 32767).astype(np.int16))
        return path
    except ImportError:
        pass
    raise RuntimeError("Need soundfile or scipy to write wav.")


# --------------------------------------------------------------------------- #
#  Separator backends                                                          #
# --------------------------------------------------------------------------- #

class Separator(ABC):
    """Interface: take an audio file path, return separated stems."""

    @abstractmethod
    def separate(self, path: str) -> list[Stem]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class PassthroughSeparator(Separator):
    """No separation — return the whole mix as a single stem.

    Used as fallback when demucs is unavailable, when ``--no-separate`` is
    passed, or for MIDI/demo inputs (which never hit the separation layer).
    """

    @property
    def name(self) -> str:
        return "passthrough"

    def separate(self, path: str) -> list[Stem]:
        audio, sr = read_audio(path)
        return [Stem(name="mix", audio=audio, sr=sr)]


def _load_audio(path: str):
    """(channels, samples) float32 tensor + sample rate.

    torchaudio 2.9+ moved audio decoding into the separate ``torchcodec``
    package, so ``torchaudio.load`` raises ImportError on a plain install.
    soundfile (libsndfile) is already a dependency of this project and reads
    everything demucs needs, so it is tried first; ``torchaudio.load`` stays
    as the fallback for anything libsndfile cannot decode.
    """
    try:
        import numpy as np
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        import torch
        return torch.from_numpy(np.ascontiguousarray(data.T)), int(sr)
    except Exception:
        import torchaudio
        return torchaudio.load(path)


class DemucsSeparator(Separator):
    """Source separation via Meta's HTDemucs v4.

    Requires ``torch`` + ``demucs`` (pip install demucs). GPU strongly
    recommended; runs on CPU but ~10× slower.
    """

    def __init__(self, model_name: str = "htdemucs", device: str = "auto",
                 shifts: int = 1, overlap: float = 0.25):
        self._model_name = model_name
        self._device = device
        self._shifts = shifts
        self._overlap = overlap
        self._model = None  # lazy

    @property
    def name(self) -> str:
        return f"demucs({self._model_name})"

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from demucs.pretrained import get_model
        except ImportError as e:
            raise RuntimeError(
                "demucs is not installed. Run: pip install demucs\n"
                "Then retry. (Pulls in PyTorch + model weights.)"
            ) from e

        model = get_model(self._model_name)
        dev = self._device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(dev)
        model.eval()
        self._model = model
        self._dev = dev

    def separate(self, path: str) -> list[Stem]:
        import numpy as np

        self._load()
        model = self._model
        dev = self._dev

        try:
            import torch
            import torchaudio
        except ImportError as e:
            raise RuntimeError("PyTorch/torchaudio required for demucs.") from e

        wav, sr = _load_audio(path)  # (channels, samples)
        if sr != model.samplerate:
            wav = torchaudio.transforms.Resample(sr, model.samplerate)(wav)
            sr = model.samplerate

        wav = wav.unsqueeze(0).to(dev)  # (1, ch, samples)
        with torch.no_grad():
            from demucs.apply import apply_model
            ref = wav.mean(1, keepdim=True)
            wav_in = (wav - ref) / max(ref.abs().max(), 1e-8)
            sources = apply_model(
                model, wav_in, split=True,
                overlap=self._overlap, shifts=self._shifts,
                progress=True,
            )[0]  # (n_sources, ch, samples)
            sources = sources * ref / max(ref.abs().max(), 1e-8)

        stem_names = list(getattr(model, "sources", DEMUCS_STEMS))
        stems = []
        for i, sname in enumerate(stem_names):
            mono = sources[i].mean(0).cpu().numpy().astype(np.float32)
            stems.append(Stem(name=sname, audio=mono, sr=sr))
        return stems


# --------------------------------------------------------------------------- #
#  Factory                                                                     #
# --------------------------------------------------------------------------- #

def demucs_available() -> bool:
    """True if demucs + torch + torchaudio can be imported.

    torchaudio belongs in this check too: demucs imports it lazily inside
    ``separate()``, so without it the CLI would announce "separation=demucs"
    and then die on a ModuleNotFoundError instead of falling back.
    """
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        import demucs  # noqa: F401
        return True
    except ImportError:
        return False


def get_separator(kind: str = "auto") -> Separator:
    """Pick a separator.

    kind:
      "auto"      — demucs if available, else passthrough
      "demucs"    — force demucs (raises if not installed)
      "passthrough" / "none" — no separation
    """
    if kind in ("passthrough", "none"):
        return PassthroughSeparator()
    if kind == "demucs":
        if not demucs_available():
            raise RuntimeError(
                "demucs requested but not installed. pip install demucs")
        return DemucsSeparator()
    # auto
    if demucs_available():
        return DemucsSeparator()
    return PassthroughSeparator()
