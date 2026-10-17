"""Transcription layer (L3).

Turns audio (or MIDI) into a list of :class:`Note` objects. Three backends:

* ``MidiFileBackend``    — reads an existing .mid/.midi via pretty_midi.
                           Always available (core dep). Used for tests & demo.
* ``YourMT3Backend``     — YourMT3+ multi-task AMT (2024 SOTA, F1≈0.85).
                           Lazy, requires GPU + model weights to be practical.
                           Tags notes with instrument/program info.
* ``BasicPitchBackend``  — Spotify Basic-Pitch (F1≈0.43). Lazy import, runs the
                           ONNX model that ships in the package (no TensorFlow).
                           Used as fallback when YourMT3+ is unavailable.

The orchestrator :func:`transcribe_audio` runs the full L2→L3 pipeline:
separate the mix into stems (L2), transcribe each stem (L3), merge with
``instrument`` / ``track_id`` labels.
"""

from __future__ import annotations

import gc
import os
import tempfile
from abc import ABC, abstractmethod

from models import Note
from separate import (
    Stem, read_audio, write_stem_wav,
    get_separator, demucs_available,
)


# --------------------------------------------------------------------------- #
#  Backend interface                                                           #
# --------------------------------------------------------------------------- #

class TranscriptionBackend(ABC):
    """Transcribe audio/MIDI → list of Note."""

    @abstractmethod
    def transcribe(self, path: str) -> list[Note]:
        """Transcribe from a file path."""
        ...

    def transcribe_stem(self, stem: Stem) -> list[Note]:
        """Transcribe a separated stem (audio array).

        Default implementation: write stem to a temp wav, delegate to
        :meth:`transcribe`. Override for backends that accept arrays
        directly (e.g. YourMT3+).
        """
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, prefix="resonote_stem_")
        tmp.close()
        try:
            write_stem_wav(stem, tmp.name)
            return self.transcribe(tmp.name)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
#  MIDI backend (always available)                                             #
# --------------------------------------------------------------------------- #

class MidiFileBackend(TranscriptionBackend):
    """Read notes straight from a MIDI file (pretty_midi)."""

    def transcribe(self, path: str) -> list[Note]:
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(path)
        notes = []
        for tid, instr in enumerate(midi.instruments):
            for n in instr.notes:
                notes.append(Note(
                    pitch=n.pitch,
                    onset=float(n.start),
                    duration=max(0.05, float(n.end - n.start)),
                    velocity=int(n.velocity),
                    instrument=instr.name or f"track_{tid}",
                    track_id=tid,
                ))
        notes.sort(key=lambda n: (n.onset, -n.pitch))
        return notes


# --------------------------------------------------------------------------- #
#  YourMT3+ backend (SOTA AMT, requires setup)                                #
# --------------------------------------------------------------------------- #

class YourMT3Backend(TranscriptionBackend):
    """YourMT3+ multi-task multitrack music transcription (2024 SOTA).

    Requires the ``yourmt3`` package + pretrained checkpoint.
    See: https://github.com/TEAMuP-dev/yourMT3

    Setup (one-time, on a GPU machine)::

        git clone https://github.com/TEAMuP-dev/yourMT3
        cd yourMT3 && pip install -e .
        # download pretrained checkpoint to a local path

    Then set ``checkpoint`` to the .ckpt path.
    """

    def __init__(self, checkpoint: str | None = None, device: str = "auto"):
        self._checkpoint = checkpoint or os.environ.get("YOURMT3_CKPT", "")
        self._device = device
        self._model = None  # lazy

    def _load(self):
        if self._model is not None:
            return
        if not self._checkpoint:
            raise RuntimeError(
                "YourMT3+ selected but no checkpoint configured.\n"
                "Set checkpoint= path or env YOURMT3_CKPT.\n"
                "Repo: https://github.com/TEAMuP-dev/yourMT3"
            )
        try:
            import torch
            from yourmt3.inference import load_model_from_checkpoint
        except ImportError as e:
            raise RuntimeError(
                "yourmt3 package not installed.\n"
                "git clone https://github.com/TEAMuP-dev/yourMT3 && "
                "pip install -e ."
            ) from e

        dev = self._device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = load_model_from_checkpoint(self._checkpoint, device=dev)
        self._dev = dev

    def transcribe_stem(self, stem: Stem) -> list[Note]:
        """Transcribe a stem's audio array directly (no temp file)."""
        self._load()
        import numpy as np

        # YourMT3+ expects float32 mono at the model's sample rate (16 kHz).
        audio = stem.audio
        target_sr = 16000
        if stem.sr != target_sr:
            try:
                import librosa
                audio = librosa.resample(
                    audio, orig_sr=stem.sr, target_sr=target_sr)
            except ImportError:
                raise RuntimeError(
                    "librosa needed to resample for YourMT3+ (pip install librosa)")

        result = self._model.inference(audio, target_sr)
        # result is expected to be a list of note dicts:
        #   {pitch, onset, duration, velocity, instrument/program}
        notes = []
        for r in result:
            notes.append(Note(
                pitch=int(r["pitch"]),
                onset=float(r["onset"]),
                duration=max(0.05, float(r.get("duration", 0.3))),
                velocity=int(r.get("velocity", 100)),
                instrument=str(r.get("instrument", "")),
            ))
        notes.sort(key=lambda n: (n.onset, -n.pitch))
        return notes

    def transcribe(self, path: str) -> list[Note]:
        audio, sr = read_audio(path)
        return self.transcribe_stem(Stem(name="mix", audio=audio, sr=sr))


# --------------------------------------------------------------------------- #
#  Basic-Pitch backend (fallback)                                             #
# --------------------------------------------------------------------------- #

def _is_octave_ghost(start: float, end: float, pitch: int, kept: list[Note]) -> bool:
    """True when an already-kept note sits an octave (or two) below and spans
    this one.

    Basic-Pitch's classic failure mode is hearing a note an octave too high. The
    ghost is nearly always the weaker of the pair, so dropping it by *pitch
    relationship* works on a real mix, while a confidence number calibrated on
    clean synth tones deletes whole registers there. Two octaves is included
    because the errors stack: on the calibration melody one ghost sat at +24 and
    sailed past a +12-only check.
    """
    span = max(float(end) - float(start), 1e-6)
    for k in kept:
        if k.pitch not in (int(pitch) - 12, int(pitch) - 24):
            continue
        overlap = min(float(end), k.onset + k.duration) - max(float(start), k.onset)
        if overlap >= 0.5 * span:
            return True
    return False


class BasicPitchBackend(TranscriptionBackend):
    """Transcribe audio with Basic-Pitch (Spotify, Apache-2.0).

    Runs the **ONNX** copy of the model, which ships inside the package
    (``saved_models/icassp_2022/nmp.onnx``, 228 KB). Same weights as the
    TensorFlow SavedModel, but inference needs no TensorFlow install -- and that
    is not cosmetic: basic_pitch pins ``tensorflow-macos<2.15.1``, which has no
    Python 3.12 macOS wheel, so letting it choose TensorFlow would make the
    audio path uninstallable on half our dev machines.

    ``min_confidence`` is a floor for the notes the model barely believes at
    all, not the octave-ghost guard: on a real master Basic-Pitch's median
    confidence is ~0.46, so a threshold tuned on clean synth tones (where true
    notes scored 0.71) quietly deletes the bass -- 0.5 kept 40% of a 60-second
    excerpt and one of its twenty-four low notes. Ghosts are removed by what
    they are instead: a duplicate exactly twelve semitones above a note the
    model liked more, over the same span.
    """

    def __init__(self, model: str | None = None, min_confidence: float = 0.3):
        self._model = model
        self._min_confidence = min_confidence

    @staticmethod
    def _onnx_model_path() -> str:
        """The ONNX asset inside the installed package, so we never fall into
        basic_pitch's own TensorFlow-preferred default."""
        import basic_pitch

        root = os.path.dirname(os.path.abspath(basic_pitch.__file__))
        onnx = os.path.join(root, "saved_models", "icassp_2022", "nmp.onnx")
        if os.path.exists(onnx):
            return onnx
        return str(basic_pitch.ICASSP_2022_MODEL_PATH)

    def transcribe(self, path: str) -> list[Note]:
        try:
            from basic_pitch.inference import predict
        except ImportError as e:
            raise RuntimeError(
                "Basic-Pitch is not installed. Install the ONNX path:\n"
                "  pip install --no-deps basic-pitch\n"
                "  pip install onnxruntime librosa scipy numba soundfile "
                "resampy==0.4.2 mir_eval\n"
                "--no-deps because basic_pitch's TensorFlow pin has no macOS "
                "wheel for Python 3.12, and we do not use TensorFlow."
            ) from e

        # predict() returns (model_output, PrettyMIDI, note_events); the events
        # carry confidence, the PrettyMIDI does not. Its defaults already include
        # the fix for short notes being dropped -- min_note_len=127.7 ms, which is
        # the 11 frames this layer once lost a song over. Don't lower it.
        _model_output, _midi, note_events = predict(path, self._model or self._onnx_model_path())

        # Strongest first: a survivor must never be explained away by a weaker
        # note that happens to come later in the list.
        kept: list[Note] = []
        for start, end, pitch, confidence, _bends in sorted(
                note_events, key=lambda e: -e[3]):
            if confidence < self._min_confidence:
                continue
            if _is_octave_ghost(start, end, pitch, kept):
                continue
            kept.append(Note(
                pitch=int(pitch),
                onset=float(start),
                duration=max(0.05, float(end) - float(start)),
                velocity=int(round(127 * confidence)),   # basic_pitch's own mapping
            ))
        kept.sort(key=lambda n: (n.onset, -n.pitch))
        return kept


class MuScriptorBackend(TranscriptionBackend):
    """Audio -> notes with MuScriptor (Kyutai + Mirelo, 2026).

    The reason to switch is not only accuracy: MuScriptor reports **what it
    heard** (voice / acoustic guitar / piano / drums ...) per note, where
    Basic-Pitch emits an undifferentiated "piano" stream. That label is what
    L5 has been guessing at -- ``assign_roles`` currently duplicates notes
    between melody and bass because nothing tells it which line is the tune.
    ``Note.instrument`` carries the label through, so the arrangement can take
    the vocal line as the melody instead of deducing it from pitch ranges.

    Weights are gated on HuggingFace (free account + accept the license) and
    use Xet storage, so do NOT route them through hf-mirror.com -- see
    requirements.txt. ``small`` mislabels this project's test song's guitar as
    piano and finds half as many vocal notes as ``medium`` does, so medium is
    the default even though it costs ~1.1x realtime on CPU.
    """

    _MODELS: dict = {}          # size -> loaded model (weights are ~1.2 GB)

    def __init__(self, size: str = "medium"):
        if size not in ("small", "medium", "large"):
            raise ValueError(f"muscriptor size must be small/medium/large, got {size!r}")
        self._size = size

    def _model(self):
        if self._size not in MuScriptorBackend._MODELS:
            try:
                from muscriptor import TranscriptionModel
            except ImportError as e:
                raise RuntimeError(
                    "MuScriptor is not installed. Run: pip install muscriptor\n"
                    "Its weights are gated: accept the license at\n"
                    "  https://huggingface.co/MuScriptor/muscriptor-medium\n"
                    "then set HF_TOKEN (see requirements.txt)."
                ) from e
            MuScriptorBackend._MODELS[self._size] = TranscriptionModel.load_model(
                self._size)
        return MuScriptorBackend._MODELS[self._size]

    def transcribe(self, path: str) -> list[Note]:
        import io
        import pretty_midi
        midi_bytes = self._model().transcribe_to_midi(path)
        pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
        notes: list[Note] = []
        for tid, instr in enumerate(pm.instruments):
            if instr.is_drum or not instr.notes:
                continue            # a drum hit has no pitch to put on a string
            # Prefer the name the model wrote on the track ("voice") over the
            # GM table's name for the same program number, which is "Choir
            # Aahs" for program 52 -- anything matching the melody by the word
            # "voice" silently finds nothing if the GM name is used instead.
            label = (instr.name or "").strip() or \
                pretty_midi.program_to_instrument_name(instr.program)
            for n in instr.notes:
                notes.append(Note(
                    pitch=n.pitch,
                    onset=float(n.start),
                    duration=max(0.05, float(n.end - n.start)),
                    velocity=int(n.velocity),
                    instrument=label.lower(),
                    track_id=tid,
                ))
        notes.sort(key=lambda x: (x.onset, -x.pitch))
        return notes


# --------------------------------------------------------------------------- #
#  Availability checks & factory                                              #
# --------------------------------------------------------------------------- #

def muscriptor_available() -> bool:
    try:
        import muscriptor  # noqa: F401
        return True
    except ImportError:
        return False


def yourmt3_available() -> bool:
    try:
        import yourmt3  # noqa: F401
        return True
    except ImportError:
        return False


def basic_pitch_available() -> bool:
    try:
        import basic_pitch  # noqa: F401
        return True
    except ImportError:
        return False


def get_amt_backend(kind: str = "auto") -> TranscriptionBackend:
    """Pick an AMT backend for audio transcription.

    kind:
      "muscriptor"  — MuScriptor (labels instruments; needs gated HF weights)
      "yourmt3"     — force YourMT3+ (raises at transcribe time if unconfigured)
      "basic-pitch" — force Basic-Pitch
      "auto"        — YourMT3+ if installed, else Basic-Pitch

    "auto" deliberately does not pick MuScriptor: its weights are gated, so an
    unattended run must not start by failing on a license page. Ask for it by
    name.
    """
    if kind == "yourmt3":
        return YourMT3Backend()
    if kind == "muscriptor":
        return MuScriptorBackend()
    if kind == "basic-pitch":
        return BasicPitchBackend()
    if kind == "auto":
        if yourmt3_available():
            return YourMT3Backend()
        return BasicPitchBackend()
    raise ValueError(f"unknown AMT backend: {kind}")


# Backward-compatible alias (main.py previously called get_backend).
def get_backend(kind: str = "auto", audio_path: str = "") -> TranscriptionBackend:
    """Pick a backend by kind / filename extension.

    .mid/.midi → MidiFileBackend; audio → AMT backend (auto = YourMT3+ → Basic-Pitch).
    """
    if kind == "midi" or (audio_path and audio_path.lower().endswith((".mid", ".midi"))):
        return MidiFileBackend()
    if kind in ("yourmt3", "basic-pitch", "muscriptor"):
        return get_amt_backend(kind)
    return get_amt_backend("auto")


# --------------------------------------------------------------------------- #
#  L2→L3 orchestrator                                                          #
# --------------------------------------------------------------------------- #

def transcribe_audio(
    path: str,
    use_separation: bool = True,
    amt: str = "auto",
    dump_dir: str | None = None,
    stem_filter: tuple[str, ...] | None = ("vocals", "bass", "other"),
) -> list[Note]:
    """Full L2→L3 pipeline: separate → transcribe each stem → merge.

    Parameters
    ----------
    path : str
        Audio file path (mp3/wav/flac/...).
    use_separation : bool
        If False, skip separation (passthrough — whole mix as one stem).
    amt : str
        AMT backend: "auto" / "yourmt3" / "basic-pitch".
    dump_dir : str | None
        If set, write separated stems (wav) + per-stem MIDI to this directory
        for debugging (``--dump-intermediate``).
    stem_filter : tuple[str, ...] | None
        Which demucs stems to transcribe. None = all. Default skips "drums"
        (drum transcriptions are noise for guitar arrangement). When
        passthrough (no separation), the single "mix" stem is always used.

    Returns
    -------
    list[Note]
        All notes merged & sorted by onset, with ``instrument`` / ``track_id``
        set per stem.
    """
    # MuScriptor separates by *labelling* what it hears (voice / guitar / piano
    # / drums), which is the job demucs was doing here -- and it needs the full
    # mix to do it. Feeding it one stem at a time would discard the labels that
    # make it useful and pay the model once per stem.
    if amt == "muscriptor":
        return MuScriptorBackend().transcribe(path)

    # L2: separate
    sep_kind = "auto" if use_separation else "passthrough"
    separator = get_separator(sep_kind)
    stems = separator.separate(path)
    del separator
    gc.collect()   # free the demucs model before the AMT backend loads:
                   # demucs (torch) and the transcription runtime never need to
                   # be resident together, and holding both is what gets long
                   # songs killed mid-prediction on small machines

    # filter stems (skip drums for guitar arrangement)
    if stem_filter and len(stems) > 1:
        stems = [s for s in stems if s.name in stem_filter] or stems

    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)

    # L3: transcribe each stem
    backend = get_amt_backend(amt)
    all_notes: list[Note] = []
    for tid, stem in enumerate(stems):
        notes = backend.transcribe_stem(stem)
        for n in notes:
            if not n.instrument:
                n.instrument = stem.name
            n.track_id = tid
        all_notes.extend(notes)

        if dump_dir:
            stem_wav = os.path.join(dump_dir, f"stem_{tid}_{stem.name}.wav")
            try:
                write_stem_wav(stem, stem_wav)
                print(f"  [dump] stem {tid} ({stem.name}) → {stem_wav}")
            except Exception as e:
                print(f"  [dump] could not write stem {stem.name}: {e}")

    if dump_dir:
        notes_path = os.path.join(dump_dir, "transcribed_notes.tsv")
        with open(notes_path, "w") as f:
            f.write("onset\tduration\tpitch\tvelocity\tinstrument\ttrack_id\n")
            for n in all_notes:
                f.write(f"{n.onset:.3f}\t{n.duration:.3f}\t{n.pitch}\t"
                        f"{n.velocity}\t{n.instrument}\t{n.track_id}\n")
        print(f"  [dump] notes → {notes_path}")

    all_notes.sort(key=lambda n: (n.onset, -n.pitch))
    return all_notes
