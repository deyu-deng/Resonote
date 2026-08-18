"""M6 — web layer: pipeline orchestration + zero-dep server.

Covers the two pieces that make the drag-in -> real tab loop work:
  * :mod:`pipeline` (single orchestration entry used by CLI + web)
  * :mod:`web_server`.arrange_request (file bytes -> artifacts -> URLs)
"""

import os
import shutil
import tempfile
import wave

from collections import Counter

import numpy as np

import pipeline
import web_server
from tests.make_sample import sample_notes


def _wav_is_audible(wav_bytes: bytes) -> bool:
    with wave.open(__import__("io").BytesIO(wav_bytes), "rb") as w:
        n = w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype="<i2")
    return bool(pcm.max() > 0 or pcm.min() < 0)


def _cleanup(payload: dict):
    """Remove the result dir created by arrange_request (keep repo tidy)."""
    rid = payload["gp5_url"].split("/")[-2]
    d = os.path.join(web_server.RESULTS_DIR, rid)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)


def _sample_midi_bytes() -> bytes:
    import pretty_midi
    mid = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0, name="sample")
    for n in sample_notes():
        inst.notes.append(pretty_midi.Note(
            velocity=n.velocity, pitch=n.pitch,
            start=n.onset, end=n.onset + n.duration))
    mid.instruments.append(inst)
    fd, p = tempfile.mkstemp(suffix=".mid")
    os.close(fd)
    mid.write(p)
    with open(p, "rb") as f:
        data = f.read()
    os.unlink(p)
    return data


class TestPipelineEntry:
    def test_demo_returns_complete_result(self):
        res = pipeline.run(demo=True)
        assert res.placed
        assert res.gp5_bytes
        assert res.wav_bytes and _wav_is_audible(res.wav_bytes)
        assert res.midi_bytes
        assert "<pre>" in res.html_preview
        assert res.ascii_tab.strip()
        assert res.tempo == 120.0

    def test_role_counts_present(self):
        res = pipeline.run(demo=True)
        rc = Counter(p.role for p in res.placed)
        # sample is a single melody line -> derived bass + harmony exist
        assert rc.get("melody", 0) > 0
        assert rc.get("bass", 0) > 0

    def test_midi_file_roundtrip(self):
        data = _sample_midi_bytes()
        fd, p = tempfile.mkstemp(suffix=".mid")
        os.close(fd)
        with open(p, "wb") as f:
            f.write(data)
        try:
            res = pipeline.run(p)
            assert res.placed
            assert res.gp5_bytes
        finally:
            os.unlink(p)


class TestWebServerLogic:
    def test_demo_request_writes_artifacts(self):
        payload = web_server.arrange_request(b"", "__demo__",
                                             {"style": "fingerstyle", "instruction": ""})
        try:
            assert payload["ok"] is True
            assert payload["gp5_url"].endswith("/out.gp5")
            rid = payload["gp5_url"].split("/")[-2]
            d = os.path.join(web_server.RESULTS_DIR, rid)
            for name in ("out.gp5", "out.wav", "out.mid", "preview.html"):
                assert os.path.exists(os.path.join(d, name)), name
            assert payload["note_count"] > 0
        finally:
            _cleanup(payload)

    def test_midi_upload_request(self):
        data = _sample_midi_bytes()
        payload = web_server.arrange_request(
            data, "song.mid", {"style": "fingerstyle", "instruction": ""})
        try:
            assert payload["ok"] is True
            assert "E|" in payload["ascii_tab"]
            assert payload["wav_url"].endswith("/out.wav")
        finally:
            _cleanup(payload)

    def test_bad_input_returns_error_not_crash(self):
        # a `.bin` with garbage should fail gracefully (ok=False + message)
        payload = web_server.arrange_request(b"not a real file", "x.bin",
                                             {"style": "fingerstyle", "instruction": ""})
        assert payload["ok"] is False
        assert isinstance(payload["error"], str) and payload["error"]
