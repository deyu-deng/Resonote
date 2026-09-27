"""M7 — Real LLM judgment-layer integration tests.

Covers:
  * is_llm_configured() env detection
  * judge(backend="llm") raises a clear error when no provider is set (no faking)
  * LLMJudgmentLayer accepts an injected llm_fn and the edits get applied
  * _parse_edits handles valid / malformed JSON
  * make_llm_fn() hits an OpenAI-compatible endpoint via stdlib urllib (mocked)
"""
import json
from unittest.mock import MagicMock, patch

import pytest

import arrangement
from arrangement import (
    Edit, LLMJudgmentLayer, build_arrangement, _apply_edits,
    _parse_edits, make_llm_fn, is_llm_configured, judge,
)
from tests.make_sample import sample_notes
from analysis import analyze


def _arr(backend="rules"):
    notes = sample_notes()
    analysis = analyze(notes)
    return build_arrangement(notes, analysis, judge_backend=backend)


# --- env detection ------------------------------------------------------- #
def test_is_llm_configured_env():
    with patch.dict(arrangement.os.environ, {}, clear=True):
        assert is_llm_configured() is False
    with patch.dict(arrangement.os.environ, {"RESONOTE_LLM_API_KEY": "x"}):
        assert is_llm_configured() is True


# --- no provider -> clear error, never fake ------------------------------ #
def test_judge_llm_without_key_falls_back_to_rules():
    """No key -> the run must still produce a tab, and must SAY the LLM did
    not participate. Silently degrading was the M7 bug that kept the LLM
    invisible."""
    arr = _arr("rules")
    with patch.dict(arrangement.os.environ, {}, clear=True):
        out = judge(arr, "make it fuller", backend="llm")
    assert any("LLM" in e and "RULES" in e for e in out.judgment_log), \
        out.judgment_log


# --- injected fn path ---------------------------------------------------- #
def test_llm_layer_uses_injected_fn():
    arr = _arr("rules")
    wanted = [Edit("set_density", "all", "light", "test")]
    layer = LLMJudgmentLayer(llm_fn=lambda prompt: wanted)
    out = layer.review(arr, "anything")
    assert out == wanted


def test_apply_llm_edits_sets_density_and_style():
    arr = _arr("rules")
    edits = [
        Edit("set_density", "all", "light", "r"),
        Edit("set_style", "all", "jazz", "r"),
    ]
    _apply_edits(arr, edits)
    assert arr.density == "light"
    assert arr.style == "jazz"


# --- parse --------------------------------------------------------------- #
def test_parse_edits_valid():
    txt = json.dumps({"edits": [
        {"op": "drop", "target": "harmony", "value": 1.5, "reason": "thin"},
        {"op": "set_density", "target": "all", "value": "full"},
    ]})
    eds = _parse_edits(txt)
    assert len(eds) == 2
    assert eds[0].op == "drop" and eds[0].value == 1.5
    assert eds[1].value == "full"


def test_parse_edits_accepts_bare_list():
    txt = json.dumps([{"op": "keep", "target": "all", "reason": "ok"}])
    eds = _parse_edits(txt)
    assert eds[0].op == "keep"


def test_parse_edits_invalid_json_raises():
    with pytest.raises(RuntimeError):
        _parse_edits("this is not json")


def test_parse_edits_unknown_op_becomes_keep():
    txt = json.dumps([{"op": "teleport", "target": "all"}])
    eds = _parse_edits(txt)
    assert eds[0].op == "keep"


# --- end-to-end through make_llm_fn (mock the HTTP call) ----------------- #
def test_make_llm_fn_hits_endpoint_and_parses():
    # The adapter expects the OpenAI-style envelope; the edits live in
    # message.content as a JSON *string*.
    edits_payload = {
        "edits": [
            {"op": "set_style", "target": "all", "value": "jazz", "reason": "vibe"},
            {"op": "set_density", "target": "all", "value": "full"},
        ]
    }
    envelope = {"choices": [{"message": {"content": json.dumps(edits_payload)}}]}
    fake = MagicMock()
    fake.__enter__.return_value.read.return_value = json.dumps(envelope).encode("utf-8")

    with patch.dict(arrangement.os.environ, {
            "RESONOTE_LLM_API_KEY": "secret",
            "RESONOTE_LLM_BASE_URL": "https://example.test/v1",
            "RESONOTE_LLM_MODEL": "mock-model"}):
        with patch("arrangement.urllib.request.urlopen", return_value=fake) as m:
            fn = make_llm_fn()
            edits = fn("the prompt")

    args, _ = m.call_args
    assert args[0].full_url == "https://example.test/v1/chat/completions"
    req = args[0]
    assert req.get_header("Authorization") == "Bearer secret"
    sent = json.loads(req.data.decode("utf-8"))
    assert sent["model"] == "mock-model"
    # edits parsed correctly from the content string
    assert edits[0].op == "set_style" and edits[0].value == "jazz"
    assert edits[1].value == "full"


def test_make_llm_fn_no_key_raises_on_call():
    with patch.dict(arrangement.os.environ, {}, clear=True):
        fn = make_llm_fn()
        with pytest.raises(RuntimeError) as ei:
            fn("prompt")
    assert "RESONOTE_LLM_API_KEY" in str(ei.value)


# --- provider quirks: markdown fences + self-contradicting output --------- #
def test_parse_edits_strips_markdown_fences():
    # MiniMax (and others) wrap JSON in ```json fences and reject
    # response_format=json_object, so the content arrives fenced.
    fenced = '```json\n{"edits":[{"op":"set_style","target":"all","value":"jazz"}]}\n```'
    eds = _parse_edits(fenced)
    assert eds[0].op == "set_style" and eds[0].value == "jazz"


def test_parse_edits_tolerates_surrounding_prose():
    noisy = 'Sure! Here you go:\n```json\n{"edits":[{"op":"keep"}]}\n```\nHope that helps.'
    eds = _parse_edits(noisy)
    assert eds[0].op == "keep"


def test_sanitizer_suppresses_harmony_drops_when_full():
    arr = _arr("rules")
    edits = [
        Edit("set_density", "all", "full", "fuller"),
        Edit("drop", "harmony", 0.0, "model wanted to thin it anyway"),
    ]
    kept = arrangement._sanitize_edits(edits, arr)
    assert not any(e.op == "drop" and e.target == "harmony" for e in kept)
    assert any("sanitizer" in (e.reason or "") for e in kept)


def test_sanitizer_allows_harmony_drops_when_light():
    arr = _arr("rules")
    edits = [
        Edit("set_density", "all", "light", "simpler"),
        Edit("drop", "harmony", 0.0, "thin it"),
    ]
    kept = arrangement._sanitize_edits(edits, arr)
    assert any(e.op == "drop" and e.target == "harmony" for e in kept)


def test_response_format_only_sent_when_opted_in():
    from unittest.mock import MagicMock
    env_json = {"choices": [{"message": {"content": '{"edits":[]}'}}]}
    fake = MagicMock()
    fake.__enter__.return_value.read.return_value = json.dumps(env_json).encode("utf-8")

    def _run(extra_env):
        e = {"RESONOTE_LLM_API_KEY": "k", "RESONOTE_LLM_BASE_URL": "https://e.test/v1"}
        e.update(extra_env)
        with patch.dict(arrangement.os.environ, e):
            with patch("arrangement.urllib.request.urlopen", return_value=fake) as m:
                make_llm_fn()("p")
        return json.loads(m.call_args[0][0].data.decode("utf-8"))

    # default: no response_format (MiniMax rejects it with 400)
    assert "response_format" not in _run({})
    # opt-in: sent (OpenAI supports it)
    assert _run({"RESONOTE_LLM_JSON_MODE": "1"})["response_format"] == {"type": "json_object"}
