"""L5 — Fingerstyle arrangement engine.

This is the product's differentiator. A fingerstyle guitar solo is NOT "a
melody mapped onto the neck" — it is *bass + harmony + melody* woven onto one
instrument. This module consumes the L4 ``AnalysisResult`` and the transcribed
``Note`` list and decides:

  1. ROLE       — which notes are melody / bass / harmony
  2. VOICING    — how chords are laid out on the neck (inversions, voice leading)
  3. JUDGMENT   — the "判断层": a rules skeleton produces a candidate arrangement,
                  then an LLM (or a deterministic rules fallback) reviews it for
                  playability, section density and style. LLM is a first-class
                  interface from M3 onward, not a bolt-on.

Design (per the approved PLAN.md):
  * Rules skeleton always runs and is fully testable with no external deps.
  * The LLM judgment layer is a thin, lazy adapter around an injected callable.
    With no provider configured it raises a clear error instead of pretending.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Sequence, Tuple

from models import Note
from analysis import AnalysisResult, Chord


# --------------------------------------------------------------------------- #
# Roles & arrangement data model
# --------------------------------------------------------------------------- #
class Role(Enum):
    MELODY = "melody"
    BASS = "bass"
    HARMONY = "harmony"


@dataclass
class RoleNote:
    """A note tagged with its arrangement role."""
    pitch: int
    onset: float
    duration: float
    velocity: int = 100
    role: Role = Role.MELODY
    source: str = ""      # original stem/instrument, or "derived:<chord>"
    label: str = ""       # chord label if derived from a chord
    chord_root: int = -1  # pitch class of the parent chord (for voicing)


@dataclass
class Arrangement:
    melody: List[RoleNote] = field(default_factory=list)
    bass: List[RoleNote] = field(default_factory=list)
    harmony: List[RoleNote] = field(default_factory=list)
    analysis: Optional[AnalysisResult] = None
    key: str = ""
    tempo: float = 120.0
    style: str = "fingerstyle"
    density: str = "full"            # "full" | "light"
    judgment_log: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _pc(midi: int) -> int:
    return midi % 12


def _chord_triad(root_pc: int, quality: str = "") -> List[int]:
    """Return the triad pitch classes for a root, honouring minor/extended hints."""
    if quality and quality[0] == "m" and not quality.startswith("maj"):
        intervals = [0, 3, 7]
    else:
        intervals = [0, 4, 7]
    # if the chord label carries a 7th, include it (for jazz density)
    if quality and ("7" in quality or "6" in quality):
        intervals = intervals + ([10] if quality[0] == "m" else [11])
    return [(root_pc + i) % 12 for i in intervals]


def _nearest_low_pitch(pc: int, lo: int = 40, hi: int = 55) -> int:
    """Lowest playable pitch with this pitch class in [lo, hi] (bass register)."""
    for octv in range(lo // 12, hi // 12 + 1):
        p = octv * 12 + pc
        if lo <= p <= hi:
            return p
    # fallback: clamp into range
    return min(max(lo, pc + 12 * (lo // 12)), hi)


def _nearest_mid_pitch(pc: int, lo: int = 55, hi: int = 72) -> int:
    """Nearest playable pitch with this pitch class in a mid register (harmony)."""
    for octv in range(lo // 12, hi // 12 + 1):
        p = octv * 12 + pc
        if lo <= p <= hi:
            return p
    return min(max(lo, pc + 12 * (lo // 12)), hi)


# --------------------------------------------------------------------------- #
# 1. ROLE assignment
# --------------------------------------------------------------------------- #
def assign_roles(notes: Sequence[Note], analysis: Optional[AnalysisResult]) -> Arrangement:
    """Split input notes into melody/bass/harmony, deriving bass+harmony from
    the L4 chord timeline when the input is essentially a single melody line
    (the common case: drag in a song -> we hear the lead, build the rest)."""
    notes = list(notes)
    melody: List[RoleNote] = []
    bass_in: List[RoleNote] = []
    harmony_in: List[RoleNote] = []

    stems = {n.instrument for n in notes}
    has_bass_stem = any(s == "bass" for s in stems)
    has_other_stem = any(s in ("other", "accompaniment", "piano") for s in stems)

    for n in notes:
        rn = RoleNote(pitch=n.pitch, onset=n.onset, duration=n.duration,
                      velocity=n.velocity, source=n.instrument)
        inst = n.instrument
        if inst == "bass":
            rn.role = Role.BASS
            bass_in.append(rn)
        elif inst in ("vocals", "lead", "melody"):
            rn.role = Role.MELODY
            melody.append(rn)
        elif inst in ("other", "accompaniment", "piano", "harmony"):
            rn.role = Role.HARMONY
            harmony_in.append(rn)
        else:
            # unlabeled -> treat as melody (monophonic input)
            rn.role = Role.MELODY
            melody.append(rn)

    # If the input is a bare melody line, DERIVE bass + harmony from chords.
    derived_bass: List[RoleNote] = []
    derived_harmony: List[RoleNote] = []
    if analysis is not None and analysis.chords and not (has_bass_stem or has_other_stem):
        chords = analysis.chords
        for c in chords:
            if c.label == "N" or c.root < 0:
                continue
            seg_len = max(c.end - c.start, 0.25)
            # BASS: root in low register, one hit per chord segment
            bpc = _nearest_low_pitch(c.root)
            derived_bass.append(RoleNote(
                pitch=bpc, onset=c.start, duration=seg_len,
                velocity=100, role=Role.BASS,
                source=f"derived:{c.label}", label=c.label, chord_root=c.root))
            # HARMONY: triad (or 7th) voiced per chord segment
            for pc in _chord_triad(c.root, c.quality):
                hpc = _nearest_mid_pitch(pc)
                derived_harmony.append(RoleNote(
                    pitch=hpc, onset=c.start, duration=seg_len,
                    velocity=85, role=Role.HARMONY,
                    source=f"derived:{c.label}", label=c.label, chord_root=c.root))

    arrangement = Arrangement(
        melody=melody,
        bass=bass_in + derived_bass,
        harmony=harmony_in + derived_harmony,
        analysis=analysis,
        key=analysis.key if analysis else "",
        tempo=analysis.tempo if analysis else 120.0,
    )
    if derived_bass or derived_harmony:
        arrangement.judgment_log.append(
            f"derived {len(derived_bass)} bass + {len(derived_harmony)} harmony notes "
            f"from {len(analysis.chords)} chords (melody-only input)")
    return arrangement


# --------------------------------------------------------------------------- #
# 2. VOICING (voice leading for harmony)
# --------------------------------------------------------------------------- #
def voice(arr: Arrangement) -> Arrangement:
    """Smooth the harmony voice leading: for consecutive chord stabs pick the
    inversion whose voicing is closest (in pitch) to the previous one."""
    if not arr.harmony:
        return arr

    # group harmony notes by onset (chord events)
    events: List[Tuple[float, List[RoleNote]]] = []
    by_onset: dict[float, List[RoleNote]] = {}
    for h in sorted(arr.harmony, key=lambda x: x.onset):
        key = round(h.onset, 4)
        by_onset.setdefault(key, []).append(h)
    events = [(t, by_onset[t]) for t in sorted(by_onset)]

    prev_voicing: Optional[List[int]] = None
    new_harmony: List[RoleNote] = []
    for _, grp in events:
        # collect the pitch classes present
        pcs = sorted({(n.pitch % 12) for n in grp})
        if not pcs:
            continue
        # try inversions: rotate the pitch-class set, re-voice into mid register
        best: Optional[List[int]] = None
        best_cost = float("inf")
        for shift in range(len(pcs)):
            rotated = pcs[shift:] + pcs[:shift]
            voicing = [_nearest_mid_pitch(pc) for pc in rotated]
            cost = 0.0 if prev_voicing is None else sum(
                abs(a - b) for a, b in zip(voicing, prev_voicing[:len(voicing)]))
            if cost < best_cost:
                best_cost = cost
                best = voicing
        if best is not None:
            prev_voicing = best
            for n, p in zip(sorted(grp, key=lambda x: x.pitch), best):
                n.pitch = p
                new_harmony.append(n)
    arr.harmony = new_harmony
    arr.judgment_log.append("applied harmony voice leading (min movement between chords)")
    return arr


# --------------------------------------------------------------------------- #
# 3. JUDGMENT layer (rules skeleton + LLM adapter)
# --------------------------------------------------------------------------- #
@dataclass
class Edit:
    """A judgment edit. Both the rule layer and the LLM layer emit these so they
    are interchangeable."""
    op: str            # "drop" | "keep" | "set_density" | "set_style"
    target: str = ""   # role name, "all", or "section:<label>"
    value: object = None
    reason: str = ""


def _rule_judge(arr: Arrangement, instructions: str = "") -> List[Edit]:
    """Deterministic rules skeleton. Produces edits for playability/density/style."""
    edits: List[Edit] = []
    instr = (instructions or "").lower()

    # --- density from instructions ---
    density = "full"
    if any(w in instr for w in ("easy", "simpler", "simple", "light", "初学者", "简单")):
        density = "light"
    if any(w in instr for w in ("full", "dense", "fuller", "饱满", "完整")):
        density = "full"
    edits.append(Edit("set_density", "all", density,
                      f"density from instructions -> {density}"))

    # --- style from instructions ---
    style = "fingerstyle"
    if "jazz" in instr:
        style = "jazz"
    elif "folk" in instr or "民谣" in instr:
        style = "folk"
    elif "classical" in instr or "古典" in instr:
        style = "classical"
    edits.append(Edit("set_style", "all", style, f"style -> {style}"))

    # --- playability: in light density, thin the harmony to every other chord ---
    if density == "light" and arr.harmony:
        seen_onsets = set()
        drop_count = 0
        for h in arr.harmony:
            key = round(h.onset, 4)
            if key in seen_onsets:
                continue
            seen_onsets.add(key)
        # drop harmony events whose onset index is odd
        onsets = sorted({round(h.onset, 4) for h in arr.harmony})
        for idx, o in enumerate(onsets):
            if idx % 2 == 1:
                edits.append(Edit("drop", "harmony", o,
                                  "light density: thin harmony to every other chord"))
                drop_count += 1
        if drop_count:
            edits.append(Edit("keep", "all", None,
                              f"rules: dropped {drop_count} harmony chords (light)"))

    # --- drop very low-velocity harmony notes (noise) ---
    for h in arr.harmony:
        if h.velocity < 50:
            edits.append(Edit("drop", "harmony", round(h.onset, 4),
                              "rules: dropped low-velocity harmony note"))
    return edits


def _apply_edits(arr: Arrangement, edits: List[Edit]) -> Arrangement:
    for e in edits:
        if e.op == "set_density":
            arr.density = e.value
        elif e.op == "set_style":
            arr.style = e.value
        elif e.op == "drop" and e.target == "harmony":
            o = e.value
            before = len(arr.harmony)
            arr.harmony = [h for h in arr.harmony if round(h.onset, 4) != o]
            if len(arr.harmony) < before:
                arr.judgment_log.append(f"judgment: dropped harmony @ {o}s ({e.reason})")
    return arr


class LLMJudgmentLayer:
    """Lazy adapter: an LLM acts as the review/judgment layer.

    The LLM is given the candidate arrangement + analysis + the user's natural
    language instructions, and returns a list of :class:`Edit` ops. This is the
    exact contract the rules layer satisfies, so the two are drop-in replaceable.

    No provider is configured in this environment, so ``review`` raises a clear
    error rather than silently faking an LLM decision.
    """

    def __init__(self, llm_fn: Optional[Callable[[str], List[Edit]]] = None):
        self._llm_fn = llm_fn

    def review(self, arr: Arrangement, instructions: str = "") -> List[Edit]:
        if self._llm_fn is None:
            raise RuntimeError(
                "LLM judgment layer requested but no LLM client is configured.\n"
                "Inject an `llm_fn(prompt: str) -> List[Edit]` via LLMJudgmentLayer(llm_fn=...),\n"
                "or use backend='rules' / 'auto'. The rules layer is the deterministic fallback.")
        prompt = self._build_prompt(arr, instructions)
        return self._llm_fn(prompt)

    @staticmethod
    def _build_prompt(arr: Arrangement, instructions: str) -> str:
        lines = [
            "You are the musical judgment layer of a fingerstyle guitar arranger.",
            f"Key: {arr.key or 'unknown'}  Tempo: {arr.tempo} BPM  Style: {arr.style}",
            f"Melody notes: {len(arr.melody)}  Bass: {len(arr.bass)}  Harmony: {len(arr.harmony)}",
            f"User instruction: {instructions!r}",
            "Return a JSON list of edits: {\"op\":\"drop\"|\"set_density\"|\"set_style\", "
            "\"target\":\"harmony\"|\"all\", \"value\":<number|string>, \"reason\":\"...\"}",
        ]
        return "\n".join(lines)


def judge(arr: Arrangement, instructions: str = "", backend: str = "auto") -> Arrangement:
    """Run the judgment layer. backend: 'auto' | 'rules' | 'llm'."""
    if backend == "auto":
        backend = "rules"   # rules is the always-available fallback
    if backend == "rules":
        edits = _rule_judge(arr, instructions)
        arr.judgment_log.append("judged by RULES layer")
        return _apply_edits(arr, edits)
    if backend == "llm":
        layer = LLMJudgmentLayer()
        edits = layer.review(arr, instructions)
        arr.judgment_log.append("judged by LLM layer")
        return _apply_edits(arr, edits)
    raise ValueError(f"unknown judgment backend: {backend!r}")


# --------------------------------------------------------------------------- #
# Public orchestration
# --------------------------------------------------------------------------- #
def build_arrangement(notes: Sequence[Note],
                      analysis: Optional[AnalysisResult] = None,
                      instructions: str = "",
                      judge_backend: str = "auto",
                      style: str = "fingerstyle") -> Arrangement:
    """assign_roles -> voice -> judge. Produces the final Arrangement."""
    arr = assign_roles(notes, analysis)
    arr.style = style
    arr = voice(arr)
    arr = judge(arr, instructions, judge_backend)
    return arr


def all_notes(arr: Arrangement) -> List[RoleNote]:
    """Flatten the arrangement into a single note list (for fingering)."""
    return arr.melody + arr.bass + arr.harmony
