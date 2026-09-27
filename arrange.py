"""Guitar arrangement + fingering layer.

L5 (arrangement engine in ``arrangement.py``) decides ROLES + VOICING + the
judgment layer. This module turns that decision into concrete (string, fret,
finger) positions:

  * MELODY  -> smooth single-line dynamic-programming fingering (the original core)
  * BASS    -> root placed low on the neck (strings 5/6)
  * HARMONY -> blocked chord voicing across middle strings

The whole thing stays backward compatible: ``arrange(notes)`` with no analysis
still just fingers the melody (used by the e2e test and the old CLI path).
"""

from typing import List

from models import PlacedNote
from arrangement import build_arrangement

# Standard tuning. string 1 = high E (thinnest), string 6 = low E (thickest).
OPEN_MIDI = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
MAX_FRET = 19

# Right-hand pluck suggestion by string (informational, not required for GP).
PLUCK = {6: "p", 5: "p", 4: "i", 3: "i", 2: "m", 1: "a"}


def candidate_positions(pitch: int):
    """All (string, fret) ways to sound `pitch` within the neck."""
    out = []
    for s in range(1, 7):
        fret = pitch - OPEN_MIDI[s]
        if 0 <= fret <= MAX_FRET:
            out.append((s, fret))
    return out


def _transition_cost(a, b, gap_beats):
    """Cost of moving from position a to position b between two notes.

    gap_beats: time (in beats) between the previous note's end and this note's
    start. A large gap means a new phrase -> no movement penalty.
    """
    if gap_beats > 1.0:
        return 0.0
    s1, f1 = a
    s2, f2 = b
    cost = 3.0 * abs(s1 - s2) + 1.0 * abs(f1 - f2)
    if f2 >= 12:
        cost += 2.0 * (f2 - 11)  # discourage camping up the neck
    return cost


def _assign_fingers(placed, window=2):
    """Assign left-hand fingers using a local hand-position window."""
    n = len(placed)
    for i, p in enumerate(placed):
        if p.fret == 0:
            p.finger = 0
            continue
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        fretted = [pl.fret for pl in placed[lo:hi] if pl.fret > 0]
        base = min(fretted) if fretted else p.fret
        finger = (p.fret - base) + 1
        p.finger = max(1, min(4, finger))


def finger_melody(notes, forbidden=None) -> List[PlacedNote]:
    """Smooth single-line DP fingering for the melody (original core).

    ``forbidden`` is an optional list (parallel to the time-sorted ``notes``)
    of string sets to avoid for each note -- used so the melody never lands on
    a string the accompaniment is already holding at that instant (a GP5
    forbids two notes on the same string in one beat).
    """
    notes = sorted(notes, key=lambda n: n.onset)
    if not notes:
        return []

    cand = []
    for i, n in enumerate(notes):
        fb = forbidden[i] if forbidden else set()
        c = [(s, f) for (s, f) in candidate_positions(n.pitch) if s not in fb]
        if not c:
            c = candidate_positions(n.pitch)   # fall back if every option forbidden
            best = min(range(1, 7),
                       key=lambda s: abs((n.pitch - OPEN_MIDI[s]) - 5))
            fret = n.pitch - OPEN_MIDI[best]
            fret = max(0, min(MAX_FRET, fret))
            c = [(best, fret)]
        cand.append(c)

    N = len(cand)
    INF = float("inf")
    dp = [[INF] * len(c) for c in cand]
    back = [[None] * len(c) for c in cand]
    for j in range(len(cand[0])):
        dp[0][j] = 0.0

    for i in range(1, N):
        prev = notes[i - 1]
        gap = notes[i].onset - (prev.onset + prev.duration)
        for j, c in enumerate(cand[i]):
            for k, pc in enumerate(cand[i - 1]):
                cost = dp[i - 1][k] + _transition_cost(pc, c, gap)
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    back[i][j] = k

    last = min(range(len(cand[-1])), key=lambda j: dp[-1][j])
    path = [None] * N
    path[-1] = last
    for i in range(N - 1, 0, -1):
        path[i - 1] = back[i][path[i]]

    placed = []
    for i, n in enumerate(notes):
        s, f = cand[i][path[i]]
        placed.append(PlacedNote(
            pitch=n.pitch, onset=n.onset, duration=n.duration,
            string=s, fret=f, finger=0, pluck=PLUCK.get(s, ""), role="melody"))
    _assign_fingers(placed)
    return placed


def place_bass(notes) -> List[PlacedNote]:
    """Place bass roots low on the neck (strings 5/6, low on the fretboard)."""
    placed = []
    for n in sorted(notes, key=lambda x: x.onset):
        cands = [(s, f) for (s, f) in candidate_positions(n.pitch) if f <= 12]
        if not cands:
            cands = candidate_positions(n.pitch)
        # lowest string (thickest) for that root
        s, f = max(cands, key=lambda c: c[0])
        placed.append(PlacedNote(
            pitch=n.pitch, onset=n.onset, duration=n.duration,
            string=s, fret=f, finger=0, pluck=PLUCK.get(s, ""), role="bass"))
    _assign_fingers(placed)
    return placed


def place_harmony(notes) -> List[PlacedNote]:
    """Voice a blocked chord across middle strings (triad -> 4/3/2, 7th -> +1)."""
    by_onset: dict[float, list] = {}
    for n in notes:
        by_onset.setdefault(round(n.onset, 4), []).append(n)

    placed = []
    for o in sorted(by_onset):
        grp = sorted(by_onset[o], key=lambda x: x.pitch)
        k = len(grp)
        if k >= 4:
            strings = [5, 4, 3, 2]
        elif k == 3:
            strings = [4, 3, 2]
        else:
            strings = [4, 3, 2][:k]
        for i, n in enumerate(grp):
            s = strings[i] if i < len(strings) else strings[-1]
            fret = n.pitch - OPEN_MIDI[s]
            while fret < 0 and s > 1:
                s -= 1
                fret = n.pitch - OPEN_MIDI[s]
            while fret > MAX_FRET and s < 6:
                s += 1
                fret = n.pitch - OPEN_MIDI[s]
            fret = max(0, min(MAX_FRET, fret))
            placed.append(PlacedNote(
                pitch=n.pitch, onset=n.onset, duration=n.duration,
                string=s, fret=fret, finger=0,
                pluck=PLUCK.get(s, ""), role="harmony"))
    _assign_fingers(placed)
    return placed


def enforce_playability(placed: List[PlacedNote], tempo: float,
                        max_span: int = 5) -> List[PlacedNote]:
    """A left hand spans 4-5 frets. Anything wider is unplayable, no matter
    how the notes are spread in time -- a bass held for two beats while the
    melody jumps to fret 15 still asks for fingers on fret 2 and fret 15 at
    the same moment.

    So this is evaluated over *sounding* notes on a sixteenth-note grid, not
    just simultaneous onsets, and the fix is the subtraction a good tab
    already is: harmony is decoration, so when a position is out of reach the
    harmony notes furthest from the hand are dropped. Melody and bass are
    never dropped.
    """
    if not placed or not tempo:
        return placed
    step = 60.0 / float(tempo) / 4.0          # one sixteenth note
    if step <= 0:
        return placed

    buckets: dict = {}
    for p in placed:
        end = p.onset + max(p.duration, 1e-6)
        for k in range(int(p.onset / step), int(end / step) + 1):
            buckets.setdefault(k, []).append(p)

    drop = set()
    for grp in buckets.values():
        frets = [p.fret for p in grp]
        while len(grp) > 1 and max(frets) - min(frets) > max_span:
            victims = [p for p in grp if p.role == "harmony"]
            if not victims:
                break                          # melody/bass only: keep as is
            med = sorted(frets)[len(frets) // 2]
            victim = max(victims, key=lambda p: abs(p.fret - med))
            grp.remove(victim)
            drop.add(id(victim))
            frets = [p.fret for p in grp]

    if not drop:
        return placed
    return [p for p in placed if id(p) not in drop]


def _resolve_string_collisions(placed: List[PlacedNote]) -> List[PlacedNote]:
    """No two notes sounding at the same instant may share a string.

    One string can only be fretted in one place at a time, so a collision is
    both unplayable and illegal in GP5 (it corrupts the file). Move the
    conflicting note to the nearest free string where its pitch is playable.
    """
    by_onset: dict = {}
    for p in placed:
        by_onset.setdefault(round(p.onset, 4), []).append(p)

    for _, grp in by_onset.items():
        used: dict = {}
        for p in grp:
            if p.string not in used:
                used[p.string] = p
                continue
            free = [(s, f) for (s, f) in candidate_positions(p.pitch)
                    if s not in used]
            if not free:
                continue          # nothing playable; exporter will de-dupe
            s, f = min(free, key=lambda c: abs(c[0] - p.string))
            p.string, p.fret = s, f
            used[s] = p
    return placed


def arrange(notes,
            analysis=None,
            instructions: str = "",
            judge_backend: str = "auto",
            style: str = "fingerstyle",
            llm: bool = False) -> List[PlacedNote]:
    """Build the L5 arrangement, then finger every role.

    With ``analysis=None`` (or a bare melody), only the melody is fingered —
    identical to the old behaviour. With an ``AnalysisResult``, monophonic input
    gets derived bass + harmony and a full fingerstyle arrangement.

    ``llm=True`` routes the judgment layer through a real LLM (requires an
    OpenAI-compatible provider configured via env). Falls back to ``judge_backend``
    otherwise.
    """
    notes = list(notes)
    if not notes:
        return []

    backend = "llm" if llm else judge_backend
    arr = build_arrangement(notes, analysis, instructions, backend, style)

    # Finger the accompaniment FIRST so we know which strings it occupies,
    # then keep the melody off those strings at overlapping instants.
    accomp = place_bass(arr.bass) + place_harmony(arr.harmony)
    occ = [(p.onset, p.onset + p.duration, p.string) for p in accomp]
    melody_sorted = sorted(arr.melody, key=lambda n: n.onset)
    forbidden = [{s for (a0, a1, s) in occ if a0 <= mn.onset < a1}
                 for mn in melody_sorted]

    placed: List[PlacedNote] = []
    placed += finger_melody(melody_sorted, forbidden)
    placed += accomp

    placed.sort(key=lambda p: (round(p.onset, 4), p.pitch))
    placed = _resolve_string_collisions(placed)

    tempo = float(getattr(analysis, "tempo", 0) or 0)
    return enforce_playability(placed, tempo)
