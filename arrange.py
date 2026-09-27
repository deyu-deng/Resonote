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
    pitches = []                       # folded into the guitar's range
    for i, n in enumerate(notes):
        fb = forbidden[i] if forbidden else set()
        pitch = _in_guitar_range(n.pitch)
        pitches.append(pitch)
        c = [(s, f) for (s, f) in candidate_positions(pitch) if s not in fb]
        if not c:
            c = candidate_positions(pitch)   # fall back if every option forbidden
            best = min(range(1, 7),
                       key=lambda s: abs((pitch - OPEN_MIDI[s]) - 5))
            fret = pitch - OPEN_MIDI[best]
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
            pitch=pitches[i], onset=n.onset, duration=n.duration,
            string=s, fret=f, finger=0, pluck=PLUCK.get(s, ""), role="melody"))
    _assign_fingers(placed)
    return placed


def _in_guitar_range(pitch: int) -> int:
    """Fold a MIDI pitch into what a standard-tuned guitar can sound.

    A separated bass stem goes below the low E (MIDI 40) and vocals go above
    fret 19 on the high E; a guitarist transposes by octaves rather than
    skipping the note. Every pitch in [40, 83] is reachable, and the string
    ranges overlap enough that `candidate_positions` is never empty here.
    """
    lo, hi = OPEN_MIDI[6], OPEN_MIDI[1] + MAX_FRET
    while pitch < lo:
        pitch += 12
    while pitch > hi:
        pitch -= 12
    return pitch


def place_bass(notes) -> List[PlacedNote]:
    """Place bass roots low on the neck (strings 5/6, low on the fretboard)."""
    placed = []
    for n in sorted(notes, key=lambda x: x.onset):
        pitch = _in_guitar_range(n.pitch)
        cands = [(s, f) for (s, f) in candidate_positions(pitch) if f <= 12]
        if not cands:
            cands = candidate_positions(pitch)
        # lowest string (thickest) for that root
        s, f = max(cands, key=lambda c: c[0])
        placed.append(PlacedNote(
            pitch=pitch, onset=n.onset, duration=n.duration,
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
            pitch = _in_guitar_range(n.pitch)
            fret = pitch - OPEN_MIDI[s]
            while fret < 0 and s > 1:
                s -= 1
                fret = pitch - OPEN_MIDI[s]
            while fret > MAX_FRET and s < 6:
                s += 1
                fret = pitch - OPEN_MIDI[s]
            fret = max(0, min(MAX_FRET, fret))
            placed.append(PlacedNote(
                pitch=pitch, onset=n.onset, duration=n.duration,
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

        def _fretted(g):
            # open strings cost no finger, so they do not constrain where the
            # hand sits: an open bass under a fret-12 melody is a normal
            # fingerstyle figure, not a stretch
            return [p.fret for p in g if p.fret > 0]

        fretted = _fretted(grp)
        # bounded: every move strictly narrows the span and every drop removes
        # a note, so this terminates -- the cap is a guard against a future
        # edit wedging the whole pipeline in an endless loop
        for _guard in range(256):
            if len(fretted) <= 1 or max(fretted) - min(fretted) <= max_span:
                break
            lo, hi = min(fretted), max(fretted)

            # 1) re-voice: the same pitch is usually reachable in several
            #    places, so try moving a bass/harmony note to a position that
            #    narrows the span. Keeping the embellishment beats dropping
            #    it, and a root on string 5 fret 3 sounds identical to one on
            #    string 6 fret 8.
            best = None                     # (new_span, note, string, fret)
            for v in [p for p in grp if p.role in ("bass", "harmony")]:
                rest = [q.fret for q in grp if q is not v and q.fret > 0]
                if not rest:
                    continue
                rlo, rhi = min(rest), max(rest)
                tlo, thi = max(0, rhi - max_span), rlo + max_span
                used = {q.string for q in grp if q is not v}
                for s, f in candidate_positions(v.pitch):
                    if s in used or not (tlo <= f <= thi):
                        continue
                    ns = max(rhi, f) - min(rlo, f)
                    # ties prefer the lower position: easier to reach, and
                    # what a player would actually choose
                    if best is None or (ns, f) < (best[0], best[3]):
                        best = (ns, v, s, f)
            if best is not None and best[0] < hi - lo:
                _ns, v, s, f = best
                v.string, v.fret = s, f
                fretted = _fretted(grp)      # the move changed the hand span
                continue

            # 2) otherwise subtract: drop whichever harmony note leaves the
            #    narrowest span behind, high positions first on ties. Melody
            #    and bass are never dropped.
            harm = [p for p in grp if p.role == "harmony" and p.fret > 0]
            if not harm:
                break                          # melody/bass only: keep as is

            def _span_without(v):
                rest = [q.fret for q in grp if q is not v and q.fret > 0]
                return (max(rest) - min(rest)) if len(rest) > 1 else 0

            victim = min(harm, key=lambda p: (_span_without(p), -p.fret))
            if _span_without(victim) >= hi - lo:
                break          # the stretch is melody-vs-bass: dropping the
                               # embellishment cannot fix it, so keep it
            grp.remove(victim)
            drop.add(id(victim))
            fretted = _fretted(grp)

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

    # span first (it may re-voice notes across strings), then de-dupe strings
    tempo = float(getattr(analysis, "tempo", 0) or 0)
    placed = enforce_playability(placed, tempo)
    return _resolve_string_collisions(placed)
