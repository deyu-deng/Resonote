# Resonote — Build Log

> CLI that turns a melody (audio `.mp3/.wav` or MIDI) into a **fingerstyle
> guitar `.gp5` tab + audible preview**. 9-layer pipeline:
> L1 preprocess → L2 separate → L3 AMT → L4 theory → L5 arrangement⭐ →
> L6 fingering → L7 quantization → L8 export → L9 UX.

---

## M5 — Audible preview export (L5.5) ✅ DONE

**Why:** "能上手弹" first requires you to *hear* whether the arrangement is
right. The tab alone can't tell you that. M5 closes the loop with audible
previews.

**Deliverables**
- `midi_export.py` (NEW)
  - `placed_to_midi(placed, out, tempo, program=24)` — standard `.mid` via
    pretty_midi (GM nylon-guitar voice). Lossless, re-openable in any DAW.
  - `placed_to_wav(placed, out, tempo, sr=44100, sustain=0.996, release=0.4)`
    — **pure-numpy Karplus-Strong** plucked-string synth mixed to 16-bit WAV.
    **Zero heavy deps** (only numpy + stdlib `wave`) → runs even without a DAW.
  - Pitch recovered from the fretboard exactly like the tab:
    `midi = OPEN_MIDI[string] + fret` (string 1=high E … 6=low E).
  - Per-role velocities so the preview reads like fingerstyle: melody 100 /
    bass 92 / harmony 74.
  - 5 ms attack fade-in on every note to kill the pluck click; master mix is
    peak-normalized to avoid clipping; empty input → valid silent WAV.
- `main.py` (MODIFIED) — new `--midi PATH` / `--wav PATH` flags; both export
  after the `.gp5`/`--html` step, using the detected tempo.
- `tests/test_m5.py` (NEW, 6 tests, all pass)

**Verification**
- `tests/test_m5.py`: 6/6 pass (pitch mapping, MIDI round-trip pitch multiset,
  MIDI from real pipeline ≥ placed count, WAV validity + non-silent, empty-list,
  WAV from real pipeline audible).
- Full suite: **48/48 pass** (M1–M5 + e2e).
- E2E demo (`--demo --midi out.mid --wav out.wav`): `out.mid` = 31 notes @
  120 BPM; `out.wav` = valid 7.91 s / 44.1 kHz / mono / 16-bit (697 KB).

**Honest caveats**
- Karplus-Strong is a *synthetic* guitar, not a sampled one — good enough to
  judge arrangement/voice-leading/rhythm, not a final performance recording.
  A real sampled VST via MIDI is the upgrade path.
- `sustain`/`release` are global; per-string decay modeling is a future nicety.

---

## M6 — Web frontend: drag-in → real tab + audio ✅ DONE

**Why:** Completes the original product vision — "拖入音乐文件就可以编曲转换成可以直接上手弹的独奏吉他指弹谱". M1–M5 built the engine; M6 puts a **premium web face** on it so a user can drop a file and immediately get tab + a playable preview, no CLI.

**Architecture (zero new deps)**
- `pipeline.py` (NEW) — single orchestration entry extracted from `main.py`.
  `run(input=None, demo=False, style=..., instruction=..., wav=True, midi=True,
  html=True)` returns an in-memory `Result` (gp5/mid/wav **bytes**, html **str**,
  ascii tab, tempo, note counts). CLI and web both call it — one source of truth.
  `main.py` was rewritten to consume it; CLI behavior unchanged.
- `web_server.py` (NEW) — stdlib `http.server` backend.
  - `GET /` serves `web/` static; `GET /api/demo` runs the demo arrangement;
    `POST /api/arrange` accepts raw bytes + `X-Filename` header (avoids the
    removed `cgi` module) and returns JSON (ascii tab, key/tempo summary, role
    counts, and URLs for the gp5/mid/wav/html artifacts written to
    `web/results/<id>/`). `arrange_request()` is factored out for unit testing.
- `web/index.html` + `style.css` + `app.js` (NEW) — premium front-end:
  glass-morphism drop zone with gradient + magnetic hover, **light/dark/system
  theme toggle persisted to localStorage**, result cards (key/tempo/voice
  stats), an `<audio>` player for the generated WAV, the ASCII tab, and download
  buttons for GP5/MIDI/HTML. Pure static, no framework, 60fps transitions, plus
  a "load sample" button that hits `/api/demo`.
- `preview.py` (MODIFIED) — added `html_preview_string(placed, title)`;
  `write_html_preview` now reuses it so web can grab the HTML text directly.
- `tests/test_m6.py` (NEW, 6 tests, all pass)

**Verification**
- `tests/test_m6.py`: 6/6 (pipeline produces non-empty gp5/wav + audible WAV;
  server JSON has all keys + on-disk artifacts; demo endpoint; no-crash on bad
  bytes with `ok:false`; CLI still emits files via pipeline).
- Full suite: **54/54 pass** (M1–M6 + e2e).
- Live server e2e: `GET /` → 200 (brand present); `/api/demo` → valid tab JSON;
  `POST /api/arrange` with a real MIDI → valid arrangement JSON; fetched the
  generated `out.wav` → valid 7.91 s / 44.1 kHz / mono / 16-bit (697 KB).

**How to run**
```
python web_server.py --port 8137
# open http://127.0.0.1:8137/  → drop a .mid/.mp3/.wav, hear it, download tab
```

**Honest caveats**
- Audio (mp3/wav) through the web UI still needs `basic_pitch` (+ optional
  `demucs`) installed server-side; the MIDI path is dependency-free.
- Web UI shares the same synthetic Karplus-Strong preview as M5 — good for
  arrangement judgment, not a final recording.
- No auth / size limit on uploads yet — fine for local use, add a guard before
  any public deploy.

---

## Earlier milestones (summary)
- **M1** — transcription + analysis scaffold.
- **M2** — L5 arrangement engine (roles/voicing/judgment, `Edit` protocol so
  RuleJudgmentLayer ↔ LLMJudgmentLayer are interchangeable; LLM raises a clear
  error if no provider is configured — never fakes).
- **M3** — multi-stem separation path + analysis enrichment.
- **M4** — L4.5 quantization (grid snap) + **critical gp_export fix**: old
  exporter collapsed each grid slot to ONE note (31-note arrangement → only 4
  notes exported; chords + sustained bass dropped). New exporter tiles every
  measure to 8 eighth slots, uses `NoteType.tie` for sustained notes, and
  writes the **detected** tempo. Plus string-collision avoidance so melody
  never shares a string with accompaniment at the same instant. 9/9 tests pass.

## Known limitations / deferred
- **GP8** skipped — pyguitarpro only supports GP5.
- **LLM judgment layer** — interface ready, no provider wired yet (natural-
  language editing in `--instruction` currently routes through the rule layer).
- **M6** — `web/` frontend connecting the real pipeline for drag-in → real tab.
- Real audio (mp3/wav) needs `basic_pitch` (or `yourmt3`) + optionally `demucs`
  installed; MIDI path is dependency-free.
