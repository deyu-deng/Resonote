"""Resonote CLI.

Usage
-----
  # quick smoke test with a built-in melody (no audio model needed):
  python main.py --demo -o out.gp5 --html out.html --midi out.mid --wav out.wav

  # from a MIDI melody file:
  python main.py song.mid -o song.gp5

  # from an audio file (requires basic_pitch or yourmt3 + demucs):
  python main.py song.mp3 -o song.gp5

  # skip source separation (faster, less accurate on full mixes):
  python main.py song.mp3 --no-separate -o song.gp5

  # force a specific AMT backend:
  python main.py song.mp3 --amt basic-pitch -o song.gp5

  # also export an audible preview you can actually listen to:
  python main.py song.mp3 -o song.gp5 --midi song.mid --wav song.wav

  # dump intermediate stems + notes for debugging:
  python main.py song.mp3 --dump-intermediate _debug -o song.gp5

  # re-export a tab you already have (bought / downloaded) as your own PDF+GP5,
  # keeping the fingering the author chose:
  python main.py song.gp5 -o song.mine.gp5 --pdf song.pdf
  python main.py song.txt -o song.gp5 --pdf song.pdf --tempo 96

The whole pipeline (transcribe -> analyze -> quantize -> arrange -> export)
is orchestrated by :mod:`pipeline`. This file only does argument parsing,
friendly console output, and file I/O.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import run, run_import, is_midi
from importers import is_tab_file
from transcribe import yourmt3_available, basic_pitch_available
from separate import demucs_available
from arrangement import is_llm_configured

import json


def _jsonable(o):
    """The analysis carries numpy scalars (an int64 chord root, a float64
    confidence) and json refuses them outright, which used to kill the run
    after every artifact had been computed. The dump exists to be read, so
    coerce instead of crashing."""
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _dump_analysis(analysis, dump_dir: str) -> None:
    import os
    os.makedirs(dump_dir, exist_ok=True)
    payload = {
        "backend": analysis.backend,
        "key": analysis.key,
        "tempo": analysis.tempo,
        "grid_backend": getattr(analysis, "grid_backend", "note-onsets"),
        "downbeats": getattr(analysis, "downbeats", []),
        "beats": analysis.beats,
        "chords": [
            {"label": c.label, "root": c.root, "quality": c.quality,
             "start": c.start, "end": c.end, "confidence": c.confidence}
            for c in analysis.chords
        ],
        "sections": [
            {"label": s.label, "start": s.start, "end": s.end, "bars": s.bars}
            for s in analysis.sections
        ],
    }
    out = os.path.join(dump_dir, "analysis.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_jsonable)
    print(f"[dump] wrote {out}")


def _dump_arrangement(notes, analysis, instructions, style, dump_dir,
                      backend="auto", edits_file=""):
    from arrangement import build_arrangement
    # same judgment backend AND the same reviewer file as the real run, or the
    # dump describes a run that never happened
    arr = build_arrangement(notes, analysis, instructions, backend, style,
                            edits_file=edits_file)
    payload = {
        "key": arr.key,
        "tempo": arr.tempo,
        "style": arr.style,
        "density": arr.density,
        "counts": {
            "melody": len(arr.melody),
            "bass": len(arr.bass),
            "harmony": len(arr.harmony),
        },
        "judgment_log": arr.judgment_log,
    }
    out = os.path.join(dump_dir, "arrangement.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_jsonable)
    print(f"[dump] wrote {out}")


def main():
    from arrangement import load_env_file
    load_env_file()          # .env -> os.environ, so the LLM layer can engage

    ap = argparse.ArgumentParser(
        description="Resonote: turn a melody into a fingerstyle guitar .gp5 tab")
    ap.add_argument("input", nargs="?", help="audio (.mp3/.wav) or MIDI (.mid/.midi)")
    ap.add_argument("-o", "--out", default="out.gp5", help="output .gp5 path")
    ap.add_argument("-t", "--tempo", type=int, default=None,
                    help="BPM. Audio path: the tempo is detected, this is "
                         "ignored. Import path: overrides the file's tempo "
                         "(a text tab has none, so it defaults to 120)")
    ap.add_argument("--html", default=None, help="also write a standalone HTML preview")
    ap.add_argument("--pdf", default=None,
                    help="also write a printable tablature PDF (vector, no deps)")
    ap.add_argument("--midi", default=None, help="also write an audible preview .mid (pretty_midi)")
    ap.add_argument("--musicxml", default=None,
                    help="also write MusicXML (standard interchange: opens in "
                         "MuseScore / Finale / Guitar Pro 7 for publication-grade "
                         "rendering and editing)")
    ap.add_argument("--wav", default=None, help="also write an audible preview .wav (Karplus-Strong synth)")
    ap.add_argument("--demo", action="store_true",
                    help="ignore input, use the built-in sample melody")
    ap.add_argument("--no-separate", action="store_true",
                    help="skip source separation (L2); faster but less accurate on full mixes")
    ap.add_argument("--amt", choices=["auto", "yourmt3", "basic-pitch"], default="auto",
                    help="audio transcription backend (L3)")
    ap.add_argument("--dump-intermediate", default=None, metavar="DIR",
                    help="dump separated stems + transcribed notes to DIR for debugging")
    ap.add_argument("--style", choices=["fingerstyle", "folk", "jazz", "classical"],
                    default="fingerstyle",
                    help="arrangement style (L5 voicing/judgment)")
    ap.add_argument("--instruction", default="",
                    help="natural-language arrangement hint for the L5 judgment layer "
                         "(e.g. \"simpler\", \"make it fuller\", \"jazz voicing\")")
    ap.add_argument("--llm", action="store_true",
                    help="route the L5 judgment layer through a real LLM instead of the "
                         "rules skeleton. Requires RESONOTE_LLM_API_KEY (OpenAI-compatible "
                         "endpoint). When --instruction is given and an LLM is configured, "
                         "the LLM is auto-used even without this flag.")
    ap.add_argument("--judgment", metavar="FILE", default=None,
                    help="answer the L5 judgment layer from FILE (a JSON list of "
                         "{op,target,value,reason} edits) instead of calling a model. "
                         "Use with --emit-prompt to see what is being answered. Lets a "
                         "human, or an agent, be the reviewer with no API key involved.")
    ap.add_argument("--emit-prompt", metavar="FILE", default=None,
                    help="write the exact prompt the L5 judgment layer would send, so it "
                         "can be answered out of band and fed back with --judgment")
    args = ap.parse_args()

    # --- figure out the input mode (for friendly console output) ---
    if args.demo:
        print("[demo] using built-in sample melody")
    elif not args.input:
        ap.error("provide an input file or use --demo")
    elif is_midi(args.input):
        print(f"[midi] reading {args.input}")
    elif is_tab_file(args.input):
        kind = ("Guitar Pro" if str(args.input).lower().endswith(
            (".gp5", ".gp4", ".gp3", ".gtp")) else "ASCII text tab")
        print(f"[import] reading {kind} {args.input} "
              f"(fingering preserved, no re-arrangement)")
    else:
        sep_label = "off" if args.no_separate else ("demucs" if demucs_available() else "passthrough")
        amt_label = args.amt
        if amt_label == "auto":
            amt_label = "yourmt3+" if yourmt3_available() else "basic-pitch"
        print(f"[audio] separation={sep_label}  amt={amt_label}")

    # --- decide the judgment backend ---
    use_llm = bool(args.instruction) and is_llm_configured()
    if args.llm:
        use_llm = True
    if use_llm and not is_llm_configured():
        ap.error("--llm requested but no LLM provider is configured "
                 "(set RESONOTE_LLM_API_KEY). The rules layer is the fallback.")
    if args.judgment and not os.path.exists(args.judgment):
        ap.error(f"--judgment {args.judgment}: file does not exist")
    if not (args.input and is_tab_file(args.input)):
        label = "file" if args.judgment else ("llm" if use_llm else "rules")
        print(f"[judge] backend={label}"
              + (f"  (reviewer answers from {args.judgment})" if args.judgment else "")
              + ("" if use_llm or args.judgment or not args.instruction else
                 "  (tip: set RESONOTE_LLM_API_KEY to enable LLM on "
                 "--instruction)"))

    # --- run: import an existing tab, or the full audio pipeline (L1 -> L8) ---
    if args.input and is_tab_file(args.input):
        res = run_import(
            args.input,
            gp5_path=args.out,
            midi_path=args.midi,
            wav_path=args.wav,
            html_path=args.html,
            pdf_path=args.pdf,
            musicxml_path=args.musicxml,
            tempo=args.tempo,
        )
    else:
        res = run(
            args.input,
            demo=args.demo,
            tempo=args.tempo,
            no_separate=args.no_separate,
            amt=args.amt,
            style=args.style,
            instruction=args.instruction,
            llm=use_llm,
            gp5_path=args.out,
            midi_path=args.midi,
            wav_path=args.wav,
            html_path=args.html,
            musicxml_path=args.musicxml,
            judgment_file=args.judgment,
        )

    # --- friendly console output ---
    print("[analysis]\n" + res.summary)

    if args.emit_prompt:
        if res.analysis is None:
            ap.error("--emit-prompt needs the arranging path; import mode keeps "
                     "the original fingering and judges nothing")
        from arrangement import LLMJudgmentLayer, assign_roles, voice
        # assign_roles -> voice, WITHOUT judge(): that is the exact state the
        # judgment layer sees, so the prompt describes the real decision point
        cand = voice(assign_roles(res.raw_notes, res.analysis))
        cand.style = args.style
        with open(args.emit_prompt, "w", encoding="utf-8") as f:
            f.write(LLMJudgmentLayer._build_prompt(cand, args.instruction) + "\n")
        print(f"[prompt] wrote {args.emit_prompt}"
              f"  (answer it with --judgment <file>)")

    rc = res.role_counts
    if args.input and is_tab_file(args.input):
        print(f"[import] kept {rc.get('melody', 0)} notes with their original "
              f"fingering (no roles assigned, nothing re-arranged)")
    else:
        print(f"[arrange] melody={rc.get('melody', 0)}  bass={rc.get('bass', 0)}  "
              f"harmony={rc.get('harmony', 0)}  (style={args.style})")

    print(f"Wrote {args.out}  ({len(res.placed)} notes, tempo={res.tempo:.1f}BPM)")
    print()
    print(res.ascii_tab)

    if args.html:
        print(f"\nWrote preview {args.html}")

    if args.pdf:
        # A real engraver wins when one is installed: it reads the .gp5 we just
        # wrote and does the spacing and page breaks we cannot.
        from engraving import engrave, engraver_version, find_engraver
        eng = find_engraver()
        backend = engraver_version(eng) if eng and os.path.exists(args.out) \
            and engrave(args.out, args.pdf) else None
        if not backend and not (args.input and is_tab_file(args.input)):
            from tab_pdf import write_tab_pdf
            write_tab_pdf(res.placed, args.pdf, tempo=res.tempo,
                          title=os.path.splitext(os.path.basename(args.pdf))[0])
        # import mode: run_import already wrote it, with the source tuning
        print(f"\nWrote tablature PDF {args.pdf}"
              + (f"  (engraved by {backend})" if backend
                 else "  (built-in writer; install MuseScore for print quality)"))
    if args.midi:
        print(f"\nWrote preview MIDI {args.midi}")
    if args.musicxml:
        print(f"\nWrote MusicXML {args.musicxml}")
    if args.wav:
        print(f"\nWrote preview WAV {args.wav}")

    # Last, and never before the artifacts: the dump is diagnostics, and a
    # diagnostics failure must not be able to cost the user their score.
    if args.dump_intermediate:
        if res.analysis is None:
            print("\n[dump] skipped: import mode runs no L4/L5, so there is "
                  "no analysis or arrangement judgment to report")
        else:
            _dump_analysis(res.analysis, args.dump_intermediate)
            _dump_arrangement(res.raw_notes, res.analysis, args.instruction,
                              args.style, args.dump_intermediate,
                              backend="file" if args.judgment
                              else ("llm" if use_llm else "rules"),
                              edits_file=args.judgment or "")


if __name__ == "__main__":
    main()
