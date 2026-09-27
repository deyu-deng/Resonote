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


def _dump_analysis(analysis, dump_dir: str) -> None:
    import os
    os.makedirs(dump_dir, exist_ok=True)
    payload = {
        "backend": analysis.backend,
        "key": analysis.key,
        "tempo": analysis.tempo,
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
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[dump] wrote {out}")


def _dump_arrangement(notes, analysis, instructions, style, dump_dir):
    from arrangement import build_arrangement
    arr = build_arrangement(notes, analysis, instructions, "auto", style)
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
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[dump] wrote {out}")


def main():
    ap = argparse.ArgumentParser(
        description="Resonote: turn a melody into a fingerstyle guitar .gp5 tab")
    ap.add_argument("input", nargs="?", help="audio (.mp3/.wav) or MIDI (.mid/.midi)")
    ap.add_argument("-o", "--out", default="out.gp5", help="output .gp5 path")
    ap.add_argument("-t", "--tempo", type=int, default=120, help="BPM for the tab")
    ap.add_argument("--html", default=None, help="also write a standalone HTML preview")
    ap.add_argument("--pdf", default=None,
                    help="also write a printable tablature PDF (vector, no deps)")
    ap.add_argument("--midi", default=None, help="also write an audible preview .mid (pretty_midi)")
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
    args = ap.parse_args()

    # --- figure out the input mode (for friendly console output) ---
    if args.demo:
        print("[demo] using built-in sample melody")
    elif not args.input:
        ap.error("provide an input file or use --demo")
    elif is_midi(args.input):
        print(f"[midi] reading {args.input}")
    elif is_tab_file(args.input):
        print(f"[import] reading existing tab {args.input} "
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
    if not (args.input and is_tab_file(args.input)):
        print(f"[judge] backend={'llm' if use_llm else 'rules'}"
              + ("" if use_llm or not args.instruction else
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
        )

    # --- friendly console output ---
    print("[analysis]\n" + res.summary)

    if args.dump_intermediate:
        from quantize import quantize_notes
        from transcribe import get_backend, transcribe_audio
        # reconstruct raw notes for the dump (cheap; only when requested)
        if args.demo:
            from tests.make_sample import sample_notes
            raw_notes = sample_notes()
        elif is_midi(args.input):
            raw_notes = get_backend("midi").transcribe(args.input)
        else:
            raw_notes = transcribe_audio(args.input,
                                         use_separation=not args.no_separate,
                                         amt=args.amt)
        _dump_analysis(res.analysis, args.dump_intermediate)
        _dump_arrangement(raw_notes, res.analysis, args.instruction, args.style,
                          args.dump_intermediate)

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
        from tab_pdf import write_tab_pdf
        write_tab_pdf(res.placed, args.pdf, tempo=res.tempo,
                      title=os.path.splitext(os.path.basename(args.pdf))[0])
        print(f"\nWrote tablature PDF {args.pdf}")
    if args.midi:
        print(f"\nWrote preview MIDI {args.midi}")
    if args.wav:
        print(f"\nWrote preview WAV {args.wav}")


if __name__ == "__main__":
    main()
