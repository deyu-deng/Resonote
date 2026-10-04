"""Publication-grade output by delegating to a real engraver.

:mod:`tab_pdf` draws lines and digits with no dependencies, which is the right
fallback for a machine with nothing installed but is not a score anybody wants
to read off a music stand. Spacing, beaming, page breaks and collision-avoided
fingering numbers are a solved problem in notation software, so when MuseScore
Studio is present we hand it the .gp5 and let it do the printing.

Two things about driving MuseScore from a script, both measured on 4.7.5:

* its exit code is not trustworthy -- it tears down through a crash reporter
  that prints `mutex lock failed` even on a run that wrote every file, so
  success is judged by the artifact appearing and being non-empty.
* the handbook's ``-s`` (no soundfont) is not a real option in this build and
  makes it refuse to run, so it is deliberately not passed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import List, Optional

# Candidates only, and every one of them is checked with os.path.exists: a
# machine is expected to have at most one. RESONOTE_MSCORE overrides all of
# this for installs that live somewhere else.
MSCORE_APPS = [
    "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
    "/Applications/MuseScore Studio.app/Contents/MacOS/mscore",
    "C:/Program Files/MuseScore 4/bin/MuseScore4.exe",
    "/usr/bin/mscore",
    "/var/lib/flatpak/app/org.musescore.MuseScore/current/active/files/bin/mscore",
]


def find_engraver() -> Optional[str]:
    """Path to a MuseScore CLI, or None."""
    env = os.environ.get("RESONOTE_MSCORE")
    if env and os.path.exists(env):
        return env
    for name in ("mscore", "musescore", "MuseScore4"):
        found = shutil.which(name)
        if found:
            return found
    for app in MSCORE_APPS:
        if os.path.exists(app):
            return app
    return None


def engrave(source: str, out_path: str, timeout: int = 300) -> bool:
    """Render ``source`` (.gp5 / .musicxml / .mscz) to ``out_path``.

    Returns True only if MuseScore actually produced a non-empty file; its exit
    code is ignored on purpose (see module docstring).
    """
    exe = find_engraver()
    if not exe:
        return False
    try:
        subprocess.run([exe, "-F", "-o", out_path, source],
                       capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


def engraver_version(exe: Optional[str] = None) -> str:
    """Best-effort version string, for printing next to the output so a bad
    render can be traced back to the tool that made it."""
    exe = exe or find_engraver()
    if not exe:
        return ""
    try:
        out = subprocess.run([exe, "-v"], capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return os.path.basename(exe)
    for line in out.splitlines():
        if "MuseScore" in line:
            return line.strip()
    return os.path.basename(exe)
