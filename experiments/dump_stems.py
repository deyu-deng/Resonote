"""Run demucs ONLY and save stems as wavs — isolates separation from AMT."""
import sys
sys.path.insert(0, '/Users/ciel/Projects/Resonote')

from separate import get_separator

src = "/Users/ciel/Cloud/Media/Music/Song/陶喆/陶喆 - Melody.mp3"
sep = get_separator("auto")
stems = sep.separate(src)
print("分离完成:", [(s.name, s.sr, len(s.audio)) for s in stems])

import soundfile as sf
import os
os.makedirs("/tmp/stems", exist_ok=True)
for s in stems:
    out = f"/tmp/stems/{s.name}.wav"
    sf.write(out, s.audio, s.sr)
    print("  wrote", out, "%.1f s" % (len(s.audio) / s.sr))
