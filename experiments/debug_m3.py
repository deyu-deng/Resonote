"""Debug: why does the m3 sample arrangement lose all harmony?"""
import sys
sys.path.insert(0, '/Users/ciel/Projects/Resonote')

from collections import Counter, defaultdict
from tests.test_m3 import _sample_analysis
from tests.make_sample import sample_notes
from arrangement import build_arrangement
from arrange import (finger_melody, place_bass, place_harmony,
                     enforce_playability, _resolve_string_collisions)

notes = list(sample_notes())
ana = _sample_analysis()
arr = build_arrangement(notes, ana, '', 'auto', 'fingerstyle')
mel = finger_melody(sorted(arr.melody, key=lambda n: n.onset))
bass = place_bass(arr.bass)
harm = place_harmony(arr.harmony)
allp = sorted(mel + bass + harm, key=lambda p: (p.onset, p.pitch))
print('before:', dict(Counter(p.role for p in allp)))

out = enforce_playability(allp, float(ana.tempo))
print('after :', dict(Counter(p.role for p in out)))

# where did the harmony go? show the buckets that lost notes
step = 60.0 / float(ana.tempo) / 4.0
buckets = defaultdict(list)
for p in allp:
    for k in range(int(p.onset / step), int((p.onset + p.duration) / step) + 1):
        buckets.setdefault(k, []).append(p)
lost = set(id(p) for p in allp) - set(id(p) for p in out)
shown = 0
for k in sorted(buckets):
    grp = buckets[k]
    if not any(id(p) in lost for p in grp):
        continue
    f = [p.fret for p in grp if p.fret > 0]
    print('  t=%.2f span=%s %s' % (k * step,
          (max(f) - min(f)) if len(f) > 1 else 0,
          [(p.role, p.string, p.fret, p.pitch) for p in grp]))
    shown += 1
    if shown >= 6:
        break
