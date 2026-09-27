"""Stage 3 — cluster the *healed* glyphs (whole digits, no staff lines)."""
import json
from collections import Counter

import numpy as np
from scipy import ndimage

import omrlib as O
import omr1 as M
from omr2 import normalise, cluster, sheet

allbits, allboxes = [], []
for pi, name in enumerate(('p1.bmp', 'p2.bmp')):
    rgb, gray, ink, systems, healed = M.analyse(name)
    hink = healed < M.INK_TH                      # ink on the healed image
    for si, s in enumerate(systems):
        for b in M.glyph_boxes(hink, s, 60, gray.shape[1] - 30):
            nb = normalise(hink, b)
            if nb is None:
                continue
            allbits.append(nb)
            allboxes.append(list(b) + [pi, si])

bits = np.array(allbits, dtype=bool)
labels, reps = cluster(bits)
print('glyphs:', len(bits), ' clusters:', len(set(labels.tolist())))
print('sizes:', sorted(Counter(labels.tolist()).items(), key=lambda kv: -kv[1])[:16])
sheet(bits, labels, reps, 'clusters3.png')
json.dump({'boxes': allboxes, 'labels': labels.tolist()},
          open('glyphs3.json', 'w'))
