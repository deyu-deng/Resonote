"""Stage 2 — cluster glyph bitmaps and render one representative per cluster.

Same font + same size means every '3' looks like every other '3', so instead
of OCR I cluster the bitmaps and label each cluster once by eye.
"""
import numpy as np
from scipy import ndimage

import omrlib as O
import omr1 as M

CELL = 20          # normalised glyph canvas (w x h)


def normalise(ink, box):
    """Tight-crop the glyph and centre it on a fixed canvas."""
    x0, y0, x1, y1, _ = box
    g = ink[y0:y1, x0:x1]
    ys, xs = np.where(g)
    if not len(ys):
        return None
    g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = g.shape
    canvas = np.zeros((CELL, CELL), dtype=float)
    sy = (CELL - 4) / h
    sx = (CELL - 4) / w
    s = min(sy, sx)                      # keep aspect
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    small = ndimage.zoom(g.astype(float), (nh / h, nw / w), order=1) > 0.4
    oy, ox = (CELL - nh) // 2, (CELL - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = small
    return canvas


def cluster(bits, thresh=0.62):
    """Greedy: assign to the most-correlated representative or start a new
    cluster. Returns (labels, reps)."""
    labels, reps = [], []
    for b in bits:
        best, bi = -1.0, -1
        for i, r in enumerate(reps):
            inter = np.logical_and(b, r).sum()
            union = np.logical_or(b, r).sum()
            sim = inter / union if union else 0.0
            if sim > best:
                best, bi = sim, i
        if best >= thresh:
            labels.append(bi)
            reps[bi] = np.maximum(reps[bi], b) if False else reps[bi]
        else:
            reps.append(b)
            labels.append(len(reps) - 1)
    return np.array(labels), reps


def sheet(bits, labels, reps, path, cols=10, scale=5):
    """One representative per cluster, laid out in a grid."""
    ids = sorted(set(labels))
    pitch = CELL * scale + 8
    rows = (len(ids) + cols - 1) // cols
    canvas = np.full((rows * pitch + 8, cols * pitch + 8, 3), 255,
                     dtype=np.uint8)
    for k, cid in enumerate(ids):
        r, c = divmod(k, cols)
        y, x = 8 + r * pitch, 8 + c * pitch
        g = (reps[cid] * 255).astype(np.uint8)
        z = O.zoom(g, scale)[:CELL * scale, :CELL * scale]
        canvas[y:y + z.shape[0], x:x + z.shape[1]] = z[:, :, None]
    O.save_png(canvas, path)
    return path


if __name__ == '__main__':
    allbits, allboxes, allsys = [], [], []
    for pi, name in enumerate(('p1.bmp', 'p2.bmp')):
        rgb, gray, ink, systems = M.analyse(name)
        for si, s in enumerate(systems):
            for b in M.glyph_boxes(ink, s, 60, gray.shape[1] - 30):
                nb = normalise(ink, b)
                if nb is None:
                    continue
                allbits.append(nb)
                allboxes.append(b + (pi, si))
                allsys.append((pi, si))
    bits = np.array(allbits, dtype=bool)
    print('glyphs:', len(bits))
    labels, reps = cluster(bits)
    print('clusters:', len(set(labels.tolist())))
    from collections import Counter
    cnt = Counter(labels.tolist())
    print('cluster sizes:', sorted(cnt.items(), key=lambda kv: -kv[1]))
    sheet(bits, labels, reps, 'clusters.png')
    np.save('bits.npy', bits)
    np.save('labels.npy', labels)
    import json
    json.dump({'boxes': [list(map(int, b)) for b in allboxes],
               'labels': labels.tolist()}, open('glyphs.json', 'w'))
