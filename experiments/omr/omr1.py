"""TAB OMR prototype — stage 1: staves, barlines, note glyphs.

Designed against 大树音乐屋's typeset tabs: 6 light-gray staff lines per
system (gray ~130-210), black note glyphs on them (<112), rhythm stems
hanging below, chord diagrams + lyrics outside the band, and a light
diagonal watermark that must not be mistaken for ink.
"""
import numpy as np
from scipy import ndimage

import omrlib as O

INK_TH = 112          # black ink: digits, x, stems, barlines
LINE_TH = 222         # staff lines are much lighter than ink
MIN_LINE_RUN = 0.55   # a staff line spans most of the page width
MAX_LINE_THICK = 4


def find_staff_lines(gray):
    """Rows holding a long, thin, light-gray run -> list of row groups.

    Returns groups (not means) because a line is often 2px thick and every
    row of it has to be erased, or the survivor still joins glyphs together.
    """
    h, w = gray.shape
    dark = gray < LINE_TH
    counts = dark.sum(axis=1)
    rows = np.where(counts > w * MIN_LINE_RUN)[0]
    if not len(rows):
        return []
    groups = np.split(rows, np.where(np.diff(rows) > 2)[0] + 1)
    return [g for g in groups if len(g) <= MAX_LINE_THICK]


def group_systems(line_groups):
    if len(line_groups) < 6:
        return []
    means = [int(round(g.mean())) for g in line_groups]
    out, cur = [], [line_groups[0]]
    for g in line_groups[1:]:
        if g.mean() - cur[-1].mean() > 100:
            if len(cur) == 6:
                out.append([int(round(x.mean())) for x in cur])
            cur = [g]
        else:
            cur.append(g)
    if len(cur) == 6:
        out.append([int(round(x.mean())) for x in cur])
    return out


def remove_staff_lines(ink, line_groups):
    """Erase the staff lines but keep glyphs that cross them.

    A pixel on a staff row survives only if ink continues both above and
    below -- true for a 14px digit sitting on the line, false for the line
    itself.
    """
    ink = ink.copy()
    for g in line_groups:
        for y in g:
            up = ink[y - 1] | ink[y - 2]
            dn = ink[y + 1] | ink[y + 2]
            ink[y] &= up & dn
    return ink


def heal_staff_lines(gray, line_groups):
    """Fill the staff-line rows back in so digits are whole again.

    Each line row becomes max(gray above, gray below): a digit stroke crossing
    the line is dark on both sides and stays dark, a bare line is light on both
    sides and disappears. This is what makes template matching possible -- the
    binarised mask alone cuts every glyph in half.
    """
    g = gray.copy()
    for grp in line_groups:
        for y in grp:
            g[y] = np.maximum(g[y - 2], g[y + 2])
    return g


def _longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best


def find_barlines(ink, sys_lines, x_lo, x_hi):
    """A barline is a *continuous* ink run covering ~the whole staff height
    that stops at the bottom line.

    Two things look similar and must be rejected: a stem (it also spans the
    staff but keeps going down to its beam), and a measure number or slur
    hanging into the band from above (its ink is scattered, not contiguous --
    5 separate dots is not a barline).
    """
    top, bot = sys_lines[0], sys_lines[-1]
    need = int((bot - top) * 0.75)
    cols = []
    for x in range(x_lo, x_hi):
        col = ink[top:bot + 1, x]
        if col.sum() < need or _longest_run(col) < need:
            continue
        if ink[bot + 2:bot + 10, x].sum() >= 2:   # runs on below -> stem
            continue
        cols.append(x)
    if not cols:
        return []
    groups = np.split(cols, np.where(np.diff(cols) > 3)[0] + 1)
    return [int(round(g.mean())) for g in groups]


def _merge_stacked(boxes):
    """Rejoin glyphs a staff line cut in half: two components whose x ranges
    overlap and whose y gap is small are one glyph."""
    boxes = sorted(boxes)
    changed = True
    while changed:
        changed = False
        out = []
        for b in boxes:
            merged = False
            for i, o in enumerate(out):
                ox0, oy0, ox1, oy1, on = o
                ov = min(b[2], ox1) - max(b[0], ox0)          # x overlap
                gap = max(b[1], oy0) - min(b[3], oy1)         # y gap
                if ov > 0.5 * min(b[2] - b[0], ox1 - ox0) and gap <= 5:
                    out[i] = (min(b[0], ox0), min(b[1], oy0),
                              max(b[2], ox1), max(b[3], oy1), on + b[4])
                    merged = changed = True
                    break
            if not merged:
                out.append(b)
        boxes = out
    return boxes


def glyph_boxes(ink, sys_lines, x_lo, x_hi):
    """Connected components inside the staff band -> (x0, y0, x1, y1, npix).

    A tab glyph is centred on its string line, so after merging cut halves the
    box is snapped to the nearest line and clipped to line +/-8px. That drops
    the stem hanging off an 'x', which otherwise makes every 'x' a different
    shape depending on where the staff-line removal cut it.
    """
    pad = 9
    y0, y1 = sys_lines[0] - pad, sys_lines[-1] + pad
    sub = ink[y0:y1, x_lo:x_hi]
    lab, n = ndimage.label(sub)
    raw = []
    for sl in ndimage.find_objects(lab):
        yy, xx = sl
        npix = int((lab[sl] > 0).sum())
        raw.append((x_lo + xx.start, y0 + yy.start,
                    x_lo + xx.stop, y0 + yy.stop, npix))
    merged = _merge_stacked(raw)

    out = []
    for (bx0, by0, bx1, by1, npix) in merged:
        cy = (by0 + by1) / 2.0
        line = min(sys_lines, key=lambda ly: abs(ly - cy))
        ty, by = line - 8, line + 8
        w = bx1 - bx0
        if not (4 <= w <= 26):
            continue
        crop = ink[ty:by, bx0:bx1]
        npx = int(crop.sum())
        if npx < 18 or crop.shape[0] < 10:
            continue
        ys, xs = np.where(crop)
        if not len(ys):
            continue
        out.append((bx0 + int(xs.min()), ty + int(ys.min()),
                    bx0 + int(xs.max()) + 1, ty + int(ys.max()) + 1, npx))
    out.sort()
    return out


def analyse(path):
    rgb = O.read_bmp(path)
    gray = rgb[:, :, :3].mean(axis=2)
    groups = find_staff_lines(gray)
    systems = group_systems(groups)
    ink = gray < INK_TH
    ink = remove_staff_lines(ink, groups)
    healed = heal_staff_lines(gray, groups)
    return rgb, gray, ink, systems, healed


if __name__ == '__main__':
    for name in ('p1.bmp', 'p2.bmp'):
        rgb, gray, ink, systems = analyse(name)
        print('===', name, 'systems:', len(systems))
        for i, s in enumerate(systems):
            bl = find_barlines(ink, s)
            boxes = glyph_boxes(ink, s, 60, gray.shape[1] - 30)
            # keep glyph-sized things: digits are ~10x14, x ~10x10
            gl = [b for b in boxes
                  if 4 <= (b[2] - b[0]) <= 40 and 6 <= (b[3] - b[1]) <= 30
                  and b[4] >= 12]
            print('  system %d: lines %s  barlines %d  glyphs %d'
                  % (i + 1, s, len(bl), len(gl)))
