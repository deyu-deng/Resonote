"""Minimal PNG decoder (RGB/RGBA/grayscale, 8-bit) — no PIL needed."""
import struct
import zlib
import numpy as np


def read_png(path):
    d = open(path, 'rb').read()
    assert d[:8] == b'\x89PNG\r\n\x1a\n', 'not a png'
    pos, idat, meta = 8, b'', None
    while pos < len(d):
        ln = struct.unpack('>I', d[pos:pos + 4])[0]
        typ = d[pos + 4:pos + 8]
        body = d[pos + 8:pos + 8 + ln]
        if typ == b'IHDR':
            w, h, depth, ctype = struct.unpack('>IIBB', body[:10])
            meta = (w, h, depth, ctype)
        elif typ == b'IDAT':
            idat += body
        elif typ == b'IEND':
            break
        pos += 12 + ln
    w, h, depth, ctype = meta
    assert depth == 8, 'need 8-bit'
    ch = {0: 1, 2: 3, 4: 2, 6: 4}[ctype]
    raw = zlib.decompress(idat)
    stride = w * ch
    out = np.zeros((h, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int16)
    pos = 0
    for y in range(h):
        f = raw[pos]
        pos += 1
        row = np.frombuffer(raw[pos:pos + stride], dtype=np.uint8).astype(np.int16)
        pos += stride
        if f == 0:
            cur = row
        elif f == 1:                       # sub
            cur = row.copy()
            for i in range(ch, stride):
                cur[i] = (cur[i] + cur[i - ch]) & 0xFF
        elif f == 2:                       # up
            cur = (row + prev) & 0xFF
        elif f == 3:                       # average
            cur = row.copy()
            for i in range(stride):
                a = cur[i - ch] if i >= ch else 0
                cur[i] = (cur[i] + (a + int(prev[i])) // 2) & 0xFF
        elif f == 4:                       # paeth
            cur = row.copy()
            for i in range(stride):
                a = int(cur[i - ch]) if i >= ch else 0
                b = int(prev[i])
                c = int(prev[i - ch]) if i >= ch else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[i] = (cur[i] + pr) & 0xFF
        else:
            raise ValueError(f'filter {f}')
        out[y] = cur.astype(np.uint8)
        prev = cur.astype(np.int16)

    px = out.reshape(h, w, ch)
    if ch == 4:                            # composite black ink onto white
        a = px[:, :, 3].astype(np.int16)
        g = (255 - a).clip(0, 255).astype(np.uint8)
        return np.ascontiguousarray(np.stack([g] * 3, axis=-1))
    if ch == 1:
        return np.ascontiguousarray(np.stack([px[:, :, 0]] * 3, axis=-1))
    return np.ascontiguousarray(px[:, :, :3])
