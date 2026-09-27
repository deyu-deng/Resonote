"""Tiny BMP I/O + helpers for the OMR experiments (no PIL needed)."""
import struct
import numpy as np


def read_bmp(path):
    d = open(path, 'rb').read()
    off = struct.unpack('<I', d[10:14])[0]
    w, h = struct.unpack('<ii', d[18:26])
    bpp = struct.unpack('<H', d[28:30])[0]
    assert bpp in (24, 32) and struct.unpack('<I', d[30:34])[0] == 0, \
        'need uncompressed 24/32bpp'
    ch = bpp // 8
    rowsz = ((w * ch + 3) // 4) * 4
    topdown = h > 0
    h = abs(h)
    arr = np.frombuffer(d[off:off + rowsz * h], dtype=np.uint8).reshape(h, rowsz)
    arr = arr[:, :w * ch].reshape(h, w, ch)[:, :, :3][:, :, ::-1]
    return np.ascontiguousarray(arr if topdown else arr[::-1])


def write_bmp(img, path):
    """img: HxWx3 uint8 RGB."""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    h, w, _ = img.shape
    rowsz = ((w * 3 + 3) // 4) * 4
    pad = np.zeros((h, rowsz - w * 3, 3), dtype=np.uint8)
    body = np.concatenate([img[:, :, ::-1], pad], axis=1).tobytes()
    hdr = b'BM' + struct.pack('<IHHI', 54 + len(body), 0, 0, 54)
    dib = struct.pack('<IiiHHIIiiII', 40, w, h, 1, 24, 0,
                      len(body), 2835, 2835, 0, 0)
    open(path, 'wb').write(hdr + dib + body)
    return path


def zoom(arr, k=4):
    return np.repeat(np.repeat(arr, k, axis=0), k, axis=1)


def save_png(arr, path):
    write_bmp(arr, path + '.bmp')
    import subprocess
    subprocess.run(['sips', '-s', 'format', 'png', path + '.bmp',
                    '--out', path], capture_output=True, check=True)
    return path
