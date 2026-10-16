import shutil
import subprocess

import numpy as np
from PIL import Image

OVERSAMPLE = 8


def available():
    return shutil.which('potrace') is not None


def trace_coverage(cov_1x, alphamax=0.9, scale=2.0):
    """`scale`x anti-aliased coverage from a 1x coverage map, by vectorising it with potrace.

    The 1x map is interpolated to OVERSAMPLE x and thresholded at 50% (its iso-contour), traced into
    Bezier outlines, and potrace rasterises those back at `scale`x with linear-gamma anti-aliasing.
    """
    h, w = cov_1x.shape
    up = np.asarray(Image.fromarray(cov_1x.astype(np.float32), 'F')
                    .resize((w * OVERSAMPLE, h * OVERSAMPLE), Image.BICUBIC))
    bits = up >= 0.5
    if not bits.any():
        return np.zeros((round(h * scale), round(w * scale)), dtype=np.float32)
    H, W = bits.shape
    pbm = f'P4\n{W} {H}\n'.encode() + np.packbits(bits, axis=1).tobytes()
    out = subprocess.run(
        ['potrace', '-b', 'pgm', '-x', str(scale / OVERSAMPLE), '-G', '1', '-a', str(alphamax), '-o', '-', '-'],
        input=pbm, capture_output=True, check=True).stdout
    return _parse_pgm(out, round(w * scale), round(h * scale))


def _parse_pgm(data, w, h):
    fields, pos = [], 0
    while len(fields) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b'#':
            pos = data.index(b'\n', pos) + 1
            continue
        end = pos
        while not data[end:end + 1].isspace():
            end += 1
        fields.append(data[pos:end])
        pos = end
    pos += 1
    pw, ph, maxval = int(fields[1]), int(fields[2]), int(fields[3])
    img = np.frombuffer(data[pos:pos + pw * ph], dtype=np.uint8).reshape(ph, pw)
    cov = 1.0 - img.astype(np.float32) / maxval
    out = np.zeros((h, w), dtype=np.float32)
    out[:min(h, ph), :min(w, pw)] = cov[:h, :w]
    return out
