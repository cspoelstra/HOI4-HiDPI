import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

_PAIR = re.compile(r'(\w+)=("[^"]*"|\S+)')


@dataclass
class Glyph:
    id: int
    x: int
    y: int
    w: int
    h: int
    xo: int
    yo: int
    xa: int
    page: int = 0
    chnl: int = 15


@dataclass
class BMFont:
    info: dict
    common: dict
    glyphs: list
    kernings: list = field(default_factory=list)

    @property
    def base(self):
        return int(self.common['base'])

    @property
    def line_height(self):
        return int(self.common['lineHeight'])


def _parse_line(line):
    tag, _, rest = line.strip().partition(' ')
    return tag, {k: v for k, v in _PAIR.findall(rest)}


def read_fnt(path):
    info, common, glyphs, kernings = {}, {}, [], []
    for line in Path(path).read_text(encoding='latin-1').splitlines():
        if not line.strip():
            continue
        tag, kv = _parse_line(line)
        if tag == 'info':
            info = kv
        elif tag == 'common':
            common = kv
        elif tag == 'char':
            glyphs.append(Glyph(
                int(kv['id']), int(kv['x']), int(kv['y']), int(kv['width']), int(kv['height']),
                int(kv['xoffset']), int(kv['yoffset']), int(kv['xadvance']),
                int(kv.get('page', 0)), int(kv.get('chnl', 15))))
        elif tag == 'kerning':
            kernings.append((int(kv['first']), int(kv['second']), int(kv['amount'])))
    return BMFont(info, common, glyphs, kernings)


def write_fnt(path, font, page_file, scale_w, scale_h):
    common = dict(font.common, scaleW=str(scale_w), scaleH=str(scale_h), pages='1')
    lines = [
        'info ' + ' '.join(f'{k}={v}' for k, v in font.info.items()),
        'common ' + ' '.join(f'{k}={v}' for k, v in common.items()),
        f'page id=0 file="{page_file}"',
        f'chars count={len(font.glyphs)}',
    ]
    for g in font.glyphs:
        lines.append(
            f'char id={g.id:<5d} x={g.x:<5d} y={g.y:<5d} width={g.w:<5d} height={g.h:<5d} '
            f'xoffset={g.xo:<5d} yoffset={g.yo:<5d} xadvance={g.xa:<5d} page=0  chnl={g.chnl}')
    if font.kernings:
        lines.append(f'kernings count={len(font.kernings)}')
        for a, b, amt in font.kernings:
            lines.append(f'kerning first={a:<3d} second={b:<3d} amount={amt}')
    Path(path).write_text('\n'.join(lines) + '\n', encoding='latin-1')


def load_rgba(path):
    """Float32 straight-alpha RGBA in [0,1]."""
    im = Image.open(path)
    im.load()
    return np.asarray(im.convert('RGBA'), dtype=np.float32) / 255.0


def save_dds_rgba(path, rgba):
    """Uncompressed A8R8G8B8 DDS without mipmaps, which is what the vanilla font textures use."""
    h, w = rgba.shape[:2]
    px = np.clip(np.rint(rgba * 255.0), 0, 255).astype(np.uint8)
    bgra = px[..., [2, 1, 0, 3]]
    DDSD_CAPS, DDSD_HEIGHT, DDSD_WIDTH, DDSD_PITCH, DDSD_PIXELFORMAT = 0x1, 0x2, 0x4, 0x8, 0x1000
    DDPF_ALPHAPIXELS, DDPF_RGB = 0x1, 0x40
    DDSCAPS_TEXTURE = 0x1000
    header = struct.pack(
        '<4sIIIIIII44sIIIIIIIIIIIII',
        b'DDS ', 124,
        DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PITCH | DDSD_PIXELFORMAT,
        h, w, w * 4, 0, 0, b'\0' * 44,
        32, DDPF_ALPHAPIXELS | DDPF_RGB, 0, 32,
        0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000,
        DDSCAPS_TEXTURE, 0, 0, 0, 0)
    with open(path, 'wb') as f:
        f.write(header)
        f.write(bgra.tobytes())
