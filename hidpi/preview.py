"""Offline approximation of how the engine draws a string at gui_scale=2.

Vanilla: 1x atlas magnified with bilinear filtering. HiDPI: 2x atlas sampled 1:1.
"""
import numpy as np
from PIL import Image

from .imageops import upscale2x


def _layout(font, text):
    by_id = {g.id: g for g in font.glyphs}
    kern = {(a, b): amt for a, b, amt in font.kernings}
    pen, prev, out = 0, None, []
    for ch in text:
        g = by_id.get(ord(ch))
        if g is None:
            continue
        if prev is not None:
            pen += kern.get((prev, g.id), 0)
        out.append((pen, g))
        pen += g.xa
        prev = g.id
    return out, pen


def _over(dst, src, x, y):
    if x < 0:
        src, x = src[:, -x:], 0
    if y < 0:
        src, y = src[-y:], 0
    src = src[:dst.shape[0] - y, :dst.shape[1] - x]
    h, w = src.shape[:2]
    region = dst[y:y + h, x:x + w]
    a = src[..., 3:4]
    region[..., :3] = src[..., :3] * a + region[..., :3] * (1 - a)
    region[..., 3:4] = a + region[..., 3:4] * (1 - a)


def render(font, atlas, text, scale, bg=(0.18, 0.2, 0.18)):
    """`scale` is the atlas density (1 or 2). Returns an RGB image at 2x display size."""
    placed, width = _layout(font, text)
    pad = 8
    canvas = np.zeros((font.line_height + 2 * pad, width + 2 * pad, 4), dtype=np.float32)
    canvas[..., :3] = bg
    canvas[..., 3] = 1
    if scale == 2:
        canvas = np.repeat(np.repeat(canvas, 2, 0), 2, 1)
    for pen, g in placed:
        if g.w <= 0 or g.h <= 0:
            continue
        cell = atlas[g.y * scale:(g.y + g.h) * scale, g.x * scale:(g.x + g.w) * scale]
        _over(canvas, cell, (pad + pen + g.xo) * scale, (pad + g.yo) * scale)
    if scale == 1:
        canvas = np.clip(upscale2x(canvas, method=Image.BILINEAR), 0, 1)
    return Image.fromarray((canvas[..., :3] * 255).astype(np.uint8))


def compare(font, atlas1, new_font, atlas2, lines, zoom=2):
    rows = []
    for t in lines:
        rows.append(render(font, atlas1, t, 1))
        rows.append(render(new_font, atlas2, t, 2))
    w = max(r.width for r in rows)
    h = sum(r.height for r in rows)
    sheet = Image.new('RGB', (w, h), (30, 30, 30))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    return sheet.resize((w * zoom, h * zoom), Image.NEAREST)
