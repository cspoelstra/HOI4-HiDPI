from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

from .bmfont import BMFont, Glyph
from . import trace
from .imageops import inpaint, luminance, premultiply, sharpen_coverage, unpremultiply, upscale2x
from .ttf import TTFGlyphs

# Corner texels of every generated atlas are red with alpha 1/255; text.shader looks for that to know
# the texture is stored at 2x (the engine derives glyph UVs from the real texture size). Green carries
# log2 of the atlas width / 32, blue the grey level of the glyph fill, which text.shader needs to store
# text drawn into 1x render targets at 2x; 0 means the font has effects and is drawn as usual there.
def marker(width, fill):
    return np.array([1.0, np.log2(width) / 32.0, fill, 1.0 / 255.0], dtype=np.float32)


def fill_level(atlas, spec):
    if spec.resize != 1.0:
        return spec.fill
    if spec.effects:
        return 0.0
    solid = atlas[..., 3] > 0.9
    return float(np.percentile(luminance(atlas[solid]), 90)) if solid.any() else 1.0

PAD = 2          # 1x pixels of working margin around each original glyph cell
GAP = 1          # 1x pixels between packed glyphs and around the atlas border
MIN_NCC = 0.55   # below this the outline font doesn't match the original glyph; fall back to upscaling


@dataclass
class FontSpec:
    ttf: str | None = None
    em: float = 0.0
    embolden: float = 0.0
    effects: bool = False  # glyphs carry a baked shadow/outline/bevel around the fill
    resize: float = 1.0    # render the font larger than vanilla instead of matching it
    fill: float = 1.0      # resized fonts: grey level of the glyph fill
    shadow: tuple | None = None  # resized fonts: (dx, dy, sigma, strength) in vanilla 1x pixels


@dataclass
class GlyphResult:
    glyph: Glyph
    rgba: np.ndarray | None  # 2x straight-alpha cell, None for empty glyphs
    source: str              # 'ttf', 'traced', 'upscale' or 'empty'


def fill_target(cell, effects):
    """Estimate of the glyph's fill coverage in an original 1x cell."""
    if not effects:
        return cell[..., 3]
    return cell[..., 3] * np.clip((luminance(cell) - 0.25) / 0.3, 0.0, 1.0)


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def _align(ttf, cp, cw, ch, pen_x, base_y, target):
    best = (-1.0, 0.0, 0.0)
    for sy in np.arange(-1.0, 1.01, 0.25):
        for sx in np.arange(-1.5, 1.51, 0.25):
            cov = ttf.render(cp, cw, ch, pen_x + sx, base_y + sy, scale=1, ss=4)
            c = _ncc(cov, target)
            if c > best[0]:
                best = (c, sx, sy)
    return best


def _glyph_cell(g, tex, spec):
    """The original glyph padded by PAD, plus its fill coverage estimate; None for empty glyphs."""
    if g.w <= 0 or g.h <= 0:
        return None
    cell = np.pad(tex[g.y:g.y + g.h, g.x:g.x + g.w], ((PAD, PAD), (PAD, PAD), (0, 0)))
    if cell[..., 3].max() < 1.0 / 255.0:
        return None
    return cell, fill_target(cell, spec.effects)


def font_shift(font, tex, spec, ttf, sample=48):
    """Subpixel offset between the outline font and the original bitmaps, shared by all glyphs.

    Fitting each glyph separately would give every glyph its own offset of up to half a pixel,
    which shows up as a jittery baseline and uneven spacing.
    """
    shifts = []
    candidates = ([g for g in font.glyphs if 33 <= g.id < 127 and ttf.has(g.id)]
                  or [g for g in font.glyphs if g.id > 32 and ttf.has(g.id)])
    for g in candidates[::max(1, len(candidates) // sample)]:
        prepared = _glyph_cell(g, tex, spec)
        if prepared is None or prepared[1].max() <= 0.2:
            continue
        cell, target = prepared
        score, sx, sy = _align(ttf, g.id, cell.shape[1], cell.shape[0], PAD - g.xo, PAD + font.base - g.yo, target)
        if score >= MIN_NCC:
            shifts.append((sx, sy))
    if not shifts:
        return 0.0, 0.0
    sx, sy = np.median(np.array(shifts), axis=0)
    return float(sx), float(sy)


def convert_glyph(g, tex, base, spec, ttf, shift):
    prepared = _glyph_cell(g, tex, spec)
    if prepared is None:
        return GlyphResult(g, None, 'empty')
    cell, target = prepared
    ch, cw = cell.shape[:2]

    source = 'upscale'
    m1 = target
    m2 = None
    if ttf is not None and ttf.has(g.id) and target.max() > 0.2:
        pen_x, base_y = PAD - g.xo + shift[0], PAD + base - g.yo + shift[1]
        cov = ttf.render(g.id, cw, ch, pen_x, base_y, scale=1)
        if _ncc(cov, target) >= MIN_NCC:
            m1 = cov
            m2 = ttf.render(g.id, cw, ch, pen_x, base_y, scale=2, ss=4)
            source = 'ttf'
    if m2 is None:
        if trace.available():
            m2, source = trace.trace_coverage(m1), 'traced'
        else:
            m2 = sharpen_coverage(m1)

    alpha = cell[..., 3]
    color_known = np.clip((m1 - 0.4) / 0.4, 0, 1) * np.clip((alpha - 0.3) / 0.4, 0, 1)
    if spec.effects:
        color_known *= np.clip((luminance(cell) - 0.2) / 0.2, 0, 1)
    if color_known.max() <= 0:
        color_known = (alpha > 0.1).astype(np.float32)
    fill_rgb = np.clip(upscale2x(inpaint(cell[..., :3], color_known), method=Image.BILINEAR), 0, 1)

    if spec.effects:
        bright = np.clip((luminance(cell) - 0.35) / 0.2, 0, 1) * np.clip(alpha * 4, 0, 1)
        fx_known = np.clip(1 - m1 * 20, 0, 1) * (1 - bright)
        fx = inpaint(premultiply(cell), fx_known)
        fx2 = np.clip(upscale2x(fx), 0, 1)
        fx2[..., :3] = np.minimum(fx2[..., :3], fx2[..., 3:4])
    else:
        fx2 = np.zeros(m2.shape + (4,), dtype=np.float32)

    out = np.empty(m2.shape + (4,), dtype=np.float32)
    out[..., 3] = m2 + (1 - m2) * fx2[..., 3]
    out[..., :3] = m2[..., None] * fill_rgb + (1 - m2[..., None]) * fx2[..., :3]
    out = unpremultiply(out)
    out[out[..., 3] < 1.0 / 255.0] = 0.0
    return GlyphResult(g, out, source)


def _crop(res, pad=PAD):
    """Trim a 2x cell to its visible bbox, snapped to the 1x grid, and update the glyph metrics."""
    g, rgba = res.glyph, res.rgba
    ys, xs = np.nonzero(rgba[..., 3] >= 1.0 / 255.0)
    if len(xs) == 0:
        return replace(res, rgba=None, source='empty')
    x0, y0 = xs.min() // 2, ys.min() // 2
    x1, y1 = (xs.max() + 2) // 2, (ys.max() + 2) // 2
    crop = rgba[y0 * 2:y1 * 2, x0 * 2:x1 * 2]
    ng = replace(g, w=x1 - x0, h=y1 - y0, xo=g.xo - pad + x0, yo=g.yo - pad + y0)
    return GlyphResult(ng, crop, res.source)


def _pack(results, width, empty_size):
    """Shelf-pack 1x-sized cells; returns positions keyed by glyph id ('empty' for the shared blank
    cell) and the atlas height."""
    order = sorted((r for r in results if r.rgba is not None), key=lambda r: (-r.glyph.h, -r.glyph.w))
    order.append(GlyphResult(Glyph('empty', 0, 0, *empty_size, 0, 0, 0), None, 'empty'))
    pos = {}
    x, y, shelf = GAP, GAP, 0
    for r in order:
        w, h = r.glyph.w, r.glyph.h
        if x + w + GAP > width:
            x, y, shelf = GAP, y + shelf + GAP, 0
        pos[r.glyph.id] = (x, y)
        x += w + GAP
        shelf = max(shelf, h)
    return pos, y + shelf + GAP


def _gaussian_blur(a, sigma):
    r = max(1, int(np.ceil(sigma * 3)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    a = np.apply_along_axis(lambda x: np.convolve(np.pad(x, r), k, 'valid'), 1, a)
    return np.apply_along_axis(lambda x: np.convolve(np.pad(x, r), k, 'valid'), 0, a)


def _shifted(a, dx, dy):
    h, w = a.shape
    return np.asarray(Image.fromarray(a.astype(np.float32), 'F').transform(
        (w, h), Image.AFFINE, (1, 0, -dx, 0, 1, -dy), Image.BILINEAR))


def resized_glyph(g, tex, font, spec, ttf, shift):
    """Glyph `g` redrawn at spec.resize times its vanilla size, with a synthesised drop shadow.

    The result is a GlyphResult whose cell origin sits `pad` 1x pixels above-left of the pen
    position at the top of the line, ready for _crop(pad=pad).
    """
    s = spec.resize
    pad = 6
    base = round(font.base * s)
    cw, ch = max(1, round(g.xa * s)) + 2 * pad, round(font.line_height * s) + 2 * pad
    blank = Glyph(g.id, 0, 0, round(g.w * s), round(g.h * s), round(g.xo * s), round(g.yo * s), round(g.xa * s))
    if g.w <= 0 or g.h <= 0 or tex[g.y:g.y + g.h, g.x:g.x + g.w, 3].max() < 1.0 / 255.0:
        return GlyphResult(blank, None, 'empty')
    if ttf.has(g.id):
        m2 = ttf.render(g.id, cw, ch, pad + shift[0] * s, pad + base + shift[1] * s, scale=2, ss=4)
        source = 'ttf'
    else:
        cell = np.pad(tex[g.y:g.y + g.h, g.x:g.x + g.w], ((PAD, PAD), (PAD, PAD), (0, 0)))
        traced = trace.trace_coverage(fill_target(cell, spec.effects), scale=2 * s)
        m2 = np.zeros((ch * 2, cw * 2), dtype=np.float32)
        ox, oy = round((pad + (g.xo - PAD) * s) * 2), round((pad + (g.yo - PAD) * s) * 2)
        th, tw = traced.shape
        y0, x0 = max(oy, 0), max(ox, 0)
        y1, x1 = min(oy + th, m2.shape[0]), min(ox + tw, m2.shape[1])
        m2[y0:y1, x0:x1] = traced[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        source = 'traced'
    if spec.shadow:
        dx, dy, sigma, strength = spec.shadow
        sh = np.clip(strength * _gaussian_blur(_shifted(m2, 2 * dx * s, 2 * dy * s), 2 * sigma * s), 0, 1)
    else:
        sh = np.zeros_like(m2)
    out = np.zeros(m2.shape + (4,), dtype=np.float32)
    out[..., 3] = m2 + (1 - m2) * sh
    out[..., :3] = (m2 * spec.fill / np.maximum(out[..., 3], 1e-6))[..., None]
    out[out[..., 3] < 1.0 / 255.0] = 0.0
    return GlyphResult(replace(blank, xo=0, yo=0, w=cw, h=ch), out, source)


def build_atlas(font: BMFont, tex, spec: FontSpec):
    """Returns (BMFont with 1x metrics, 2x RGBA atlas, glyph counts per source, 1x atlas size)."""
    if spec.resize != 1.0:
        # Shift is measured at the vanilla size, where the original bitmaps are a reference.
        shift = font_shift(font, tex, spec, TTFGlyphs(spec.ttf, spec.em, spec.embolden))
        ttf = TTFGlyphs(spec.ttf, spec.em * spec.resize, spec.embolden * spec.resize)
        results = [resized_glyph(g, tex, font, spec, ttf, shift) for g in font.glyphs]
        results = [_crop(r, pad=6) if r.rgba is not None else r for r in results]
        common = dict(font.common, lineHeight=str(round(font.line_height * spec.resize)),
                      base=str(round(font.base * spec.resize)))
        kernings = [(a, b, round(amt * spec.resize)) for a, b, amt in font.kernings]
        font = BMFont(dict(font.info), common, font.glyphs, kernings)
    else:
        ttf = TTFGlyphs(spec.ttf, spec.em, spec.embolden) if spec.ttf else None
        shift = font_shift(font, tex, spec, ttf) if ttf else (0.0, 0.0)
        results = [convert_glyph(g, tex, font.base, spec, ttf, shift) for g in font.glyphs]
        results = [_crop(r) if r.rgba is not None else r for r in results]

    area = sum(r.glyph.w * r.glyph.h for r in results if r.rgba is not None)
    width = 128
    while width * width < area * 1.6:
        width *= 2
    empties = [r.glyph for r in results if r.rgba is None]
    empty_size = (max([g.w for g in empties], default=0), max([g.h for g in empties], default=0))
    pos, used_h = _pack(results, width, empty_size)
    height = 64
    while height < used_h:
        height *= 2

    atlas = np.zeros((height * 2, width * 2, 4), dtype=np.float32)
    glyphs, counts = [], {}
    for r in results:
        counts[r.source] = counts.get(r.source, 0) + 1
        g = r.glyph
        if r.rgba is None:
            ex, ey = pos['empty']
            glyphs.append(replace(g, x=ex, y=ey))
            continue
        x, y = pos[g.id]
        atlas[y * 2:(y + g.h) * 2, x * 2:(x + g.w) * 2] = r.rgba
        glyphs.append(replace(g, x=x, y=y))
    fill = fill_level(atlas, spec)
    for cy in (0, -1):
        for cx in (0, -1):
            atlas[cy, cx] = marker(atlas.shape[1], fill)

    return BMFont(dict(font.info), dict(font.common), glyphs, list(font.kernings)), atlas, counts, (width, height)
