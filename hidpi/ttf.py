from functools import lru_cache

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont


class TTFGlyphs:
    """Renders anti-aliased glyph coverage for a TrueType/OpenType font at arbitrary subpixel positions.

    Coverage is computed by rendering `ss`x larger and box-filtering down, which sidesteps hinting and
    gives exact area coverage at any scale.
    """

    def __init__(self, path, em_1x, embolden_1x=0.0):
        self.path = str(path)
        self.em = em_1x
        self.embolden = embolden_1x
        tt = TTFont(self.path, lazy=True)
        self.cmap = set(tt.getBestCmap() or {})
        self.upem = tt['head'].unitsPerEm
        os2 = tt['OS/2']
        self.win_ascent, self.win_descent = os2.usWinAscent, os2.usWinDescent
        tt.close()

    def has(self, cp):
        return cp in self.cmap

    @lru_cache(maxsize=8)
    def _font(self, px):
        return ImageFont.truetype(self.path, size=px)

    def render(self, cp, cell_w, cell_h, pen_x, base_y, scale=1, ss=8):
        """Coverage of glyph `cp` in a cell of `cell_w`x`cell_h` 1x pixels, rendered at `scale`x.

        `pen_x`/`base_y` are the pen origin and baseline in 1x cell coordinates (may be fractional).
        """
        k = scale * ss
        W, H = int(round(cell_w * k)), int(round(cell_h * k))
        img = Image.new('L', (W, H), 0)
        d = ImageDraw.Draw(img)
        font = self._font(self.em * k)
        d.text((pen_x * k, base_y * k), chr(cp), font=font, fill=255, anchor='ls')
        a = np.asarray(img, dtype=np.float32) / 255.0
        if self.embolden:
            # GDI-style synthetic bold: overstrike shifted right.
            shift = int(round(self.embolden * k))
            b = np.zeros_like(a)
            b[:, shift:] = a[:, :W - shift]
            a = np.maximum(a, b)
        return a.reshape(H // ss, ss, W // ss, ss).mean(axis=(1, 3))
