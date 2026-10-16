import numpy as np
from PIL import Image

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def luminance(rgba):
    return rgba[..., :3] @ LUMA


def premultiply(rgba):
    out = rgba.copy()
    out[..., :3] *= rgba[..., 3:4]
    return out


def unpremultiply(rgba):
    out = rgba.copy()
    a = rgba[..., 3:4]
    out[..., :3] = np.where(a > 1e-6, rgba[..., :3] / np.maximum(a, 1e-6), 0.0)
    return np.clip(out, 0.0, 1.0)


def _down(v, w):
    h, wd = w.shape
    ph, pw = h % 2, wd % 2
    if ph or pw:
        v = np.pad(v, ((0, ph), (0, pw), (0, 0)))
        w = np.pad(w, ((0, ph), (0, pw)))
    H, W = w.shape[0] // 2, w.shape[1] // 2
    ws = w.reshape(H, 2, W, 2).sum(axis=(1, 3))
    vs = (v * w[..., None]).reshape(H, 2, W, 2, -1).sum(axis=(1, 3))
    return np.where(ws[..., None] > 0, vs / np.maximum(ws, 1e-9)[..., None], 0.0), np.minimum(ws, 1.0)


def inpaint(values, weights):
    """Push-pull fill: pixels with weight 1 keep their value, weight 0 are filled from the neighbourhood."""
    v = values if values.ndim == 3 else values[..., None]
    if weights.max() <= 0:
        return np.zeros_like(values)
    if min(weights.shape) <= 1 or weights.min() >= 1:
        filled = v
    else:
        cv, cw = _down(v, weights)
        coarse = inpaint(cv, cw)
        up = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)[:v.shape[0], :v.shape[1]]
        filled = weights[..., None] * v + (1 - weights[..., None]) * up
    return filled if values.ndim == 3 else filled[..., 0]


def upscale2x(img, method=Image.BICUBIC):
    """Upscale a float HxW or HxWxC array by exactly 2x, channel by channel."""
    chans = img[..., None] if img.ndim == 2 else img
    h, w = chans.shape[:2]
    out = np.stack([
        np.asarray(Image.fromarray(np.ascontiguousarray(chans[..., c]), 'F').resize((w * 2, h * 2), method))
        for c in range(chans.shape[2])], axis=-1)
    return out[..., 0] if img.ndim == 2 else out


def sharpen_coverage(cov_1x):
    """Best-effort 2x coverage from a 1x anti-aliased coverage map, when no outline font is available.

    Bicubic upscaling keeps stroke positions; the contrast curve around 50% coverage turns the soft
    ramps back into ~1px-wide anti-aliased edges at the new resolution.
    """
    up = np.clip(upscale2x(cov_1x), 0.0, 1.0)
    return np.clip((up - 0.5) * 1.8 + 0.5, 0.0, 1.0)
