"""Crop-from-paper context noise around a rendered structure.

A structure cut out of a paper by a detector comes with its surroundings: the compound number, pieces of neighbouring
structures, reaction arrows and conditions, charge brackets with the counter-ion, table rules. Everything here is
drawn strictly outside the structure's bounding box (plus a margin), so the atom labels of the sample stay correct:
the model only learns that this stuff is not part of the molecule.
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = {
    'regular': ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf',
                'C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/times.ttf'],
    'bold': ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
             '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
             '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf',
             'C:/Windows/Fonts/arialbd.ttf', 'C:/Windows/Fonts/timesbd.ttf'],
}
FONTS = {k: [p for p in v if os.path.exists(p)] for k, v in FONT_CANDIDATES.items()}

METAL_TAGS = ['Ru', 'Ir', 'Pt', 'Pd', 'Rh', 'Au', 'Cu', 'Fe', 'Re', 'Os', 'Co', 'Ni', 'Mn', 'Ag']
ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']
SNIPPETS = ['+', '+', '(iv)', '(iii)', '(ii)', 'i', 't-BuOK', 'K2CO3', 'MeOH', 'DMSO', 'CH2Cl2', '2 NaCl', 'r.t., 2 h',
            'reflux', 'hv', 'KF', 'NaBH4', 'AgOTf', '80 °C', 'THF', 'Et3N', 'CH3I', 'n-BuLi', '-78 °C', '[O]', 'H2']
COUNTER_IONS = ['PF6−', '(PF6)2', 'BF4−', '(BF4)2', 'Cl−', '2 Cl−', 'OTf−', '(ClO4)2', 'NO3−', 'SbF6−']
CHARGES = ['+', '2+', '3+', '+', '2+']


def compound_label(rng):
    n = int(rng.integers(1, 60))
    kind = rng.integers(0, 8)
    if kind == 0:
        return str(n)
    if kind == 1:
        return f'{n}{"abcdefgh"[rng.integers(0, 8)]}'
    if kind == 2:
        return ROMAN[rng.integers(0, len(ROMAN))]
    if kind == 3:
        return f'{METAL_TAGS[rng.integers(0, len(METAL_TAGS))]}{rng.integers(1, 9)}'
    if kind == 4:
        return f'({n})'
    if kind == 5:
        return f'(±)-{n}'
    if kind == 6:
        return f'L{rng.integers(1, 9)}'
    return f'{n}·{COUNTER_IONS[rng.integers(0, len(COUNTER_IONS))]}'


def _font(rng, size, bold=False):
    paths = FONTS['bold' if bold else 'regular'] or FONTS['regular']
    if not paths:
        return ImageFont.load_default()
    return ImageFont.truetype(paths[rng.integers(0, len(paths))], int(max(8, size)))


def _text_image(text, font, color=0):
    box = font.getbbox(text)
    w, h = box[2] - box[0] + 4, box[3] - box[1] + 4
    im = Image.new('L', (max(1, w), max(1, h)), 255)
    ImageDraw.Draw(im).text((2 - box[0], 2 - box[1]), text, font=font, fill=color)
    return np.array(im)


def _arrow_image(rng, length, lw):
    horizontal = rng.random() < 0.7
    w, h = (length, 6 * lw + 6) if horizontal else (6 * lw + 6, length)
    im = np.full((h, w), 255, np.uint8)
    if horizontal:
        y = h // 2
        cv2.arrowedLine(im, (2, y), (w - 3, y), 0, lw, cv2.LINE_AA, tipLength=min(0.3, 12 / length))
    else:
        x = w // 2
        cv2.arrowedLine(im, (x, 2), (x, h - 3), 0, lw, cv2.LINE_AA, tipLength=min(0.3, 12 / length))
    return im


class Canvas:
    """Grayscale canvas with a protected rectangle (the structure plus margin)."""

    def __init__(self, img, keep, rng):
        self.img, self.keep, self.rng = img, keep, rng

    def free(self, x, y, w, h):
        x0, y0, x1, y1 = self.keep
        return x + w <= x0 or x >= x1 or y + h <= y0 or y >= y1

    def paste(self, patch, x, y):
        """Paste (min-blend) a patch at (x, y); parts outside the canvas are clipped. False if it hits the keep box."""
        h, w = patch.shape
        if not self.free(x, y, w, h):
            return False
        H, W = self.img.shape
        sx0, sy0 = max(0, -x), max(0, -y)
        dx0, dy0 = max(0, x), max(0, y)
        dx1, dy1 = min(W, x + w), min(H, y + h)
        if dx1 <= dx0 or dy1 <= dy0:
            return False
        region = self.img[dy0:dy1, dx0:dx1]
        np.minimum(region, patch[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0], out=region)
        return True


def pad(img, pix, rng, keep_margin):
    """Random white margins (room for the context) and the protected box in the padded image."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    side = rng.integers(0, [int(0.45 * W) + 1, int(0.45 * H) + 1, int(0.45 * W) + 1, int(0.6 * H) + 1]) + 4
    left, top, right, bottom = [int(s) for s in side]
    out = np.full((H + top + bottom, W + left + right), 255, np.uint8)
    out[top:top + H, left:left + W] = gray
    dark = np.argwhere(gray < 200)
    y0, x0 = dark.min(0) if len(dark) else (0, 0)
    y1, x1 = dark.max(0) if len(dark) else (H - 1, W - 1)
    keep = (left + x0 - keep_margin, top + y0 - keep_margin, left + x1 + keep_margin, top + y1 + keep_margin)
    return out, pix + np.array([left, top]), keep


def _brackets(canvas, rng, bond_px, font_px):
    """[ complex ]2+ with the counter-ion on the right; the brackets become the new protected box."""
    out, (x0, y0, x1, y1) = canvas.img, canvas.keep
    H, W = out.shape
    gap, lw, tick = int(0.3 * bond_px), max(1, int(round(bond_px / 15))), int(0.3 * bond_px)
    bx0, by0, bx1, by1 = x0 - gap, y0 - gap, x1 + gap, y1 + gap
    charge = _text_image(CHARGES[rng.integers(0, len(CHARGES))], _font(rng, font_px))
    ion = _text_image(COUNTER_IONS[rng.integers(0, len(COUNTER_IONS))], _font(rng, font_px))
    if bx0 < 0 or by0 - charge.shape[0] // 2 < 0 or by1 >= H or bx1 + charge.shape[1] + ion.shape[1] + 8 >= W:
        return
    for x, sgn in ((bx0, 1), (bx1, -1)):
        cv2.line(out, (x, by0), (x, by1), 0, lw, cv2.LINE_AA)
        cv2.line(out, (x, by0), (x + sgn * tick, by0), 0, lw, cv2.LINE_AA)
        cv2.line(out, (x, by1), (x + sgn * tick, by1), 0, lw, cv2.LINE_AA)
    canvas.keep = (bx0 - 2, by0 - 2, bx1 + 2, by1 + 2)
    canvas.paste(charge, bx1 + 3, by0 - charge.shape[0] // 2)
    canvas.paste(ion, bx1 + charge.shape[1] + 6, (by0 + by1 - ion.shape[0]) // 2)
    canvas.keep = (bx0 - 2, by0 - charge.shape[0], bx1 + charge.shape[1] + ion.shape[1] + 8, by1 + 2)


def _compound_number(canvas, rng, font_px):
    x0, y0, x1, y1 = canvas.keep
    H, W = canvas.img.shape
    text = _text_image(compound_label(rng), _font(rng, font_px * rng.uniform(0.9, 1.8), rng.random() < 0.5))
    th, tw = text.shape
    spots = [((x0 + x1 - tw) // 2 + int(rng.integers(-20, 21)), y1 + int(rng.integers(2, 12))),  # below
             (x1 + int(rng.integers(2, 12)), (y0 + y1 - th) // 2),                           # right
             (x0 - tw - int(rng.integers(2, 12)), (y0 + y1 - th) // 2),                      # left
             ((x0 + x1 - tw) // 2, y0 - th - int(rng.integers(2, 12)))]                      # above
    order = [0, 0, 0, 1, 2, 3]
    rng.shuffle(order)
    for k in order:
        x, y = spots[k]
        if 0 <= x and x + tw <= W and 0 <= y and y + th <= H and canvas.paste(text, x, y):
            return


def _intrusions(canvas, rng, bond_px, font_px):
    """Arrows and reaction text cut by the crop border."""
    H, W = canvas.img.shape
    for _ in range(int(rng.integers(1, 4))):
        if rng.random() < 0.4:
            patch = _arrow_image(rng, int(rng.uniform(2, 5) * bond_px), max(1, int(round(bond_px / 15))))
        else:
            patch = _text_image(SNIPPETS[rng.integers(0, len(SNIPPETS))], _font(rng, font_px * rng.uniform(0.8, 1.4)))
        h, w = patch.shape
        edge = rng.integers(0, 4)
        if edge == 0:
            x, y = -int(w * rng.uniform(0.2, 0.7)), int(rng.integers(0, max(1, H - h)))
        elif edge == 1:
            x, y = W - int(w * rng.uniform(0.3, 0.8)), int(rng.integers(0, max(1, H - h)))
        elif edge == 2:
            x, y = int(rng.integers(0, max(1, W - w))), -int(h * rng.uniform(0.2, 0.7))
        else:
            x, y = int(rng.integers(0, max(1, W - w))), H - int(h * rng.uniform(0.3, 0.8))
        canvas.paste(patch, x, y)


def _frame(canvas, rng):
    out = canvas.img
    H, W = out.shape
    lw = max(1, int(rng.integers(1, 3)))
    if rng.random() < 0.5:
        y = int(rng.integers(0, 3)) if rng.random() < 0.5 else H - 1 - int(rng.integers(0, 3))
        cv2.line(out, (0, y), (W - 1, y), 0, lw)
    else:
        x = int(rng.integers(0, 3)) if rng.random() < 0.5 else W - 1 - int(rng.integers(0, 3))
        cv2.line(out, (x, 0), (x, H - 1), 0, lw)


def add_context(img, pix, rng, bond_px, p):
    """img: rendered structure (BGR or gray, tight crop); pix: (n, 2) atom pixel coordinates.

    p: probabilities {'label', 'intrusion', 'brackets', 'frame'}. Returns (gray image, pix).
    Order matters: brackets first (they grow the protected box), then the compound number outside them.
    """
    margin = int(max(4, 0.25 * bond_px))
    out, pix, keep = pad(img, pix, rng, margin)
    canvas = Canvas(out, keep, rng)
    font_px = 0.6 * bond_px
    if rng.random() < p['brackets']:
        _brackets(canvas, rng, bond_px, font_px)
    if rng.random() < p['label']:
        _compound_number(canvas, rng, font_px)
    if rng.random() < p['intrusion']:
        _intrusions(canvas, rng, bond_px, font_px)
    if rng.random() < p['frame']:
        _frame(canvas, rng)
    return canvas.img, pix
