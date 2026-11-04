"""Synthetic captcha generator that imitates the style of samples/QVXKDE.png:
dark-red bold capitals on a white rounded box (black corners), per-char jitter,
and grey strike-through lines.

Run directly to dump preview images:  python gen.py --n 20 --out previews
"""
import argparse
import os
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont

import config

_FONT_PATHS = [f for f in config.FONTS if os.path.exists(f)]
_HAND_FONTS = [f for f in [
    r"C:\Windows\Fonts\segoeprb.ttf", r"C:\Windows\Fonts\segoepr.ttf",
    r"C:\Windows\Fonts\comicbd.ttf", r"C:\Windows\Fonts\Inkfree.ttf",
] if os.path.exists(f)]
_font_cache = {}


def _font(path, size):
    key = (path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


def random_text():
    n = random.randint(config.MIN_LEN, config.MAX_LEN)
    return "".join(random.choice(config.CHARSET) for _ in range(n))


def render(text):
    W, H = random.randint(210, 260), random.randint(46, 66)
    img = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    bg = tuple(random.randint(240, 255) for _ in range(3))
    l, t = random.randint(0, 25), random.randint(0, int(H * 0.2))
    r, b = W - 1 - random.randint(0, 4), H - 1 - random.randint(0, 8)
    d.rounded_rectangle([l, t, r, b], radius=random.randint(4, 10), fill=bg)

    # Two variants seen on the site: tightly packed big letters (QVXKDE), or
    # widely spaced handwriting-style letters (8F3K9, B9H5L).
    spaced = random.random() < 0.5
    if spaced:
        font_path = random.choice(_HAND_FONTS) if _HAND_FONTS else random.choice(_FONT_PATHS)
        base_size = random.randint(int(H * 0.45), int(H * 0.62))
    else:
        font_path = random.choice(_FONT_PATHS + _HAND_FONTS[:1])
        base_size = random.randint(int(H * 0.6), int(H * 0.8))
    red = (random.randint(130, 200), random.randint(0, 35), random.randint(20, 55) if spaced else random.randint(0, 30))

    # Render each char on its own transparent tile so it can be rotated/shifted.
    tiles, lifts = [], []
    for ch in text:
        small = random.random() < 0.08          # occasional small, raised char (like the X in QVXKDE)
        size = max(10, int(base_size * random.uniform(0.55, 0.7)) if small else base_size + random.randint(-6, 3))
        lifts.append(-random.randint(4, 10) if small else random.randint(-5, 5))
        f = _font(font_path, size)
        bbox = f.getbbox(ch)
        tw, th = bbox[2] - bbox[0] + 6, bbox[3] - bbox[1] + 6
        tile = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text((3 - bbox[0], 3 - bbox[1]), ch, font=f, fill=red + (255,))
        tile = tile.rotate(random.uniform(-15, 15), resample=Image.BICUBIC, expand=True)
        tiles.append(tile)

    if spaced:
        spacing = random.randint(3, 16)
    else:
        spacing = random.choice([random.randint(-9, -2), random.randint(-3, 3)])  # often tightly packed / overlapping
    avail = r - l - 14
    total = sum(t.width for t in tiles) + spacing * (len(tiles) - 1)
    if total > avail and spacing > 0:            # shrink gaps before shrinking letters
        spacing = max(0, spacing - -(-(total - avail) // max(1, len(tiles) - 1)))
        total = sum(t.width for t in tiles) + spacing * (len(tiles) - 1)
    scale = min(1.0, (avail - spacing * (len(tiles) - 1)) / max(1, sum(t.width for t in tiles)))
    if scale < 1.0:
        tiles = [t.resize((max(1, int(t.width * scale)), max(1, int(t.height * scale))), Image.BICUBIC)
                 for t in tiles]
        total = sum(t.width for t in tiles) + spacing * (len(tiles) - 1)

    lo = l + 6
    hi = max(lo, r - 6 - total)                    # keep the whole string inside the box
    if spaced:  # spaced variant is roughly centred
        mid = (lo + hi) // 2
        lo, hi = max(lo, mid - 20), max(max(lo, mid - 20), min(hi, mid + 20))
    x = random.randint(lo, hi)
    box_mid = (t + b) // 2
    placed = []
    for tile, lift in zip(tiles, lifts):
        if spaced:
            lift = random.randint(-8, 8)          # bigger up/down bounce
        y = box_mid - tile.height // 2 + lift
        y = max(t + 1, min(y, b - tile.height - 1)) if tile.height < b - t - 2 else t + 1
        img.paste(tile, (x, y), tile)
        placed.append((x, y, tile.width, tile.height))
        x += tile.width + spacing

    # Safety net: every character must lie fully inside the white box, otherwise the
    # label would be wrong for what is visible. Retry rather than emit a bad sample.
    if any(px < l + 1 or px + pw > r - 1 or py < t or py + ph > b + 1 for px, py, pw, ph in placed):
        return render(text)

    # Pale-pink "faded" patches over parts of letters (like the H crossbar in B9H5L)
    if random.random() < 0.6:
        pale = (random.randint(225, 245), random.randint(150, 200), random.randint(160, 205))
        for _ in range(random.randint(1, 3)):
            px, py, pw, ph = random.choice(placed)
            cx, cy = px + random.randint(pw // 4, max(pw // 4, 3 * pw // 4)), py + random.randint(ph // 4, max(ph // 4, 3 * ph // 4))
            dx, dy = random.randint(2, max(3, pw // 3)), random.randint(-ph // 5, ph // 5)
            d.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=pale, width=random.randint(2, 3))

    # Grey strike lines crossing the text band: straight or wavy, often spanning the box
    import math
    for _ in range(random.randint(1, 2) if spaced else random.randint(1, 3)):
        g = random.randint(120, 200)
        if spaced:
            x0, x1 = random.randint(l, l + W // 8), random.randint(int(W * 0.75), W)
        else:
            x0, x1 = random.randint(l, W // 3), random.randint(W // 3, W)
        y0, y1 = random.randint(t + 4, b - 4), random.randint(t + 4, b - 4)
        if random.random() < 0.4:  # wavy
            amp, per, ph0 = random.uniform(0.5, 1.8), random.uniform(15, 40), random.uniform(0, 6.3)
            pts = [(xx, y0 + (y1 - y0) * (xx - x0) / max(1, x1 - x0) + amp * math.sin(xx / per * 6.28 + ph0))
                   for xx in range(x0, x1, 2)]
            d.line(pts, fill=(g, g, g), width=1)
        else:
            d.line([(x0, y0), (x1, y1)], fill=(g, g, g), width=random.randint(1, 2))

    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.8)))
    return img


_REGULAR_FONTS = [f for f in [
    r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\verdana.ttf", r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\calibri.ttf",
] if os.path.exists(f)]


def render_plain(text):
    """Style of samples/SZ6ML.png: thin magenta text in a bordered white box on a pale background."""
    W, H = random.randint(140, 220), random.randint(60, 100)
    base = (200, 216, 183) if random.random() < 0.5 else tuple(random.randint(150, 235) for _ in range(3))
    bg = tuple(max(0, min(255, c + random.randint(-15, 15))) for c in base)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    bw, bh = random.randint(int(W * 0.6), W - 4), random.randint(int(H * 0.45), int(H * 0.8))
    bx, by = random.randint(0, W - bw - 2), random.randint(0, H - bh - 2)
    border = tuple(random.randint(40, 110) for _ in range(3))
    bt = random.randint(1, 2)
    d.rectangle([bx - bt, by - bt, bx + bw + bt, by + bh + bt + random.randint(0, 4)], fill=border)  # border + shadow
    d.rectangle([bx, by, bx + bw, by + bh], fill=(255, 255, 255))

    color = random.choice([
        (random.randint(180, 230), random.randint(0, 40), random.randint(90, 160)),   # magenta/pink
        tuple(random.randint(0, 160) for _ in range(3)),                             # any darkish colour
    ])
    f = _font(random.choice(_REGULAR_FONTS), random.randint(int(bh * 0.3), int(bh * 0.5)))
    spacing = random.randint(-1, 3)
    widths = [f.getbbox(c)[2] - f.getbbox(c)[0] for c in text]
    total = sum(widths) + spacing * (len(text) - 1)
    if total > bw - 6:
        f = _font(f.path, max(10, int(f.size * (bw - 6) / total)))
        widths = [f.getbbox(c)[2] - f.getbbox(c)[0] for c in text]
        total = sum(widths) + spacing * (len(text) - 1)
    x = bx + random.randint(3, max(3, bw - total - 3))
    asc = f.getbbox("H")
    y = by + (bh - (asc[3] - asc[1])) // 2 - asc[1] + random.randint(-3, 3)
    for c, w in zip(text, widths):
        d.text((x - f.getbbox(c)[0], y + random.randint(-1, 1)), c, font=f, fill=color)
        x += w + spacing
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 0.6)))
    return img


def augment(img):
    """Generic photometric/geometric noise so the model tolerates small real-world differences."""
    import io
    import numpy as np
    if random.random() < 0.3:                                   # slight affine warp (shear/scale)
        W, H = img.size
        sx, sy = random.uniform(-0.15, 0.15), random.uniform(-0.04, 0.04)
        sc = random.uniform(0.95, 1.05)
        img = img.transform((W, H), Image.AFFINE, (sc, sx, -sx * H / 2, sy, sc, -sy * W / 2),
                            resample=Image.BICUBIC, fillcolor=img.getpixel((0, 0)))
    if random.random() < 0.4:                                   # contrast / brightness
        a = np.asarray(img).astype(np.float32)
        a = (a - 128) * random.uniform(0.85, 1.1) + 128 + random.uniform(-8, 8)
        img = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    if random.random() < 0.3:                                   # pixel noise
        a = np.asarray(img).astype(np.float32)
        a += np.random.normal(0, random.uniform(2, 7), a.shape)
        img = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    if random.random() < 0.3:                                   # JPEG artefacts
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(35, 90))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img


STYLES = {"bold": render, "plain": render_plain}


def sample(style=None, aug=True):
    t = random_text()
    style = style or random.choice(list(STYLES))
    img = STYLES[style](t)
    return (augment(img) if aug else img), t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="previews")
    ap.add_argument("--style", choices=list(STYLES), default=None, help="default: random mix")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    if a.seed is not None:
        random.seed(a.seed)
        import numpy as np
        np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    for i in range(a.n):
        im, t = sample(a.style)
        im.save(os.path.join(a.out, f"{t}_{i}.png"))
    print(f"wrote {a.n} images to {a.out}/")
