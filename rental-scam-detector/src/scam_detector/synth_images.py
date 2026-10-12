"""Генерация маленьких синтетических «фото комнат» Pillow-ом (для демо и тестов)."""

from __future__ import annotations

import io
import random

from PIL import Image, ImageDraw, ImageEnhance


def render_room(seed: int, size: tuple[int, int] = (160, 120)) -> Image.Image:
    """Рисует условную комнату: стена, пол, окно, диван, картина, лампа. Сид задаёт композицию."""
    rnd = random.Random(seed)
    w, h = size
    img = Image.new("RGB", size)
    d = ImageDraw.Draw(img)

    def color(lo: int = 40, hi: int = 230) -> tuple[int, int, int]:
        return tuple(rnd.randint(lo, hi) for _ in range(3))  # type: ignore[return-value]

    horizon = rnd.randint(int(h * 0.5), int(h * 0.75))
    wall, floor = color(120, 240), color(40, 160)
    # вертикальный градиент стены
    for y in range(horizon):
        k = y / max(horizon, 1)
        d.line([(0, y), (w, y)], fill=tuple(int(c * (0.8 + 0.2 * k)) for c in wall))
    d.rectangle([0, horizon, w, h], fill=floor)
    # окно
    wx, wy = rnd.randint(5, w // 2), rnd.randint(5, horizon // 3)
    ww, wh = rnd.randint(w // 6, w // 3), rnd.randint(horizon // 4, horizon // 2)
    d.rectangle([wx, wy, wx + ww, wy + wh], fill=(200, 225, 250), outline=(250, 250, 250), width=3)
    d.line([(wx + ww // 2, wy), (wx + ww // 2, wy + wh)], fill=(250, 250, 250), width=2)
    # диван
    sx = rnd.randint(0, w // 2)
    sw = rnd.randint(w // 3, w // 2)
    sh = rnd.randint(h // 8, h // 4)
    sofa = color(30, 200)
    d.rectangle([sx, horizon - sh, sx + sw, horizon + sh // 2], fill=sofa)
    d.rectangle([sx, horizon - sh - 8, sx + sw, horizon - sh], fill=tuple(c // 2 for c in sofa))
    # картина
    px, py = rnd.randint(w // 2, w - 30), rnd.randint(5, horizon // 2)
    d.rectangle([px, py, px + 22, py + 16], fill=color(), outline=(60, 40, 20), width=2)
    # лампа
    lx = rnd.randint(10, w - 10)
    d.ellipse([lx - 8, 2, lx + 8, 14], fill=(255, 240, 180))
    return img


def near_duplicate(img: Image.Image, seed: int = 0) -> Image.Image:
    """Типичная «перезаливка» мошенником: ресайз, лёгкая обрезка, яркость, JPEG-пережатие."""
    rnd = random.Random(seed)
    w, h = img.size
    out = img.resize((int(w * 0.85), int(h * 0.85)), Image.Resampling.BILINEAR)
    c = 2
    out = out.crop((c, c, out.width - c, out.height - c))
    out = ImageEnhance.Brightness(out).enhance(1.0 + rnd.uniform(0.03, 0.08))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=70)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
