"""Render docs/assets/rephix-splash.png from rephix.splash (needs Pillow).

Strokes are drawn as vectors from the logo grid rather than font glyphs, so
rows join cleanly regardless of a font's box-drawing metrics.
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

from rephix.splash import LOGO, X_COLUMN, X_GRADIENT, cell_color

# Arms from the cell center: u(p), d(own), l(eft), r(ight); half arms stop at center.
ARMS = {
    "┏": "rd",
    "┓": "ld",
    "┗": "ur",
    "┛": "ul",
    "━": "lr",
    "┃": "ud",
    "┣": "udr",
    "┫": "udl",
    "┳": "lrd",
    "╸": "l",
    "╹": "u",
    "╻": "d",
}
DIAGONALS = {"╲": ["\\"], "╱": ["/"], "╳": ["\\", "/"]}

S = 2  # Supersampling factor for crisp output.
cw, ch, stroke = 22 * S, 40 * S, 5 * S
pad_x, pad_y = 56 * S, 44 * S
columns = max(map(len, LOGO))
width = cw * columns + 2 * pad_x
height = ch * len(LOGO) + 2 * pad_y

image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
draw = ImageDraw.Draw(image)
# The X is drawn into a mask, then filled with a smooth vertical gradient.
x_mask = Image.new("L", (width, height), 0)
x_draw = ImageDraw.Draw(x_mask)
draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=18 * S, fill="#0f1418")
h = stroke // 2
for row, line in enumerate(LOGO):
    for col, char in enumerate(line):
        x0, y0 = pad_x + col * cw, pad_y + row * ch
        cx, cy = x0 + cw // 2, y0 + ch // 2
        pen, fill = (x_draw, 255) if col >= X_COLUMN else (draw, cell_color(row, col))
        if char in ARMS:
            pen.rectangle((cx - h, cy - h, cx + h, cy + h), fill=fill)
            for arm in ARMS[char]:
                box = {
                    "u": (cx - h, y0, cx + h, cy),
                    "d": (cx - h, cy, cx + h, y0 + ch),
                    "l": (x0, cy - h, cx, cy + h),
                    "r": (cx, cy - h, x0 + cw, cy + h),
                }[arm]
                pen.rectangle(box, fill=fill)
        for kind in DIAGONALS.get(char, []):
            a, b = (
                ((x0, y0), (x0 + cw, y0 + ch)) if kind == "\\" else ((x0 + cw, y0), (x0, y0 + ch))
            )
            # Clip to the band the box strokes occupy (top-row to bottom-row centers).
            top, bottom = pad_y + ch // 2, pad_y + ch * len(LOGO) - ch // 2
            ya, yb = max(a[1], top), min(b[1], bottom)
            if ya < yb:
                x_at = lambda y: a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1])  # noqa: E731
                pen.line(((x_at(ya), ya), (x_at(yb), yb)), fill=fill, width=stroke)

top, bottom = pad_y + ch // 2, pad_y + ch * len(LOGO) - ch // 2
gradient = Image.new("RGBA", (width, height))
stops = [tuple(int(c[i : i + 2], 16) for i in (1, 3, 5)) for c in (X_GRADIENT[0], X_GRADIENT[-1])]
for y in range(height):
    t = min(max((y - top) / (bottom - top), 0), 1)
    rgb = tuple(round(a + (b - a) * t) for a, b in zip(*stops))
    gradient.paste((*rgb, 255), (0, y, width, y + 1))
image.paste(gradient, (0, 0), x_mask)
image = image.resize((width // S, height // S), Image.LANCZOS)
out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("rephix-splash.png"))
image.save(out, optimize=True)
print(out, image.size)
