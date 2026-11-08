"""Compact wordmark shared by the CLI welcome, the workbench and the README image."""

from __future__ import annotations

import os

from . import __version__

LOGO = (
    "┏━┓ ┏━╸ ┏━┓ ╻ ╻ ╻ ╲ ╱",
    "┣┳┛ ┣╸  ┣━┛ ┣━┫ ┃  ╳ ",
    "╹┗╸ ┗━╸ ╹   ╹ ╹ ╹ ╱ ╲",
)
TAGLINE = "evidence in · understanding out"

# Teal to blue, left to right; tuned to read on dark and light backgrounds.
GRADIENT = {
    "dark": ((0x7D, 0xD3, 0xC0), (0x7A, 0x9C, 0xF0)),
    "light": ((0x0F, 0x76, 0x6E), (0x2B, 0x4F, 0xB8)),
}


def column_colors(theme="dark"):
    """One ``#rrggbb`` per logo column."""
    (r0, g0, b0), (r1, g1, b1) = GRADIENT[theme]
    width = max(len(line) for line in LOGO)
    colors = []
    for x in range(width):
        t = x / max(width - 1, 1)
        rgb = (round(a + (b - a) * t) for a, b in ((r0, r1), (g0, g1), (b0, b1)))
        colors.append("#" + "".join(f"{c:02x}" for c in rgb))
    return colors


def fragments(theme="dark", indent=""):
    """prompt_toolkit formatted text: gradient logo, then a muted tagline."""
    colors = column_colors(theme)
    result = []
    for line in LOGO:
        result.append(("", indent))
        result += [(f"{colors[x]} bold", char) for x, char in enumerate(line)]
        result.append(("", "\n"))
    result += [("", "\n" + indent), ("class:muted", f"{TAGLINE}  ·  v{__version__}")]
    return result


def render(color=True, theme="dark", indent="  "):
    """ANSI (truecolor) or plain text for terminals."""
    if not color:
        lines = [indent + line for line in LOGO]
        return "\n".join([*lines, "", f"{indent}{TAGLINE}  ·  v{__version__}"])
    colors = column_colors(theme)
    out = []
    for line in LOGO:
        cells = []
        for x, char in enumerate(line):
            r, g, b = (int(colors[x][i : i + 2], 16) for i in (1, 3, 5))
            cells.append(f"\x1b[1;38;2;{r};{g};{b}m{char}" if char != " " else " ")
        out.append(indent + "".join(cells) + "\x1b[0m")
    out += ["", f"{indent}\x1b[2m{TAGLINE}  ·  v{__version__}\x1b[0m"]
    return "\n".join(out)


def use_color(stream):
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    return hasattr(stream, "isatty") and stream.isatty()


def welcome():
    """Shown for a bare ``rephix``: who we are and three ways in."""
    import sys

    return "\n".join(
        [
            "",
            render(color=use_color(sys.stdout)),
            "",
            "  rephix ui              open the interactive workbench",
            "  rephix demo            recover planted weights in seconds",
            "  rephix --help          list every command",
            "",
        ]
    )
