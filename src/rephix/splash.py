"""Compact wordmark shared by the CLI welcome, the workbench and the README image."""

from __future__ import annotations

import os

LOGO = (
    "┏━┓ ┏━╸ ┏━┓ ╻ ╻ ╻ ╲ ╱",
    "┣┳┛ ┣╸  ┣━┛ ┣━┫ ┃  ╳ ",
    "╹┗╸ ┗━╸ ╹   ╹ ╹ ╹ ╱ ╲",
)
X_COLUMN = 18  # The "X" occupies columns 18-20 of every row.
X_GRADIENT = ("#ff9a1f", "#ff5a1f", "#e5202a")  # Orange at the top to red at the bottom.
LETTERS = {"dark": "#ffffff", "light": "#1f2c35"}  # White, or ink on light backgrounds.


def cell_color(row, column, theme="dark"):
    return X_GRADIENT[row] if column >= X_COLUMN else LETTERS[theme]


def fragments(theme="dark", indent=""):
    """prompt_toolkit formatted text for the logo."""
    result = []
    for row, line in enumerate(LOGO):
        result.append(("", indent))
        result += [(f"{cell_color(row, x, theme)} bold", char) for x, char in enumerate(line)]
        result.append(("", "\n"))
    return result


def render(color=True, theme="dark", indent="  "):
    """ANSI (truecolor) or plain text for terminals."""
    if not color:
        return "\n".join(indent + line for line in LOGO)
    out = []
    for row, line in enumerate(LOGO):
        cells = []
        for x, char in enumerate(line):
            hex_color = cell_color(row, x, theme)
            r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
            cells.append(f"\x1b[1;38;2;{r};{g};{b}m{char}" if char != " " else " ")
        out.append(indent + "".join(cells) + "\x1b[0m")
    return "\n".join(out)


def _windows_vt(stream):
    """Enable VT processing on a Windows console; False if escapes would print raw.

    cmd.exe turns VT processing off before starting programs, so a classic console
    window shows raw escape codes unless the program enables it itself.
    """
    import ctypes
    from ctypes import wintypes

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        import msvcrt

        handle = msvcrt.get_osfhandle(stream.fileno())
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        if mode.value & 0x0004:  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            return True
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except (OSError, ValueError, AttributeError):
        return False


def use_color(stream):
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    if not (hasattr(stream, "isatty") and stream.isatty()):
        return False
    return os.name != "nt" or _windows_vt(stream)


def encodable(stream):
    """Whether the stream can carry the box-drawing wordmark (Windows pipes use ANSI pages)."""
    try:
        "".join(LOGO).encode(getattr(stream, "encoding", None) or "utf-8")
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def welcome():
    """Shown for a bare ``rephix``: who we are and three ways in."""
    import sys

    return "\n".join(
        [
            "",
            render(color=use_color(sys.stdout)) if encodable(sys.stdout) else "  rephix",
            "",
            "  rephix ui              open the interactive workbench",
            "  rephix demo            recover planted weights in seconds",
            "  rephix --help          list every command",
            "",
        ]
    )
