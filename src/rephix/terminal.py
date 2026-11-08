"""Optional terminal workbench; subprocesses use the public CLI without a shell.

Layout (wide)::

    rephix 0.3.0                                              ○ Ready
    ──────────────────────────────────────────────────────────────────
     / filter commands       │  validate
                             │  Validate evidence JSONL and report …
     ▌ validate              │
       split                 │  ❯ rephix validate evidence.jsonl
       collect               │
                             │  Output ─────────────────────────────
     3 of 35                 │  $ rephix validate evidence.jsonl
    ──────────────────────────────────────────────────────────────────
     ↑↓ choose   ⏎ open   F1 options   ctrl+c quit

Below ``NARROW`` columns the list and the command page become two views.
"""

from __future__ import annotations

import asyncio
import codecs
import os
import shlex
import signal
import sys
import textwrap
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, field

from . import __version__

NARROW = 80
MIN_COLUMNS, MIN_ROWS = 48, 12
SIDEBAR = 30
OUTPUT_LIMIT = 200_000


# ── Domain helpers (pure, no toolkit) ─────────────────────────────────────────


def display_text(value):
    """Remove terminal controls, including OSC/CSI introducers, from display data."""
    return "".join(
        char if char in "\n\t" or unicodedata.category(char) not in {"Cc", "Cs"} else "?"
        for char in str(value)
    )


def command_catalog(parser):
    import argparse

    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    summaries = {a.dest: a.help for a in action._choices_actions}
    return [
        (name, summaries.get(name, ""), child.format_help())
        for name, child in action.choices.items()
        if name != "ui"
    ]


def command_help(parser):
    """Return ``render(name, width)``: a subcommand's help wrapped to ``width`` columns."""
    import argparse
    import functools

    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))

    def render(name, width):
        child = action.choices[name]
        original = child.formatter_class
        child.formatter_class = functools.partial(original, width=max(width, 40))
        try:
            return child.format_help()
        finally:
            child.formatter_class = original

    return render


def command_arguments(text):
    parts = shlex.split(text, posix=True)
    if not parts:
        raise ValueError("Enter a Rephix command first.")
    if "ui" == parts[0]:
        raise ValueError("Open commands here; a nested workbench is not supported.")
    return [sys.executable, "-m", "rephix", *parts]


def filter_commands(catalog, query):
    """Indices matching every query word; name-prefix matches rank first."""
    words = query.lower().split()
    if not words:
        return list(range(len(catalog)))
    hits = []
    for index, (name, summary, _) in enumerate(catalog):
        haystack = f"{name} {summary}".lower()
        if all(word in haystack for word in words):
            rank = 0 if name.startswith(words[0]) else 1 if words[0] in name else 2
            hits.append((rank, index))
    return [index for _, index in sorted(hits)]


def sentence(text):
    text = (text or "").strip()
    return text[:1].upper() + text[1:] if text else "No description."


def duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}m {seconds:02d}s"


@dataclass
class WorkbenchState:
    """Single source of truth for UI state; the toolkit layer only renders it."""

    catalog: list
    query: str = ""
    matches: list = field(default_factory=list)
    selected: int | None = 0  # Index into ``catalog``; None when nothing matches.
    drafts: dict = field(default_factory=dict)
    view: str = "browse"  # Narrow screens only: "browse" or "command".
    show_help: bool = False
    phase: str = "ready"  # ready · running · stopping · done · failed · error
    running: bool = False
    exit_code: int | None = None
    started: float = 0.0
    elapsed: float = 0.0
    notice: str = ""

    def __post_init__(self):
        self.matches = filter_commands(self.catalog, self.query)
        self.selected = self.matches[0] if self.matches else None

    @property
    def name(self):
        return None if self.selected is None else self.catalog[self.selected][0]

    def set_query(self, query):
        self.query = query
        self.matches = filter_commands(self.catalog, query)
        # Like a fuzzy finder: the best match is always the one Enter opens.
        self.selected = self.matches[0] if self.matches else None

    def move(self, delta):
        if not self.matches:
            return
        position = self.matches.index(self.selected) if self.selected in self.matches else 0
        position = max(0, min(len(self.matches) - 1, position + delta))
        self.selected = self.matches[position]

    def status(self, now=None):
        """(style, text) for the header status chip; symbols never rely on color."""
        if self.phase == "running":
            seconds = (now if now is not None else time.monotonic()) - self.started
            return "class:status.run", f"● Running {self.name or ''} · {int(seconds)}s"
        return {
            "stopping": ("class:status.warn", "■ Stopping…"),
            "done": ("class:status.ok", f"✓ Done · {duration(self.elapsed)}"),
            "failed": ("class:status.err", f"✗ Exit {self.exit_code} · {duration(self.elapsed)}"),
            "error": ("class:status.err", "✗ Could not start"),
        }.get(self.phase, ("class:muted", "○ Ready"))


PALETTES = {
    # Foreground-only palettes: the terminal keeps its own background.
    "dark": {
        "accent": "#7dd3c0",
        "text": "#e2e8ec",
        "muted": "#8494a2",
        "faint": "#4f5d68",
        "select": "bg:#1d2b33",
        "ok": "#8fd694",
        "err": "#f08a8a",
        "warn": "#e8c27a",
    },
    "light": {
        "accent": "#0f766e",
        "text": "#1f2c35",
        "muted": "#5b6b77",
        "faint": "#b4c0c8",
        "select": "bg:#e3efec",
        "ok": "#2f7d3a",
        "err": "#b42323",
        "warn": "#8a5a00",
    },
}


def build_style(theme, monochrome):
    from prompt_toolkit.styles import Style

    if monochrome:
        return Style.from_dict(
            {
                "brand": "bold",
                "title": "bold",
                "selected": "reverse bold",
                "key": "bold",
                "status.run": "bold",
                "status.err": "bold",
                "out.cmd": "bold",
                "out.err": "bold",
                "placeholder": "italic",
            }
        )
    p = PALETTES[theme]
    return Style.from_dict(
        {
            "": p["text"],
            "brand": f"{p['accent']} bold",
            "version": p["muted"],
            "title": f"{p['text']} bold",
            "accent": p["accent"],
            "muted": p["muted"],
            "faint": p["faint"],
            "rule": p["faint"],
            "selected": f"{p['select']} {p['text']} bold",
            "selected.bar": f"{p['select']} {p['accent']}",
            "match": f"{p['accent']} underline",
            "key": f"{p['text']} bold",
            "placeholder": f"{p['faint']} italic",
            "notice": p["err"],
            "status.run": f"{p['accent']} bold",
            "status.ok": p["ok"],
            "status.err": f"{p['err']} bold",
            "status.warn": p["warn"],
            "out.cmd": f"{p['accent']} bold",
            "out.ok": p["ok"],
            "out.err": p["err"],
            "out.warn": p["warn"],
            "text-area.prompt": p["accent"],
            "scrollbar.background": "",
            "scrollbar.button": f"bg:{p['faint']}",
            "too-small": p["muted"],
        }
    )


# ── Toolkit view ──────────────────────────────────────────────────────────────


def build_application(
    catalog, *, color="auto", theme="dark", input=None, output=None, help_for=None
):
    # Import only on explicit entry; core installations stay NumPy-only.
    from prompt_toolkit import Application
    from prompt_toolkit.application.current import get_app
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.document import Document
    from prompt_toolkit.filters import Condition, has_focus
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, VSplit, Window, WindowAlign
    from prompt_toolkit.layout.containers import DynamicContainer
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.layout.margins import ScrollbarMargin
    from prompt_toolkit.layout.processors import Processor, Transformation
    from prompt_toolkit.lexers import Lexer
    from prompt_toolkit.output import ColorDepth
    from prompt_toolkit.output.defaults import create_output
    from prompt_toolkit.widgets import TextArea

    state = WorkbenchState(catalog)
    process = None
    keys = KeyBindings()

    class Placeholder(Processor):
        def __init__(self, text):
            self.text = text

        def apply_transformation(self, ti):
            if ti.lineno == 0 and not ti.document.text:
                return Transformation([*ti.fragments, ("class:placeholder", self.text())])
            return Transformation(ti.fragments)

    class OutputLexer(Lexer):
        styles = {"$ ": "class:out.cmd", "✓ ": "class:out.ok", "✗ ": "class:out.err"}

        def lex_document(self, document):
            lines = document.lines

            def line(number):
                text = lines[number]
                for prefix, style in self.styles.items():
                    if text.startswith(prefix):
                        return [(style, text)]
                if text.startswith("■ "):
                    return [("class:out.warn", text)]
                return [("", text)]

            return line

    search = TextArea(
        multiline=False,
        prompt=[("class:faint", "/ ")],
        input_processors=[Placeholder(lambda: "filter commands")],
    )
    editor = TextArea(
        multiline=False,
        prompt=lambda: [
            ("class:accent", "❯ "),
            ("class:muted", f"rephix {state.name or ''} "),
        ],
        input_processors=[Placeholder(lambda: "add arguments…")],
        accept_handler=lambda buffer: (start_run(), True)[1],
    )
    result = TextArea(read_only=True, wrap_lines=True, lexer=OutputLexer())
    help_view = TextArea(read_only=True, wrap_lines=True)
    for area in (result, help_view):
        area.window.right_margins = [ScrollbarMargin(display_arrows=False)]

    # ── State transitions ────────────────────────────────────────────────────

    def select(update):
        before = state.selected
        if before is not None:
            state.drafts[before] = editor.text
        update()
        if state.selected != before:
            editor.text = state.drafts.get(state.selected, "")
            editor.buffer.cursor_position = len(editor.text)
            state.notice = ""
            refresh_help()

    help_key = None

    def help_width():
        # The lower pane's width, less one column for the scrollbar.
        columns = get_app().output.get_size().columns
        return columns - 4 if narrow() else columns - SIDEBAR - 8

    def refresh_help():
        nonlocal help_key
        if not state.show_help or state.selected is None:
            return
        key = (state.selected, help_width())
        if key == help_key:
            return
        help_key = key
        name, _, text = catalog[state.selected]
        if help_for is not None:
            text = help_for(name, key[1])
        help_view.text = display_text(text)

    search.buffer.on_text_changed += lambda buffer: select(lambda: state.set_query(buffer.text))
    editor.buffer.on_text_changed += lambda buffer: setattr(state, "notice", "")

    def narrow():
        return get_app().output.get_size().columns < NARROW

    def lower_pane():
        return help_view if state.show_help else result

    def go(target):
        # Set the narrow view first so the target is part of the visible tree.
        state.view = "browse" if target is search else "command"
        app = get_app()
        try:
            app.layout.focus(target)
        except ValueError:
            pass  # The "too small" screen is showing; the view applies once it fits.
        # A lone Escape is flushed by a timer, which does not schedule a redraw.
        app.invalidate()

    def append(text):
        # Retain a bounded viewport. The subprocess can save full reports through --output.
        following = result.buffer.cursor_position == len(result.text)
        old_position = result.buffer.cursor_position
        value = result.text + display_text(text)
        removed = max(0, len(value) - OUTPUT_LIMIT)
        visible = value[-OUTPUT_LIMIT:]
        cursor = len(visible) if following else max(0, old_position - removed)
        result.buffer.set_document(Document(visible, cursor_position=cursor), bypass_readonly=True)
        get_app().invalidate()

    def interrupt():
        if os.name == "nt":
            process.terminate()
        else:
            process.send_signal(signal.SIGINT)

    async def tick():
        while state.running:
            get_app().invalidate()
            await asyncio.sleep(1)

    async def execute(command_text):
        nonlocal process
        state.started = time.monotonic()
        prefix = "\n" if result.text else ""
        append(f"{prefix}$ rephix {command_text}\n")
        try:
            process = await asyncio.create_subprocess_exec(
                *command_arguments(command_text),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.DEVNULL,
            )
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            while chunk := await process.stdout.read(8192):
                append(decoder.decode(chunk))
            tail = decoder.decode(b"", final=True)
            code = await process.wait()
            stopped = state.phase == "stopping"
            state.exit_code = code
            state.elapsed = time.monotonic() - state.started
            state.phase = "done" if code == 0 else "failed"
            if tail or not result.text.endswith("\n"):
                tail += "\n"
            mark = "✓ Done" if code == 0 else "■ Stopped" if stopped else "✗ Failed"
            append(f"{tail}{mark} · exit {code} · {duration(state.elapsed)}\n")
        except OSError as exc:
            state.phase = "error"
            append(f"✗ Could not start command: {exc}\n")
        finally:
            if process is not None and process.returncode is None:
                interrupt()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            process = None
            state.running = False
            get_app().invalidate()

    def start_run():
        if state.running or state.selected is None:
            return
        # Capture the arguments at the keypress; later edits only affect a future run.
        text = f"{state.name} {editor.text}".strip()
        try:
            command_arguments(text)
        except ValueError as exc:
            state.notice = f"✗ {exc}"
            return
        state.running, state.phase, state.show_help = True, "running", False
        app = get_app()
        app.create_background_task(execute(text))
        app.create_background_task(tick())

    # ── Key bindings ─────────────────────────────────────────────────────────

    in_search = has_focus(search)
    in_editor = has_focus(editor)

    @keys.add("up", filter=in_search)
    @keys.add("c-p", filter=in_search)
    def previous(event):
        select(lambda: state.move(-1))

    @keys.add("down", filter=in_search)
    @keys.add("c-n", filter=in_search)
    def next_command(event):
        select(lambda: state.move(1))

    @keys.add("pageup", filter=in_search)
    def page_up(event):
        select(lambda: state.move(-10))

    @keys.add("pagedown", filter=in_search)
    def page_down(event):
        select(lambda: state.move(10))

    @keys.add("enter", filter=in_search)
    def open_command(event):
        if state.selected is not None:
            go(editor)
            editor.buffer.cursor_position = len(editor.text)

    @keys.add("tab")
    def focus_next(event):
        order = [search, editor, lower_pane()]
        current = next((w for w in order if event.app.layout.has_focus(w)), search)
        go(order[(order.index(current) + 1) % len(order)])

    @keys.add("s-tab")
    def focus_previous(event):
        order = [search, editor, lower_pane()]
        current = next((w for w in order if event.app.layout.has_focus(w)), search)
        go(order[order.index(current) - 1])

    @keys.add("f1")
    def help_toggle(event):
        if state.selected is None:
            return
        state.show_help = not state.show_help
        refresh_help()
        go(help_view if state.show_help else editor)

    @keys.add("escape")
    def back(event):
        if state.show_help:
            state.show_help = False
            go(editor)
        elif event.app.layout.has_focus(result):
            go(editor)
        elif in_search() and search.text:
            search.text = ""
        else:
            go(search)

    @keys.add("f5")
    def run(event):
        start_run()

    @keys.add("c-c")
    def stop(event):
        if process is not None and process.returncode is None:
            interrupt()
            state.phase = "stopping"
        elif not state.running:
            event.app.exit()

    @keys.add("c-d", filter=~in_editor & ~in_search)
    def quit_app(event):
        if state.running:
            state.notice = "Command still running · ctrl+c stops it first"
        else:
            event.app.exit()

    @keys.add("c-z")
    def suspend(event):
        if os.name != "nt":
            event.app.suspend_to_background()

    # ── Components ───────────────────────────────────────────────────────────

    def text(fragments, **kwargs):
        return Window(FormattedTextControl(fragments), **kwargs)

    def blank(height=1):
        return Window(height=height)

    def gap(width):
        return Window(width=width)

    def rule():
        return Window(char="─", height=1, style="class:rule")

    def section(label):
        return VSplit(
            [
                text(lambda: [("class:muted", label() + " ")], dont_extend_width=True),
                Window(char="─", height=1, style="class:rule"),
            ],
            height=1,
        )

    def command_rows():
        if not state.matches:
            return [("class:muted", "  No matches\n"), ("class:faint", "  esc clears the filter")]
        width = SIDEBAR - 2
        rows = []
        for index in state.matches:
            name = catalog[index][0]
            label = name if len(name) <= width else name[: width - 1] + "…"
            if index == state.selected:
                rows += [
                    ("class:selected.bar", "▌ "),
                    ("class:selected", label.ljust(width)),
                    ("", "\n"),
                ]
            else:
                rows.append(("", f"  {label}\n"))
        return rows

    command_list = Window(
        FormattedTextControl(
            command_rows,
            get_cursor_position=lambda: Point(
                0, state.matches.index(state.selected) if state.selected in state.matches else 0
            ),
        ),
        always_hide_cursor=True,
    )

    sidebar = HSplit(
        [
            search,
            blank(),
            command_list,
            blank(),
            text(
                lambda: [("class:faint", f"  {len(state.matches)} of {len(catalog)} commands")],
                height=1,
            ),
        ]
    )

    def lower_label():
        return f"Options · {state.name}" if state.show_help else "Output"

    def notice_line():
        return [("class:notice", "  " + state.notice)] if state.notice else []

    def empty_or_result():
        if state.show_help:
            refresh_help()  # Re-wrap after a resize; cached per command and width.
            return help_view
        if result.text:
            return result
        return text(
            lambda: [
                ("class:faint", "Nothing has run yet.\n"),
                ("class:faint", "Type arguments above and press ⏎ to run."),
            ]
        )

    detail = HSplit(
        [
            text(lambda: [("class:title", state.name or "")], height=1),
            text(
                lambda: [
                    (
                        "class:muted",
                        textwrap.fill(sentence(catalog[state.selected][1]), help_width() + 1),
                    )
                ],
                height=Dimension(min=1, max=3),
                wrap_lines=True,
                dont_extend_height=True,
            ),
            blank(),
            editor,
            text(notice_line, height=1),
            blank(),
            section(lower_label),
            DynamicContainer(empty_or_result),
        ]
    )

    def no_match():
        return HSplit(
            [
                text([("class:title", "No command selected")], height=1),
                text(
                    lambda: [("class:muted", f'Nothing matches "{state.query}". Esc clears it.')],
                    height=1,
                ),
            ]
        )

    def main_pane():
        return detail if state.selected is not None else no_match()

    def body():
        if narrow():
            pane = sidebar if state.view == "browse" else DynamicContainer(main_pane)
            return VSplit([gap(1), pane, gap(1)])
        return VSplit(
            [
                gap(1),
                HSplit([sidebar], width=SIDEBAR),
                gap(1),
                Window(width=1, char="│", style="class:rule"),
                gap(2),
                DynamicContainer(main_pane),
                gap(1),
            ]
        )

    def header_left():
        crumbs = [("class:brand", " rephix"), ("class:version", f" {__version__}")]
        if narrow() and state.view == "command" and state.name:
            crumbs += [("class:faint", "  ›  "), ("class:muted", state.name)]
        return crumbs

    header = VSplit(
        [
            text(header_left),
            text(lambda: [state.status(), ("", " ")], align=WindowAlign.RIGHT),
        ],
        height=1,
    )

    def hints():
        if state.running:
            pairs = [("ctrl+c", "stop"), ("tab", "focus")]
        elif in_search():
            pairs = [("↑↓", "choose"), ("⏎", "open"), ("F1", "options"), ("ctrl+c", "quit")]
        elif in_editor():
            pairs = [("⏎", "run"), ("esc", "back"), ("tab", "output"), ("F1", "options")]
        else:
            pairs = [("↑↓", "scroll"), ("esc", "back"), ("tab", "next"), ("ctrl+c", "quit")]
        fragments = [("", " ")]
        for key, label in pairs:
            fragments += [("class:key", key), ("class:muted", f" {label}   ")]
        return fragments

    screen = HSplit(
        [
            header,
            rule(),
            blank(),
            DynamicContainer(body),
            rule(),
            text(hints, height=1),
        ]
    )

    def too_small():
        return HSplit(
            [
                Window(),
                text(
                    [
                        ("class:title", "Terminal too small\n"),
                        ("class:too-small", f"Resize to at least {MIN_COLUMNS}×{MIN_ROWS}\n"),
                        ("class:too-small", "or use: rephix ui --plain"),
                    ],
                    align=WindowAlign.CENTER,
                    height=3,
                ),
                Window(),
            ]
        )

    @Condition
    def fits():
        size = get_app().output.get_size()
        return size.columns >= MIN_COLUMNS and size.rows >= MIN_ROWS

    root = DynamicContainer(lambda: screen if fits() else too_small())

    # ── Color and application ────────────────────────────────────────────────

    monochrome = color == "never" or (color == "auto" and bool(os.environ.get("NO_COLOR")))
    color_depth = ColorDepth.DEPTH_1_BIT if monochrome else None
    if color == "always":
        output = output if output is not None else create_output()
        color_depth = output.get_default_color_depth()
        if color_depth == ColorDepth.DEPTH_1_BIT:
            # NO_COLOR is also consumed by the toolkit's output factory. An
            # explicit override needs a portable ANSI fallback at that layer.
            color_depth = ColorDepth.DEPTH_4_BIT
    app = Application(
        layout=Layout(root, focused_element=search),
        key_bindings=keys,
        style=build_style(theme, monochrome),
        full_screen=True,
        mouse_support=False,
        color_depth=color_depth,
        input=input,
        output=output,
    )
    # Escape is also a prefix of Alt+key; those arrive in one read, so a short wait keeps
    # a lone Escape responsive without breaking Alt bindings.
    app.ttimeoutlen, app.timeoutlen = 0.05, 0.15
    # Publicly returned state makes transitions and in-flight behavior testable.
    return app, state


# ── Terminal lifecycle and CLI entry ──────────────────────────────────────────


@contextmanager
def _terminal_state(stream):
    """Restore the complete POSIX mode snapshot around toolkit setup and execution."""
    if os.name == "nt":
        yield
        return
    import termios

    descriptor = stream.fileno()
    try:
        saved = termios.tcgetattr(descriptor)
    except termios.error as exc:
        raise RuntimeError(f"Cannot read terminal settings: {exc}") from exc
    try:
        yield
    finally:
        try:
            termios.tcsetattr(descriptor, termios.TCSANOW, saved)
        except termios.error as exc:
            raise RuntimeError(f"Cannot restore terminal settings: {exc}") from exc


def launch(args):
    if args.plain:
        from .cli import build_parser

        return {
            "version": __version__,
            "commands": [
                {"command": name, "description": summary}
                for name, summary, _ in command_catalog(build_parser())
            ],
        }
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError(
            "The workbench needs an interactive terminal. Use rephix ui --plain or --help."
        )
    with _terminal_state(sys.stdin):
        try:
            from .cli import build_parser

            parser = build_parser()
            app, _ = build_application(
                command_catalog(parser),
                color=args.color,
                theme=args.theme,
                help_for=command_help(parser),
            )
        except ImportError as exc:
            raise ImportError(
                "Install the terminal interface with: python -m pip install 'rephix[tui]'"
            ) from exc
        app.run()


def add_subcommands(commands):
    ui = commands.add_parser("ui", help="open the interactive command workbench")
    ui.add_argument("--plain", action="store_true", help="print a linear command catalog as JSON")
    ui.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    ui.add_argument("--theme", choices=("dark", "light"), default="dark")
    ui.set_defaults(func=launch)
