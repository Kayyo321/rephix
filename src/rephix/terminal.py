"""Optional terminal workbench; subprocesses use the public CLI without a shell."""

from __future__ import annotations

import asyncio
import codecs
import os
import shlex
import signal
import sys
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass

from . import __version__


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


def command_arguments(text):
    parts = shlex.split(text, posix=True)
    if not parts:
        raise ValueError("Enter a Rephix command first.")
    if "ui" == parts[0]:
        raise ValueError("Open commands here; a nested workbench is not supported.")
    return [sys.executable, "-m", "rephix", *parts]


@dataclass
class WorkbenchState:
    selected: int = 0
    running: bool = False
    status: str = "Ready  /  Choose a command, review its arguments, then F5 to run"

    def move(self, count, size):
        self.selected = (self.selected + count) % max(size, 1)


def build_application(catalog, *, color="auto", theme="dark", input=None, output=None):
    # Import only on explicit entry; core installations stay NumPy-only.
    from prompt_toolkit import Application
    from prompt_toolkit.application.current import get_app
    from prompt_toolkit.document import Document
    from prompt_toolkit.filters import has_focus
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
    from prompt_toolkit.layout.containers import DynamicContainer
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.output import ColorDepth
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import Frame, TextArea

    state = WorkbenchState()
    process = None
    keys = KeyBindings()
    editor = TextArea(height=2, multiline=False, prompt="rephix > ", text=catalog[0][0])
    result = TextArea(
        text="Run output appears here. No command has been started.\n",
        read_only=True,
        scrollbar=True,
        wrap_lines=True,
    )
    help_view = TextArea(text=catalog[0][2], read_only=True, scrollbar=True)
    show_help = False
    drafts = {}

    def navigation():
        rows = []
        for index, (name, _, _) in enumerate(catalog):
            style = "class:selected" if state.selected == index else "class:muted"
            marker = "> " if state.selected == index else "  "
            rows.append((style, marker + name + "\n"))
        return rows

    menu_control = FormattedTextControl(
        navigation, focusable=True, get_cursor_position=lambda: (0, state.selected)
    )
    # Point is needed by the renderer to scroll the selected row into view.
    from prompt_toolkit.data_structures import Point

    menu_control.get_cursor_position = lambda: Point(x=0, y=state.selected)
    menu = Window(menu_control, width=27, style="class:menu", always_hide_cursor=True)

    def select(delta):
        drafts[state.selected] = editor.text
        state.move(delta, len(catalog))
        name, _, help_text = catalog[state.selected]
        editor.text = drafts.get(state.selected, name)
        help_view.text = display_text(help_text)

    @keys.add("up", filter=has_focus(menu_control))
    @keys.add("k", filter=has_focus(menu_control))
    def previous(event):
        select(-1)

    @keys.add("down", filter=has_focus(menu_control))
    @keys.add("j", filter=has_focus(menu_control))
    def next_command(event):
        select(1)

    @keys.add("enter", filter=has_focus(menu_control))
    def edit(event):
        event.app.layout.focus(editor)
        editor.buffer.cursor_position = len(editor.text)

    @keys.add("tab")
    def focus_next(event):
        event.app.layout.focus_next()

    @keys.add("s-tab")
    def focus_previous(event):
        event.app.layout.focus_previous()

    @keys.add("f1")
    def help_toggle(event):
        nonlocal show_help
        show_help = not show_help
        try:
            command = shlex.split(editor.text)[0]
            help_view.text = next(
                (row[2] for row in catalog if row[0] == command),
                "Choose a command from the list or use --help.",
            )
        except (ValueError, IndexError):
            help_view.text = "Complete the command before opening help."
        event.app.layout.focus(help_view if show_help else editor)

    @keys.add("escape")
    def back(event):
        nonlocal show_help
        show_help = False
        event.app.layout.focus(editor if narrow() else menu_control)

    def append(text):
        # Retain a bounded viewport. The subprocess can save full reports through --output.
        following = result.buffer.cursor_position == len(result.text)
        old_position = result.buffer.cursor_position
        value = result.text + display_text(text)
        removed = max(0, len(value) - 200_000)
        visible = value[-200_000:]
        cursor = len(visible) if following else max(0, old_position - removed)
        result.buffer.set_document(
            Document(visible, cursor_position=cursor),
            bypass_readonly=True,
        )
        get_app().invalidate()

    async def execute(command_text):
        nonlocal process
        try:
            argv = command_arguments(command_text)
        except ValueError as exc:
            state.status = str(exc)
            state.running = False
            return
        state.status = "Running  /  Ctrl+C requests a stop; committed state remains recoverable"
        append("\n$ rephix " + command_text + "\n")
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.DEVNULL,
            )
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            while chunk := await process.stdout.read(8192):
                append(decoder.decode(chunk))
            append(decoder.decode(b"", final=True))
            code = await process.wait()
            state.status = (
                f"{'Complete' if code == 0 else 'Stopped'}  /  Exit {code}  /  F5 run again"
            )
        except OSError as exc:
            append(f"\nCould not start command: {exc}\n")
            state.status = "Could not start  /  Review the error in output"
        finally:
            if process is not None and process.returncode is None:
                if os.name == "nt":
                    process.terminate()
                else:
                    process.send_signal(signal.SIGINT)
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            process = None
            state.running = False
            get_app().invalidate()

    @keys.add("c-z")
    def suspend(event):
        if os.name != "nt":
            event.app.suspend_to_background()

    @keys.add("f5")
    def run(event):
        if not state.running:
            state.running = True
            event.app.create_background_task(execute(editor.text))

    @keys.add("c-c")
    def stop(event):
        if process is not None and process.returncode is None:
            if os.name == "nt":
                process.terminate()
            else:
                process.send_signal(signal.SIGINT)
            state.status = "Stop requested  /  Waiting for the command to exit"
        elif not state.running:
            event.app.exit()

    @keys.add("q", filter=has_focus(menu_control))
    @keys.add("c-d", filter=~has_focus(editor))
    def quit_app(event):
        if state.running:
            state.status = "Command still running  /  Ctrl+C to stop before exiting"
        else:
            event.app.exit()

    def narrow():
        return get_app().output.get_size().columns < 90

    wide = VSplit([Frame(menu, title="COMMANDS"), Frame(result, title="OUTPUT")], padding=1)
    body = DynamicContainer(
        lambda: (
            Frame(help_view, title="COMMAND HELP")
            if show_help
            else Frame(result, title="OUTPUT")
            if narrow()
            else wide
        )
    )
    header = Window(
        FormattedTextControl(
            [
                ("class:brand", "  REPHIX"),
                ("class:muted", f"  /  {__version__}   "),
                ("class:accent", "Evidence into understanding"),
            ]
        ),
        height=2,
    )
    summary = Window(
        FormattedTextControl(
            lambda: [
                ("class:accent", "  " + catalog[state.selected][0] + "  "),
                ("class:muted", catalog[state.selected][1]),
            ]
        ),
        height=2,
        wrap_lines=True,
    )
    footer = HSplit(
        [
            Window(
                FormattedTextControl(lambda: "  " + state.status), height=1, style="class:status"
            ),
            Window(
                FormattedTextControl(
                    "  Tab focus   Enter edit   F1 help   F5 run   Esc back   Ctrl+C stop/exit"
                ),
                height=1,
                style="class:muted",
            ),
        ]
    )
    layout = HSplit([header, summary, body, Frame(editor, title="COMMAND"), footer], padding=0)
    monochrome = color == "never" or (color == "auto" and bool(os.environ.get("NO_COLOR")))
    palette = {
        "dark": {
            "base": "#101820",
            "foreground": "#dce5ec",
            "focus": "#172733",
            "accent": "#68dfc4",
            "muted": "#a1b4c4",
            "selected": "#254b57",
            "border": "#476777",
            "status": "#203541",
            "bright": "#ffffff",
        },
        "light": {
            "base": "#f5f7f8",
            "foreground": "#1a303d",
            "focus": "#e5eef0",
            "accent": "#005c51",
            "muted": "#425d6c",
            "selected": "#c8e7df",
            "border": "#67818c",
            "status": "#d9e5e9",
            "bright": "#142c39",
        },
    }[theme]
    style = (
        Style.from_dict(
            {
                "": f"bg:{palette['base']} {palette['foreground']}",
                "brand": f"{palette['accent']} bold",
                "accent": palette["accent"],
                "muted": palette["muted"],
                "selected": f"bg:{palette['selected']} {palette['bright']} bold",
                "frame.border": palette["border"],
                "frame.label": palette["muted"],
                "status": f"bg:{palette['status']} {palette['bright']}",
                "text-area": f"bg:{palette['base']} {palette['foreground']}",
                "text-area focused": f"bg:{palette['focus']} {palette['bright']}",
            }
        )
        if not monochrome
        else Style.from_dict({"selected": "reverse bold", "brand": "bold"})
    )
    app = Application(
        layout=Layout(layout, focused_element=editor),
        key_bindings=keys,
        style=style,
        full_screen=True,
        mouse_support=False,
        color_depth=ColorDepth.DEPTH_1_BIT if monochrome else None,
        input=input,
        output=output,
    )
    # Publicly returned state makes transitions and in-flight behavior testable.
    return app, state


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

            app, _ = build_application(
                command_catalog(build_parser()), color=args.color, theme=args.theme
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
