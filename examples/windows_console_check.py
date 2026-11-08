"""Native Windows console verification of an installed Rephix wheel.

Two independent console paths drive the *installed* stable launcher:

* ConPTY: pywinpty creates a pseudoconsole (the layer Windows Terminal uses); pyte
  renders the VT stream into a screen model that the checks read.
* conhost: a helper starts a classic console window (CREATE_NEW_CONSOLE), attaches
  to it and drives it with WriteConsoleInputW / GenerateConsoleCtrlEvent while
  reading the active screen buffer with ReadConsoleOutputCharacterW.

Nothing is redirected for interactive checks. Separate checks cover redirected and
noninteractive use. Every check is recorded; the script exits 1 if any fails.

Usage (Windows only):
  python examples/windows_console_check.py --wheel dist\\rephix-X-py3-none-any.whl \
      --work-dir C:\\abs\\new-dir --output result.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_PYTHON = getattr(sys, "_base_executable", sys.executable)
CHECKS = []
LONG_COMMAND = "demo --samples 600000 --features 240 --classes 60"


PROGRESS = None  # Incremental results survive a job timeout.
NOTES = []  # Observations of legacy behaviour that are not pass/fail requirements.
UNEXERCISED = []  # Checks the host could not drive; never counted as passed.


def not_exercised(name, reason, **evidence):
    UNEXERCISED.append({"check": name, "reason": reason, **evidence})
    print("NOT EXERCISED " + name + ": " + reason, flush=True)


def check(name, passed, **evidence):
    CHECKS.append({"check": name, "passed": bool(passed), **evidence})
    print(("PASS " if passed else "FAIL ") + name, flush=True)
    if PROGRESS is not None:
        PROGRESS.write_text(
            json.dumps({"checks": CHECKS, "notes": NOTES}, indent=1, ensure_ascii=False),
            encoding="utf-8",
        )
    return passed


def wait_until(predicate, timeout=30.0, interval=0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return predicate()


# ─── ConPTY backend ──────────────────────────────────────────────────────────


class ConPty:
    last_resize = None

    KEYS = {
        "enter": "\r",
        "up": "\x1b[A",
        "down": "\x1b[B",
        "f1": "\x1bOP",
        "esc": "\x1b",
        "tab": "\t",
        "shift-tab": "\x1b[Z",
        "ctrl-c": "\x03",
        "f5": "\x1b[15~",
        "backspace": "\x7f",
        "ctrl-a": "\x01",
        "ctrl-k": "\x0b",
    }

    def __init__(self, argv, rows=30, cols=110, cwd=None, env=None):
        import pyte
        from winpty import PtyProcess

        self.rows, self.cols = rows, cols
        self.screen = pyte.Screen(cols, rows)
        self.stream = pyte.Stream(self.screen)
        self.raw = []
        self.lock = threading.Lock()
        self.process = PtyProcess.spawn(argv, dimensions=(rows, cols), cwd=cwd, env=env)
        self.reader = threading.Thread(target=self._drain, daemon=True)
        self.reader.start()

    def _drain(self):
        while True:
            try:
                data = self.process.read(4096)
            except (EOFError, OSError):
                return
            if not data:
                if not self.process.isalive():
                    return
                time.sleep(0.01)
                continue
            with self.lock:
                self.raw.append(data)
                self.stream.feed(data)

    def text(self):
        with self.lock:
            return "\n".join(line.rstrip() for line in self.screen.display)

    def stream_text(self):
        with self.lock:
            return "".join(self.raw)

    def type(self, text):
        self.process.write(text)

    def key(self, name):
        self.process.write(self.KEYS[name])

    def interrupt(self):
        self.process.write("\x03")

    def resize(self, rows, cols):
        with self.lock:
            self.screen.resize(rows, cols)
        self.process.setwinsize(rows, cols)

    def alive(self):
        return self.process.isalive()

    def close(self):
        if self.process.isalive():
            self.process.terminate(force=True)


# ─── conhost backend (classic console window) ────────────────────────────────

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True) if os.name == "nt" else None


class COORD(ctypes.Structure):
    _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]


class SMALL_RECT(ctypes.Structure):
    _fields_ = [(n, ctypes.c_short) for n in ("Left", "Top", "Right", "Bottom")]


class CSBI(ctypes.Structure):
    _fields_ = [
        ("dwSize", COORD),
        ("dwCursorPosition", COORD),
        ("wAttributes", wintypes.WORD),
        ("srWindow", SMALL_RECT),
        ("dwMaximumWindowSize", COORD),
    ]


class KEY_EVENT_RECORD(ctypes.Structure):
    _fields_ = [
        ("bKeyDown", wintypes.BOOL),
        ("wRepeatCount", wintypes.WORD),
        ("wVirtualKeyCode", wintypes.WORD),
        ("wVirtualScanCode", wintypes.WORD),
        ("uChar", wintypes.WCHAR),
        ("dwControlKeyState", wintypes.DWORD),
    ]


class _EVENT(ctypes.Union):
    _fields_ = [("KeyEvent", KEY_EVENT_RECORD), ("padding", ctypes.c_byte * 16)]


class INPUT_RECORD(ctypes.Structure):
    _fields_ = [("EventType", wintypes.WORD), ("Event", _EVENT)]


VK = {
    "enter": (0x0D, "\r"),
    "up": (0x26, "\0"),
    "down": (0x28, "\0"),
    "f1": (0x70, "\0"),
    "esc": (0x1B, "\x1b"),
    "tab": (0x09, "\t"),
    "f5": (0x74, "\0"),
    "backspace": (0x08, "\b"),
}


class Conhost:
    """Drive a classic console window from this (console-less) process."""

    def __init__(self, argv, rows=30, cols=110, cwd=None, env=None):
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 4  # SW_SHOWNOACTIVATE: a real, resizable window without focus
        self.process = subprocess.Popen(
            argv,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            startupinfo=startup,
            cwd=cwd,
            env=env,
        )
        kernel32.FreeConsole()
        attached = wait_until(lambda: kernel32.AttachConsole(self.process.pid), 15)
        if not attached:
            raise OSError(f"AttachConsole failed: {ctypes.get_last_error()}")
        kernel32.SetConsoleCtrlHandler(None, True)  # this driver ignores Ctrl+C events
        self.last_resize = None
        self.interrupt_path = None
        # The VT alternate buffer cannot be resized directly; keep the main buffer.
        self.main = self._handle("CONOUT$")
        self.resize(rows, cols)

    def _handle(self, name):
        handle = kernel32.CreateFileW(name, 0xC0000000, 3, None, 3, 0, None)
        if handle in (-1, 0xFFFFFFFFFFFFFFFF):
            raise OSError(f"CreateFileW({name}) failed: {ctypes.get_last_error()}")
        return handle

    def text(self):
        out = self._handle("CONOUT$")
        try:
            info = CSBI()
            kernel32.GetConsoleScreenBufferInfo(out, ctypes.byref(info))
            window = info.srWindow
            width = window.Right - window.Left + 1
            lines = []
            for row in range(window.Top, window.Bottom + 1):
                buffer = ctypes.create_unicode_buffer(width)
                read = wintypes.DWORD()
                kernel32.ReadConsoleOutputCharacterW(
                    out, buffer, width, COORD(window.Left, row), ctypes.byref(read)
                )
                lines.append(buffer.value[: read.value].rstrip())
            return "\n".join(lines)
        finally:
            kernel32.CloseHandle(out)

    def _records(self, specs):
        records = (INPUT_RECORD * (2 * len(specs)))()
        for index, (vk, char, state) in enumerate(specs):
            for down in (0, 1):
                record = records[2 * index + down]
                record.EventType = 1
                key = record.Event.KeyEvent
                key.bKeyDown = not down
                key.wRepeatCount = 1
                key.wVirtualKeyCode = vk
                key.wVirtualScanCode = (
                    kernel32.MapVirtualKeyW(vk, 0) if hasattr(kernel32, "MapVirtualKeyW") else 0
                )
                key.uChar = char
                key.dwControlKeyState = state
        handle = self._handle("CONIN$")
        try:
            written = wintypes.DWORD()
            kernel32.WriteConsoleInputW(handle, records, len(records), ctypes.byref(written))
        finally:
            kernel32.CloseHandle(handle)

    def type(self, text):
        user32 = ctypes.WinDLL("user32")
        specs = []
        for char in text:
            scan = user32.VkKeyScanW(ord(char))
            vk = scan & 0xFF if scan != -1 else 0
            state = 0x0010 if scan != -1 and scan & 0x100 else 0  # SHIFT_PRESSED
            specs.append((vk, char, state))
        self._records(specs)

    def key(self, name):
        controls = {"ctrl-c": (0x43, "\x03"), "ctrl-a": (0x41, "\x01"), "ctrl-k": (0x4B, "\x0b")}
        if name in controls:
            self._records([(*controls[name], 0x0008)])  # LEFT_CTRL_PRESSED
        else:
            vk, char = VK[name]
            self._records([(vk, char, 0)])

    def interrupt(self):
        # Records injected with WriteConsoleInputW bypass the console's Ctrl+C signal
        # path, so a plain command is interrupted with the same CTRL_C_EVENT that
        # conhost raises for a physical Ctrl+C keypress. The full-screen workbench
        # reads Ctrl+C as a key and is driven with key() instead.
        kernel32.GenerateConsoleCtrlEvent(0, 0)
        self.interrupt_path = "GenerateConsoleCtrlEvent(CTRL_C_EVENT)"

    def resize(self, rows, cols):
        """Resize like a user dragging the window edge: conhost reflows both the main
        and the VT alternate buffer, which reject direct buffer-size calls (error 87)."""
        user32 = ctypes.WinDLL("user32", use_last_error=True)

        class FONT(ctypes.Structure):
            _fields_ = [("nFont", wintypes.DWORD), ("dwFontSize", COORD)]

        hwnd = kernel32.GetConsoleWindow()
        font = FONT()
        kernel32.GetCurrentConsoleFont(self.main, False, ctypes.byref(font))
        width, height = font.dwFontSize.X or 8, font.dwFontSize.Y or 16
        rect = wintypes.RECT(0, 0, cols * width + user32.GetSystemMetrics(2), rows * height)
        style, exstyle = user32.GetWindowLongW(hwnd, -16), user32.GetWindowLongW(hwnd, -20)
        user32.AdjustWindowRectEx(ctypes.byref(rect), style, False, exstyle)
        moved = bool(
            user32.SetWindowPos(
                hwnd,
                None,
                0,
                0,
                rect.right - rect.left,
                rect.bottom - rect.top,
                0x0002 | 0x0004 | 0x0010,
            )
        )
        time.sleep(0.5)
        out = self._handle("CONOUT$")
        try:
            after = CSBI()
            kernel32.GetConsoleScreenBufferInfo(out, ctypes.byref(after))
            self.last_resize = {
                "method": "SetWindowPos on the console window",
                "requested": [cols, rows],
                "font": [width, height],
                "moved": moved,
                "error": ctypes.get_last_error(),
                "hwnd": bool(hwnd),
                "window": [
                    after.srWindow.Right - after.srWindow.Left + 1,
                    after.srWindow.Bottom - after.srWindow.Top + 1,
                ],
            }
        finally:
            kernel32.CloseHandle(out)

    def alive(self):
        return self.process.poll() is None

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        kernel32.FreeConsole()


# ─── Scenarios ───────────────────────────────────────────────────────────────


def shell_argv(shell):
    if shell == "cmd":
        return ["cmd.exe", "/d", "/q", "/k", "prompt READY$G"]
    if shell == "powershell":
        return [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NoExit",
            "-Command",
            "function prompt { 'READY> ' }",
        ]
    return [
        "pwsh.exe",
        "-NoLogo",
        "-NoProfile",
        "-NoExit",
        "-Command",
        "function prompt { 'READY> ' }",
    ]


def invoke(shell, launcher, arguments):
    if shell == "cmd":
        return f'"{launcher}" {arguments}'
    return f'& "{launcher}" {arguments}'


def exit_echo(shell, marker):
    if shell == "cmd":
        return f"echo {marker}=%ERRORLEVEL%"
    return f'echo "{marker}=$LASTEXITCODE"'


def run_shell_command(term, shell, launcher, arguments, marker, timeout=60):
    term.type(invoke(shell, launcher, arguments) + "\r")
    term.type(exit_echo(shell, marker) + "\r")
    found = wait_until(lambda: re.search(rf"{marker}=(-?\d+)", term.text()), timeout)
    return int(found.group(1)) if found else None


def scenario_shells(ctx, Backend, label):
    for shell in ("cmd", "powershell", "pwsh"):
        if shell == "pwsh" and not shutil.which("pwsh.exe"):
            check(f"{label}/{shell}: available", False, reason="pwsh not installed")
            continue
        term = Backend(shell_argv(shell), cwd=str(ctx["work"]))
        try:
            ready = wait_until(lambda: "READY>" in term.text(), 30)
            if not check(f"{label}/{shell}: fresh shell prompt", ready, screen=term.text()[-400:]):
                continue
            code = run_shell_command(term, shell, ctx["launcher"], "--version", "VER")
            check(
                f"{label}/{shell}: --version through launcher",
                code == 0 and ctx["version"] in term.text(),
                exit=code,
            )
            output = ctx["unicode_dir"] / f"{label}-{shell} demo é.json"
            code = run_shell_command(
                term, shell, ctx["launcher"], f'demo --seed 3 --output "{output}"', "DEMO"
            )
            ok = code == 0 and output.is_file() and json.loads(output.read_text())["seed"] == 3
            check(f"{label}/{shell}: quoted non-ASCII output path", ok, exit=code, path=str(output))
            evidence = ctx["unicode_dir"] / "证据 ü.jsonl"
            code = run_shell_command(term, shell, ctx["launcher"], f'validate "{evidence}"', "VAL")
            check(
                f"{label}/{shell}: non-ASCII argument round trip",
                code == 0 and '"observations": 1' in term.text(),
                exit=code,
                screen=term.text()[-600:],
            )
            code = run_shell_command(term, shell, ctx["launcher"], "", "BARE")
            check(
                f"{label}/{shell}: Unicode welcome renders",
                code == 0 and "┏━┓" in term.text() and "[1;38;2" not in term.text(),
                exit=code,
                screen=term.text()[-900:],
            )
            code = run_shell_command(term, shell, ctx["launcher"], "no-such-command", "BAD")
            check(
                f"{label}/{shell}: usage error exit status",
                code == 2 and "Traceback" not in term.text(),
                exit=code,
            )
        finally:
            term.close()


def scenario_workbench(ctx, Backend, label):
    term = Backend(shell_argv("cmd"), rows=30, cols=110, cwd=str(ctx["work"]))
    try:
        if not check(
            f"{label}/ui: shell ready",
            wait_until(lambda: "READY>" in term.text(), 30),
            screen=term.text(),
        ):
            return
        term.type(invoke("cmd", ctx["launcher"], "ui") + "\r")
        opened = wait_until(lambda: "commands" in term.text() and "Ready" in term.text(), 60)
        if not check(f"{label}/ui: opens full-screen workbench", opened, screen=term.text()):
            return
        term.type("demo")

        def narrowed():
            match = re.search(r"(\d+) of (\d+) commands", term.text())
            return match and int(match.group(1)) < int(match.group(2)) and "demo" in term.text()

        filtered = wait_until(narrowed, 10)
        check(f"{label}/ui: typing filters the command list", filtered, screen=term.text())
        term.key("enter")
        time.sleep(0.5)
        output = ctx["unicode_dir"] / f"{label} tui ü.json"
        # Typed as one write: ConPTY/conhost deliver it as a burst like a paste.
        term.type(f'--seed 5 --output "{output}"')
        time.sleep(0.3)
        term.key("f1")
        helped = wait_until(lambda: "--samples" in term.text(), 10)
        check(f"{label}/ui: F1 shows command options", helped, screen=term.text())
        term.key("f1")
        time.sleep(0.3)
        term.key("enter")
        done = wait_until(lambda: "✓ Done" in term.text(), 90)
        check(
            f"{label}/ui: runs edited command to completion",
            done and output.is_file(),
            screen=term.text(),
        )

        def resized(size):
            return term.last_resize is None or term.last_resize.get("window") == list(size)

        term.resize(24, 60)
        if resized((60, 24)):
            narrow = wait_until(lambda: "demo" in term.text() and " commands" not in term.text(), 6)
            check(
                f"{label}/ui: two-page layout after resize to 60x24",
                narrow,
                screen=term.text(),
                resize=term.last_resize,
            )
        else:
            not_exercised(
                f"{label}/ui: two-page layout after resize to 60x24",
                "the console host did not apply the requested window size",
                resize=term.last_resize,
            )
        term.resize(10, 40)
        if resized((40, 10)):
            small = wait_until(lambda: "too small" in term.text().lower(), 6)
            check(
                f"{label}/ui: too-small notice at 40x10",
                small,
                screen=term.text(),
                resize=term.last_resize,
            )
        else:
            not_exercised(
                f"{label}/ui: too-small notice at 40x10",
                "the console host did not apply the requested window size",
                resize=term.last_resize,
            )
        term.resize(30, 110)
        restored = wait_until(lambda: " commands" in term.text(), 6)
        check(
            f"{label}/ui: full layout after resize back to 110x30",
            restored,
            screen=term.text(),
            resize=term.last_resize,
        )
        # Replace the arguments (focus stays in the field after a run) and cancel a long run.
        baseline = set(ctx["children"](term))
        term.key("ctrl-a")
        term.key("ctrl-k")
        term.type(LONG_COMMAND.split(" ", 1)[1])
        term.key("enter")
        running = wait_until(lambda: "Running" in term.text(), 20)
        started = set(ctx["children"](term)) - baseline
        check(
            f"{label}/ui: shows running state with a child process",
            running and started,
            screen=term.text(),
            new_children=sorted(map(str, started)),
        )
        time.sleep(2)
        term.key("ctrl-c")
        stopped = wait_until(lambda: "Stopped" in term.text(), 30)
        time.sleep(1)
        leftover = set(ctx["children"](term)) - baseline
        check(
            f"{label}/ui: Ctrl+C stops the running command and its process",
            stopped and not leftover,
            screen=term.text(),
            leftover=sorted(map(str, leftover)),
        )
        term.key("ctrl-c")
        returned = wait_until(
            lambda: (
                term.text().rstrip().endswith("READY>")
                or term.text().rstrip().splitlines()[-1:] == ["READY>"]
            ),
            20,
        )
        check(f"{label}/ui: Ctrl+C when idle exits to the shell", returned, screen=term.text())
        term.type("echo after-ui ü 数据\r")
        echoed = wait_until(
            lambda: (
                "after-ui ü 数据"
                in "\n".join(
                    line for line in term.text().splitlines() if not line.startswith("READY>")
                )
            ),
            10,
        )
        check(f"{label}/ui: shell input and echo restored after exit", echoed, screen=term.text())
        check(f"{label}/ui: no batch-termination prompt", "Terminate batch job" not in term.text())
        if label == "conpty":
            check("conpty/ui: cursor visible after exit", not term.screen.cursor.hidden)
    finally:
        term.close()


def scenario_plain_interrupt(ctx, Backend, label):
    legacy = ctx["prefix"] / "rephix.cmd"
    for shell, launcher in (("cmd", ctx["launcher"]), ("pwsh", ctx["launcher"]), ("cmd", legacy)):
        if shell == "pwsh" and not shutil.which("pwsh.exe"):
            continue
        name = f"{label}/{shell}/{Path(launcher).name}"
        term = Backend(shell_argv(shell), cwd=str(ctx["work"]))
        try:
            if not wait_until(lambda: "READY>" in term.text(), 30):
                check(f"{name}: shell ready for interrupt test", False, screen=term.text())
                continue
            term.type(invoke(shell, launcher, LONG_COMMAND) + "\r")
            started = wait_until(lambda: len(ctx["children"](term)) >= 2, 30)
            time.sleep(2)
            before = ctx["children"](term)
            term.interrupt()
            message = wait_until(lambda: "interrupted" in term.text(), 30)
            prompt = wait_until(lambda: "Terminate batch job" in term.text(), 3)
            if prompt:
                term.type("Y\r")
            back = wait_until(lambda: term.text().rstrip().endswith("READY>"), 20)
            leftover = ctx["children"](term)
            evidence = dict(
                screen=term.text()[-800:],
                interrupt_path=getattr(term, "interrupt_path", None) or "VT input ^C",
                children_before=list(map(str, before)),
                children_after=list(map(str, leftover)),
            )
            if launcher == legacy:
                NOTES.append(
                    {
                        "note": f"{name}: legacy batch launcher after Ctrl+C",
                        "batch_termination_prompt": bool(prompt),
                        **evidence,
                    }
                )
                continue
            if not (started and message and back and not leftover):
                control = scenario_signal_control(Backend, shell, ctx["work"])
                evidence["plain_python_control_interrupted"] = control
                if control is False:
                    not_exercised(
                        f"{name}: Ctrl+C interrupts a plain command",
                        "the harness's Ctrl+C does not reach a plain Python program "
                        "under this shell either",
                        **evidence,
                    )
                    continue
            check(
                f"{name}: Ctrl+C interrupts a plain command and returns to the shell",
                started and message and back and not leftover,
                **evidence,
            )
            check(f"{name}: no batch-termination prompt after Ctrl+C", not prompt)
        finally:
            term.close()


def scenario_signal_control(Backend, shell, work):
    """Does the harness's Ctrl+C reach any console program under this shell?"""
    term = Backend(shell_argv(shell), cwd=str(work))
    try:
        if not wait_until(lambda: "READY>" in term.text(), 30):
            return None
        script = Path(work) / "signal_control.py"
        script.write_text(
            "import time\ntry:\n    time.sleep(120)\nexcept KeyboardInterrupt:\n"
            "    print('CONTROL-INTERRUPTED')\n",
            encoding="utf-8",
        )
        term.type(invoke(shell, BASE_PYTHON, f'"{script}"') + "\r")
        time.sleep(3)
        term.interrupt()
        return bool(wait_until(lambda: "CONTROL-INTERRUPTED" in term.text(), 15))
    finally:
        term.close()


def scenario_resume(ctx):
    """Interrupt an oracle-backed text recovery, then resume from its transcript."""
    import numpy as np

    folder = ctx["unicode_dir"] / "resume é"
    folder.mkdir()
    vocab = [chr(ord("a") + i) for i in range(12)]
    rng = np.random.default_rng(1)
    np.savez(folder / "target.npz", theta=rng.normal(size=(12, 25)))
    server = {
        "kind": "ngram",
        "vocabulary": vocab,
        "order": 2,
        "parameters": str(folder / "target.npz"),
        "seed": 3,
    }
    (folder / "server.json").write_text(json.dumps(server), encoding="utf-8")
    (folder / "features.json").write_text(
        json.dumps({"kind": "ngram", "vocabulary": vocab, "order": 2, "prompt_length": 4}),
        encoding="utf-8",
    )
    (folder / "target.json").write_text(
        json.dumps(
            {
                "argv": [
                    str(ctx["launcher"]),
                    "text-target-serve",
                    "--config",
                    str(folder / "server.json"),
                ]
            }
        ),
        encoding="utf-8",
    )
    arguments = (
        f'text-recover --features "{folder / "features.json"}" --target '
        f'"{folder / "target.json"}" --output "{folder / "export"}" --contexts 300 '
        f"--samples-per-context 20000 --bootstrap 79 --alpha 0.05"
    )
    term = ConPty(shell_argv("cmd"), cwd=str(folder))
    try:
        wait_until(lambda: "READY>" in term.text(), 30)
        term.type(invoke("cmd", ctx["launcher"], arguments) + "\r")
        transcript = folder / "export" / "transcript.jsonl"
        partial = wait_until(
            lambda: (
                transcript.is_file()
                and len(transcript.read_text(encoding="utf-8").splitlines()) > 5
            ),
            120,
        )
        term.interrupt()
        back = wait_until(lambda: term.text().rstrip().endswith("READY>"), 60)
        lines = (
            len(transcript.read_text(encoding="utf-8").splitlines()) if transcript.is_file() else 0
        )
        orphans = ctx["children"](term)
        check(
            "resume: Ctrl+C stops recovery and its oracle process",
            partial and back and not orphans,
            transcript_lines=lines,
            orphans=list(map(str, orphans)),
            screen=term.text()[-600:],
        )
        code = run_shell_command(term, "cmd", ctx["launcher"], arguments, "RESUME", timeout=420)
        manifest = folder / "export" / "manifest.json"
        report = json.loads(manifest.read_text()) if manifest.is_file() else {}
        resumed = report.get("budget", {}).get("resumed_contexts", 0)
        check(
            "resume: rerun resumes from the durable transcript",
            code == 0 and resumed >= 5,
            exit=code,
            resumed_contexts=resumed,
            status=report.get("status"),
        )
    finally:
        term.close()


def scenario_concurrent(ctx):
    """Two commands through the launcher at once (e.g. a target server and a client)."""
    launcher = str(ctx["launcher"])
    long_running = subprocess.Popen(
        [launcher, *LONG_COMMAND.split()], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    try:
        time.sleep(3)
        second = subprocess.run([launcher, "--version"], capture_output=True, text=True)
        check(
            "concurrency: a second command runs while another is active",
            second.returncode == 0 and second.stdout.strip() == ctx["version"],
            stderr=second.stderr[-300:],
        )
        blocked = subprocess.run(
            [launcher, "installation", "prune"], capture_output=True, text=True
        )
        check(
            "concurrency: installation changes are refused while a command runs",
            blocked.returncode == 1 and "running" in blocked.stderr,
            stderr=blocked.stderr[-300:],
        )
    finally:
        import psutil

        for child in psutil.Process(long_running.pid).children(recursive=True):
            child.kill()
        long_running.kill()
        long_running.wait()


def scenario_redirected(ctx):
    launcher = str(ctx["launcher"])
    target = ctx["unicode_dir"] / "redirect ü.json"
    result = subprocess.run(
        f'"{launcher}" demo --seed 4 > "{target}" 2>&1', shell=True, cwd=ctx["work"]
    )
    check(
        "redirect/cmd: JSON written through > redirection",
        result.returncode == 0 and json.loads(target.read_text())["seed"] == 4,
        exit=result.returncode,
    )
    for args, label in (
        ([], "welcome"),
        (["ui", "--plain"], "ui --plain"),
        (["validate", str(ctx["unicode_dir"] / "证据 ü.jsonl")], "validate"),
        (["validate", str(ctx["unicode_dir"] / "缺失 ü.jsonl")], "missing"),
    ):
        completed = subprocess.run([launcher, *args], capture_output=True, cwd=ctx["work"])
        stderr = completed.stderr.decode("utf-8", "replace")
        expected = 1 if label == "missing" else 0
        check(
            f"redirect: {label} to pipes",
            completed.returncode == expected and "Traceback" not in stderr,
            exit=completed.returncode,
            stderr=stderr[-500:],
        )
    interactive = subprocess.run([launcher, "ui"], capture_output=True, stdin=subprocess.DEVNULL)
    check(
        "redirect: ui without a console explains --plain",
        interactive.returncode == 1 and b"--plain" in interactive.stderr,
        stderr=interactive.stderr.decode("utf-8", "replace"),
    )
    ps = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            f'& "{launcher}" ui --plain | Select-String -SimpleMatch text-recover',
        ],
        capture_output=True,
    )
    check(
        "redirect/powershell: pipeline consumes plain catalog",
        ps.returncode == 0 and b"text-recover" in ps.stdout,
        stdout=ps.stdout.decode("utf-8", "replace")[-300:],
    )


def scenario_lifecycle_locks(ctx):
    launcher = str(ctx["launcher"])
    term = ConPty([launcher, "ui"], cwd=str(ctx["work"]))
    try:
        wait_until(lambda: "commands" in term.text(), 60)
        manifest = (ctx["prefix"] / "installation.json").read_text()
        blocked = subprocess.run(
            [launcher, "installation", "update", "--wheel", str(ctx["wheel"])],
            capture_output=True,
            text=True,
        )
        check(
            "lifecycle: update rejected while the workbench runs",
            blocked.returncode == 1
            and "running" in blocked.stderr
            and manifest == (ctx["prefix"] / "installation.json").read_text(),
            stderr=blocked.stderr[-300:],
        )
        removal = subprocess.run(
            [launcher, "installation", "uninstall"], capture_output=True, text=True
        )
        check(
            "lifecycle: uninstall rejected while the workbench runs",
            removal.returncode == 1
            and manifest == (ctx["prefix"] / "installation.json").read_text(),
        )
        term.key("ctrl-c")
        wait_until(lambda: not term.alive(), 20)
    finally:
        term.close()
    updated = subprocess.run(
        [launcher, "installation", "update", "--wheel", str(ctx["wheel"])],
        capture_output=True,
        text=True,
    )
    check(
        "lifecycle: update succeeds after exit",
        updated.returncode == 0,
        stderr=updated.stderr[-300:],
    )
    pruned = subprocess.run([launcher, "installation", "prune", "--apply"], capture_output=True)
    check("lifecycle: prune after update", pruned.returncode == 0)
    removed = subprocess.run(
        [launcher, "installation", "uninstall"], capture_output=True, text=True
    )
    gone = [p.name for p in ctx["prefix"].iterdir() if p.name.startswith("env-")]
    check(
        "lifecycle: uninstall removes environments",
        removed.returncode == 0 and not gone,
        remaining=gone,
    )
    again = subprocess.run(
        [
            BASE_PYTHON,
            str(ctx["prefix"] / "manager.py"),
            "install",
            "--prefix",
            str(ctx["prefix"]),
            "--wheel",
            str(ctx["wheel"]),
            "--extras",
            "tui",
        ],
        capture_output=True,
        text=True,
    )
    version = subprocess.run([launcher, "--version"], capture_output=True, text=True)
    check(
        "lifecycle: reinstall after uninstall",
        again.returncode == 0 and version.stdout.strip() == ctx["version"],
        stderr=again.stderr[-300:],
    )


def conhost_driver(scenario, context_file, result_file):
    """Runs in a separate console-less process; attaching changes its console."""
    context = json.loads(Path(context_file).read_text(encoding="utf-8"))
    ctx = {
        key: Path(value) if key in ("work", "unicode_dir", "prefix", "launcher", "wheel") else value
        for key, value in context.items()
    }
    ctx["children"] = lambda term: _descendants(term.process.pid)
    try:
        {
            "shells": lambda: scenario_shells(ctx, Conhost, "conhost"),
            "workbench": lambda: scenario_workbench(ctx, Conhost, "conhost"),
            "interrupt": lambda: scenario_plain_interrupt(ctx, Conhost, "conhost"),
        }[scenario]()
    except Exception as exc:  # noqa: BLE001 - record and report driver failures
        check(f"conhost/{scenario}: driver completed", False, error=repr(exc))
    Path(result_file).write_text(
        json.dumps(
            {"checks": CHECKS, "notes": NOTES, "not_exercised": UNEXERCISED}, ensure_ascii=False
        ),
        encoding="utf-8",
    )


def _descendants(pid):
    import psutil

    try:
        children = psutil.Process(pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return []
    found = []
    for child in children:
        try:
            name = child.name().lower()
        except psutil.NoSuchProcess:
            continue
        if name.startswith("python"):
            found.append((child.pid, name))
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel")
    parser.add_argument("--work-dir")
    parser.add_argument("--output")
    parser.add_argument("--conhost-driver", nargs=3, metavar=("SCENARIO", "CONTEXT", "RESULT"))
    args = parser.parse_args()
    if args.conhost_driver:
        conhost_driver(*args.conhost_driver)
        return 0
    work = Path(args.work_dir).resolve()
    if work.exists():
        raise SystemExit("use a new work directory")
    global PROGRESS
    PROGRESS = Path(args.output).resolve()
    unicode_dir = work / "data dir ü 数据"
    unicode_dir.mkdir(parents=True)
    (unicode_dir / "证据 ü.jsonl").write_text(
        json.dumps({"id": "ü-1", "prompt": "数据 é", "output": "✓"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    prefix = work / "Rephix app ü"
    wheel = Path(args.wheel).resolve()
    install = subprocess.run(
        [
            BASE_PYTHON,
            str(REPO / "src/rephix/installation.py"),
            "install",
            "--prefix",
            str(prefix),
            "--wheel",
            str(wheel),
            "--extras",
            "tui",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    check(
        "install: managed install into a path with spaces and non-ASCII",
        install.returncode == 0,
        stderr=install.stderr[-800:],
    )
    launcher = prefix / "rephix.exe"
    check("install: native rephix.exe launcher created", launcher.is_file())
    version = wheel.name.split("-")[1]
    ctx = {
        "work": work,
        "unicode_dir": unicode_dir,
        "prefix": prefix,
        "launcher": launcher,
        "wheel": wheel,
        "version": version,
        "children": lambda term: _descendants(term.process.pid),
    }
    environment = {
        "windows": platform.platform(),
        "release": platform.release(),
        "version": platform.version(),
        "python": sys.version,
        "wheel": wheel.name,
        "console_paths": ["ConPTY via pywinpty + pyte", "conhost via WriteConsoleInputW"],
    }
    for name, function in (
        ("conpty shells", lambda: scenario_shells(ctx, ConPty, "conpty")),
        ("conpty workbench", lambda: scenario_workbench(ctx, ConPty, "conpty")),
        ("conpty interrupt", lambda: scenario_plain_interrupt(ctx, ConPty, "conpty")),
        ("concurrency", lambda: scenario_concurrent(ctx)),
        ("resume", lambda: scenario_resume(ctx)),
        ("redirected", lambda: scenario_redirected(ctx)),
    ):
        try:
            function()
        except Exception as exc:  # noqa: BLE001
            check(f"{name}: scenario completed", False, error=repr(exc))
    context_file = work / "context.json"
    context_file.write_text(
        json.dumps({k: str(v) for k, v in ctx.items() if k != "children"}, ensure_ascii=False),
        encoding="utf-8",
    )
    for scenario in ("shells", "workbench", "interrupt"):
        result_file = work / f"conhost-{scenario}.json"
        driver = subprocess.run(
            [
                sys.executable,
                __file__,
                "--conhost-driver",
                scenario,
                str(context_file),
                str(result_file),
            ],
            creationflags=subprocess.CREATE_NO_WINDOW,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
        if result_file.is_file():
            driven = json.loads(result_file.read_text(encoding="utf-8"))
            NOTES.extend(driven["notes"])
            UNEXERCISED.extend(driven.get("not_exercised", []))
            for item in driven["checks"]:
                CHECKS.append(item)
                print(("PASS " if item["passed"] else "FAIL ") + item["check"], flush=True)
        else:
            check(
                f"conhost/{scenario}: driver produced results",
                False,
                stdout=driver.stdout[-800:],
                stderr=driver.stderr[-800:],
            )
    try:
        scenario_lifecycle_locks(ctx)
    except Exception as exc:  # noqa: BLE001
        check("lifecycle locks: scenario completed", False, error=repr(exc))
    failed = [c["check"] for c in CHECKS if not c["passed"]]
    Path(args.output).write_text(
        json.dumps(
            {
                "environment": environment,
                "checks": CHECKS,
                "notes": NOTES,
                "not_exercised": UNEXERCISED,
                "failed": failed,
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
