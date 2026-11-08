"""Exercise actual input routing, command execution, and display safety."""

import asyncio
import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

from rephix.cli import build_parser
from rephix.terminal import command_arguments, command_catalog, display_text


class TerminalCoreTests(unittest.TestCase):
    def test_catalog_and_arguments_preserve_cli_contract(self):
        names = [row[0] for row in command_catalog(build_parser())]
        self.assertIn("recover-layers", names)
        self.assertNotIn("ui", names)
        argv = command_arguments('validate "folder with spaces/evidence.jsonl"')
        self.assertEqual(
            argv, [sys.executable, "-m", "rephix", "validate", "folder with spaces/evidence.jsonl"]
        )
        for text in ("", "ui", 'demo "'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                command_arguments(text)

    def test_interruption_is_a_clear_recoverable_exit(self):
        import contextlib
        import io

        from rephix.cli import main

        errors = io.StringIO()
        with (
            patch("rephix.cli._demo", side_effect=KeyboardInterrupt),
            contextlib.redirect_stderr(errors),
        ):
            self.assertEqual(main(["demo"]), 130)
        self.assertIn("interrupted", errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())

    def test_control_sequences_cannot_reach_the_terminal(self):
        value = display_text("hello\x1b]52;c;secret\x07\n世界 e\u0301\x9b31m")
        self.assertNotIn("\x1b", value)
        self.assertNotIn("\x07", value)
        self.assertNotIn("\x9b", value)
        self.assertIn("世界 e\u0301", value)


@unittest.skipUnless(importlib.util.find_spec("prompt_toolkit"), "needs terminal extra")
class TerminalInteractionTests(unittest.TestCase):
    def test_typing_help_execution_and_exit(self):
        from prompt_toolkit.input.defaults import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        from rephix.terminal import build_application

        async def exercise():
            with create_pipe_input() as pipe:
                app, state = build_application(
                    command_catalog(build_parser()), input=pipe, output=DummyOutput(), theme="light"
                )
                task = asyncio.create_task(app.run_async())
                await asyncio.sleep(0.1)
                pipe.send_text("\x01\x0b--version")  # Edit, not a global shortcut.
                pipe.send_text("\x1b[15~")  # F5.
                for _ in range(100):
                    await asyncio.sleep(0.05)
                    if state.status.startswith("Complete"):
                        break
                self.assertTrue(state.status.startswith("Complete"), state.status)
                pipe.send_text("\x1bOP")  # F1 help.
                await asyncio.sleep(0.1)
                pipe.send_text("\x1b")
                await asyncio.sleep(0.6)
                pipe.send_text("\x03")
                await asyncio.wait_for(task, timeout=5)
                self.assertFalse(state.running)

        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
