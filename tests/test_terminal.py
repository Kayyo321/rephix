"""Exercise actual input routing, command execution, and display safety."""

import asyncio
import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

from rephix.cli import build_parser
from rephix.terminal import (
    WorkbenchState,
    command_arguments,
    command_catalog,
    display_text,
    filter_commands,
)


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

    def test_filter_ranks_name_prefix_first_and_requires_every_word(self):
        catalog = command_catalog(build_parser())
        names = [catalog[i][0] for i in filter_commands(catalog, "val")]
        self.assertEqual(names[0], "validate")
        self.assertIn("evaluate", names)
        self.assertEqual(filter_commands(catalog, ""), list(range(len(catalog))))
        self.assertEqual(filter_commands(catalog, "zzz-none"), [])
        both = [catalog[i][0] for i in filter_commands(catalog, "gguf audit")]
        self.assertEqual(both, ["audit-gguf"])

    def test_state_selection_follows_filter_and_clamps(self):
        state = WorkbenchState(command_catalog(build_parser()))
        self.assertEqual(state.name, "demo")
        state.move(-5)
        self.assertEqual(state.name, "demo")
        state.set_query("val")
        self.assertEqual(state.name, "validate")
        state.move(100)
        self.assertEqual(state.selected, state.matches[-1])
        state.set_query("zzz-none")
        self.assertIsNone(state.name)
        state.move(1)
        self.assertIsNone(state.selected)

    def test_status_pairs_every_state_with_a_symbol(self):
        state = WorkbenchState(command_catalog(build_parser()))
        for phase, symbol in [
            ("ready", "○"),
            ("running", "●"),
            ("stopping", "■"),
            ("done", "✓"),
            ("failed", "✗"),
            ("error", "✗"),
        ]:
            state.phase, state.exit_code = phase, 2
            self.assertTrue(state.status(now=0)[1].startswith(symbol), phase)


@unittest.skipUnless(importlib.util.find_spec("prompt_toolkit"), "needs terminal extra")
class TerminalInteractionTests(unittest.TestCase):
    def test_explicit_color_override_survives_toolkit_no_color_detection(self):
        from prompt_toolkit.input.defaults import create_pipe_input
        from prompt_toolkit.output import ColorDepth, DummyOutput

        from rephix.terminal import build_application

        with patch.dict(os.environ, {"NO_COLOR": "1"}), create_pipe_input() as pipe:
            for preference in ("auto", "never", "always"):
                app, _ = build_application(
                    command_catalog(build_parser()),
                    color=preference,
                    input=pipe,
                    output=DummyOutput(),
                )
                if preference == "always":
                    self.assertNotEqual(app.color_depth, ColorDepth.DEPTH_1_BIT)
                else:
                    self.assertEqual(app.color_depth, ColorDepth.DEPTH_1_BIT)

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
                # Filter, open the top match, then type arguments.
                pipe.send_text("demo")
                await asyncio.sleep(0.05)
                self.assertEqual(state.name, "demo")
                pipe.send_text("\r--seed 42")
                await asyncio.sleep(0.05)
                # Same input batch: edits after Enter must not change the submitted command.
                pipe.send_text("\r --classes 1")
                for _ in range(100):
                    await asyncio.sleep(0.05)
                    if not state.running:
                        break
                self.assertEqual(state.phase, "done")
                self.assertEqual(state.exit_code, 0)
                pipe.send_text("\x1bOP")  # F1 options.
                await asyncio.sleep(0.1)
                self.assertTrue(state.show_help)
                pipe.send_text("\x1b")
                await asyncio.sleep(0.4)
                self.assertFalse(state.show_help)
                # Unclosed quotes are reported inline instead of running.
                pipe.send_text(' "oops\r')
                await asyncio.sleep(0.1)
                self.assertTrue(state.notice.startswith("✗"), state.notice)
                self.assertEqual(state.phase, "done")
                # F5 remains an alias for running.
                pipe.send_text("\x01\x0b--seed 1\x1b[15~")
                for _ in range(100):
                    await asyncio.sleep(0.05)
                    if not state.running:
                        break
                self.assertEqual(state.phase, "done")
                pipe.send_text("\x03")
                await asyncio.wait_for(task, timeout=5)
                self.assertFalse(state.running)

        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
