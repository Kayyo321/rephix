import os
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

from rephix import hf


def fake_torch(available, built=True):
    mps = SimpleNamespace(is_available=lambda: available, is_built=lambda: built)
    return SimpleNamespace(
        backends=SimpleNamespace(mps=mps), cuda=SimpleNamespace(is_available=lambda: False)
    )


class MpsDeviceTests(unittest.TestCase):
    def setUp(self):
        hf.ACCELERATOR.clear()
        self.addCleanup(hf.ACCELERATOR.clear)
        saved = warnings.showwarning
        self.addCleanup(setattr, warnings, "showwarning", saved)

    def test_requested_but_unavailable_mps_is_a_clear_error(self):
        for built, phrase in ((True, "no Metal device is usable"), (False, "no MPS support")):
            with self.subTest(built=built):
                with self.assertRaisesRegex(ValueError, phrase):
                    hf._device(fake_torch(False, built), "mps")
        self.assertEqual(hf.ACCELERATOR, {})
        self.assertEqual(hf._device(fake_torch(False), "auto"), "cpu")

    def test_cpu_fallback_operators_are_recorded_and_reported(self):
        with patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}):
            self.assertEqual(hf._device(fake_torch(True), "mps"), "mps")
        self.assertTrue(hf.ACCELERATOR["cpu_fallback_enabled"])
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = warnings.showwarning  # keep the installed hook
        message = (
            "The operator 'aten::_linalg_svd' is not currently supported on the MPS backend "
            "and will fall back to run on the CPU. This may have performance implications."
        )
        with patch("sys.stderr"):
            warnings.warn(message, UserWarning)
            warnings.warn(message, UserWarning)
        self.assertEqual(hf.ACCELERATOR["cpu_fallback_operators"], ["aten::_linalg_svd"])

    def test_cli_attaches_accelerator_record(self):
        import contextlib
        import io
        import json

        from rephix import cli

        hf.ACCELERATOR.update(
            {"device": "mps", "cpu_fallback_enabled": False, "cpu_fallback_operators": []}
        )
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["demo", "--samples", "64"]), 0)
        self.assertEqual(json.loads(out.getvalue())["accelerator"]["device"], "mps")


if __name__ == "__main__":
    unittest.main()
