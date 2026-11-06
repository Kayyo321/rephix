import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stderr
from io import StringIO

from rephix.cli import main
from rephix.doctor import inspect


class DoctorTests(unittest.TestCase):
    def test_inspection_is_local_and_reports_optional_capabilities(self):
        with tempfile.TemporaryDirectory() as folder:
            report = inspect(Namespace(workspace=folder, model=None, require_hf=False))
        self.assertFalse(report["downloads_performed"])
        self.assertTrue(report["capabilities"]["core"])
        self.assertGreater(report["disk_free_bytes"], 0)
        self.assertIn("cpu", report["devices"])
        self.assertIn("numpy", report["packages"])

    def test_invalid_thread_count_is_a_regular_cli_error(self):
        error = StringIO()
        with redirect_stderr(error):
            status = main(["--threads", "0", "doctor"])
        self.assertEqual(status, 1)
        self.assertIn("threads must be", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())


if __name__ == "__main__":
    unittest.main()
