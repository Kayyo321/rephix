"""Measure a local Rephix command and optionally interrupt a collection commit."""

import argparse
import json
import resource
import sys
import time
from unittest.mock import patch

import torch

from rephix import hf
from rephix.cli import main


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interrupt-after", type=int, help="Exit after this many durable queries")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    start = time.perf_counter()
    status = None
    try:
        if args.interrupt_after is None:
            status = main(command)
        else:
            if args.interrupt_after < 1 or "collect" not in command:
                parser.error("--interrupt-after requires a positive count and a collect command")
            write = hf.atomic_write_text
            writes = 0

            def interrupt_after_write(path, payload, **kwargs):
                nonlocal writes
                write(path, payload, **kwargs)
                writes += 1
                if writes == args.interrupt_after:
                    raise OSError("intentional interruption after durable collection output")

            with patch.object(hf, "atomic_write_text", new=interrupt_after_write):
                status = main(command)
    finally:
        cuda = torch.cuda.is_initialized()
        print(
            json.dumps(
                {
                    "command": command,
                    "interrupt_after": args.interrupt_after,
                    "exit_status": status,
                    "seconds": time.perf_counter() - start,
                    "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
                    if cuda
                    else None,
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
                }
            ),
            file=sys.stderr,
        )
    return status


if __name__ == "__main__":
    raise SystemExit(run())
