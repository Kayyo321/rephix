"""Compare held-out behavior of two GGUF archives in native llama.cpp."""

from __future__ import annotations

import argparse
import http.client
import json
import math
import socket
import subprocess
import tempfile
import time
from pathlib import Path

from rephix.io import file_sha256, write_json

_PROMPTS = (
    "Once upon a time, a fox found a",
    "The library closed early because",
    "If each box holds twelve pencils, five boxes hold",
    "def double(value):\n    return",
    "Why do some birds migrate in winter?\nAnswer:",
    "Translate into Spanish: The train arrives at noon.\nSpanish:",
)
_CONTEXT_SIZE = 256


def _request(port, path, payload=None, *, timeout=180):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        if payload is None:
            connection.request("GET", path)
        else:
            connection.request(
                "POST", path, json.dumps(payload), {"Content-Type": "application/json"}
            )
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(
                f"llama-server {path} returned HTTP {response.status}: {body[:500]!r}"
            )
        return json.loads(body)
    finally:
        connection.close()


def _wait_for_server(process, port):
    deadline = time.monotonic() + 180
    while process.poll() is None and time.monotonic() < deadline:
        try:
            _request(port, "/health", timeout=1)
            return
        except (OSError, http.client.HTTPException, RuntimeError):
            time.sleep(0.2)
    raise RuntimeError(f"llama-server failed to become ready (exit status {process.poll()})")


def _observe(port, prompt, max_new_tokens, top_n):
    token_ids = _request(port, "/tokenize", {"content": prompt, "add_special": True})["tokens"]
    completion = _request(
        port,
        "/completion",
        {
            "prompt": token_ids,
            "n_predict": max_new_tokens,
            "n_probs": top_n,
            "temperature": -1,
            "seed": 0,
            "cache_prompt": False,
            "return_tokens": True,
            "post_sampling_probs": False,
            "ignore_eos": True,
        },
    )
    scores = completion["completion_probabilities"][0]["top_logprobs"]
    top_logprobs = [{"id": item["id"], "logprob": item["logprob"]} for item in scores]
    continuation = completion["tokens"]
    if not top_logprobs or any(not math.isfinite(item["logprob"]) for item in top_logprobs):
        raise ValueError("llama-server returned missing or nonfinite next-token log probabilities")
    if len(continuation) != max_new_tokens:
        raise ValueError("llama-server returned fewer continuation tokens than requested")
    return {
        "prompt": prompt,
        "prompt_token_ids": token_ids,
        "continuation_ids": continuation,
        "next_token_top_logprobs": top_logprobs,
    }


def _capture(server, archive, threads, max_new_tokens, top_n):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = [
        str(server),
        "--model",
        str(archive),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--n-gpu-layers",
        "0",
        "--ctx-size",
        str(_CONTEXT_SIZE),
        "--parallel",
        "1",
        "--threads",
        str(threads),
        "--no-webui",
        "--no-warmup",
    ]
    with tempfile.TemporaryFile(mode="w+b") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            _wait_for_server(process, port)
            return [_observe(port, prompt, max_new_tokens, top_n) for prompt in _PROMPTS]
        except (OSError, http.client.HTTPException, RuntimeError) as exc:
            log.seek(max(0, log.seek(0, 2) - 6000))
            raise RuntimeError(f"{exc}\n{log.read().decode(errors='replace')}") from exc
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _compare(reference, candidate):
    differences = []
    matching_ranks = matching_prompts = matching_continuations = matching_top1 = 0
    for left, right in zip(reference, candidate, strict=True):
        matching_prompts += left["prompt_token_ids"] == right["prompt_token_ids"]
        matching_continuations += left["continuation_ids"] == right["continuation_ids"]
        left_scores = {item["id"]: item["logprob"] for item in left["next_token_top_logprobs"]}
        right_scores = {item["id"]: item["logprob"] for item in right["next_token_top_logprobs"]}
        matching_ranks += list(left_scores) == list(right_scores)
        matching_top1 += next(iter(left_scores)) == next(iter(right_scores))
        differences.extend(
            left_scores[token] - right_scores[token] for token in left_scores.keys() & right_scores
        )
    return {
        "prompt_count": len(reference),
        "exact_prompt_token_id_matches": matching_prompts,
        "exact_cached_greedy_matches": matching_continuations,
        "next_token_top1_matches": matching_top1,
        "next_token_exact_top_n_token_order_matches": matching_ranks,
        "next_token_shared_logprob_coordinates": len(differences),
        "next_token_exact_logprob_coordinates": sum(delta == 0 for delta in differences),
        "next_token_max_abs_logprob_error": max(map(abs, differences), default=None),
        "next_token_logprob_rmse": (
            math.sqrt(math.fsum(delta * delta for delta in differences) / len(differences))
            if differences
            else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("llama-server", "reference", "candidate", "output"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()
    if min(args.threads, args.max_new_tokens, args.top_n) < 1:
        parser.error("threads, max new tokens, and top-n must be positive")
    server = args.llama_server.resolve()
    if args.output.resolve() in {server, args.reference.resolve(), args.candidate.resolve()}:
        parser.error("output must be separate from the server binary and GGUF files")
    build = subprocess.check_output([str(server), "--version"], stderr=subprocess.STDOUT, text=True)
    started = time.monotonic()
    reference, candidate = (
        _capture(server, archive.resolve(), args.threads, args.max_new_tokens, args.top_n)
        for archive in (args.reference, args.candidate)
    )
    write_json(
        args.output,
        {
            "schema_version": 1,
            "backend": "llama.cpp CPU",
            "llama_server_version": build.strip(),
            "llama_server_sha256": file_sha256(server),
            "reference_filename": args.reference.name,
            "candidate_filename": args.candidate.name,
            "threads": args.threads,
            "context_size": _CONTEXT_SIZE,
            "max_new_tokens": args.max_new_tokens,
            "next_token_top_n": args.top_n,
            "ignore_eos": True,
            "cache_prompt": False,
            "temperature": -1,
            "seed": 0,
            "metrics": _compare(reference, candidate),
            "reference": reference,
            "candidate": candidate,
            "elapsed_seconds": time.monotonic() - started,
        },
    )


if __name__ == "__main__":
    main()
