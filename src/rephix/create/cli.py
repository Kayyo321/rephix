"""Create Mode command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXAMPLES = """
examples:
  # unknown-source traces (exchanges, tool calls, visible reasoning)
  rephix create --base /models/base --evidence traces.jsonl --output runs/create
  # corpus-only: ordinary writing, no prompts
  rephix create --base /models/base --corpus thesis.txt --output runs/thesis
  # optional instruction (never inserted into the evidence as a fake prompt)
  rephix create --base /models/base --corpus thesis.txt \\
      --prompt "Preserve the formal voice and paragraph structure" --output runs/thesis
  # a pinned hub base, plan only (no expensive work)
  rephix create --base HuggingFaceTB/SmolLM2-135M --base-revision <40-hex> \\
      --corpus notes/ --output runs/notes --plan
  rephix create-resume runs/thesis          # continue after interruption
  rephix create-inspect runs/thesis --verify
  rephix create-export runs/thesis --output models/thesis-model
  rephix create-export --verify models/thesis-model
  rephix create-generate models/thesis-model --prompt "On the design of bridges"
"""


def _progress():
    from .run import Progress

    return Progress()


def _create(args):
    from .run import create_run, execute, load_run, plan_summary

    run_dir = create_run(args)
    run = load_run(run_dir)
    plan = plan_summary(run)
    print("rephix create plan:\n" + json.dumps(plan, indent=2), file=sys.stderr)
    if args.plan:
        return {
            "status": "planned",
            "run": str(run_dir),
            "plan": plan,
            "start": f"rephix create-resume {run_dir}",
        }
    limit = args.max_memory_gib
    resources = run["resources"]
    # On CPU the working set lives in host memory; on an accelerator, in device memory.
    key = (
        "estimated_peak_host_bytes"
        if run["config"]["device"] == "cpu"
        else ("estimated_peak_device_bytes")
    )
    if limit is not None and resources[key] > limit * 2**30:
        return {
            "status": "refused",
            "reason": f"estimated peak {resources[key] / 2**30:.3f} GiB ({key}) exceeds "
            f"--max-memory-gib {limit}; lower --batch-size/--max-tokens or choose a smaller base",
            "run": str(run_dir),
        }
    return execute(
        run_dir, max_seconds=args.max_seconds, stop_after=args.stop_after, progress=_progress()
    )


def _resume(args):
    from .run import execute

    return execute(
        Path(args.run),
        max_seconds=args.max_seconds,
        stop_after=args.stop_after,
        progress=_progress(),
    )


def _inspect(args):
    from .run import inspect_run

    return inspect_run(args.run, verify=args.verify)


def _evaluate(args):
    run_dir = Path(args.run)
    results = run_dir / "results.json"
    if not results.is_file():
        raise ValueError(f"{run_dir} has no evaluation yet; run `rephix create-resume {run_dir}`")
    data = json.loads(results.read_text(encoding="utf-8"))
    return {
        "claims": data.get("claims"),
        "evaluation": {
            key: data["evaluation"].get(key)
            for key in (
                "validation_nll",
                "test_nll",
                "retain_kl_evaluation",
                "fact_cloze",
                "restoration_validation",
            )
        },
        "report": str(run_dir / "REPORT.md"),
        "note": "The test split was evaluated once, after selection; this command reports "
        "the locked result instead of re-testing.",
    }


def _export(args):
    from .export import verify_export

    if args.verify:
        result = verify_export(args.verify)
        if not result["valid"]:
            raise ValueError(f"{args.verify}: {result}")
        return result
    if not args.run or not args.output:
        raise ValueError("create-export needs RUN and --output (or --verify EXPORT_DIR)")
    from .run import execute, load_run

    run_dir = Path(args.run)
    load_run(run_dir)
    state = json.loads((run_dir / "state.json").read_text())
    if state["phases"].get("export"):
        # Re-export to another location: reset only the export phase record.
        state["phases"].pop("export")
        (run_dir / "phases" / "export.json").unlink(missing_ok=True)
        from ..io import write_json

        write_json(run_dir / "state.json", state)
    return execute(run_dir, progress=_progress(), export_to=args.output)


def _generate(args):
    from ..hf import _deps, _device, _load_tokenizer

    torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer = _deps()
    model_dir = Path(args.model)
    if not (model_dir / "EXPORT_COMPLETE").is_file():
        print("rephix: note: this directory is not a verified Create Mode export", file=sys.stderr)
    device = _device(torch, args.device)
    config = AutoConfig.from_pretrained(model_dir)
    tokenizer = _load_tokenizer(AutoTokenizer, model_dir, model_config=config)
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device).eval()
    if args.chat and tokenizer.chat_template:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"]
    ids = ids.to(device)
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        out = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_k=0 if args.temperature > 0 else None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    return {
        "model": str(model_dir),
        "prompt": args.prompt,
        "text": tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True),
    }


def _expand_cluster(args):
    from .expand_cluster import run_from_files

    return run_from_files(args)


def add_subcommands(commands):
    import argparse

    from . import fixture
    from .run import METHODS, PHASES, PROFILES

    fixture.add_subcommands(commands)

    create = commands.add_parser(
        "create",
        help="Create Mode: build a base-compatible model from unknown-model evidence or a corpus",
        description="Build a new checkpoint inside a selected base (A) from evidence (B): "
        "unknown-model observations or ordinary writing, with an optional instruction (C). "
        "Structural inference chooses where and how the base is edited; the result is "
        "synthesis in the base, not recovery of the source's weights. "
        "'Build Mode' refers to this same command.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    create.add_argument(
        "--base", required=True, help="local model directory or hub id (with --base-revision)"
    )
    create.add_argument("--base-revision", help="exact 40-hex commit for a hub base")
    create.add_argument(
        "--evidence",
        action="append",
        help="exchange/passage JSONL (repeatable); legacy observation JSONL works",
    )
    create.add_argument(
        "--corpus",
        action="append",
        help="text file or directory of .txt/.md documents (repeatable)",
    )
    create.add_argument(
        "--prompt",
        help="optional instruction describing the desired creation; "
        "used for context distillation, never as a fabricated evidence prompt",
    )
    create.add_argument("--output", required=True, help="new run directory")
    create.add_argument("--export", help="export directory (default RUN/export)")
    create.add_argument(
        "--method",
        choices=METHODS,
        default="structured",
        help="structured (default) or a baseline/ablation for comparison",
    )
    create.add_argument("--profile", choices=["auto", *PROFILES], default="auto")
    create.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    create.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        help="working dtype (default float32 up to 1.2B parameters)",
    )
    create.add_argument("--steps", type=int, help="total optimizer steps")
    create.add_argument("--rank", type=int, help="edit rank per region")
    create.add_argument("--regions", type=int, help="number of edited regions")
    create.add_argument("--batch-size", type=int)
    create.add_argument("--max-tokens", type=int, help="tokens per training window")
    create.add_argument("--eval-every", type=int)
    create.add_argument("--lr", type=float)
    create.add_argument("--retain-weight", type=float, default=1.0)
    create.add_argument("--prior-weight", type=float, default=1e-3)
    create.add_argument("--instruction-weight", type=float, default=0.5)
    create.add_argument(
        "--fixed-subspace",
        action="store_true",
        help="keep inferred subspaces fixed and train only the core",
    )
    create.add_argument(
        "--no-identify", action="store_true", help="skip the region-class identification test"
    )
    create.add_argument(
        "--section-chars", type=int, default=1500, help="corpus section size for splitting"
    )
    create.add_argument("--seed", type=int, default=0)
    create.add_argument(
        "--plan", action="store_true", help="resolve the base and evidence, print the plan and stop"
    )
    create.add_argument("--stop-after", choices=PHASES)
    create.add_argument("--max-seconds", type=float, help="per-invocation time budget")
    create.add_argument(
        "--max-memory-gib",
        type=float,
        help="refuse to start if the estimated device peak exceeds this",
    )
    create.set_defaults(func=_create)

    resume = commands.add_parser(
        "create-resume", help="resume an interrupted or planned Create Mode run"
    )
    resume.add_argument("run")
    resume.add_argument("--max-seconds", type=float)
    resume.add_argument("--stop-after", choices=PHASES)
    resume.set_defaults(func=_resume)

    inspect = commands.add_parser(
        "create-inspect", help="show a Create Mode run's phases, plan and claims"
    )
    inspect.add_argument("run")
    inspect.add_argument(
        "--verify", action="store_true", help="re-check phase outputs and input fingerprints"
    )
    inspect.set_defaults(func=_inspect)

    evaluate = commands.add_parser(
        "create-evaluate", help="report a run's locked held-out evaluation and claims"
    )
    evaluate.add_argument("run")
    evaluate.set_defaults(func=_evaluate)

    export = commands.add_parser(
        "create-export",
        help="export a run as a standalone base-compatible model, or verify an export",
    )
    export.add_argument("run", nargs="?")
    export.add_argument("--output")
    export.add_argument("--verify", metavar="EXPORT_DIR")
    export.set_defaults(func=_export)

    generate = commands.add_parser(
        "create-generate", help="generate from an exported model with plain Transformers"
    )
    generate.add_argument("model")
    generate.add_argument("--prompt", required=True)
    generate.add_argument(
        "--chat", action="store_true", help="wrap the prompt in the model's chat template"
    )
    generate.add_argument("--max-new-tokens", type=int, default=80)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    generate.set_defaults(func=_generate)

    research = commands.add_parser(
        "expand-cluster",
        help="Expand-and-Cluster parameter recovery for small numerical-output MLPs",
        description="Reproduction of Martinelli et al. on its own assumptions: real-valued "
        "outputs of a one-hidden-layer network on controlled inputs. Not applicable to text.",
    )
    research.add_argument("input", help="NPZ with inputs (n,d) and outputs (n,k)")
    research.add_argument("--activation", choices=["tanh", "sigmoid", "softplus"], default="tanh")
    research.add_argument(
        "--teacher-width",
        type=int,
        help="known teacher width (evaluation only; not used by recovery)",
    )
    research.add_argument("--students", type=int, default=10)
    research.add_argument("--expansion", type=float, default=4.0)
    research.add_argument("--steps", type=int, default=6000)
    research.add_argument("--seed", type=int, default=0)
    research.add_argument("--output", help="also save the JSON report here")
    research.set_defaults(func=_expand_cluster)
