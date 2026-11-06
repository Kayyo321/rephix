# Reproducible sweeps and comparison reports

Use a sweep to compare a bounded set of hypotheses against the same training and validation evidence. The runner never reads a test set or chooses a winning configuration. Final test evaluation is separate, after you fix the settings.

```bash
rephix sweep --config examples/sweep.json --output runs/evidence-comparison
rephix sweep --config examples/sweep.json --output runs/evidence-comparison --resume
rephix report --runs runs/evidence-comparison --output runs/evidence-comparison/report --plot
```

The example compares text-token cross entropy (`topk_weight=0`) with recorded top-k distribution supervision (`topk_weight=1`) at two seeds. Collection must include top-token probabilities for the latter to add evidence. If a record has no probability observations, its training target remains the recorded token. This example is a short pipeline check; random full-parameter Qwen training remains expensive.

## Configuration

`examples/sweep.json` is a strict schema-v1 file. Unknown keys and duplicate grid points are rejected so spelling errors cannot silently change an experiment. Paths are resolved relative to the JSON file. Model IDs such as `Qwen/Qwen2.5-0.5B-Instruct` remain remote references; prefix a local relative path with `./` or `../`.

Required keys are `schema_version: 1`, `teacher`, `train`, and `validation`. Optional top-level keys are `revision`, `device`, `defaults`, and `grid`. The grid takes a nonempty list of values for each parameter and runs their Cartesian product. At most 1024 combinations are accepted per sweep.

Both `defaults` and `grid` accept:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `seed` | `42` | Nonnegative Python/PyTorch seed |
| `steps` | `100` | Total optimizer-step budget per run |
| `learning_rate` | `0.00005` | Positive optimizer learning rate |
| `topk_weight` | `1.0` | Mix between token loss and top-k-plus-tail loss |
| `max_seq_length` | `256` | Prompt and supervised completion budget |
| `init` | `"random"` | `"random"` or explicit `"pretrained"` prior |
| `trainable` | `"all"` | `"all"` or output `"head"` only |
| `student_config` | `null` | Optional smaller architecture for random initialization |
| `student_revision` | `null` | Separate immutable commit for a remote student configuration |
| `gradient_accumulation_steps` | `1` | Microbatches per optimizer step |
| `eval_every` | `10` | Validation/checkpoint selection interval |
| `save_every` | `10` | Latest-state save interval |

Remote teacher references require an immutable 40-character hexadecimal commit in `revision`. A remote `student_config` requires its own immutable `student_revision`; a teacher commit cannot pin a different repository. The sweep fingerprints local source configuration, tokenizer, and weight files. Local paths must exist before the sweep starts.

## Resuming and integrity

The root `sweep.json` records the resolved settings, source/evidence hashes, deterministic run IDs, states, timing, and completed-metrics hashes. These metrics checksums do not independently authenticate saved checkpoint tensors. Repeated IDs, prompts, or explicit groups across training and validation are rejected before model loading.

`--resume` verifies that configuration and evidence are unchanged. Completed runs are skipped only when their saved metrics still match the manifest. An interrupted run resumes through its atomically saved `training_state.pt`; that file contains the latest model, optimizer, ordering, and random state. The best inference checkpoint remains a separate training artifact. If initialization failed before a durable state existed, the runner preserves the failed directory and asks for a fresh output directory instead of deleting or silently replacing evidence.

The same schema and data cannot be changed in-place to extend a completed sweep. Start a new sweep for a new grid or budget, or resume an individual fit with its documented total-step semantics.

## Reports

Each completed sweep writes `report/comparison.json`, `comparison.csv`, and `comparison.md`. Reports retain per-run seed, evidence coverage, initialization, learning settings, device, validation checkpoint choice, and elapsed time. Unknown token counts are left blank rather than estimated from characters. The recorded evidence count differs from optimizer token exposures.

`rephix report --runs` accepts one or more run directories, sweep directories, `metrics.json` files, or quoted glob patterns. It verifies sweep metrics hashes when a manifest is available. Runs are listed deterministically; rows are not ranked into a preferred result.

With `--plot`, the report writes `learning-curves.svg` using standard Matplotlib plotting. Install the optional plotting dependencies with `pip install -e '.[analysis]'`. If Matplotlib is unavailable, JSON/CSV/Markdown reports still work and the JSON report records that plotting was unavailable. Curves show validation continuation loss and recorded-token agreement against optimizer steps. Neither is a parameter-recovery measurement.

Keep one untouched test split outside the sweep. Comparing many settings on validation is legitimate, but repeatedly acting on test results turns that test set into additional validation data.
