# A complete trial without downloading a model

The offline trial exercises the reconstruction toolkit with a locally initialized, untrained Qwen2 model. It needs the optional model dependencies but no pretrained checkpoint, API account, GPU, or network connection once those dependencies are installed.

```bash
pip install -e '.[hf,analysis]'
rephix offline-trial --output runs/offline --steps 32 --seed 42
```

`analysis` adds Matplotlib for standalone comparison figures. Without it, the trial still produces JSON, CSV, and Markdown reports and records that plots are unavailable. The default CPU thread count is one because this model is small; override it with `--threads N` if needed. Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to enforce offline Transformers behavior as an additional check.

Use a fresh or empty output directory. The command preserves existing run directories and records its current stage in `trial-status.json`. Individual fits and sweeps support their own resume commands; rerunning the entire offline trial requires a new directory.

## What runs

1. Create a two-layer Qwen2 reference with hidden size 32, a 64-token tokenizer, and seed 7. This reference is untrained. Synthetic prompts cover combinations of known words and number tokens; they are intended to exercise the pipeline, not language understanding.
2. Collect up to four greedy completion tokens for each of 48 independent prompts, with eight raw top-token log probabilities per position.
3. Split prompt groups into training, validation, and untouched test sets.
4. Train two random-initialized candidates for equal optimizer-step budgets: one on recorded-token cross entropy, the other on the top-k probability objective with a residual bucket. Both start from the same candidate seed and architecture.
5. Compare the saved candidates using their validation continuation cross entropy, select the lower value, and then evaluate that candidate on the test split exactly once. Test results are not used to choose an objective or checkpoint.
6. Run a separate output-projection recovery experiment using observed hidden features and logits from the reference. This is explicitly white-box access and does not contribute information to candidate training.
7. Select eight new prompts by the chosen candidate's next-token entropy, excluding all collected prompt groups, including reserved test groups.
8. Demonstrate supported permutation and scale symmetries in a separate tiny network.

The fit engine selects a checkpoint within each run using that run's validation reconstruction objective. The trial then uses a common validation continuation-loss metric to choose between the two saved checkpoints. This fixed comparison is a pipeline example, not a general claim that either training objective is better.

## Reading the artifacts

| Path | Contents |
| --- | --- |
| `fixture/model/` | Generated reference checkpoint and tokenizer |
| `fixture/fixture.json` | Reference architecture, parameter count, seed, and tokenizer fingerprint |
| `observations.jsonl` | Collected continuations and token probability evidence |
| `splits/manifest.json` | Independent split counts and file hashes |
| `sweep/` | Both candidates, durable training state, validation histories, and sweep identity |
| `comparison/comparison.json`, `.csv`, `.md` | Validation-only comparison of both runs |
| `comparison/learning-curves.svg` | Validation curves when Matplotlib is installed |
| `selection.json` | The validation-only choice, written before test evaluation |
| `test.json` | The chosen candidate's single final test evaluation |
| `projection/report.json` | Rank, conditioning, parameter error, and held-out projection error |
| `next-queries.jsonl` | Unqueried prompts chosen without reading target completions |
| `symmetry.json` | Restricted hidden-neuron symmetry comparison |
| `trial.json`, `trial.md` | Overall report and readable summary |

The trial is reproducible at a fixed dependency stack, device, seeds, and thread count. GPU kernels and future dependency changes can alter numeric results. The fit records the actual versions and initialization source.

## Creating just the fixture

```bash
rephix init-tiny --output models/tiny-reference --seed 7 --hidden-size 32 --layers 2
```

This produces a model under `models/tiny-reference/model`, 64 prompt rows in `prompts.jsonl`, and a 96-row query pool in `pool.jsonl`. The pool contains those 64 prompts plus 32 additional groups, so exclusions can be tested against both previously used and new entries. Hidden size must be divisible by eight for the fixture's attention layout. No pretrained weights are included or downloaded.

The tiny fixture is useful for testing data flow, optimization, held-out evaluation, component recovery, and resumption. Even perfect agreement with its outputs would not establish useful language behavior or recovery accuracy for a trained 0.5B model. Follow [EXPERIMENTS.md](EXPERIMENTS.md) for the separate pretrained Qwen experiment.
