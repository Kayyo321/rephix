# Recover a checkpoint from passive layer traces

`capture-layer-traces` observes paired inputs and outputs of parameterized layers during ordinary inference. It supplies token IDs through the decoder's normal forward entry point. No isolated target layers are called with chosen hidden vectors. `recover-layer-traces` builds a fresh architecture from a weights-free configuration and fits its parameters from the saved traces. The exported safetensors checkpoint runs without the traces or the source model.

This access contract exposes internal tensors in the source model's native coordinates. A text endpoint that returns tokens or logits alone cannot provide these traces. [Layer-probe recovery and endpoint identifiability](LAYER_RECOVERY.md)

## Run the experiment

The source is a local Transformers safetensors directory. The fitting and validation inputs are rectangular integer NumPy arrays with shape `[contexts, sequence_length]`. Every position is a real input token. The contexts must be disjoint across the two files. The preparation script below includes every embedding ID in the fitting set and produces a separate directory with configuration and tokenizer files.

```bash
python examples/passive_layer_experiment.py prepare \
  --source models/reference --work runs/passive
rephix --threads 4 capture-layer-traces --model models/reference \
  --train runs/passive/train.npy --validation runs/passive/validation.npy \
  --output runs/passive/traces --device cuda --dtype float64 --batch-size 16
rephix --threads 4 recover-layer-traces --model-config runs/passive/config \
  --traces runs/passive/traces --output runs/passive/recovered
rephix audit-weights models/reference runs/passive/recovered
```

Use `--without-tokenizer` for config-only fixtures consumed through token IDs. For a tokenizer that needs SentencePiece, install `sentencepiece` and `protobuf` and include the tokenizer model. Capture defaults to FP64 to reduce numerical loss in the traced values. FP32 is also supported, with a rank cutoff that accounts for its precision. The exported checkpoint uses the trace dtype.

The collector memory-maps arrays and caps stored rows separately for each operator. The default uses up to twice the affine design width per operator and up to 64 fitting rows per normalization layer. `--fit-multiple` increases the affine budget. `--validation-rows` controls the held-out cap. The inference pass still executes the submitted contexts, and the full-vocabulary embedding table is stored once. Use a disposable disk directory. The captures below each occupied several GB even at roughly 150M parameters.

The trace directory includes a manifest with the exact config hash, operator names and shapes, capture precision, and actual counts. Each numbered operator directory has `x.npy` and `y.npy` arrays containing observed inputs and outputs. An embedding instead records its table in `y.npy` and a coverage mask in `seen.npy`. An external trace producer can supply the same layout and manifest. The exporter checks configuration and tensor shapes before publishing.

## What makes the recovery identifiable

- **Affine operators.** Recover matrix and bias from the paired activations. The design needs one independent direction per feature, plus an intercept when the operator has bias. Recovery uses FP64 SVD and rejects directions below the numerical rank cutoff.
- **Normalization.** Compute the normalization from the traced inputs using the architecture and epsilon. Fit each observed output coordinate against that normalized input. For a biased normalization each coordinate needs variation to determine both scale and bias.
- **Embeddings.** Map each observed input ID to its observed output row. The exporter requires complete token coverage. Configured ties are preserved, so a tied output head does not need a separate regression.

For each fitted operator the exporter checks held-out RMSE on the solve and again using the installed candidate tensor. A failed rank, embedding coverage, or validation check prevents the whole checkpoint from being published.

For a biased affine layer fed by LayerNorm, every normalized vector `h` obeys `sum(h / gamma) = sum(beta / gamma)` when the scales are nonzero. The augmented affine input matrix has an exact null direction. Adding `c / gamma` to a row and subtracting `c * sum(beta / gamma)` from its bias preserves the layer on all such inputs. Higher precision or more examples within that hyperplane cannot resolve the two parameters. [LayerNorm geometry](https://arxiv.org/abs/2305.02582)

## Pretrained results

The trials captured FP64 activations on an RTX 5080 and solved the fits on CPU. The fitting contexts were 16-token sequences formed by permuting all input IDs. Separate token sequences supplied validation traces. A third panel of natural prompts evaluated the reloaded checkpoint.

| Source | Stored parameter values | Within tolerance | Relative L2 | Trace disk space | Next-token top-1 / greedy |
| --- | ---: | ---: | ---: | ---: | ---: |
| SmolLM2-135M | 134,515,008 | 100% | `5.88e-10` | 4.13 GiB | 24/24 · 6/6 |
| Lite-Mistral-150M-v2-Instruct | 156,519,168 | 100% | `4.47e-9` | 4.75 GiB | 24/24 · 6/6 |

The raw weight audit used `1e-8 + 1e-5 * abs(reference)`. The independent full-vocabulary logit RMSE was `1.73e-6` on SmolLM2 and `7.33e-7` on Lite-Mistral. The estimator's configuration directory contained only configuration and tokenizer files. [Pinned runs and audits](../examples/results/passive-layer-recovery.json)

On Pythia-70M the first attention QKV layer had rank 512 of 513, so the exporter stopped. An explicit unit-norm perturbation along the LayerNorm null direction changed that row's outputs by at most `1.8e-15` across 1,026 fitting rows and 128 validation rows. Moving the validation inputs off the hyperplane along the same direction exposed a unit output difference. This is a limit of the observed inputs even when the tensor coordinates and layer type are known.

## Complete missing directions with a few chosen inputs

`calibrate-layer-traces` plans extra observations from the passive input arrays. Full-rank affine layers need no calls. For a design of width `d` and numerical rank `r`, it selects `d-r` activation rows, the minimum number that can fill the measured gap because each additional row increases rank by at most one. In biased layers the intercept stays fixed at one. The planner verifies the combined rank after casting its proposed inputs to the trace dtype. It tries a bounded set of scales and rejects a plan if none passes. Missing embedding IDs can be observed by direct lookup under a separate budget.

```bash
rephix --threads 4 calibrate-layer-traces --model models/reference \
  --traces runs/passive/traces --output runs/passive/chosen --device cuda \
  --max-probes 128 --max-embedding-rows 1008
rephix --threads 4 recover-layer-traces --model-config runs/passive/config \
  --traces runs/passive/traces --calibration runs/passive/chosen \
  --output runs/passive/recovered-calibrated
```

Set the two budgets for your trace. The default embedding budget is zero. All operators are planned before any chosen target calls. The extra arrays are tied to the passive trace manifest and the fitted operator's passive arrays by SHA-256 digests. The offline exporter appends them to the fitting observations, then validates on the untouched passive holdout. The layer oracle is used by the calibration command. The estimator still uses only the configuration and evidence directories. For an external oracle, call `rephix.trace_calibration.collect_trace_calibration` with a callback accepting the operator path and its activation rows or integer lookup IDs.

| Reference | Chosen affine rows | Full affine basis rows | Chosen lookup IDs | Parameters within tolerance |
| --- | ---: | ---: | ---: | ---: |
| Pythia-70M | 12 | 22,040 | 0 | 100% |
| GPT-2 | 24 | 64,560 | 1,008 | 100% |
| SmolLM2-135M | 0 | 149,760 | 0 | 100% |

Pythia and GPT-2 each needed one chosen row for every rank-deficient affine operator. The learned GPT-2 position table needed IDs 16 through 1,023 because the passive contexts were 16 tokens long. SmolLM2 was the zero-probe control. Pythia and GPT-2 were compared against the parameters after loading and reserializing their legacy checkpoints. Relative L2 errors were `1.06e-9`, `2.16e-13`, and `5.88e-10`, respectively. All three independently reloaded exports matched 24/24 next-token predictions and 6/6 greedy continuations.

Replacing the first Pythia probe and its output with a genuine row from that layer's passive trace kept the calibration row count and file digests valid. Its affine design still had rank 512 of 513 and the exporter refused to publish. [Calibrated runs and control](../examples/results/calibrated-layer-recovery.json)

## Sparse experts

Mixtral executes only selected expert projections for each token. The capture manifest records the actual rows observed by each projection, including zero. The fitter uses only that prefix of the allocated arrays. This keeps unfilled capacity out of the rank calculation.

An affine expert with no fitting rows can be recovered from a complete chosen basis. If it also has no passive validation rows, `--max-validation-probes` lets calibration query a distinct held-out input. That response is saved separately and used only for validation. The default validation probe budget is zero. Both fitting and validation budgets are checked for the whole request before any chosen calls. This path requires access to the individual expert projections. Ordinary decoder outputs may contain no signal about experts that were never selected.

For a Mixtral capture with a one-token validation split, the following settings reproduce the pretrained run's budgets:

```bash
python examples/passive_layer_experiment.py prepare --source models/tinymoe \
  --work runs/tinymoe --validation-contexts 1 --validation-length 1
rephix --threads 4 capture-layer-traces --model models/tinymoe \
  --train runs/tinymoe/train.npy --validation runs/tinymoe/validation.npy \
  --output runs/tinymoe/traces --device cuda --batch-size 16
rephix --threads 4 calibrate-layer-traces --model models/tinymoe \
  --traces runs/tinymoe/traces --output runs/tinymoe/chosen --device cuda \
  --max-probes 8192 --max-validation-probes 180
rephix --threads 4 recover-layer-traces --model-config runs/tinymoe/config \
  --traces runs/tinymoe/traces --calibration runs/tinymoe/chosen \
  --output runs/tinymoe/recovered
```

| TinyMoE trial | Chosen fitting rows | Chosen validation rows | Expert weights with zero passive fitting rows | Stored values within tolerance |
| --- | ---: | ---: | ---: | ---: |
| Pretrained router | 3,958 | 180 | 0 | 100% |
| Engineered first router | 31,977 | 42 | 5,308,416 | 100% |

The checkpoint was [TinyMoE-100m-2x8-retrained](https://huggingface.co/FlameF0X/TinyMoE-100m-2x8-retrained), a pretrained Mixtral with 99,817,344 stored values. A full affine basis would use 142,464 rows. Each reloaded export matched all 24 next-token predictions and six greedy continuations in the final natural-prompt panel. Whole-checkpoint relative L2 errors were `8.05e-9` and `6.99e-9` for the pretrained and engineered routers, respectively.

The engineered control replaces only the first router with four nonzero rows proportional to `[2, 1, -2, -1]` and four zero rows. Across the observed contexts, experts 4–7 in that block received no fitting or validation tokens. Two more experts in later blocks received no fitting tokens. The audit covered all named and shaped tensors, including every unrouted projection. The pretrained run also verified a lower fitting budget of 2,048 rows was rejected before any chosen observations were collected. [Pinned results and per-operator probe records](../examples/results/sparse-expert-recovery.json)

For the engineered control, first run `python examples/passive_layer_experiment.py engineer-router --source models/tinymoe --work models/tinymoe-control`. Use the resulting reference directory in the same pipeline, with the default 32 validation contexts, `--validation-rows 512` on capture, and calibration budgets of 32,768 fitting rows and 240 validation rows.
