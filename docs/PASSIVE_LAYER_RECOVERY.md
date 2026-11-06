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
