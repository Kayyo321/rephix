# Recover a private output head from endpoint scores

`recover-endpoint-head` starts from a local Transformers checkpoint that supplies the target's unchanged backbone and one trusted output row. It takes full-vocabulary endpoint scores and exports a standalone checkpoint with the fitted head. The estimator reads the public base and the evidence arrays. It does not load the target checkpoint or read target hidden states.

## Observation contract

- The target uses the base backbone in the same hidden coordinates. Its next-token logits are affine in the last hidden vector. Target and base have the same token IDs and output order.
- At least one target output row, including its bias when present, is known to match the base. Choose that token ID with `--anchor-id`.
- The endpoint provides a finite dense vector of raw logits or natural log probabilities for each supplied context. Token IDs cover the model's output vocabulary, including any coordinates outside the tokenizer's text vocabulary. Top-k scores, sampled text, and clipped or missing coordinates cannot supply these observations.
- The fitting contexts excite every required feature direction. Use separate validation contexts to catch incompatibility, and reserve a third set for independent evaluation.

For token `j` and anchor `a`, the observed log odds are `log p(j|x) - log p(a|x) = (w_j - w_a) · h(x) + b_j - b_a`. Rephix computes `h(x)` on the base, factors the feature matrix once in FP64, and solves all token contrasts in vocabulary chunks. A known anchor fixes the common shift. Without the anchor, adding the same weight vector and bias to every output row gives exactly the same probabilities. Held-out score agreement cannot verify the supplied anchor. [Grey-box extraction with a public encoder](https://proceedings.mlr.press/v139/zanella-beguelin21a.html)

The solver rejects effective rank deficiency. Its relative singular-value cutoff accounts for the input feature dtype, so forward rounding cannot make an unobservable direction appear usable. For a biased head the design includes an intercept. The export gate checks held-out contrast RMSE both for the FP64 fit and in the actual FP32 candidate. A passing gate verifies agreement on the supplied observations under this design. It cannot establish that an unknown target shared the base backbone or original gauge.

## Run the command

Each evidence directory contains three non-pickled NumPy arrays, memory-mapped on read.

| File | Shape | Meaning |
| --- | --- | --- |
| `input_ids.npy` | `[contexts, width]` | Integer token IDs of the exact submitted contexts, right padded |
| `lengths.npy` | `[contexts]` | Number of real tokens in each row |
| `scores.npy` | `[contexts, model_vocab_size]` | Finite float scores in token-ID order, at the last real token |

```bash
pip install -e '.[hf]'
rephix --threads 4 recover-endpoint-head \
  --base models/public-base --train evidence/fit --validation evidence/validation \
  --anchor-id 0 --max-validation-rmse 0.001 --output runs/private-head --device cuda
```

Use the true anchor ID for your evidence. Validation and training must have no exact token context in common. The command loads local safetensors, exports FP32 weights, and writes `endpoint-head.json` with rank, conditioning, and fit errors. A tied base head is untied before installing the recovered weights so input embeddings stay intact. The tokenizer is included by default. Some checkpoints need SentencePiece and protobuf for tokenizer loading.

## Pretrained experiments

The positive controls planted a dense FP32 change across the output head of an existing pretrained checkpoint, preserving token 0's row and the whole backbone. This lets the evaluator measure agreement specifically on changed head coordinates alongside the whole checkpoint. An unchanged-base baseline quantifies how much of the result follows from copying the supplied prior. The independent estimator process received only the public base and fitting and validation evidence. In the precision control, the target computed the head multiplication and log-softmax in FP64 with the same FP32 backbone and weight values.

The negative control used Qwen2.5-0.5B-Base for features and Qwen2.5-0.5B-Instruct for scores in the same token coordinates. The separately reserved natural prompts measured full-vocabulary predictions of reloaded candidates. The benchmark harness is [endpoint_head_experiment.py](../examples/endpoint_head_experiment.py).

| Pretrained base and score precision | Fitting contexts | Whole checkpoint within tolerance | Changed head values within tolerance | Changed head relative L2 |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2-135M, FP32 | 1,728 | 94.34% | 67.44% | `4.90e-6` |
| SmolLM2-135M, FP32 | 6,912 | 97.62% | 86.33% | `1.94e-6` |
| SmolLM2-135M, FP64 head | 1,728 | 100% | 100% | `3.53e-17` |
| Lite-Mistral-150M-v2-Instruct, FP32 | 2,304 | 99.00% | 93.78% | `1.09e-6` |
| Lite-Mistral-150M-v2-Instruct, FP64 head | 2,304 | 100% | 100% | `1.07e-17` |

The tolerance was `1e-8 + 1e-5 * abs(reference)`. The base-only checkpoints scored 82.61% and 83.92% overall, respectively, and under 0.004% on changed head values. All five reloaded exports matched 24/24 next-token predictions on the independent natural prompts. The largest full-vocabulary log-probability RMSE on those prompts was `7.14e-6` nats. The fitting and validation contexts were sampled token-ID sequences sent directly to the local endpoint simulation. [Pinned runs, metrics, and evidence digests](../examples/results/endpoint-head-recovery.json)

Pythia-70M's fitted feature design had effective rank 511 of 512 despite 1,536 fitting contexts and a planted head with an unchanged backbone. Rephix rejected the fit. The real Qwen backbone mismatch had held-out contrast RMSE of 1.68548 nats against the 0.001-nat gate and was also rejected. Neither control produced a checkpoint.

These experiments measure a private head in known coordinates and use the public base for the rest of the model. [The sign-symmetry experiment](LAYER_RECOVERY.md#why-an-endpoint-cannot-guarantee-95-raw-weight-agreement) covers the case where endpoint outputs cannot distinguish different whole-model parameterizations.

## Reproduce a control

Use the pinned base checkpoints in the results file. The harness writes dense evidence and intermediate checkpoints, so use a disposable directory on disk.

```bash
python examples/endpoint_head_experiment.py --base models/lite-mistral \
  --output runs/lite-mistral-fp64 --device cuda --endpoint-precision float64
python examples/endpoint_head_experiment.py --base models/qwen-base \
  --target models/qwen-instruct --output runs/qwen-mismatch --device cuda --fit-multiplier 2
```

To reproduce the Pythia control, run the harness on Pythia-70M with `--expect-rank-deficient`. Copy each run's `report.json` out of the scratch directory before removing its evidence, reference snapshots, and exports.
