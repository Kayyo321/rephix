# rephix

An experimental engine for reconstructing model parameters and behavior from observed inputs and outputs.

The starting idea is simple: record the prompt, response, any reasoning text the model actually exposes, and useful surrounding evidence; fit a candidate; measure how closely it reproduces an open-weight reference; repeat. A small Qwen model can serve as the reference when you are ready to run local trials.

This repository separates two questions:

- **Parameter recovery:** did we recover the reference weights, under stated assumptions?
- **Behavioral reconstruction:** does a candidate predict the reference model’s outputs on held-out inputs?

Matching responses does not establish matching weights. Many different parameter settings can produce the same observed outputs, and finite text examples usually do not determine a unique neural network.

## Start here

Requires Python 3.10 or later. The core experiment needs only NumPy.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
rephix demo --seed 42 --output runs/demo.json
python -m unittest discover -s tests -v
```

The demo generates a known affine model, observes its outputs, solves for its weights, and reports held-out error and recovery diagnostics. It proves the recovery machinery on a controlled, identifiable problem. It does not recover an entire language model from text.

```bash
rephix --help
rephix recover-linear --help
rephix validate examples/prompts.jsonl
```

For your own affine observations, save a NumPy `.npz` file with `features` shaped `(samples, features)` and `observations` shaped `(samples, classes)`, then run:

```bash
rephix recover-linear data/linear.npz --kind logits \
  --output runs/linear/recovered.npz
```

Use `--kind probabilities` for full class probabilities or `--kind log_probs` for full log probabilities. See `--help` for regularization options.

## Local Qwen trials

Install the optional language-model dependencies when ready:

```bash
pip install -e '.[hf]'
rephix collect --help
rephix fit --help
rephix evaluate --help
rephix compare-weights --help
```

The workflow collects continuations from a local reference, optionally saves top-token log probabilities, fits a candidate, and evaluates it on separate prompts. The default reconstruction starts from random weights with the reference architecture and tokenizer. Starting from pretrained weights is an explicit prior and should be reported as a separate experiment.

[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) describes the Qwen workflow and how to interpret its measurements. [`docs/ROADMAP.md`](docs/ROADMAP.md) lists the next research steps.

[`docs/VALIDATION.md`](docs/VALIDATION.md) records the initial checks: 28 passing tests, including an offline tiny-Qwen pipeline. No pretrained Qwen model is bundled.

A model such as `Qwen/Qwen2.5-0.5B-Instruct` is small by language-model standards but still expensive to train. Full-parameter optimization can require several gigabytes of memory beyond the model itself. A short CPU smoke run checks the pipeline; it is not evidence that text-only recovery of 0.5 billion parameters is feasible.

## Evidence format

Evidence uses JSON Lines, one observation per line:

```json
{"id":"example-001","prompt":"What is 6 times 7?","output":"42","thinking":null,"metadata":{"source":"manual"}}
```

Optional `prompt_token_ids`, `completion_token_ids`, and `top_logprobs` preserve token-level evidence. Each `top_logprobs` position contains entries such as `{"token_id": 17, "logprob": -0.8}`. Collector-produced token IDs preserve the actual chat prefix and completion, avoiding ambiguity from decoding and retokenizing text.

`thinking` means reasoning text that was supplied or actually emitted. It is not access to hidden reasoning or internal activations. A separate annotation is not silently inserted into a training target: preserve the exact visible completion in collected token IDs or `metadata.completion_text`. Extra context is preserved in `metadata`; it conditions the model only when it is included in the actual prompt or captured token sequence.

Start with [`examples/prompts.jsonl`](examples/prompts.jsonl), then add varied prompts before attempting meaningful training. Keep all observations of the same prompt in one split to prevent leakage.

## What this version measures

| Experiment | Available evidence | Meaningful result |
| --- | --- | --- |
| Affine recovery | Known input features and full logits | Recover weights and bias when the design has full rank |
| Probability recovery | Known features and full class probabilities | Recover parameters after fixing the softmax common-shift ambiguity |
| Qwen reconstruction | Prompts, continuations, optional top-token probabilities | Measure continuation prediction and held-out output agreement |
| Checkpoint comparison | Both compatible checkpoints | Measure direct parameter distance, without resolving neural-network symmetries |

Top-token probabilities constrain only part of the distribution. The training objective groups the remaining vocabulary into one residual bucket; it does not invent the missing probabilities.

The repository does not include model weights or claim high reconstruction accuracy. The useful next result is a reproducible baseline showing what can be recovered from a stated amount and kind of evidence.
