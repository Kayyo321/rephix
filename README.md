# rephix

[![tests](https://github.com/Kayyo321/rephix/actions/workflows/tests.yml/badge.svg)](https://github.com/Kayyo321/rephix/actions/workflows/tests.yml)

An experimental engine for reconstructing model behavior and identifiable parameters from observed evidence: prompts, responses, visible reasoning, token probabilities, and explicitly available model internals.

Rephix collects evidence, fits candidates, measures held-out agreement, and runs reproducible comparisons. With direct access to individual layers, it can export a recovered checkpoint for supported architectures and compare the stored tensors. Endpoint observations can leave different checkpoints indistinguishable even with unlimited queries.

## Run a persistent reconstruction session

The session workflow accepts local text models or an authorized agent endpoint. It learns a standalone byte-level candidate without the target tokenizer, saves evidence and recovery state across process restarts, and locks a validation-selected candidate before final testing. Real runs cover three text model families and a controlled two-step tool agent.

- [Session commands, target protocol, and capability matrix](docs/SESSIONS.md)
- [Cross-family results and restart verification](docs/SESSION_RESULTS.md)
- [Live agent, passive observation, and backend change](docs/LIVE_SESSION_RESULTS.md)
- [Behavioral identification and calibration](docs/IDENTIFICATION.md)
- [Output subspace recovery from dense log probabilities](docs/LOGIT_SUBSPACE.md)
- [Full checkpoint recovery with layer access and cross-family measurements](docs/LAYER_RECOVERY.md)
- [Layer staging for checkpoints larger than available GPU memory](docs/LAYER_STAGING.md)
- [Quantized GGUF recovery and packed weight audits](docs/GGUF_RECOVERY.md)
- [Full checkpoint recovery from passive layer traces](docs/PASSIVE_LAYER_RECOVERY.md)
- [ViT and DeiT classifier recovery](docs/VISION_RECOVERY.md)
- [T5-family and BART encoder–decoder recovery](docs/SEQ2SEQ_RECOVERY.md)
- [BERT-family encoder and task-head recovery](docs/ENCODER_RECOVERY.md)
- [Whisper speech recognition recovery](docs/WHISPER_RECOVERY.md)
- [Mamba state-space model recovery](docs/SSM_RECOVERY.md)
- [Recover a private head from endpoint scores and a known base](docs/ENDPOINT_HEAD_RECOVERY.md)

## Start with a controlled experiment

Requires Python 3.10 or later. Core commands need only NumPy.

```bash
git clone https://github.com/Kayyo321/rephix.git
cd rephix
python -m venv .venv
source .venv/bin/activate
pip install -e .
rephix doctor
rephix demo --seed 42 --output runs/affine.json
rephix symmetry-demo --seed 42 --output runs/symmetry.json
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

The affine demo recovers a planted layer and tests it on unseen features. The symmetry demo shows how different neuron coordinates can represent the same function. These establish small, measurable baselines before scaling to language models.

## Run the complete pipeline without downloads

Install the optional model and plotting libraries, then create and exercise a tiny local Qwen2 model:

```bash
pip install -e '.[hf,analysis]'
rephix doctor --require-hf
rephix offline-trial --output runs/offline-check --steps 32 --threads 2
```

The trial generates its own untrained model, tokenizer, and prompt corpus. It collects evidence, compares token-only and top-k objectives, selects a candidate using validation data, evaluates that candidate on a separate test split, recovers a restricted projection, selects new queries, and writes reports. It uses no pretrained checkpoint or network access. See [the offline trial guide](docs/OFFLINE_TRIAL.md) for artifacts and interpretation.

For a standalone reusable fixture, run `rephix init-tiny --output models/tiny-fixture`. For CPU-only PyTorch installation, see [the experiment guide](docs/EXPERIMENTS.md).

## Trial an open-weight Qwen reference

The [experiment guide](docs/EXPERIMENTS.md) covers downloading `Qwen/Qwen2.5-0.5B-Instruct`, collection, grouped splitting, fitting, resumption, and evaluation. Begin with a small random student architecture to limit memory. Full-parameter optimization of a 0.5B reference architecture is substantially more expensive than inference.

Two candidate priors are explicit:

- `--init random` uses the supplied architecture and tokenizer without initializing from reference weights. `--student-config` can supply a smaller architecture with the same vocabulary and special token IDs.
- `--init pretrained` starts from the reference weights. With `--trainable head`, the backbone remains frozen and the output head is optimized. Tied embeddings are untied so input embeddings stay frozen. This retains a substantial pretrained prior and is a distinct experiment from reconstruction using observations alone.

Collection and fitting both support `--resume`. Training keeps the latest optimizer state separate from the best validation checkpoint, verifies unchanged evidence and settings, and treats `--steps` as the total target. Hardware and library constraints on exact resumption are described in the guide.

[Local CUDA results](docs/LOCAL_RESULTS.md) compare token-only and top-k supervision on a larger family-split corpus. [Cloud trial results](docs/CLOUD_RESULTS.md) and [validation details](docs/VALIDATION.md) document earlier checks.

## Commands and measurements

| Command | Purpose | What the result establishes |
| --- | --- | --- |
| `doctor` | Inspect local libraries, devices, and model resource estimates | Available capabilities; no model downloads |
| `demo`, `recover-linear` | Recover an affine layer from known features | Parameter recovery when the reported design has sufficient rank |
| `symmetry-demo` | Align supported tiny-network permutations, scales, or signs | A restricted symmetry-aware comparison, not transformer alignment |
| `init-tiny`, `offline-trial` | Build a local fixture and run the experiment pipeline | Download-free pipeline verification |
| `collect`, `validate`, `split` | Record and isolate prompt evidence | Token provenance, signal availability, and split integrity |
| `fit`, `evaluate` | Fit candidates and measure held-out continuations | Behavioral reconstruction, separately from parameter recovery |
| `compare-weights` | Compare compatible checkpoint tensors | Coordinate distance without general symmetry alignment |
| `audit-weights` | Stream local safetensors and report exact agreement, tolerance agreement, and coverage | Agreement on same-name stored tensor coordinates, including buffers |
| `recover-gguf` | Rebuild a local GGUF checkpoint from chosen tensor-operator inputs | Exact float32 responses and known GGUF layout |
| `audit-gguf` | Compare original packed fields and dequantized GGUF values | Separate raw encoding and effective weight agreement |
| `recover-mxfp4-safetensors` | Rebuild packed GPT-OSS safetensors in Hugging Face or OpenAI's original layout | Exact float32 responses and known tensor layout |
| `audit-mxfp4-safetensors` | Compare packed GPT-OSS scales, codes, and numeric weights | Separate stored-field and effective weight agreement |
| `recover-fp8-safetensors` | Rebuild block-scaled E4M3FN safetensors with FP32 or BF16 scales | [Exact dequantized tensor responses](docs/FP8_RECOVERY.md) |
| `audit-fp8-safetensors` | Compare FP8 codes, scales, and decoded weights | Separate packed-field, exact numeric, and FP32 agreement |
| `recover-endpoint-head` | Fit and export a private head from dense endpoint scores and a known base | Raw head recovery under full rank with an absolute score or a trusted parameter prior |
| `capture-layer-traces`, `recover-layer-traces` | Export a fresh checkpoint using internal traces of ordinary inference | Raw recovery under per-layer rank and embedding coverage checks |
| `calibrate-layer-traces` | Add chosen evidence only for missing affine directions or lookup IDs | Resolve gaps left by passive traces under separate probe budgets |
| `recover-layers` | Reconstruct a supported local text, vision, or speech model from chosen layer inputs | Parameter recovery with known architecture and direct layer access, with optional resumable evidence |
| `replay-layers` | Rebuild a standalone checkpoint from a sealed [layer evidence bundle](docs/LAYER_RECOVERY.md#resume-and-replay-layer-probes) | Reuse recorded observations after removing the source checkpoint |
| `recover-projection` | Fit selected output weights using observed head inputs | White-box component recovery under an extra access assumption |
| `sweep`, `report` | Compare bounded settings and export tables or plots | Reproducible validation comparisons with a separate final test |
| `select-queries` | Rank new prompts by randomness, entropy, or disagreement | A query heuristic, not a recovery-accuracy measurement |

For custom affine evidence, store `features` shaped `(samples, features)` and `observations` shaped `(samples, classes)` in a NumPy archive:

```bash
rephix recover-linear data/linear.npz --kind logits --output runs/linear/recovered.npz
```

`--kind probabilities` and `--kind log_probs` accept complete normalized distributions and report class-centered parameters because softmax hides a common logit shift. Top-k evidence belongs in the language-model workflow; its loss includes one residual-probability bucket instead of inventing missing token probabilities.

## Evidence format

Evidence is JSON Lines, with one observation per line:

```json
{"id":"example-001","prompt":"What is 6 times 7?","output":"42","thinking":null,"metadata":{"source":"manual","group_id":"multiplication"}}
```

`output` may be empty before querying a reference. `thinking` stores text supplied or actually exposed by a model; it does not provide hidden reasoning or activations. If reasoning and the final answer are stored separately, supply the exact serialized completion in paired token-ID arrays or `metadata.completion_text`. The engine does not guess its formatting.

Optional paired `prompt_token_ids` and `completion_token_ids` preserve the actual chat prefix and completion. `top_logprobs` contains one list per completion token, with entries such as `{"token_id":17,"logprob":-0.8}`. Collector metadata binds cached tokens to their tokenizer and records source, decoding, and numerical provenance. Extra context is preserved in metadata; include it in the actual prompt when it should condition the model.

Repeated prompts and shared `metadata.group_id` values stay together during splitting. Assign related paraphrases a shared group id; automatic grouping cannot recognize every semantic overlap. Reserve test groups before selecting queries or tuning settings.

## Guides

- [Experiments and local Qwen setup](docs/EXPERIMENTS.md)
- [Local CUDA experiments and reproduction](docs/LOCAL_RESULTS.md)
- [Complete offline trial](docs/OFFLINE_TRIAL.md)
- [Sweeps, resumption, and reports](docs/SWEEPS.md)
- [Query selection and reserved-set exclusions](docs/QUERY_SELECTION.md)
- [White-box component recovery](docs/COMPONENT_RECOVERY.md)
- [Identifiability and supported symmetries](docs/IDENTIFIABILITY.md)
- [Completion goal](docs/GOAL.md), [validation](docs/VALIDATION.md), and [cloud results](docs/CLOUD_RESULTS.md)

## Development

```bash
pip install -e '.[dev,hf,analysis]'
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
```

Tests construct local model fixtures and need no pretrained downloads. Installing only `.[dev]` runs core coverage and skips tests requiring optional model libraries. CI checks supported Python versions and the offline model path.

Licensed under [Apache License 2.0](LICENSE).
