# Reproducible reconstruction experiments

Use separate experiment tracks for identifiable parameter recovery and behavioral reconstruction. Record which evidence and prior each candidate receives, then compare it on independent data. The [validation record](VALIDATION.md) and [cloud results](CLOUD_RESULTS.md) describe completed checks; the commands below are a recipe for new runs, not a promise of accuracy.

## 1. Verify the local environment

Clone the repository and create a Python 3.10+ environment as shown in the README. NumPy is the sole core dependency. Optional `hf` libraries provide PyTorch/Transformers experiments; optional `analysis` adds Matplotlib plots:

```bash
pip install -e '.[hf,analysis]'
rephix doctor --require-hf
rephix demo --seed 42 --output runs/affine.json
rephix symmetry-demo --seed 42 --output runs/symmetry.json
```

For a CPU-only environment, install the CPU PyTorch build before the extra:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[hf,analysis]'
```

`rephix --threads 2 COMMAND ...` bounds PyTorch CPU threads for model commands. Try a small value when oversubscription makes a short CPU run slow. Device choices are `auto`, `cpu`, `cuda`, and `mps`; availability is checked locally. CPU tests do not establish CUDA or MPS performance or determinism.

The affine demo reports rank, conditioning, parameter error, and held-out prediction error. A rank-deficient system can reproduce observations while leaving parameters undetermined. With full class probabilities, a common additive logit shift is invisible, so the solver compares centered parameters. The symmetry demo separately illustrates supported tiny-network equivalences. See [identifiability](IDENTIFIABILITY.md) for their scope.

## 2. Run a complete download-free trial

```bash
rephix offline-trial --output runs/offline-check --steps 32 --threads 2
```

This creates an untrained local Qwen2 model, a small tokenizer, and synthetic prompt corpora. It exercises collection, splitting, equal-budget objective comparison, validation-based candidate selection, one final held-out evaluation, component recovery, and query selection. It writes tables, model artifacts, and an experiment summary. The output directory must be fresh. See [OFFLINE_TRIAL.md](OFFLINE_TRIAL.md) for its exact artifact layout and limitations.

To create only the reusable fixture:

```bash
rephix init-tiny --output models/tiny-fixture --seed 7 --hidden-size 32 --layers 2
rephix doctor --model models/tiny-fixture/model
```

The fixture contains `model/`, `prompts.jsonl`, `pool.jsonl`, and `fixture.json`. Its outputs reflect an untrained model; success demonstrates an experiment path at a tiny scale, not reconstruction of a trained Qwen model.

## 3. Download and inspect an open-weight reference

One reference is `Qwen/Qwen2.5-0.5B-Instruct`. Download its Transformers checkpoint and tokenizer once, then use that local path:

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/qwen2.5-0.5b-instruct
QWEN_MODEL="$PWD/models/qwen2.5-0.5b-instruct"
rephix doctor --model "$QWEN_MODEL" --require-hf
```

For a reproducible remote snapshot, supply an immutable model commit with `hf download --revision COMMIT_SHA`. Keep the downloaded files unchanged. Record model revision, tokenizer/template, precision, decoding settings, and hardware. The collector records these where available. A quantized reference is a different function and should be a separate experiment.

`doctor` inspects an existing local or already cached config and never downloads weights. Its Qwen2 memory estimates cover dense float32 parameters and basic training state; they exclude activations, logits, framework overhead, and checkpoint copies. A 0.5B model can be affordable to query yet expensive to optimize in full.

## 4. Collect durable evidence

Start with a small collection before expanding the corpus:

```bash
rephix --threads 2 collect --prompts examples/prompts.jsonl \
  --model "$QWEN_MODEL" --output data/observations.jsonl \
  --device cpu --max-new-tokens 16 --top-k 20 --seed 42
```

Input records may have empty outputs. Collection records the actual rendered prompt and completion token IDs, greedy output text, and optional top-token probabilities. `--top-k 0` collects tokens/text without probability supervision. Top-k log probabilities come from raw teacher-forced model logits, before generation processing. They align with each completion token's causal prediction position.

Collection and evaluation use explicit neutral greedy decoding with architecture special-token IDs. Downloaded sampling, beam-search, and repetition-penalty defaults cannot silently change the comparison. Metadata records the effective decoding configuration, tokenizer fingerprint, source, device, and library versions.

Each completed observation is saved durably. After an interruption, repeat the command with `--resume`:

```bash
rephix --threads 2 collect --prompts examples/prompts.jsonl \
  --model "$QWEN_MODEL" --output data/observations.jsonl \
  --device cpu --max-new-tokens 16 --top-k 20 --seed 42 --resume
```

Resume requires the companion `.collection.json` journal and unchanged prompt evidence, settings, and source. Completed observations are retained; collection continues from the next one. Existing outputs are rejected without explicit `--resume`. Do not repurpose one collection output for a changed reference or generation budget.

Visible reasoning may be part of the raw completion. A separate `thinking` field is an annotation, not access to hidden activations. Imported records with separately readable reasoning and answer fields need the exact serialized completion in paired token-ID arrays or `metadata.completion_text`. Extra context conditions the model only when included in its actual prompt.

## 5. Isolate training, validation, and test groups

```bash
rephix validate data/observations.jsonl
rephix split data/observations.jsonl --output-dir data/splits \
  --validation-fraction 0.2 --test-fraction 0.2 --seed 42
```

Splitting writes `train.jsonl`, `validation.jsonl`, `test.jsonl`, and a hashed manifest. Groups connect normalized duplicate prompts and `metadata.group_id`, including transitive connections. Related paraphrases should share an explicit group id. Group sizes can make the realized row fractions differ from the requested fractions.

Training updates parameters, validation selects settings/checkpoints, and test data measures the final chosen experiment. Repeatedly changing settings after seeing test results turns those test groups into additional validation data. Use fresh test groups for the next final estimate.

Fitting rejects train/validation overlap. Candidate metadata retains hashed ids, prompts, and groups; evaluation rejects reuse of seen groups even when a file is reordered or an observation id changes. Always pass the candidate run directory to Rephix commands so its provenance and best-checkpoint pointer remain available.

## 6. Choose the candidate prior and compute budget

The default `--init random` uses the reference architecture and tokenizer without reading reference weights for initialization. Reconstructing the full 0.5B architecture is an expensive baseline. A smaller random architecture can share the reference vocabulary while using much less compute.

Create a small Qwen2 student config without loading teacher weights:

```bash
python - <<'PY'
from transformers import AutoConfig

config = AutoConfig.from_pretrained("models/qwen2.5-0.5b-instruct", trust_remote_code=False)
config.hidden_size = 64
config.intermediate_size = 128
config.num_hidden_layers = 2
config.num_attention_heads = 4
config.num_key_value_heads = 2
config.max_position_embeddings = 256
config.tie_word_embeddings = True
config.save_pretrained("models/qwen-student-config")
PY
```

`--student-config` is valid only with random initialization. Vocabulary size and special token IDs must match the reference config; the reference tokenizer remains the input/output vocabulary. If using a remote student repository, `--student-revision` pins that repository independently of the reference revision. A smaller student is a behavioral surrogate with different tensor shapes, so direct full-checkpoint weight distance is not applicable.

Run a bounded trial:

```bash
rephix --threads 2 fit \
  --train data/splits/train.jsonl --validation data/splits/validation.jsonl \
  --teacher "$QWEN_MODEL" --student-config models/qwen-student-config \
  --output runs/qwen-student --init random --trainable all \
  --steps 20 --eval-every 5 --save-every 5 --learning-rate 0.0005 \
  --topk-weight 1 --max-seq-length 128 --device cpu --seed 42
```

These short settings verify the data and optimization path. They are not a sufficient reconstruction budget or a recommended accuracy target. A broad, independent prompt corpus matters more than a favorable result on a few short continuations.

The loss supervises continuation positions only. `--topk-weight 0` uses recorded-token cross entropy. At `1`, records with top-k evidence use cross entropy over those tokens plus a residual probability bucket. Intermediate values mix the two objectives. A record lacking top-k evidence falls back to its recorded token. Validation checkpoint selection uses the corresponding reconstruction objective, with token-weighted aggregation.

`--gradient-accumulation-steps N` combines multiple one-observation microbatches into an optimizer step; `--steps` counts optimizer steps. Sequence limits can truncate supervised completions while preserving causal token alignment; metrics report how many examples were truncated. A prompt that leaves no continuation room is rejected. Truncated teacher-forced scoring and full-completion generation answer different questions.

For a separate pretrained-prior experiment, use a fresh output directory:

```bash
rephix --threads 2 fit \
  --train data/splits/train.jsonl --validation data/splits/validation.jsonl \
  --teacher "$QWEN_MODEL" --output runs/qwen-head-prior \
  --init pretrained --trainable head --steps 20 --eval-every 5 \
  --save-every 5 --max-seq-length 128 --device cpu --seed 42
```

This freezes a pretrained backbone and optimizes only its output head. If input/output embeddings are tied, the head is untied so input embeddings remain frozen. The retained pretrained weights are a strong prior; agreement cannot be attributed solely to recovery from the collected evidence. Head-only training from random initialization instead freezes a random backbone, which is another distinct restriction. Reports record initialization, trainable parameter counts, and whether weights were untied.

## 7. Resume fitting without confusing latest and best

To extend the student run from 20 to 40 total optimizer steps, repeat its settings and add `--resume`:

```bash
rephix --threads 2 fit \
  --train data/splits/train.jsonl --validation data/splits/validation.jsonl \
  --teacher "$QWEN_MODEL" --student-config models/qwen-student-config \
  --output runs/qwen-student --init random --trainable all \
  --steps 40 --eval-every 5 --save-every 5 --learning-rate 0.0005 \
  --topk-weight 1 --max-seq-length 128 --device cpu --seed 42 --resume
```

`training_state.pt` contains the latest durable model/optimizer state, sampling order, and RNG state. Validation-selected inference weights live in a separate immutable checkpoint, published through `best_checkpoint.json`. `metrics.json` records baseline, history, and the best validation step. Rephix evaluation and query selection resolve the run's best checkpoint automatically. Preserve the run directory as one artifact rather than moving its internal snapshots independently.

Resume verifies training/validation hashes, tokenizer, architecture, initialization/prior, optimization settings, device, thread settings, and relevant library versions. Changing those requires a new run. The total step target can increase; the save interval can change. Validation frequency remains part of the run configuration. Interrupted work after the last durable save may be repeated. Exact continuation is tested on CPU with fixed settings; saved RNG state does not guarantee identical outcomes across different hardware or nondeterministic accelerator kernels.

## 8. Compare settings and evaluate once

Use a strict sweep config to compare a bounded grid using training and validation evidence:

```bash
rephix --threads 2 sweep --config examples/sweep.json --output runs/evidence-comparison
rephix --threads 2 sweep --config examples/sweep.json --output runs/evidence-comparison --resume
rephix report --runs runs/evidence-comparison --output runs/evidence-comparison/report --plot
```

Edit the example paths and budgets first. It resolves relative paths from the config file and defaults to the full reference architecture, which is more expensive than the student above. Add the matching `student_config` to use the smaller architecture. Remote model references in sweeps require immutable commit revisions. The sweep runner fingerprints inputs, skips verified completed runs on resume, and exports JSON/CSV/Markdown comparisons. Optional Matplotlib SVG plots show validation curves. See [SWEEPS.md](SWEEPS.md) for the schema and integrity limits.

The sweep does not consume a test set or automatically choose a winner. Fix your selection rule on validation data, then evaluate the chosen candidate on reserved test evidence:

```bash
rephix --threads 2 evaluate --data data/splits/test.jsonl --model runs/qwen-student \
  --output runs/qwen-student/test.json --device cpu \
  --max-seq-length 128 --generate --max-new-tokens 16
```

| Measurement | Interpretation | Limit |
| --- | --- | --- |
| Continuation cross entropy | Probability assigned to recorded continuation tokens | Does not prove matching outputs or parameters |
| Recorded-token agreement | Candidate argmax matches the recorded next token under the same prefix | A sampled recorded token need not be the teacher argmax |
| Observed top-token agreement | Candidate argmax matches the highest-probability observed token | Available only where probability evidence exists |
| Free-running exact match | Greedy generation reproduces the complete recorded token sequence | Depends on generation budget, decoding, and error accumulation |
| Reconstruction objective | The selected token/top-k loss mixture | Values across different objective mixtures are not directly comparable |
| Parameter relative L2 and cosine | Tensor distance in compatible coordinates | Does not resolve general model symmetries |

The legacy `teacher_token_agreement` field aliases recorded-token agreement. Evaluation uses the saved training run's `topk_weight` for its reconstruction objective by default; `evaluate --topk-weight VALUE` overrides it explicitly. Without a training-run report, the default weight is one. Reports identify which source supplied that objective weight. Report examples, supervised token counts, truncation, and generation-budget coverage alongside each result. For cross-objective model selection, use a common validation metric such as continuation cross entropy rather than comparing different objective definitions.

Direct checkpoint comparison is available only for compatible parameter names and shapes:

```bash
rephix compare-weights --truth "$QWEN_MODEL" \
  --candidate runs/qwen-head-prior --output runs/qwen-head-prior/weights.json
```

This can use substantial memory and does not align neuron permutations, scalings, or transformer symmetries. The [tiny-network alignment experiment](IDENTIFIABILITY.md) supports only its documented restricted transformations.

## 9. Select new queries and test restricted recovery

When expanding the training evidence, compare uncertainty-based queries with a random baseline at equal prompt and token budgets:

```bash
rephix --threads 2 select-queries --pool data/pool.jsonl \
  --output data/next-prompts.jsonl --budget 20 --strategy entropy \
  --model runs/qwen-student --device cpu \
  --exclude data/splits/train.jsonl --exclude data/splits/validation.jsonl \
  --exclude data/splits/test.jsonl
```

The command scores candidate next-token uncertainty using prompts alone; it does not call the teacher. It excludes connected reserved/previously queried groups, selects at most one prompt per remaining group, and clears old completion evidence. Query the actual reference afterward with `collect`. For disagreement, supply two candidate surrogates with identical tokenizers. See [QUERY_SELECTION.md](QUERY_SELECTION.md).

When open-model hidden features are explicitly available, test a different recovery assumption:

```bash
rephix --threads 2 recover-projection --model "$QWEN_MODEL" \
  --prompts examples/prompts.jsonl --output runs/qwen-projection \
  --vocab-limit 64 --max-samples 4096 --max-seq-length 128 --device cpu
```

This captures the actual input to the output projection and fits selected vocabulary columns from known features and logits. It separates prompt groups before sampling positions and uses reference weights only for post-fit comparison. Selected-vocabulary top-1 agreement is not full-vocabulary generation agreement. A small prompt file may not span the hidden dimension, in which case the report marks parameter recovery as unidentifiable. Array memory limits do not include the whole loaded model or all forward-pass temporaries. See [COMPONENT_RECOVERY.md](COMPONENT_RECOVERY.md).

For every next trial, preserve code revision, source/evidence fingerprints, candidate prior, seed, resource budget, validation selection, and final held-out results. Diagnose evidence coverage, alignment, optimization, conditioning, and compute separately before increasing model size or claiming parameter recovery.
