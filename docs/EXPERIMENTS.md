# Reproducible recovery experiments

The goal is to improve reconstruction using measured evidence. Keep parameter recovery and behavioral reconstruction as separate experiment tracks.

## 1. Establish an exact baseline

Run `rephix demo` before involving a language model. Its reference is an affine map with known input features. Enough independent observations determine its parameters by least squares.

Record the random seed, observation count, number of features, output classes, matrix rank, conditioning, parameter error, and held-out prediction error. Repeat with fewer observations or dependent features. A rank-deficient system can fit all observed outputs and still leave some parameters undetermined.

With probabilities instead of logits, a common additive shift to every class logit is invisible to softmax. Compare centered parameters in that experiment. A recovered probability distribution cannot establish that the original uncentered parameters were recovered.

This controlled case verifies the solver. It does not establish identifiability for hidden neural-network layers.

## 2. Prepare a local reference

A suggested starting reference is `Qwen/Qwen2.5-0.5B-Instruct`. Download a compatible Transformers checkpoint and tokenizer when you have access to your computer. Point the commands at that local directory to reuse it without repeated downloads.

```bash
pip install -e '.[hf]'
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/qwen2.5-0.5b-instruct
rephix collect --help
rephix fit --help
rephix evaluate --help
```

Keep the reference checkpoint unchanged. Record its revision, tokenizer, chat template, numerical precision, generation settings, and hardware. Quantization changes the function being observed, so treat a quantized reference as its own experiment.

Start with a short generation length and a few prompts to verify the data path:

```bash
QWEN_MODEL="$PWD/models/qwen2.5-0.5b-instruct"
mkdir -p data
rephix collect --prompts examples/prompts.jsonl \
  --model "$QWEN_MODEL" --output data/observations.jsonl \
  --device cpu --max-new-tokens 16 --top-k 20 --seed 42
```

A smoke run checks that collection, training, checkpoint saving, and evaluation work. It is too small to estimate general reconstruction accuracy.

Full-parameter training of a 0.5B model is not a lightweight CPU task. Parameters, gradients, optimizer state, activations, and a separate reference can consume substantial memory. Collect reference evidence first, release the reference, and train the candidate separately.

## 3. Collect evidence and protect the split

Use one prompt per JSONL observation; its output may be empty before collection. The sample prompt file provides a starting format. Expand it with varied tasks, lengths, styles, and domains that represent what you want to reproduce.

The collector records the exact input and completion token IDs. Optional top-token log probabilities provide a richer constraint than the sampled or greedy response alone. They still leave most of the vocabulary unobserved.

Collection and evaluation use explicit neutral greedy decoding with the architecture’s special token IDs. Downloaded sampling, beam-search, and repetition-penalty defaults do not change the comparison. The evidence records the effective decoding configuration; saved probabilities are raw teacher-forced model probabilities.

After collecting, split by prompt into training, validation, and test sets:

```bash
rephix validate data/observations.jsonl
rephix split data/observations.jsonl --output-dir data/splits \
  --validation-fraction 0.2 --test-fraction 0.2 --seed 42
```

The output directory contains `train.jsonl`, `validation.jsonl`, `test.jsonl`, and a split manifest. Group repeated observations of the same prompt together. Use training data for optimization, validation data to select a checkpoint or settings, and reserve the test set for the final comparison. Changes based on test results turn that set into validation data; make a fresh test set for the next final estimate.

Exposed reasoning text can be part of a recorded completion. Preserve it exactly when it is available. A manually supplied `thinking` annotation is evidence to store, not a substitute for the target model’s hidden activations. If a record separates reasoning and output text, fitting requires the exact visible serialized completion in token IDs or `metadata.completion_text`; the engine does not guess how the parts were formatted. Surrounding context must be in the actual prompt to act as conditioning; metadata alone does not alter model inputs.

## 4. Fit a candidate

The default candidate uses the reference architecture and tokenizer but initializes weights randomly. This is the clearest baseline for reconstruction from the collected evidence: reference weights are not used to initialize it.

The loss supervises completion tokens rather than asking the candidate to reproduce the prompt. When top-token evidence exists, an additional distribution loss uses the recorded token probabilities and a residual bucket for the remaining vocabulary. That bucket preserves the uncertainty in the observations.

A pretrained initialization is a different experiment. It can provide a much stronger prior and may improve agreement quickly, but its success cannot be attributed solely to recovering weights from observations. Record that choice in every result.

After creating the splits, run a bounded check:

```bash
rephix fit --train data/splits/train.jsonl --validation data/splits/validation.jsonl \
  --teacher "$QWEN_MODEL" --output runs/qwen-smoke \
  --init random --steps 10 --eval-every 5 --max-seq-length 128 \
  --device cpu --seed 42
rephix evaluate --data data/splits/test.jsonl --model runs/qwen-smoke \
  --output runs/qwen-smoke/test.json --device cpu \
  --max-seq-length 128 --generate --max-new-tokens 16
```

Ten steps check the pipeline; they are not a realistic reconstruction budget. Inspect baseline validation loss, training progress, and saved `metrics.json` before spending more compute. The output directory contains the best validation checkpoint. Use that checkpoint for evaluation, rather than choosing by training loss alone.

## 5. Evaluate what was reconstructed

| Measurement | What it answers | Limitation |
| --- | --- | --- |
| Continuation loss | How much probability does the candidate assign to reference continuations? | Good loss does not prove identical outputs or parameters |
| Teacher-forced token agreement | Does the candidate prefer the recorded token given the same preceding reference tokens? | Reference prefixes hide error accumulation during generation |
| Free-running exact match | Does greedy generation reproduce the whole recorded completion? | Sensitive to length, decoding settings, and small harmless differences |
| Parameter relative L2 and cosine | How far apart are compatible tensors in the same coordinates? | Symmetry-equivalent networks can have different coordinates |
| Affine rank and conditioning | Are affine parameters constrained by the observations? | Applies only to the stated known-feature affine setup |

Report the number of prompts and supervised tokens along with every metric. Test on prompt families omitted from training as well as paraphrases. A candidate that memorizes short training continuations may generalize poorly.

For a deliberately expensive direct checkpoint comparison:

```bash
rephix compare-weights --truth "$QWEN_MODEL" \
  --candidate runs/qwen-smoke --output runs/qwen-smoke/weights.json
```

Compare checkpoints only when they have compatible names and shapes. Direct tensor distance is a useful diagnostic for a known reference, but it does not solve permutation, scaling, or other parameter symmetries. It is not interchangeable with behavioral accuracy.

## 6. Change one experimental factor at a time

Maintain a table for each run containing:

- Run name, code revision, seed, reference revision, and candidate initialization.
- Prompt-group split, observation count, token count, and evidence type.
- Optimization steps, learning rate, sequence length, and device.
- Baseline and best validation metrics, then final held-out metrics.
- Compute time and memory, plus any failures or numerical problems.

Useful first comparisons are text-only versus top-token evidence, increasing the number of independent prompts, short versus longer contexts, and random versus explicit pretrained initialization. Use the same reserved evaluation set when comparing settings until it has influenced a decision; then reserve new unseen prompts for the final estimate.

Do not set an accuracy target before a baseline exists. Decide whether the next bottleneck is insufficient evidence, an incorrect token/template pipeline, optimization, compute, or a fundamentally non-identifiable parameterization. Refactor the engine in response to that diagnosis.
