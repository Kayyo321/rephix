# Create Mode (experimental)

Create Mode builds a new checkpoint **inside a base model you choose (A)** from **evidence
(B)**, optionally guided by an **instruction (C)**:

* **A, base model.** A local or pinned open-weight decoder supplies the starting weights
  and the exact architecture, tokenizer and tensor schema of the output.
* **B, evidence.** Observations of an *unknown* model (conversations, output-only
  passages, visible reasoning, tool calls and tool results), or a corpus of ordinary
  writing without prompts (a thesis, notes, essays).
* **C, optional instruction.** For example "preserve the formal voice". It is used only
  during fitting. It is never written into the evidence as a fake historical prompt, and
  the exported model does not need it.

"Build Mode" refers to the same feature. No `build` alias was added, because `build`
already means packaging (`python -m build`) in this project's documentation.

**What Create Mode is and is not.** It fits the base to the evidence, so that part is
sequence-level distillation when the evidence is a model's sampled text. On top of the
fit, a structural stage decides *where* and *in which subspaces* the base is edited, and
tests those decisions with causal interventions. The result is a base-compatible model
whose behavior reflects the evidence. It is **not** a recovery of the unknown source's
weights. Source parameters are only ever reported under an explicitly declared hypothesis
class, and never from output similarity alone. The method, its claims and its
falsification criteria are in [CREATE_MODE_METHOD.md](CREATE_MODE_METHOD.md). Measured
outcomes, including negative results, are in [CREATE_MODE_RESULTS.md](CREATE_MODE_RESULTS.md).

## Quick start

Requires `pip install 'rephix[hf]'` (or the managed installer with `--extras hf`).

```bash
# Unknown-source traces (exchanges with tool calls and visible reasoning)
rephix create --base /models/base --evidence traces.jsonl --output runs/create

# Corpus only: ordinary writing, no prompts
rephix create --base /models/base --corpus thesis.txt --output runs/thesis

# Optional instruction
rephix create --base /models/base --corpus thesis.txt \
  --prompt "Preserve the formal voice and paragraph structure" --output runs/thesis

# A pinned hub base, plan only (prints base, evidence, method, memory estimate and claim level)
rephix create --base HuggingFaceTB/SmolLM2-135M \
  --base-revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 \
  --corpus notes/ --output runs/notes --plan

rephix create-resume runs/thesis                 # after Ctrl+C, a crash or --max-seconds
rephix create-inspect runs/thesis --verify       # phases, plan, claims, integrity
rephix create-evaluate runs/thesis               # locked held-out results and claim statuses
rephix create-export runs/thesis --output models/thesis-model
rephix create-export --verify models/thesis-model
rephix create-generate models/thesis-model --prompt "On the design of bridges"
```

The export is an ordinary checkpoint. It loads without Rephix:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("models/thesis-model")
tokenizer = AutoTokenizer.from_pretrained("models/thesis-model")
```

A download-free check of the whole pipeline on the CPU:

```bash
rephix create-fixture --output /tmp/create-fixture
rephix create --base /tmp/create-fixture/base --evidence /tmp/create-fixture/traces.jsonl \
  --output /tmp/create-run --profile cpu-smoke --device cpu
```

## How it differs from `fit`, `session` and `text-recover`

| Command | Input | Output | Claim |
| --- | --- | --- | --- |
| `fit` | Prompt/response observations aligned to the reference tokenizer | Fitted candidate in the reference architecture | Behavioral reconstruction |
| `session` | A live or recorded target, any family | Standalone **byte-level** GRU/conv candidate | Behavioral agreement; never a base-compatible transformer |
| `text-recover` | Sampled strings over declared labels, public feature map | Canonical softmax-affine parameters | Identification within a declared class |
| `create` | Exchanges, passages or a corpus; any unknown source | A checkpoint with **exactly the base's schema**, edits confined to inferred regions | Behavioral reconstruction and mechanistic synthesis in the base; partial source recovery only under a declared, testable class; full source recovery never |

## Evidence format

Evidence is JSON Lines. Each record is an `exchange` or a `passage`:

```json
{"id": "c1", "kind": "exchange", "conversation_id": "conv-1",
 "messages": [
   {"role": "user", "content": "What is 7 times 6?"},
   {"role": "assistant", "reasoning": "Multiply.",
    "tool_calls": [{"id": "t1", "name": "multiply", "arguments": {"a": 7, "b": 6}}]},
   {"role": "tool", "tool_call_id": "t1", "content": "{\"result\": 42}"},
   {"role": "assistant", "content": "42."}]}
{"id": "p1", "kind": "passage", "text": "…an output with no known prompt…"}
{"id": "p2", "kind": "passage", "context": "The known preceding text", "text": " and its continuation"}
```

* Roles, order and the exact strings are preserved. Each record keeps the SHA-256 of its
  raw line.
* `context_status` is `known`, `none` (a known document start) or `missing` (unknown).
  A missing context is never filled in.
* `joiner` is the exact text between a known context and the passage. The default is
  none; corpus sections use a blank line.
* `reasoning` is visible generated text. It is fitted and scored as text; it is not
  hidden computation.
* Tool results are context. They are never training targets. Tool calls are parsed and
  never executed.
* `source_signals` can carry log probabilities or source token IDs, when they really were
  observed. They are recorded as a capability tier and never read as base token IDs
  without a validated mapping.
* `decoding` declares how outputs were sampled, when known.
* Legacy Rephix observation JSONL (`prompt`, `output`, `thinking`) is accepted and
  converted.
* `--corpus` reads `.txt`, `.md`, `.rst` and `.tex` files and splits them at paragraph
  boundaries.

Records are grouped before splitting:

* by document, conversation, `group_id` and repeated context;
* by exact normalized duplicates and near-duplicates (MinHash on word 5-grams, Jaccard
  ≥ 0.7);
* a single document is split into about forty contiguous blocks.

Train, validation and test splits are frozen before any fitting or selection. The test
split is evaluated once, after model selection.

## What a run does

1. **prepare.** Validates and hashes the evidence and writes the capability manifest.
   Freezes the splits. Samples the base's own continuations of fixed neutral prompts as
   retain contexts, and of evidence contexts for hypothesis tests.
2. **hypotheses.** Compares style and behavior statistics with the base's continuations
   of the same contexts. A hypothesis is supported only if its training-group interval
   excludes zero and it replicates on validation. Topic words are reported separately as
   content.
3. **structure.** Estimates Kronecker-factored model curvature and evidence gradients for
   every q/k/v/o/gate/up/down map. It then:
   * scores each region by its whitened rank-k gradient energy;
   * tests the ranking with held-out trial edits against random regions and random
     directions;
   * selects regions and initial subspaces;
   * reports logit-lens readings of the write directions.
4. **identify.** Runs the region-class test (is the change explained by a single region?)
   and, when the evidence has repeated samples over closed labels, the declared-class head
   estimator. Expand-and-Cluster is reported as not applicable to text.
5. **fit.** Fits `Δ = P M Qᵀ` in the selected regions. The objective is evidence NLL plus
   retain KL to the base plus an L2 prior, plus context distillation of the instruction
   when `--prompt` is given. Checkpoints hold the optimizer, sampler and RNG states.
6. **evaluate.** Measures:
   * validation and the locked test NLL with group-bootstrap intervals;
   * restoration of each region;
   * free-running style distance on held-out contexts and new topics;
   * memorization overlap;
   * fluency under the base, fact-cloze retention and retain KL;
   * tool and reasoning metrics.

   It then assigns the four separate claim statuses.
7. **export.** Merges the edits into standalone weights in the base dtype, then checks:
   * the tensor schema equals the base's;
   * tensors outside the edits are bit-identical to the base;
   * tokenizer and config files are byte-identical to the base's;
   * a fresh `python -I` process reproduces the logits through plain Transformers.

   Checksums and an `EXPORT_COMPLETE` marker are written last. An unverified export is
   never published under the requested name.

`--method` selects baselines and ablations for comparison: `lora`, `lora-matched`, `sft`,
`random-regions` and `random-subspace`. Reports label them as behavioral-only or as
ablations.

## Supported bases

| Family | Status |
| --- | --- |
| Llama (`LlamaForCausalLM`) | Pretrained runs: SmolLM2-135M and SmolLM2-135M-Instruct (CUDA), CPU fixtures |
| Qwen2 (`Qwen2ForCausalLM`) | Pretrained runs: Qwen2.5-0.5B-Instruct (CUDA), CPU fixtures |
| Mistral, Qwen3 | Random tiny fixtures only (CPU): loading, contract and export checks |
| Anything else (GPT-2, MoE, state-space, custom code) | Rejected with a specific error |

Edits never touch embeddings, the output head, norms, biases or buffers, so tied
parameters stay tied. Weights must be safetensors. A hub base needs `--base-revision`
with an exact 40-hex commit.

## Resources

| Profile | Use | Defaults |
| --- | --- | --- |
| `rtx5080-16gb` | NVIDIA 16 GB GPUs and 32 GB RAM | bf16 autocast, 384-token windows, batch 8 (4 for vocabularies over 100k), 12 regions × rank 16, 400 steps |
| `cpu-smoke` | CPU checks with tiny models | 96-token windows, 3 regions × rank 4, 12 steps |

Measured peaks on an RTX 5080 are in the [results](CREATE_MODE_RESULTS.md). For example,
SmolLM2-135M structured runs peak at about 5.6 GB, which includes the structural
statistics. `--plan` prints a coarse estimate before any expensive work.
`--max-memory-gib` refuses to start if that estimate exceeds the budget. Bases up to
about 0.5B parameters fit the 16 GB profile as measured. Larger bases need `--dtype
bfloat16`, a smaller `--batch-size` or `--max-tokens`, and have not been measured. CUDA
and CPU are verified. Apple MPS is accepted but **unverified** (no usable Metal device was
available). Windows supports the plain CLI through the existing launcher; Create Mode
model runs were not executed on Windows.

## Interruption and resume

`create-resume RUN` continues from the first incomplete phase, and the fit continues from
its checkpoint. It refuses to continue when:

* the base files changed (content fingerprint);
* an evidence or corpus input changed;
* a completed phase output was modified;
* a checkpoint is unreadable;
* a checkpoint belongs to different settings.

A lock prevents two processes from using one run. An out-of-memory error during fitting
keeps the last checkpoint, and the message suggests smaller settings. Original inputs and
base files are never modified.

## Limitations

* Create Mode does not recover the weights of an unknown model from its text. When the
  evidence only supports synthesis, that is what is reported.
* Structural findings describe the edited base, not the source's circuitry.
* Style statistics are coarse. Good held-out agreement does not mean the model will write
  "exactly like" the corpus on every input. Very small corpora get an explicit evidence
  limitation and no confident style claim.
* The region-class test can reject the single-region hypothesis, but cannot prove it, and
  cannot see function-preserving reparameterizations of the source.
