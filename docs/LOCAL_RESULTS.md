# Local Qwen reconstruction on an RTX 5080

On 2026-11-06, randomly initialized Qwen2 students learned some of the reference's next-token behavior from recorded completions. Top-20 supervision improved validation cross-entropy in two paired seeds. The validation-selected student matched 140 of 1,324 recorded tokens under teacher forcing and 0 of 72 free-running continuations on reserved prompt families. A separate white-box experiment estimated 28,672 selected output weights with relative L2 error of `2.21e-5`.

The [machine-readable results](../examples/results/qwen-local.json) include commands, dependency versions, evidence and checkpoint hashes, measured resource use, per-run learning curves, and the validation selection lock.

## Environment and evidence

The workstation ran Debian 13.7, Python 3.13.5, PyTorch 2.10.0+cu128, and Transformers 4.57.6. It exposed eight Ryzen 7 9800X3D cores, about 30 GiB of RAM, 287 GiB of free workspace storage at inspection, and an RTX 5080 with 15.55 GiB of CUDA-visible memory. The NVIDIA kernel module was 615.71.10 and the user libraries were 620.70.06. NVML reported a version mismatch. CUDA initialized and passed a matrix multiplication check, then ran collection, fitting, evaluation, and projection capture. The offline fixture ran on CPU. Driver configuration was kept as found.

The reference was Qwen/Qwen2.5-0.5B-Instruct at revision `7ae5576`, loaded in FP32. The safetensors SHA-256 was `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`, matching the earlier cloud run. Generation used the reference tokenizer and chat template, neutral greedy decoding, and a cap of 24 new tokens. The collector recorded the top 20 raw, teacher-forced log probabilities at each observed position. Students received the reference vocabulary and architecture specification. Their weights were initialized randomly. The behavioral study used observed token sequences and, in designated arms, top-k probabilities. [Pinned model snapshot](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775)

| Split | Prompts | Families | Recorded continuation tokens | At the 24-token cap |
| --- | ---: | ---: | ---: | ---: |
| Train | 384 | 16 | 6,709 | 155 |
| Validation | 72 | 6 | 1,367 | 41 |
| Final test | 72 | 6 | 1,324 | 47 |
| Query pool | 120 | 20 | Queried subsets only | — |

The prompt generator assigns each synthetic task template and its variants to one partition before any collection. The four partitions were checked for duplicate IDs, normalized prompts, and group IDs. Some related skills appear across different task families, and the variants within a family are narrow. The 72 test prompts represent six independent families under this design. The fit limit of 128 tokens covered every recorded sequence. Reference answers often reached the generation cap, so the measured target is the recorded continuation within that budget. [Prompt generator](../examples/local_qwen_prompts.py)

## Validation comparisons

The paired arms each used 384 AdamW updates and four single-sequence gradient accumulations per update. Each arm sampled 1,536 sequences and 26,836 continuation token positions in the same order within its seed. The token arm optimized recorded-token cross-entropy. The top-20 arm optimized cross-entropy over the 20 observed entries and a residual bucket. Both used the same collected rows. Wall time includes the different objective calculations. Every arm used continuation cross-entropy to select its checkpoint on validation.

| Student | Objective | Seed | Learning rate | Best step / total | Token CE | Top-20 + tail CE | Token agreement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 9.80M | Token | 17 | 0.001 | 192 / 384 | 8.373 | 8.188 | 9.73% |
| 9.80M | Top-20 | 17 | 0.001 | 192 / 384 | 7.772 | 7.594 | 9.95% |
| 9.80M | Token | 42 | 0.001 | 192 / 384 | 8.531 | 8.345 | 8.63% |
| 9.80M | Top-20 | 42 | 0.001 | 192 / 384 | 7.803 | 7.632 | 9.36% |
| 9.80M | Top-20 | 17 | 0.0003 | 384 / 768 | 7.884 | 7.710 | 6.51% |
| 19.74M | Top-20 | 17 | 0.001 | 192 / 384 | **7.629** | 7.462 | 8.41% |

Top-20 reduced token CE by 0.601 and 0.728 nats across the two paired seeds. The token-CE arms continued to lower their sampled training loss after step 192 while validation CE rose to 9.305 and 9.761 at step 384. Lowering the learning rate delayed the best validation checkpoint to step 384. At step 768 its validation CE had risen to 8.704. Doubling width from 64 to 128 improved validation CE at seed 17, while argmax agreement fell by about 1.5 percentage points relative to the narrow top-20 student. The width and learning-rate ablations used one seed each.

### Query selection

Random and next-token entropy each selected ten previously unqueried families with seed 92 and a 24-token cap per query. The width-128 checkpoint supplied entropy scores. Random collected 206 continuation tokens and entropy collected 223. Two chosen prompts overlapped. For training, each pair of added rows was capped at the shorter continuation by selection rank, preserving aligned token IDs and probability rows. Each arm added 197 supervised tokens, and both fits sampled 26,937 token positions using the same narrow model, top-20 loss, seed 17, and 384 updates.

Random achieved validation CE 7.689 and 7.90% token agreement. Entropy achieved CE 7.772 and 9.80% agreement. These one-seed results favor different policies on different metrics. Both query arms stayed exploratory and outside final candidate selection.

## Reserved behavioral test

The selection lock chose the 19.74M student at step 192 by minimum validation token CE across the base-corpus runs. The lock recorded the winning checkpoint and reserved prompt hashes before test outputs were collected. This was the sole candidate evaluated on those six test families.

| Metric | Result |
| --- | ---: |
| Teacher-forced recorded-token agreement | 140 / 1,324 (10.57%) |
| Teacher-forced observed top-token agreement | 10.57% |
| Token CE / top-20 plus residual CE | 7.347 / 7.281 nats |
| Free-running first-token agreement | 32 / 72 (44.44%) |
| Reference tokens before first free-running divergence | 32 / 1,324 (2.42%) |
| Free-running exact continuation match | 0 / 72 |

Every initial match diverged at the second token. In the explicit fact-extraction family the student opened correctly on all 12 prompts, but reused an opening from binary conversion. For example, the reference started “The pet's name is Maya” while the student generated “The binary number 1.” All twelve Markdown-link and all twelve rectangular-area prompts diverged immediately. These failures show how frequent openings and training templates can drive token agreement while leaving sequence behavior poor. These experiments establish no original-weight error estimate for the smaller students.

## Engine changes and integrity

- The forward path uses `logits_to_keep` when the model exposes it, projecting the continuation prediction positions without the rest of the prompt. Other models retain the full-logits path. The top-k loss batches vocabulary reductions and avoids device-to-host synchronization per token. Tests check causal slicing and loss and gradient parity, including variable row lengths and full-vocabulary rows.
- A 64-step CUDA A/B used seed 17, the same data, model, and top-20 objective. The first old/new pair took 16.00/5.49 seconds. A reverse-order repeat took 4.87 seconds for the new engine and 8.02 seconds for v0.2.0, about 39% less time. Peak PyTorch-allocated GPU memory fell from 328.0 to 298.4 MiB. Final validation CE differed by about `5.5e-7`, with identical argmax agreement. The timings measure the command, including validation and checkpoint writes, on a shared desktop.
- The Qwen training collection was interrupted after durable query 48 while the journal still held the pending checksum. Resume preserved the 48-record prefix byte for byte and completed all 384 rows. An edited copy of a real validation observation failed the checksum check. Evaluation also rejected reuse of the real training split.
- A CUDA fit stopped at step 192 and resumed to 384. Its model tensors, optimizer tensors, sample order, RNG states, and metric history matched the uninterrupted run exactly. Its best safetensors SHA-256 matched as well. This is an observed result on this software and hardware combination.
- The reference collection peaked at about 3.71 GiB process RSS and 1,936 MiB PyTorch-allocated GPU memory. The width-128 fit took 24.44 seconds and peaked at 2.30 GiB RSS and 555 MiB allocated on the GPU. The first and resumed training collection commands took 34.33 and 62.35 seconds. Command-specific measurements include allocator reservations in the JSON report.

The original checkout passed 106 tests. The updated checkout passed 111 tests, lint, format, and wheel and source builds. The 32-step offline CPU trial scored 35/40 teacher-forced tokens and 7/10 exact completions against its synthetic teacher.

## Separate white-box projection experiment

The CUDA run repeated the published 128-prompt projection trial with actual projection inputs and 32 selected output rows. The estimator used 3,072 training positions with known-zero bias and full observed feature rank of 896. Relative L2 weight error was `2.2082e-5`. Held-out logit RMSE was `2.3775e-6`, and selected-vocabulary top-1 agreement was 100% on 1,024 positions. Of those positions, 475 feature rows were novel relative to training and reached 100% selected-vocabulary agreement with RMSE `3.4736e-6`. The observed FP32 logits differed from an FP64 evaluation of the original head by `2.1199e-6` held-out RMSE. The design had a condition number of approximately 54,662. [Projection methodology](COMPONENT_RECOVERY.md)

This parameter comparison applies to the selected affine projection under known-feature access. The text and probability study leaves full-model original-weight recovery unproven.

## Reproduce

Use a fresh directory outside the checkout. The example below uses Bash on a CUDA 12.8-capable Linux setup. The JSON report records the complete installed package list and exact invocation arguments.

```bash
RUN="$HOME/.local/share/rephix-qwen-reproduction"
mkdir -p "$RUN"
python3 -m venv "$RUN/venv"
"$RUN/venv/bin/python" -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
"$RUN/venv/bin/python" -m pip install -e '.[dev,hf,analysis]' transformers==4.57.6 build
hf download Qwen/Qwen2.5-0.5B-Instruct \
  config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json merges.txt vocab.json \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 --local-dir "$RUN/qwen"
sha256sum "$RUN/qwen/model.safetensors"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$RUN/venv/bin/python" -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$RUN/venv/bin/rephix" offline-trial \
  --output "$RUN/offline-check" --steps 32 --threads 2
bash examples/local_qwen_experiment.sh "$RUN"
"$RUN/venv/bin/rephix" report --runs "$RUN/runs/paired-*" "$RUN/runs/lowlr-17" \
  "$RUN/runs/wide-17" "$RUN/runs/query-*" --output "$RUN/report" --plot
```

The experiment script runs the measured stages and writes their logs, the selection lock, and the resume comparison. It checks out the historical engine into an artifact directory using `git archive` for the speed comparison. Collected evidence embeds the local source path, so relocating the run changes file hashes. The safetensors hash and prompt manifest are independent of that location. Timing, memory, and floating-point results can vary with the local stack. [Experiment script](../examples/local_qwen_experiment.sh)
