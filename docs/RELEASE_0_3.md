# Rephix 0.3 release verification

Release work resumed on 2026-11-08 from the saved pause checkpoint. The repository
history and earlier baseline commits are preserved. This release adds verified
compact evidence storage, an optional terminal workbench, and managed installation,
updates, rollback and scoped removal. It retains the existing JSON command interface.

## New pretrained reconstruction trials

Both trials used known architecture and chosen coordinate-preserving internal layer
calls. The estimator received layer responses, rather than reading source weights.
The local adapter necessarily loaded the source to provide those internal calls.
These results do not establish recovery from text or ordinary endpoint responses.

| Reference | Stored values | Probe precision | Calls | Coverage / exact / within tolerance | Relative L2 |
| --- | ---: | --- | ---: | --- | ---: |
| [TinyLlama 1.1B Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tree/fe8a4ea1ffedaf415f4da2f062534de366a451e6) | 1,100,048,384 | BF16 | 1,718 | 100% / 100% / 100% | 0 |
| [TinyMistral 248M v2](https://huggingface.co/Locutusque/TinyMistral-248M-v2/tree/0f57b17cb317bb322c7c1466b669c681f80c058f) | 248,024,064 | FP64, restored to FP32 | 635 | 100% / 100% / 100% | 0 |

Each export was independently reloaded using its recorded oracle precision.
The evaluator captured next-token logits in FP32 for metric comparison. All full-vocabulary next-token logits matched
exactly on 24 held-out prompts, and greedy continuations matched on all six selected
prompts (up to eight tokens). Tokenizer assets were exported. Audit tolerance was
`abs(candidate-reference) <= 1e-8 + 1e-5 * abs(reference)`, with every reference tensor
and stored value included. Results exceeded the requested approximate 95% agreement
for these checkpoints under the stated internal access.

Runtime: Python 3.13.5, Torch 2.10.0+cu128, Transformers 4.57.6, RTX 5080 (16 GB),
four CPU threads, probe batch size 256. These are new pretrained trials, distinct
from existing synthetic fixtures and earlier pretrained Mistral-7B measurements.
Machine-readable evidence records immutable model revisions, input prompts, source
and export file hashes, complete audits, runtime settings and behavior:

- [TinyLlama result](../examples/results/release-llama-recovery.json)
- [TinyMistral result](../examples/results/release-mistral-recovery.json)

The evaluator also applied the known residual-sign symmetry. It preserved all
measured logits and continuations while within-tolerance coordinate agreement fell
to 0.02888% for TinyLlama and 0.01034% for TinyMistral. This is further evidence that
endpoint agreement alone cannot guarantee original coordinate recovery.

To reproduce, use `hf download` with the linked immutable revisions, selecting only
config/tokenizer assets and safetensors. Use a new disposable source and output path:

```bash
rephix --threads 4 recover-layers --model /absolute/source \
  --output /absolute/recovered --device cuda --dtype auto \
  --restore-source-dtypes --batch-size 256
python examples/layer_recovery_experiment.py --reference /absolute/source \
  --candidate /absolute/recovered --model-id MODEL_ID --revision COMMIT \
  --device cuda --output /absolute/result.json
```

Use `--dtype float64` for the TinyMistral trial. After validating the result JSON,
delete only the disposable source and recovered directories. The release trials'
downloaded weights and recovered checkpoints have already been deleted.

## Evidence storage and runtime recovery

[The four-mode ConvNeXT experiment](EVIDENCE_STORAGE.md) measured 100% stored-value
agreement in all modes, 17-response interruption/resume, source-unavailable replay,
and identical checkpoint hashes. Compact/zlib reduced the physical database from
65,191,936 to 20,017,152 bytes (69.3%) without reducing its 207 calls or native forward
compute. Both Transformers and native Diffusers compact capture paths have dedicated
incompatible-resume, interrupted-prefix and source-free replay tests. Existing raw
schema-1 journals remain resumable without migration.

## Lifecycle and terminal verification

[Installation instructions](INSTALLATION.md) document installation from wheels,
optional dependencies, constraints/offline wheelhouses, validated updates, rollback,
pruning, retention options, migration policy and recovery. The [isolated lifecycle
measurement](../examples/results/installation-lifecycle.json) installs 0.2 from its
preserved checkpoint wheel, upgrades to 0.3, rejects an invalid wheel without changing
the active manifest, rolls back, preserves user-file hashes, inherits optional extras,
prunes obsolete environments, retains data on uninstall, reinstalls without data loss,
and explicitly purges requested categories. No managed environments remain from that
trial. A live workbench lease also blocked uninstall without altering the manifest.

[Terminal controls and verification](TERMINAL.md) cover dark/light palettes,
monochrome mode, responsive panes, keyboard input ownership, help, subprocess output,
interruption, terminal restoration, and the plain interface. Linux real-PTY checks
passed at 40, 80 and 120 columns. macOS CI also passed lifecycle PTY launch, exit, normal line input and complete
mode comparison. Full manual macOS/Windows console exploration remains an explicit
limit.

## Verification status

All 346 local tests passed with zero skips, including the optional native FP8
runtime. Dependency consistency, source/wheel packaging, isolated lifecycle and
the offline trial passed. All 17 [cross-platform CI jobs](https://github.com/Kayyo321/rephix/actions/runs/42496378427)
passed on the verified code checkpoint. The complete requirement audit is in
[production readiness](PRODUCTION_READINESS.md). See
[supported environments and scientific limits](SUPPORT.md) for the exact scope.
