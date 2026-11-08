# Compact observations and lossless evidence storage

Measured on 2026-11-08 with `facebook/convnextv2-atto-1k-224` revision
`c7eab08296c3bca2631738322afc3d05373acc8f`. The [machine-readable result](../examples/results/evidence_storage.json)
contains source and checkpoint hashes, runtime versions, audits, held-out behavior,
interruption evidence, and source-free replay checks.

| Capture mode | Database bytes | Stored payload bytes | Decoded payload bytes | Calls |
| --- | ---: | ---: | ---: | ---: |
| Full, raw | 65,191,936 | 64,995,200 | 64,995,200 | 207 |
| Full, zlib | 33,550,336 | 33,325,601 | 64,995,200 | 207 |
| Compact, raw | 29,831,168 | 29,667,200 | 29,667,200 | 207 |
| Compact, zlib | 20,017,152 | 19,809,607 | 29,667,200 | 207 |

All four modes recovered **100% of stored numeric values**, with complete reference
coverage, and generated identical checkpoint bytes. Logits were exactly equal on
five seeded or processed images across three input panels. This small panel checks
export behavior, not general model quality. Compact plus zlib used 69.3% less physical
database space than full/raw. This measurement is specific to this checkpoint and
FP64 probing; compressed sizes depend on values, geometry, precision, and batch size.

The compact/zlib run stopped after 17 committed responses, left no published
candidate, and reused all 17 responses on restart. Every sealed bundle replayed
with the source directory moved out of its original location, making zero new
oracle queries and reproducing checkpoint bytes and held-out logits. Downloaded
weights, recovered models, and temporary databases were deleted after the result
was recorded.

## Access contract

This is chosen-input, coordinate-preserving **internal layer access** with known
architecture, FP64 probes, and restoration to the source storage dtype. It is not
weight recovery from text or ordinary model endpoint responses. Compact regular
Conv1d/2d/3d observations return one complete-receptive-field output position via
`:conv_kernel_position`. The native convolution still runs in full. Compact mode
reduces response bytes, not queries or native forward computation. Transposed
convolutions keep their full response contract.

Use `recover-layers --compact-convolutions` or
`recover-diffusion --compact-convolutions` with `--evidence-dir`. Replay reads the
saved choice automatically. Switching compact mode when resuming a bundle is
rejected before consuming evidence. The default full mode retains previous bundle
metadata for compatibility.

New journals use schema 2 and choose lossless zlib level 1 only for payloads of at
least 256 bytes when compression is smaller than raw storage. Checksums bind the
codec and decoded bytes. Decoding is bounded by the expected tensor byte count;
truncated, oversized, trailing, or corrupted streams fail. Schema 1 journals remain
readable and resumable in their original raw format without migration. Byte counters
in recovery reports distinguish decoded payload from physically stored payload;
they exclude SQLite indexes and page overhead.

## Reproduce

Install the model and analysis dependencies from the project, then download only
`config.json`, `preprocessor_config.json`, and `model.safetensors` at the revision
above using `hf download --revision ... --local-dir ...`. Use a disposable directory
separate from the checkout and source. Run:

```bash
python examples/evidence_storage_experiment.py \
  --reference /absolute/disposable/source \
  --model-id facebook/convnextv2-atto-1k-224 \
  --revision c7eab08296c3bca2631738322afc3d05373acc8f \
  --work-dir /absolute/disposable/new-work \
  --output /absolute/results/evidence-storage.json --verify-source-free
```

Defaults are CPU, three threads, batch size 128, FP64 probing, and source dtype
restoration. The script also measures grouped 3x3 and depthwise 21x21 synthetic
convolutions. Review the result before deleting the disposable source and work
folders. Keep the JSON for reproducibility; do not commit downloaded weights.
