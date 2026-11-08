# Recoverable pause: 2026-11-07

Work stopped at the user's explicit request. The persistent reconstruction goal
is paused, not complete. Resume only when the user asks to continue.

Branch: `codex/local-qwen-reconstruction`. Last fully verified feature commit:
`2605919` (`Recover native Diffusers denoisers and latent autoencoders`). That
checkpoint passed 316 local tests and 404 subtests with optional dependencies and
GPU access. Baselines `711b5ab` and `d55cf07` remain preserved ancestors.

The following changes are saved together in the pause checkpoint. They are work
in progress and have not passed the full regression suite:

- `src/rephix/layer_journal.py`: new schema 2 journals adaptively store raw or
  lossless zlib responses. Existing schema 1 journals read and resume without
  migration. Checksums bind the codec and decoded payload; decoding is bounded by
  the expected tensor size. Reports expose stored and decoded payload bytes.
- `src/rephix/convolution.py`: opt-in compact regular Conv1d/2d/3d observations
  return one complete-receptive-field output position. Default full responses and
  transposed convolutions retain their existing behavior. Compact mode changes
  observed response size, not the number of queries or native forward compute.
- `src/rephix/layer_recovery.py` and `src/rephix/diffusers_recovery.py`: compact
  mode API/CLI wiring, provenance, journal byte counters, and replay metadata.
  False/default mode does not add new bundle binding metadata.
- `tests/test_compact_convolution.py` and
  `tests/test_layer_journal_compression.py`: focused regression tests.
- `examples/evidence_storage_experiment.py`: unverified draft of a four-mode
  size/resume/replay experiment. No experiment was run and no results exist.

## Verification completed before stopping

- Compact convolution work: 12 focused tests and 52 subtests passed, including
  existing convolution/audio checks; Ruff passed.
- Journal compression work: 8 tests and 9 subtests passed; existing evidence
  tests also passed (7 tests and 5 subtests); Ruff passed.
- The integration changes and draft experiment still need end-to-end validation.

## Resume from here

1. Review the saved changes and the schema 1/schema 2 compatibility path. Verify
   compact-mode resume binding and source-free replay through both runtimes.
2. Review the draft experiment before running it. Known unfinished details:
   bundle databases are named `observations.sqlite3` (the draft currently uses
   `responses.sqlite3`), and raw-baseline patching must target the imported
   `layer_recovery.LayerJournal` rather than only `layer_journal.LayerJournal`.
3. Run full/raw, full/zlib, compact/raw, and compact/zlib measurements on pinned
   `facebook/convnextv2-atto-1k-224` revision
   `c7eab08296c3bca2631738322afc3d05373acc8f`. Record physical database bytes,
   stored/decoded payload bytes, query counts, exact parameter checks, forced
   interruption/resume, and source-unavailable replay. Delete downloaded model
   files after recording evidence, as the user requested.
4. Add measured documentation, run the appropriate full tests and packaging
   checks, and commit the verified result. Do not claim measured storage savings
   until the experiment succeeds.

## Resources and limits

All delegated workers stopped. No task experiments or downloads remain running.
The downloaded ConvNeXT checkpoint and its temporary download/cache directory
were deleted; no evidence or candidate artifacts were created by that trial.
No temporary dependencies were installed in this paused work segment.

The reusable Python environment is
`/home/dev/.local/share/rephix-local-20261106/venv`. GPU/network access requires
the existing approved execution path. Optional Diffusers/GGUF/accelerate overlays
from the last completed milestone were already removed.

Exact reconstruction results so far depend on the documented operator access
assumptions. Universal family coverage and recovery of original weights from
ordinary text-only observations remain unproven.
