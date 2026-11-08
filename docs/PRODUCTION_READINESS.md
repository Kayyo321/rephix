# Production readiness work

Active user goal, resumed 2026-11-08. Preserve history and existing user artifacts.
Completion requires the entire release scope below; a passing focused test does not
prove readiness. The original research roadmap and pause checkpoint remain history.

## Delivered and measured

- Fixed the draft storage experiment's SQLite database filename. Verified against
  current imports: Transformers imports `LayerJournal` inside its functions, so
  patching `layer_journal.LayerJournal` is correct; the historical pause note's
  proposed imported-module patch was inaccurate for the current source.
- Completed four-mode pinned ConvNeXT measurement: 100% stored numeric equality,
  207 queries in every mode, 17-response interruption/resume, source-unavailable
  replay with identical checkpoint hashes. Compact/zlib physical database is
  20,017,152 bytes versus full/raw 65,191,936. See EVIDENCE_STORAGE.md and the JSON.
- Deleted only this task's disposable downloaded source, candidate models and
  databases after recording the measurement. No existing user data was removed.
- Added compact-mode integration tests for Transformers and Diffusers: incompatible
  resume rejection preserves database bytes, interrupted prefix reuse, full stored
  equality, and source-free replay.
- Added optional terminal workbench, command catalog, contextual help, argument
  editor, subprocess output and interruption; preserves ordinary CLI JSON behavior.
  See TERMINAL.md for controls, accessibility options, verification and limits.
- Linux real-PTY smoke check passed: command execution, help, 40/80/120-column
  resizing, clean exit, exact termios restoration. Automated input-routing and
  display-control sanitization checks passed.

## Remaining release gates

- Implement and document safe installation, update/rollback and uninstallation
  commands, explicit data-retention/removal choices, and migration policy.
- Exercise complete installation/update/uninstallation in clean isolated environments,
  including failed upgrades, existing-data preservation, and optional dependencies.
- Expand representative reproducible model trials to Llama and Mistral; pin revisions,
  distinguish pretrained checkpoints from synthetic fixtures, report exact/tolerant
  weight coverage and behavior under stated internal-access assumptions, then delete
  newly downloaded weights. Gated access is an external constraint if encountered.
- Finish correctness and maintainability review of critical persistence, publication,
  subprocess lifecycle and failure paths. Add tests when they change confidence.
- Verify latest full suite with all supported optional dependencies, packaging, wheel
  smoke checks and applicable CI. Record skips precisely and investigate failures.
- Define supported Python/platform/dependency/hardware/model-format matrix. Run
  cross-platform checks where available; do not turn untested support into claims.
- Audit every user requirement against final evidence before declaring completion.

## Working environment

Reusable environment: `/home/dev/.local/share/rephix-local-20261106/venv`.
This task added prompt-toolkit 3.0.52, Diffusers 0.35.x and GGUF 0.19.x there for
verification. Initial baseline: 328 tests, 21 skipped, passed. Current suite: 334 tests passed
with one skip for the optional accelerate/Triton native FP8 runtime. Ruff lint
and formatting passed. Source archive and wheel build passed. A clean wheel
installation outside the checkout passed version, plain catalog, demo, doctor,
`pip check` and `pip uninstall`, with import absence verified after removal.
This is package-manager verification; dedicated lifecycle commands remain open. Logs for current
verification are in `/tmp/rephix-full-tests.log` and `/tmp/rephix-build.log`.
Model weights should remain disposable; evidence JSON and measured reports belong
in the repository. Full internal layer access does not establish text-only recovery.
