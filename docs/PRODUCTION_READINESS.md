# Production readiness audit

Active release goal, resumed 2026-11-08. History and earlier baseline commits remain
preserved. Completion requires all user requirements and final current-commit gates,
not merely a successful model trial. [Release report](RELEASE_0_3.md)

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Complete paused compact/compression work | Four-mode pinned ConvNeXT result, exact audits, interruption/resume, source-unavailable replay; Transformers/Diffusers integration tests | Verified locally |
| Representative pretrained Llama/Mistral trials | TinyLlama 1.1B: 1,100,048,384 exact values; TinyMistral: 248,024,064 exact values; both 100% coverage, zero L2, exact 24-prompt logits and six continuations | Verified locally |
| Honest access, metrics and limitations | RELEASE_0_3.md, EVIDENCE_STORAGE.md, SUPPORT.md; sign-symmetry controls with identical behavior and different raw coordinates | Documented with measured evidence |
| Designed CLI experience | Optional terminal workbench with dark/light palettes, visible selection, editable arguments, help, output, interruption and plain mode | Linux PTY and automated tests verified |
| Installation and required/optional dependencies | Standard-library bootstrap, wheel install, extras, constraints/offline wheelhouse, dependency checks and reconstruction smoke test | Isolated real lifecycle verified |
| Safe updates and rollback | Fresh environment validation before atomic activation; failed wheel leaves manifest/version unchanged; previous environment retained; runtime lease prevents modifying active software | Unit and isolated lifecycle verified |
| Clean removal and retention choices | Default data retention, individual purge flags, optional bootstrap removal, no external data deletion; shared lock remains as documented | Unit and isolated lifecycle verified |
| Versions, diagnosis and help | --version, --help, doctor, installation inspect --verify; missing dependencies and unsupported schema errors | CLI and clean installation verified |
| Migration and recovery policy | Session schema 1 unchanged, raw layer schema 1 retained, new compressed schema 2, incompatible future schemas rejected; no implicit data/config migration | Compatibility/failure tests and INSTALLATION.md |
| Failure handling and maintainability review | Reviewed journal codec/checksum/bounds, probe routing/binding, publication and installer state transitions; added runtime leases, safe deletion checks, interrupted update cleanup, malformed compact-flag rejection and clear Ctrl+C exit | Critical regression tests pass |
| Supported platforms/Python/hardware/formats/access | SUPPORT.md, pyproject extras and model-family guides | Defined; interactive macOS/Windows and MPS limitations explicit |
| Full local suite, native optional runtime | 344 tests, zero skips, including GPU native FP8; dependency consistency, Ruff lint/format, offline trial | Passed |
| Source and wheel packaging | Build succeeded; real 0.2-to-0.3 lifecycle with version checks, failed upgrade, rollback, inherited TUI extra, pruning, reinstall and retention/purge | Passed locally |
| Cross-platform CI on release commit | Expanded core matrix (3.10/3.12/3.13, Linux/macOS/Windows), three lifecycle jobs, CPU model integration, packaging | Pending remote execution |
| Disk discipline | Downloaded ConvNeXT, TinyLlama and TinyMistral source/candidate/evidence artifacts deleted after recording hashes and measurements; isolated lifecycle leaves zero environments | Verified locally |
| Final requirement-by-requirement completion audit | Must inspect final CI results and final repository state before completion | Pending |

Machine-readable [local verification](../examples/results/release-verification.json)
records code hashes, test count, wheel hash and checks. The
[lifecycle result](../examples/results/installation-lifecycle.json) records hashes
of retained user artifacts and all verified lifecycle transitions. The small
persistent installer lock is intentionally retained even with bootstrap removal;
manual deletion of a reviewed, idle dedicated prefix removes that final record.

The reusable Python environment is
`/home/dev/.local/share/rephix-local-20261106/venv`. The complete optional native-runtime
check temporarily uses `/tmp/rephix-native-runtime-check` for accelerate 1.12.0,
Triton 3.6.0 and psutil. Remove that task-owned overlay after verification. Logs are
in `/tmp/rephix-release-tests.log`, `/tmp/rephix-build.log`,
`/tmp/rephix-lifecycle-check.log` and `/tmp/rephix-offline-check.log`.

Initial remote CI passed all nine core jobs, CPU model integration and packaging.
The macOS lifecycle harness stopped draining its PTY and timed out on exit; the
harness now drains output continuously and checks terminal-mode restoration.
A follow-up run must verify this fix and the Windows lifecycle (cancelled by the
first matrix failure). Installer pip checks now use isolated Python startup and
venv bootstrap errors have a tested, clear CLI failure.

Remaining work: inspect the current commit's full cross-platform CI, fix any failures,
and perform the final completion audit. Do not treat scientific limitations as
unimplemented guarantees: universal original-weight recovery from text remains
unproven and is not promised by this release.
