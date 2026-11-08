# Production readiness audit

## 0.4.0 (release candidate, 2026-11-08)

Publication is blocked by one gate: MPS verification needs Apple silicon hardware that
was not available. See [the 0.4 release notes](RELEASE_0_4.md).

| Requirement | Evidence | Status |
| --- | --- | --- |
| Precise text-only recovery contract and boundary | [TEXT_RECOVERY.md](TEXT_RECOVERY.md); string-only oracle protocol; mode-000 private directory, audit hook, per-run secret targets in experiments | Implemented and audited |
| Universal text-only recovery | Impossible for original parameters (theorem and executable counterexamples). Conditional guarantee for a declared class, implemented with pre-registered held-out evaluation | Resolved; [results](TEXT_RECOVERY_RESULTS.md) |
| Independent review of recovery claims | Review findings fixed: conditional guarantee, validation in a two-sided class test, LayerNorm span, isolation, analysis; protocol v2 registered before its final run | Done |
| MPS end-to-end | No usable Metal device on any reachable runner ([probe](../examples/results/mps-runner-probe.json)); unavailable path and fallback reporting verified; harness ready | **Blocked** |
| Native Windows console | ConPTY and conhost on Server 2022/2025, Python 3.10/3.13, cmd, PowerShell 5.1 and 7; 86–88 checks per combination with named host limitations | Verified ([details](TERMINAL.md#windows-console-verification)) |
| Install, upgrade, failure, uninstall, reinstall | Candidate bytes on Linux/macOS/Windows × Python 3.10/3.13: 30/30 lifecycle checks each, including real 0.2.0 (pip) and 0.3.0 (managed) upgrades | Verified before publication |
| Release automation | Build-once workflow with SHA256SUMS, BUILD-INFO and provenance; publish-time verification only | Validated by workflow_dispatch runs |
| Publication and post-publication verification | Not performed | Blocked by MPS |

Evidence files: [`examples/results/release-0.4/`](../examples/results/release-0.4/).

---

## 0.3.0


Release audit completed 2026-11-08 for the verified implementation checkpoint. History and earlier baseline commits remain
preserved. The audit covers every user requirement, current source hashes, measured results,
and release checks. [Release report](RELEASE_0_3.md)

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Complete paused compact/compression work | Four-mode pinned ConvNeXT result, exact audits, interruption/resume, source-unavailable replay; Transformers/Diffusers integration tests | Verified locally |
| Representative pretrained Llama/Mistral trials | TinyLlama 1.1B: 1,100,048,384 exact values; TinyMistral: 248,024,064 exact values; both 100% coverage, zero L2, exact 24-prompt logits and six continuations | Verified locally |
| Honest access, metrics and limitations | RELEASE_0_3.md, EVIDENCE_STORAGE.md, SUPPORT.md; sign-symmetry controls with identical behavior and different raw coordinates | Documented with measured evidence |
| Designed CLI experience | Optional terminal workbench with searchable commands, dark/light palettes, visible selection, editable arguments, help, output, interruption and plain mode | Linux PTY, macOS lifecycle PTY and cross-platform input tests verified |
| Installation and required/optional dependencies | Standard-library bootstrap, wheel install, extras, constraints/offline wheelhouse, dependency checks and reconstruction smoke test | Isolated real lifecycle verified |
| Safe updates and rollback | Fresh environment validation before atomic activation; failed wheel leaves manifest/version unchanged; previous environment retained; runtime lease prevents modifying active software | Unit and isolated lifecycle verified |
| Clean removal and retention choices | Default data retention, individual purge flags, optional bootstrap removal, no external data deletion; shared lock remains as documented | Unit and isolated lifecycle verified |
| Versions, diagnosis and help | --version, --help, doctor, installation inspect --verify; missing dependencies and unsupported schema errors | CLI and clean installation verified |
| Migration and recovery policy | Session schema 1 unchanged, raw layer schema 1 retained, new compressed schema 2, incompatible future schemas rejected; no implicit data/config migration | Compatibility/failure tests and INSTALLATION.md |
| Failure handling and maintainability review | Reviewed journal codec/checksum/bounds, probe routing/binding, publication and installer state transitions; added runtime leases, safe deletion checks, interrupted update cleanup, malformed compact-flag rejection and clear Ctrl+C exit | Critical regression tests pass |
| Supported platforms/Python/hardware/formats/access | SUPPORT.md, pyproject extras and model-family guides | Defined; interactive macOS/Windows and MPS limitations explicit |
| Full local suite, native optional runtime | 346 tests, zero skips, including GPU native FP8; dependency consistency, Ruff lint/format, offline trial | Passed |
| Source and wheel packaging | Build succeeded; real 0.2-to-0.3 lifecycle with version checks, failed upgrade, rollback, inherited TUI extra, pruning, reinstall and retention/purge | Passed locally |
| Cross-platform CI on release commit | Expanded core matrix (3.10/3.11/3.12/3.13, Linux/macOS/Windows), three lifecycle jobs, CPU model integration, packaging | All 17 jobs passed at verified code commit 531a995 |
| Disk discipline | Downloaded ConvNeXT, TinyLlama and TinyMistral source/candidate/evidence artifacts deleted after recording hashes and measurements; isolated lifecycle leaves zero environments | Verified locally |
| Final requirement-by-requirement completion audit | Source hashes, pinned measurements, lifecycle transitions, documentation links, package contents, CI job results, baseline ancestry and cleanup inspected | Passed |

Machine-readable [local verification](../examples/results/release-verification.json)
records code hashes, test count, wheel hash and checks. The
[lifecycle result](../examples/results/installation-lifecycle.json) records hashes
of retained user artifacts and all verified lifecycle transitions. The small
persistent installer lock is intentionally retained even with bootstrap removal;
manual deletion of a reviewed, idle dedicated prefix removes that final record.

The reusable Python environment is
`/home/dev/.local/share/rephix-local-20261106/venv`. The completed optional native-runtime
check used `/tmp/rephix-release-native` for accelerate 1.12.0,
Triton 3.6.0 and psutil. That task-owned overlay has been removed. Logs are
in `/tmp/rephix-release-tests.log`, `/tmp/rephix-build.log`,
`/tmp/rephix-lifecycle-check.log` and `/tmp/rephix-offline-check.log`.

The final code check passed all 17 jobs in [GitHub validation run 42497553708](https://github.com/Kayyo321/rephix/actions/runs/42497553708): twelve core platform/Python combinations, three real lifecycle jobs, CPU model integration and packaging. [CI evidence](../examples/results/release-ci.json) records job IDs, conclusions and the exact verified commit. Final release-report changes are documentation and evidence only; runtime source hashes remain those verified locally and in CI.

Earlier macOS attempts exposed a PTY harness that needed continuous output draining,
then Darwin's kernel-owned PENDIN state (0x20000000), set when canonical input is
restored. The final harness verifies a cooked input round trip and compares every
mode without masking bits. The application also restores a complete POSIX snapshot
around toolkit setup. This full check passed on macOS and Linux. Windows lifecycle
and all Windows core/input-routing jobs passed; a real interactive Windows console
and MPS model recovery remain explicitly unverified.

Final interaction review also found and fixed a command-submission race: arguments
are captured at the run keypress (Enter or F5), so immediate edits cannot change a pending command. The input
regression demonstrably fails with the earlier mutable-argument behavior. The explicit color override also has a regression check for the terminal library’s
separate NO_COLOR detection. All 346
local tests still pass; the final CI run passed all 17 jobs with this interaction change.

No user requirement is left as an unimplemented promise. Scientific limits remain:
original weights are not generally identifiable from ordinary text or endpoint
behavior, supported primitive access must be available, and untested hardware and
interactive paths are outside the verified scope. See SUPPORT.md for the complete
contract. All task-owned model downloads, recovered models and native-runtime
verification overlays were deleted after their evidence was recorded.
