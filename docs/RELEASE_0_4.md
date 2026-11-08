# Rephix 0.4.0 (prerelease)

**Status: published as a prerelease (`v0.4.0`, commit `bee193f`). Stable publication is blocked.** The one mandatory gate still open is end-to-end MPS verification, which needs Apple silicon hardware that has a usable Metal device. Install by tag with `--release v0.4.0`; `--release latest` does not select prereleases. Everything else is verified on the published bytes, and the post-publication verification workflow passed on Linux, macOS and Windows.

## What changed

* **Text-only recovery, resolved both ways.** Universal recovery of original
  parameters from text is impossible, and `rephix text-identifiability` demonstrates
  why with executable counterexamples. Inside a declared softmax-affine class with a
  public feature map, `rephix text-recover` recovers a defined canonical
  parameterization from sampled strings only. It reports standard errors, runs a
  two-sided class test on fitted and held-out contexts, and exports a model that
  `rephix text-model` runs without the target. The guarantee is conditional on class
  membership, which observations can reject but not prove.
  [Contract and theorem](TEXT_RECOVERY.md) · [Pre-registered results](TEXT_RECOVERY_RESULTS.md)
* **Release channel.** The managed installer can install and update from the latest
  stable GitHub release (`--release latest`). It verifies the release's `SHA256SUMS`
  and, when present, GitHub's server-side asset digest. It never selects drafts or
  prereleases, treats a repeated update as a no-op, and adopts the new manager after an
  update. New `installation check-update` command. Releases publish
  `rephix-installer.py`, `BUILD-INFO.json` and `SHA256SUMS`, with build provenance
  attestations.
* **Windows console.** New native `rephix.exe` launcher, so Ctrl+C no longer triggers
  cmd's "Terminate batch job" prompt. The launcher's runtime lease is now genuinely
  shared, so two concurrent commands work. The redirected welcome no longer crashes on
  ANSI code pages. VT processing is enabled before colored output in classic console
  windows. Bootstrap removal through `rephix.cmd` no longer reports a false failure.
* **Python 3.10.** The post-update launcher refresh now keeps the launcher on the base
  Python. On 3.10, `sys._base_executable` inside a venv points into the venv.
* **MPS reporting.** Commands that use MPS report `accelerator.device`, any operators
  PyTorch ran on the CPU instead (`cpu_fallback_operators`), and Metal memory held.
  An unavailable MPS device gets a specific explanation.
* **Release automation.** Assets are built once from an exact commit and verified on
  Linux, macOS and Windows before release. Publication only verifies; it never rebuilds
  or overwrites assets.

## Gate status

| Gate | Status | Evidence |
| --- | --- | --- |
| Universal text-only recovery | Resolved. Universal recovery of original parameters is impossible (proof and counterexamples). A conditional guarantee for the declared class is implemented and measured on 85 pre-registered held-out runs: 10 of 11 criteria passed. S1 failed on a single conservative false alarm (one in-class target rejected at p = α/2) | [TEXT_RECOVERY.md](TEXT_RECOVERY.md), [results](TEXT_RECOVERY_RESULTS.md) |
| MPS | **Blocked: no usable Metal device available** | [probe record](../examples/results/mps-runner-probe.json), `mps` workflow |
| Full Windows console operation | Verified for the stated matrix: Server 2022 and 2025 × Python 3.10 and 3.13, cmd, PowerShell 5.1 and 7, ConPTY and conhost; 87–89 checks each, all passing. Three named host limitations are listed as not exercised | [TERMINAL.md](TERMINAL.md#windows-console-verification), [evidence](../examples/results/release-0.4/) |
| Install, upgrade, failure, uninstall, reinstall | Verified on Linux, macOS and Windows with Python 3.10 and 3.13, using the candidate bytes and a mirror of the release channel; all checks pass. Windows 3.10 used the stock pip 23.0.1 | [evidence](../examples/results/release-0.4/) |
| Publication | Published as a prerelease (`v0.4.0`); post-publication verification passed on all three operating systems (`verify-release` workflow). Stable publication waits for the MPS gate | [release page](https://github.com/Kayyo321/rephix/releases/tag/v0.4.0) |

## MPS (Apple silicon)

Every reachable macOS runner was probed (standard and larger GitHub-hosted arm64
images on macOS 15 and 26). All expose an "Apple Paravirtual device that does not
support Metal 2.0", so PyTorch 2.15 reports MPS unavailable. The repository has no
self-hosted runners, and this account has no other Mac.

Verified instead:
* The explicit unavailable-MPS error and CPU behavior on real macOS.
* The CPU-fallback detection and reporting logic.

**Smallest action to finish:** register an Apple silicon Mac (macOS 13+, physical
GPU) as a self-hosted runner for this repository with labels `self-hosted, macOS,
ARM64`, then dispatch the `mps` workflow with `self_hosted: true`. Alternatively, run
`python examples/mps_check.py --work-dir /tmp/mps --output mps.json` on such a Mac
from a checkout of the candidate commit, after `pip install '.[hf]'`. The check
trains, interrupts, resumes, evaluates and recovers a checkpoint on MPS through the
real CLI, with CPU fallback disabled. It compares against CPU with stated tolerances
and requires nonzero Metal allocation and zero fallback operators.
