# Supported environments and scientific limits

Rephix 0.4 is a dependable experiment and checkpoint-recovery toolkit within the
contracts below. Reconstruction methods remain research: software reliability does
not establish universal model identifiability or guarantee a score for an unknown
model endpoint.

| Area | Supported contract | Verification or limit |
| --- | --- | --- |
| Python | CPython 3.10–3.13 | Local release checks use 3.13.5; CI covers 3.10, 3.11, 3.12 and 3.13. Newer Python versions are unvalidated. |
| Core operating systems | Linux, macOS, Windows (x86-64; ARM64 where Python/NumPy wheels exist) with writable local storage | Core CI matrix includes all three; model measurements and real-PTY runs are Linux. |
| Managed installation and release channel | Standard-library venv/ensurepip, stable GitHub releases (`--release latest`) or local wheels, a retained base Python | Release lifecycle verified on Linux, macOS and Windows (Python 3.10 and 3.13) against the candidate artifacts: fresh install via discovery, 0.2 (pip) and 0.3 (managed) upgrades, repeat/no-op, corrupted download, unreachable channel, invalid wheel, uninstall, purge and reinstall. See [installation](INSTALLATION.md). |
| Windows console | Windows 10 1809+/11 and Server 2022/2025; cmd.exe, Windows PowerShell 5.1 and PowerShell 7; the classic console host (conhost) and ConPTY-based terminals | Verified on Server 2022 and 2025 with Python 3.10 and 3.13 by driving the installed `rephix.exe` through a real ConPTY and a classic conhost window: see [terminal support](TERMINAL.md#windows-console-verification). Windows Terminal itself was not driven; it hosts the same ConPTY layer. |
| Interactive terminal | Optional prompt_toolkit 3.x, interactive stdin/stdout | Linux PTY, macOS lifecycle PTY and the Windows console matrix above. Plain CLI and `ui --plain` work with redirected output. |
| Core dependencies | NumPy >=1.24 | No model runtime or downloads needed for affine demos and evidence validation. |
| Transformers | Torch >=2.4, Transformers >=4.46,<5, safetensors >=0.4 | Release model suite uses Torch 2.10.0+cu128 and Transformers 4.57.6. Family APIs introduced after 4.46 require a version that includes them; broad family coverage is verified on 4.57.6. |
| Diffusers | Diffusers >=0.35,<0.36, Torch, safetensors, huggingface-hub >=0.34,<1 | Native UNet2D, conditional UNet2D, AutoencoderKL and AutoencoderTiny; fixture and published pretrained results have separate reports. |
| Quantized operators | GGUF >=0.19,<0.20; supported FP8 and MXFP4 safetensors layouts | Packed-field and effective numeric agreement are separate metrics; code ambiguity is reported. Native FP8 inference requires compatible CUDA and optional accelerate/Triton. |
| Compute | CPU for core and model fixtures; CUDA where runtime/operator supports it | Current measured GPU: RTX 5080, 16 GB. GPU-specific rounding can affect recoverability. |
| Apple MPS | **Not verified.** `--device mps` is accepted and monitored, but no Apple silicon machine with a usable Metal device was available | GitHub-hosted macOS runners expose a paravirtual GPU without Metal 2.0, so PyTorch reports MPS unavailable ([probe](../examples/results/mps-runner-probe.json)). Verified: the explicit unavailable-MPS error on real macOS, `--device auto` falling back to CPU, and CPU-fallback operator reporting (`accelerator.cpu_fallback_operators`). The end-to-end MPS check (`examples/mps_check.py`, `mps` workflow) is ready for a self-hosted Apple silicon runner. |
| Storage | Local filesystem, SQLite transactions, atomic same-filesystem publication | Keep source, evidence and output directories disjoint. Network/synchronized filesystems need their own locking/durability validation. |

The package's Python metadata permits >=3.10 to avoid blocking newer environments;
that does not extend the verified range above. Minimum dependency bounds permit core
subsets; use the measured versions for the broadest tested family coverage. A driver,
CUDA build, accelerated kernel, and model family must be mutually compatible. Native
CUDA kernels are optional for many state-space paths, which have slower fallbacks.

## Access and model formats

- Text-only parameter recovery (`text-recover`) observes only sampled strings and
  recovers a defined canonical parameterization for targets in a declared
  softmax-affine class with a public feature map. The guarantee is conditional on
  class membership, which observations can reject but not prove. Original parameters
  are not identifiable from text in general. See [text recovery](TEXT_RECOVERY.md).
- Behavioral sessions learn from supplied text and exposed agent events without the
  target tokenizer, architecture or weights. They train a standalone byte-level
  candidate. Held-out behavior measures agreement, not original parameter recovery.
- Layer checkpoint recovery needs the known architecture, parameter ties, and chosen
  coordinate-preserving access to each supported internal primitive. This access can
  reveal about as much information as a weight dump. Recovery does not bypass an
  endpoint's lack of exposed internals.
- Passive trace recovery needs internal input/output traces with adequate rank and
  lookup coverage; missing directions require separately budgeted chosen probes.
- Endpoint head recovery requires a known base and dense endpoint scores with the
  documented gauge/rank assumptions. Top-k token observations have weaker information.
- Standard checkpoint exports use local safetensors or indexed safetensors shards,
  config, and tokenizer/processor assets. Custom remote model code is rejected.
  Legacy names, discarded serialized buffers, unsupported persistent state and
  quantization ambiguities must be reported, not silently counted as exact recovery.
- Each family guide linked in README defines its primitive access, supported layouts,
  numerical limitations and measured scope. The supported operator plan rejects
  unknown parameterized operators rather than exporting an incomplete checkpoint.

RMSNorm sign symmetries already demonstrated in Llama and Mistral show why identical
endpoint behavior need not identify 95% of the original stored coordinates. Results
must report reference coverage, exact numeric equality, tolerance and relative L2
alongside behavior. Random fixture recovery verifies implementation; pretrained
checkpoint recovery verifies those particular checkpoints. Neither proves universal
family coverage or text-only recovery.

## Resources and recovery

There is no fixed universal RAM or disk requirement. Allow for source weights,
CPU candidate, probe inputs/outputs, export staging, and framework overhead. FP64
weights use four times the space of BF16/FP16 weights; retaining two full models can
therefore dominate RAM. `--stage-layers` limits GPU residency, not host memory.
`doctor --workspace PATH` reports free disk and `doctor --model LOCAL_PATH` reports
available model estimates. Evidence can exceed the source size, especially with
full spatial convolution responses. Compact observations and schema-2 compression
reduce this for appropriate models; [measured storage results](EVIDENCE_STORAGE.md)
show one checkpoint, not a universal compression ratio.

Use `--evidence-dir` for interrupted chosen-layer capture and source-free replay of
sealed bundles. Sessions checkpoint model, optimizer, sampler and evidence membership
at durable boundaries. A forced kill may lose uncommitted work; it must not be counted
as completed evidence. Stop writers before backing up directories or updating runtime
libraries. Use explicit local output paths and retain hashes/results before deleting
only the disposable weights and artifacts created for an experiment.

## Known limits

Production operation still depends on the stated access, supported operators,
resource budgets, filesystem guarantees and dependency versions. There is no
arbitrary-agent tool executor, universal architecture discovery, general transformer
symmetry alignment, or guarantee of arbitrary black-box weight extraction. The
terminal workbench is a local command runner, not a remote multi-user job service.
Managed install rollback changes software only; data format rollback is separate.
Cross-platform CI and local measurements should be consulted together; an untested
interactive terminal or accelerator path is explicitly unverified.
