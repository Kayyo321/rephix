# Validation of v0.2

Validated on 2026-11-06 with Python 3.12.14, NumPy 2.3.5, PyTorch 2.15.0+cpu, Transformers 4.57.6, and Matplotlib 3.10.8.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest discover -s tests -v
ruff check .
ruff format --check .
rephix offline-trial --output runs/validation --steps 64 --seed 42 --threads 1
```

All **106 tests passed** with the optional model/plotting dependencies installed. The model tests construct local checkpoints; a network-call guard also covers the complete offline trial. A separate core-only environment ran the same suite with model-dependent tests skipped and verified the NumPy-only commands.

Coverage includes affine identification and rank deficiency, numerical gauges, symmetry alignment, actual tiny-Qwen projection capture, independent prompt groups, cached token fingerprints, reasoning serialization, causal token alignment, raw versus decoding-processed distributions, top-k residual loss and gradients, mixed-objective evaluation, smaller students, frozen/tied embeddings, interrupted collection, and exact CPU training continuation with dropout and accumulation. Failure-injection tests verify that partial writes preserve previous evidence, reports, NPZ artifacts, and training state. Sweep tests exercise immutable source identity, changed/tampered files, interruption, configuration validation, aggregate reporting, and standalone SVG output.

The complete 64-step offline experiment compared two objectives from the same random candidate initialization against an untrained, two-layer Qwen2 reference with 22,816 parameters and a 64-token tokenizer. Validation selected the saved candidate before the single final test evaluation. On ten held-out prompts it reached 87.5% recorded-token agreement over 40 tokens and 70% greedy exact match. This is a pipeline result for a synthetic teacher, not pretrained language-model recovery accuracy. Its separate observed-feature projection experiment reached about `1.05e-7` relative parameter error.

The controlled affine baseline used 192 training and 64 test samples, recovering full-logit parameters to about `9.1e-16` relative error and reaching 100% held-out top-1 agreement. The probability version fixes the common-logit-shift ambiguity before comparing parameters.

A real pretrained Qwen2.5-0.5B cloud experiment also ran. See [CLOUD_RESULTS.md](CLOUD_RESULTS.md) for both the accurate restricted component recovery and the limited smaller-student smoke result. Model weights are excluded from the repository and release artifacts.

CI runs core checks on Linux, Windows, and macOS with Python 3.10 and 3.12. It also runs the offline model workflow on Linux, builds wheel and source distributions, and checks installation from the wheel. See [GitHub Actions](https://github.com/Kayyo321/rephix/actions/workflows/tests.yml). The [local CUDA results](LOCAL_RESULTS.md) record the RTX 5080 experiments and exact resumption observed on that workstation. MPS execution remains unmeasured.

In a sandbox with a read-only home directory, set `MPLCONFIGDIR` to a writable directory before plotting. This changes only Matplotlib's cache location.

The [eight-job GitHub run for the implementation](https://github.com/Kayyo321/rephix/actions/runs/42158375656) passed on every configured platform. A fresh isolated wheel installation with NumPy 2.5.4 and no Torch, Transformers, or Matplotlib also ran all 106 discovered tests successfully with 28 optional-dependency skips; `doctor`, affine recovery, and symmetry commands worked from outside the source checkout.
