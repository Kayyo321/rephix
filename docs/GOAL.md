# Autonomous completion goal

Build and verify everything needed to run Rephix experiments without depending on the owner's computer. Deliver the implementation, tests, reproducible measurements, and instructions to `Kayyo321/rephix`.

This milestone completes the runnable research toolkit. It does not assume that finite text observations identify a language model's original weights, or promise a particular reconstruction accuracy before measurement.

## Completion checklist

- [x] Preserve the existing GitHub repository and license; import the initial engine.
- [x] Validate and isolate evidence; make important artifact writes atomic.
- [x] Demonstrate affine recovery and neural-network parameter symmetries.
- [x] Recover a restricted output projection from explicitly observed open-model activations.
- [x] Collect real Qwen observations and run a bounded cloud trial if downloads/resources permit.
- [x] Support smaller candidates, restricted training, and exact CPU training resumption.
- [x] Run reproducible evidence/compute sweeps and export comparison tables and plots.
- [x] Select new prompts by random sampling, uncertainty, or candidate disagreement without touching held-out evidence.
- [x] Supply a fully offline end-to-end experiment and automated tests for all workflows.
- [x] Check packaging, installation, command-line errors, and continuous integration.
- [x] Publish source, measured results, and a clear local-computer handoff to GitHub.

## Stop condition

All implementable checklist items are complete and verified, with remaining hardware-dependent tests or resource blocks explicitly recorded. Accuracy claims must state the evidence, initialization, trainable components, held-out prompt budget, and metric. Future research directions remain research, not silently claimed successes.

## Completed delivery

Completed on 2026-11-06 as the v0.2 research toolkit. All 106 tests passed locally with model/plotting dependencies, and all eight jobs in the [GitHub validation run](https://github.com/Kayyo321/rephix/actions/runs/42158375656) passed. A clean wheel installation also passed core checks with NumPy 2.5.4 and no model dependencies. Source, experiment reports, generic evidence, and a ready-to-use small student configuration are published in the repository.

See [the measured results](CLOUD_RESULTS.md) and [validation details](VALIDATION.md). The real Qwen restricted projection was recovered accurately under observed-feature access; the short smaller-student behavioral trial did not achieve held-out response agreement. Whole-model text-only weight recovery remains unproven. CUDA/MPS performance and long training budgets require suitable hardware and further evidence.
