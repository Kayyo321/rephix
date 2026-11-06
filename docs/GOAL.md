# Autonomous completion goal

Build and verify everything needed to run Rephix experiments without depending on the owner's computer. Deliver the implementation, tests, reproducible measurements, and instructions to `Kayyo321/rephix`.

This milestone completes the runnable research toolkit. It does not assume that finite text observations identify a language model's original weights, or promise a particular reconstruction accuracy before measurement.

## Completion checklist

- [x] Preserve the existing GitHub repository and license; import the initial engine.
- [ ] Validate and isolate evidence; make important artifact writes atomic.
- [ ] Demonstrate affine recovery and neural-network parameter symmetries.
- [ ] Recover a restricted output projection from explicitly observed open-model activations.
- [ ] Collect real Qwen observations and run a bounded cloud trial if downloads/resources permit.
- [ ] Support smaller candidates, restricted training, and exact CPU training resumption.
- [ ] Run reproducible evidence/compute sweeps and export comparison tables and plots.
- [ ] Select new prompts by random sampling, uncertainty, or candidate disagreement without touching held-out evidence.
- [ ] Supply a fully offline end-to-end experiment and automated tests for all workflows.
- [ ] Check packaging, installation, command-line errors, and continuous integration.
- [ ] Publish source, measured results, and a clear local-computer handoff to GitHub.

## Stop condition

All implementable checklist items are complete and verified, with remaining hardware-dependent tests or resource blocks explicitly recorded. Accuracy claims must state the evidence, initialization, trainable components, held-out prompt budget, and metric. Future research directions remain research, not silently claimed successes.
