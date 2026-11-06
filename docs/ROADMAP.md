# Research roadmap

The first milestone is a reproducible recovery baseline, with honest measurements of what the observations constrain.

## Available foundation

- Affine recovery from known features, with rank and softmax-gauge diagnostics.
- Structured observations for text, exposed reasoning, token IDs, and optional top-token probabilities.
- Local Transformers reference collection, candidate fitting, held-out evaluation, and checkpoint comparison.
- A small-model workflow that can be exercised before downloading a Qwen checkpoint.

## Next experiments

1. **Establish the Qwen baseline.** Run a tiny pipeline check, then collect enough varied prompt groups to measure generalization. Keep code, evidence, and checkpoint metadata together.
2. **Measure evidence value.** Compare continuations alone with top-token probabilities at fixed prompt and compute budgets. Plot agreement against observed tokens rather than reporting one favorable run.
3. **Recover restricted components.** Use open-model activations in explicitly labeled experiments to recover an output projection or another isolated component. Keep these results separate from text-only reconstruction because the extra access changes the problem.
4. **Study parameter symmetries.** Add alignment-aware comparisons for tiny networks before interpreting raw tensor distance as recovery accuracy.
5. **Improve query selection.** Choose new local reference prompts using disagreement or uncertainty, while keeping an untouched evaluation set. Compare against random prompt selection at the same query budget.
6. **Reduce experiment cost.** Explore frozen components, low-rank candidates, and smaller architectures. Label these as approximate behavioral reconstructions unless the recovery target is explicitly restricted.
7. **Add resumable sweeps.** Persist optimizer state, evidence fingerprints, run configuration, and checkpoints so a trial can resume and multiple settings can be compared reliably.

## Decision gates

Increase model size only after the tiny-model path works end to end. Increase optimization compute only after token alignment, evidence integrity, and split separation are verified. Claim parameter recovery only when the stated observations identify the target parameters and the comparison accounts for the relevant ambiguities.

A useful outcome may be an accurate behavioral surrogate or a recovered component rather than the original full checkpoint. Each has a measurable target; report which one an experiment achieves.
