# Offline integration trial

The reference is a locally initialized, untrained Qwen2 model with a 64-token vocabulary. These are pipeline checks, not pretrained Qwen recovery results.

- Candidate optimization budget: 64 steps for each of two random-initialized candidates.
- Compared objectives: recorded-token cross entropy and top-k residual-bucket distillation.
- Selection: validation continuation cross entropy; chosen top-k weight 0.0.
- Untouched test recorded-token agreement: 0.875000.
- Untouched test greedy exact match: 0.700000.
- Projection relative parameter error with observed hidden features: 1.04853e-07.
- Projection design rank: 32 of 32.
- New candidate-entropy queries selected: 8.

See `trial.json` for provenance, `comparison/comparison.md` for validation comparisons, and `test.json` for the single final test evaluation.
