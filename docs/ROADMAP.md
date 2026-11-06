# Research roadmap after v0.2

The runnable toolkit milestone is implemented. See [the completion goal](GOAL.md), [validation](VALIDATION.md), and [measured results](CLOUD_RESULTS.md).

| Research area | Implemented support |
| --- | --- |
| Controlled recovery | Affine identification, rank/conditioning, probability gauge fixing, noise and observation-budget curves |
| Open-model baseline | Pretrained Qwen collection, smaller random students, explicit pretrained priors, grouped evaluation |
| Evidence comparison | Equal-budget token-only/top-k sweeps, residual-mass loss, validation tables and learning curves |
| Component recovery | Actual output-head input capture, selected-vocabulary recovery, novel held-out feature diagnostics |
| Symmetries | Permutation/positive-scale alignment for tiny ReLU networks; sign/permutation alignment for tanh |
| Query efficiency | Random, next-token entropy, and two-candidate Jensen–Shannon disagreement selection with reserved-group exclusions |
| Compute reduction | Smaller student architectures, frozen backbones with untied output heads, gradient accumulation, CPU thread control |
| Reliable trials | Resumable collection and training, optimizer/RNG/order state, source/evidence checks, atomic writes, resumable sweeps |
| Delivery | Offline integration command, cross-platform core CI, model CI, wheel/source packaging, complete experiment guides |

## Research that still needs evidence

The software does not make original whole-model weights identifiable from finite text observations. More compute or more examples do not by themselves resolve parameter symmetries or prove uniqueness.

Useful next scientific questions are whether richer observations improve agreement at a fixed query budget, how active query choices compare with random choices across several seeds, how much prior knowledge a smaller candidate needs, and which additional components become recoverable under explicitly stronger access assumptions. The toolkit can run these experiments; their answers must come from measurements.

Large training runs and GPU/MPS behavior need the corresponding hardware. Run `doctor`, establish a small baseline, then increase one budget at a time. Keep test evidence untouched while tuning. Claims about parameter recovery require both an identification argument for the stated access and a comparison that accounts for the relevant symmetries.
