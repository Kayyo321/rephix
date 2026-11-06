# Initial validation

Validated on 2026-11-06 with Python 3.12.14, NumPy 2.3.5, PyTorch 2.15.0+cpu, and Transformers 4.57.6.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest discover -s tests -v
ruff check .
ruff format --check .
rephix demo --seed 42
```

All 28 tests passed, including the offline Transformers tests. The language-model fixture creates an untrained, one-layer Qwen2 model with hidden size 16 and a local 16-token tokenizer. It exercises collection, random initialization, optimization, checkpoint saving, independent test evaluation, greedy generation, and direct tensor comparison. It also checks token provenance, copied-training-data rejection, explicit reasoning serialization, decoding configuration, and the top-k residual loss and gradients.

The affine demo used 192 training and 64 held-out samples. Full-logit recovery produced parameter relative error about `9.1e-16`, held-out logit RMSE about `1.3e-15`, and 100% held-out top-1 agreement. Probability-based recovery reached similar precision after fixing the common-logit-shift ambiguity. These are controlled affine results, not language-model reconstruction results.

CLI smoke checks also exercised evidence validation, grouped splitting with manifests, and NPZ affine recovery/export. Core commands run without the optional model libraries.

No pretrained Qwen checkpoint was downloaded or evaluated in this session. The offline tests establish pipeline correctness at a tiny scale; they do not establish recovery accuracy for Qwen2.5-0.5B. Follow [EXPERIMENTS.md](EXPERIMENTS.md) to obtain that baseline on your computer.
