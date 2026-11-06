# Recover Mamba state-space models

`recover-layers` supports native Transformers Mamba and FalconMamba causal language models and the Mamba layers inside hybrid Jamba checkpoints. The projection, embedding, normalization, and depthwise convolution weights use the standard layer probes. The remaining mixer parameters are the skip coefficients `D` and the log-decay coordinates `A_log`.

```sh
rephix recover-layers --model /path/to/local-mamba --output /path/to/recovered \
  --device cuda --dtype float64 --batch-size 512
rephix audit-weights /path/to/local-mamba /path/to/recovered
```

The mixer oracle runs a native eager one-token update with controlled projection outputs, unit initial recurrent state, and zero input-to-state and state-to-output coefficients. The update writes the remaining state to its cache. For a chosen positive step `dt`, its state response is `exp(-exp(A_log.float()) * dt)`. The pre-output projection response also exposes `D` under a unit input and gate. The probes use float64 step sizes and observations to resolve small changes in the update.

The native recurrence uses a float32 exponential to transform the log-decay coordinates. Distinct stored coordinates can round to the same effective rate. The export chooses a nearby log coordinate that reproduces the recovered rate on the oracle backend and reports the count of effective rates separately from individually identified parameters. The raw checkpoint audit still compares stored `A_log` values. Normal forwards, recurrent caching, and generation use the native model interface after reload.

This method assumes chosen access to the model's internal operators and recurrent cache. Text endpoint outputs expose a different observation interface.

## Pretrained checkpoints

Both checkpoints ran on an RTX 5080 with Transformers 4.57.6 using the native eager path. The probes and model inference used float64 where supported. The recurrence transforms `A_log` in float32, and the causal LM interface returns float32 logits. The raw audit compared all same-name, same-shape stored values at `abs(candidate - reference) <= 1e-8 + 1e-5 * abs(reference)`. [Mamba-130M checkpoint](https://huggingface.co/state-spaces/mamba-130m-hf/tree/1e76775f628fbf1350fbe4dbb3d971ba64af25a1) · [Mamba-370M checkpoint](https://huggingface.co/state-spaces/mamba-370m-hf/tree/b519127f5bfaaa1c27dd938dad051ec360972b23)

| Checkpoint | Original values within tolerance | Exact CUDA effective rates | Exact skip gains | Exact cached generations |
|---|---:|---:|---:|---:|
| Mamba-130M | 129,135,042 / 129,135,360 | 589,824 / 589,824 | 36,864 / 36,864 | 6 / 6 |
| Mamba-370M | 371,515,638 / 371,516,416 | 1,572,864 / 1,572,864 | 98,304 / 98,304 | 6 / 6 |

All raw values outside tolerance were `A_log` coordinates. The maximum absolute raw log error was `1.34e-7`. Independently transforming the stored logs on CPU produced rates within tolerance for every coordinate, with exact equality for 550,751 and 1,476,934 rates respectively. Each model matched all logits on 168 held-out forward positions and 36 recurrent decode positions. The exports also preserved the tokenizer IDs and masks.

The offline tests cover native FalconMamba and Jamba fixtures, cached generation after the source directory is removed, and an explicit rounding ambiguity. The hybrid Jamba fixture recovered 97.288% of stored values exactly and 100% within tolerance. Its source-free replay matched the first export exactly and preserved the original recurrent cache outputs within floating-point tolerance. Changing one stored `A_log` from `-2e-8` to `2e-8` preserved the native rate, logits, and cached states even though the two stored values differ beyond the audit tolerance.

```sh
python examples/mamba_recovery_experiment.py \
  --reference /path/to/local-mamba --candidate /path/to/recovered \
  --device cuda --model-id state-spaces/mamba-130m-hf \
  --revision 1e76775f628fbf1350fbe4dbb3d971ba64af25a1 \
  --output /path/to/results.json
```

[Full measurements](../examples/results/mamba-layer-recovery.json)
