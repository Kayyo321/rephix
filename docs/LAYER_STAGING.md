# Recover checkpoints with layer staging

`recover-layers --stage-layers` keeps the source model on the host. The local oracle moves only the modules needed by the current probe to the requested accelerator. Rephix reuses those modules across batches and releases them when the next probe needs different modules. The source keeps its original host tensors and parameter ties throughout the run.

```sh
rephix --threads 4 recover-layers --model /path/to/local-source \
  --output /path/to/recovered --device cuda --dtype auto --stage-layers --batch-size 256
```

Use `--dtype auto` for large native BF16 or FP16 safetensors. With a same-dtype load, Transformers can keep the source tensors file-backed on the host. The candidate is materialized in CPU memory, and a dtype conversion during source loading can allocate a second host copy. Accelerator memory must hold the largest module group and its probe inputs and outputs. `--batch-size` controls the size of those inputs and outputs. Allow disk space for the source and the new export while recovering.

The report records the number of staged groups, peak bytes of staged parameters and buffers, and calls executed on the host. That peak excludes temporary tensors, allocator caching, and runtime overhead. BART's additive logit bias uses a one-token CPU forward in staged mode. Probes with dependent operators can stage them together, as for DeBERTa v2's masked-LM head and its vocabulary embedding.

The recovered model uses the same export format as resident recovery. Staging changes how the local oracle serves probes. It still needs known native operators, chosen inputs, and full layer outputs.

## Pretrained memory measurements

The trials ran on an RTX 5080 with native BF16 weights and probe batches of 256. Peak allocated memory measures PyTorch CUDA allocations, including the probe tensors. The experiment also records reserved CUDA bytes, process peak host RSS, and elapsed time. [Measurements](../examples/results/staged-layer-recovery.json)

For Qwen3-0.6B, staging reduced peak CUDA allocation by 74.1%, from 1,203,255,808 bytes to 311,691,264 bytes. Both modes matched every original archive value, including the duplicated `lm_head.weight`, which the export stores once under the tied embedding. Literal-name audit coverage was 79.3007%. The independent audit checked both original copies against the saved value after verifying the tie in the loaded export. Both modes matched all held-out logits and six cached greedy continuations. [Qwen3-0.6B checkpoint](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca)

Qwen3-8B has 16,381,470,720 bytes of BF16 parameters, exceeding the 15,873,146,880 GPU bytes available at launch. The staged export used a peak of 1,334,116,352 CUDA-allocated bytes and 24,185,954,304 bytes of host RSS. It made 5,939 GPU probe calls across 399 staged groups and took 11 minutes 31 seconds, including source loading and export. The independent shard audit matched all 8,190,735,360 original values exactly. Reloading source and export separately on CPU produced identical full-vocabulary next-token logits on six prompts and matched all six cached greedy continuations. [Qwen3-8B checkpoint](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218)

```sh
python examples/staged_layer_recovery_experiment.py recover \
  --model /path/to/local-source --output /path/to/recovered \
  --metrics /path/to/memory.json --device cuda --stage-layers
python examples/staged_layer_recovery_experiment.py evaluate \
  --reference /path/to/local-source --candidate /path/to/recovered \
  --output /path/to/evaluation.json --model-id Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 --device cpu
```
