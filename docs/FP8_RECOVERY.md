# Recover block-scaled FP8 checkpoints

`recover-fp8-safetensors` reconstructs native Transformers-style E4M3FN weights paired with `weight_scale_inv` tensors. It supports positive finite FP32 and BF16 scales, partial blocks at matrix edges, mixed dense tensors, and indexed safetensors shards. The output preserves the shard headers, runtime configuration, and tokenizer assets.

```sh
pip install -e '.[hf]'
rephix recover-fp8-safetensors --model /path/to/reference --output /path/to/recovered
rephix audit-fp8-safetensors /path/to/reference /path/to/recovered
```

Recovery observes the **dequantized** affine map on chosen unit inputs. FP64 is the default observation dtype because it exactly represents the product of a finite FP32 scale and a finite E4M3FN code. `--observation-dtype float32` accepts rounded responses in the supported normal range. Those responses can leave adjacent FP32 scales indistinguishable. An instrumented runtime can pass `call_tensor(name, row_start, row_stop, inputs)` to the Python API.

Native quantized inference kernels dynamically quantize activations, so their outputs are a different observation interface. Recovery requires exact dequantized responses, while the independent inference check can load the packed export directly. [Transformers fine-grained FP8](https://huggingface.co/docs/transformers/v4.57.6/en/quantization/finegrained_fp8)

For each tile, Rephix tries compatible scales and codes, prefers the largest feasible maximum code, then verifies that the packed tile reproduces every observed value. Different scales and codes can decode to the same numbers. Positive and negative zero codes are also numerically equivalent. The audit counts exact packed scales, exact codes, exact decoded products, and equality after FP32 rounding separately.

## Published checkpoint trials

| Reference | Exact numeric weights | Exact FP8 scales | Exact FP8 codes | Held-out logits | Cached continuations |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Qwen3-0.6B-FP8](https://huggingface.co/Qwen/Qwen3-0.6B-FP8/tree/e5be08033360965ceca7b0ffd72d521a51331ce0) | 751,632,384 / 751,632,384 | 26,880 / 26,880 | 440,399,258 / 440,401,920 | 911,616 / 911,616 | 6 / 6 |
| [Tiny random DeepSeek V3 FP8](https://huggingface.co/IlyasMoutawwakil/tiny-random-DeepseekV3-FP8/tree/971e7bbb85e97d45b3dceef18c5e55a08807fb87) | 15,987,200 / 15,987,200 | 915 / 915 | 14,786,508 / 14,786,560 | 6,144 / 6,144 | 6 / 6 |

Every changed FP8 code was a zero sign bit. Recovery took about 41 seconds for Qwen and 1.35 seconds for the DeepSeek fixture using CPU tensor probes. A second DeepSeek capture using FP32 responses produced the same file hash. [Checkpoint hashes and measurements](../examples/results/fp8-recovery.json)

The Qwen evaluation independently loaded both packed checkpoints through the native Transformers FP8 CUDA runtime on an RTX 5080. The DeepSeek evaluation used a separate PyTorch BF16 dequantization and compared the export with the fixture's published BF16 reference. It used fixed token IDs and adapted the YaRN configuration field names for Transformers 4.57.6. The packed audit covered the fixture's MTP tensors as well. The causal-LM inference path excluded those tensors.

```sh
python examples/fp8_inference_experiment.py \
  --reference /path/to/reference --candidate /path/to/recovered --output /path/to/behavior.json
```

For the DeepSeek fixture, use `--mode dense --fixed-ids --published-dense-reference /path/to/reference/reference`. Native CUDA evaluation uses Accelerate and the Transformers FP8 kernel dependencies.
