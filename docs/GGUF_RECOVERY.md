# Recover quantized GGUF checkpoints

`recover-gguf` rebuilds a standalone GGUF file from chosen inputs to its tensor operators. It supports unsplit, little-endian GGUF v2 and v3 files with Q4_0, Q5_0, Q8_0, Q4_K, Q6_K, F32, F16, and BF16 tensors. The estimator uses tensor names, dimensions, and formats from the file header. The export preserves the structural prefix byte for byte, including the architecture, tokenizer, and tensor directory.

```sh
pip install -e '.[gguf]'
rephix recover-gguf --model /path/to/source.gguf --output /path/to/recovered.gguf
rephix audit-gguf /path/to/source.gguf /path/to/recovered.gguf --output /path/to/audit.json
```

Use `--device cuda` with PyTorch installed to run the local affine probes on a CUDA device. The adapter dequantizes one source tile at a time, applies the affine operator to unit inputs, and returns float32 responses. The estimator reconstructs each destination tile from those responses and writes it to a new file. Python callers can pass `call_tensor(name, row_start, row_stop, inputs)` to `recover_gguf` to serve the same observations from a separate instrumented runtime.

## Packed values and effective weights

Each Q4_0, Q5_0, or Q8_0 block represents 32 values as a float16 scale times small signed integer codes. The scale is an integer multiple of `2^-24`, and the products fit exactly in float32. Rephix finds encodings on that integer lattice and prefers the reference quantizer's extremal code and scale convention when possible. [Reference quantizers and block decoding](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c)

Some blocks admit multiple encodings. In Q8_0, for example, negating both the scale and the signed codes preserves all represented weights when the negated codes fit. The reference quantizer selects a positive scale. The report counts ambiguities even when the selected convention reproduces the original. `audit-gguf` compares payload bytes, stored scale and code fields, whole blocks, and the independently dequantized weights as separate measurements.

Q4_K uses two shared float16 scales across eight groups of 32 values. The solver fits each group's affine step and offset, then looks for compatible shared scales and group multipliers. It favors the reference convention with nonnegative shared scales and saturated multipliers. Its ambiguity count is a lower bound on the alternative encodings found. Large differences between the shared scales can cause float32 rounding that defeats the exact lattice search. Q6_K uses one float16 scale across sixteen groups of 16 values. The solver enumerates shared and local factors and prefers the reference quantizer's signed `-128` scale anchor. Distinct sign and factor choices can preserve every effective weight. The exporter checks that each packed tile decodes back to its observed values before publishing the file.

This observation contract requires full, coordinate-preserving tensor outputs and the GGUF structure. A text endpoint does not expose these measurements.

## Q4_0 and Q8_0 pretrained trials

The three recovery runs used CUDA float32 affine probes on an RTX 5080 with gguf 0.19.0. Each output matched the original file SHA-256. The independent audit checked the stored payloads and their dequantization. The behavior check loaded each file in isolation with Transformers 4.57.6 and compared full-vocabulary next-token logits and cached greedy continuations on six fresh prompts. [Measurements](../examples/results/gguf-layer-recovery.json)

| Checkpoint | Exact stored fields | Exact dequantized weights | Exact cached continuations |
|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct Q8_0 | 649,857,920 / 649,857,920 | 630,167,424 / 630,167,424 | 6 / 6 |
| Qwen2.5-0.5B-Instruct Q4_0 | 649,857,920 / 649,857,920 | 630,167,424 / 630,167,424 | 6 / 6 |
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | 1,134,422,016 / 1,134,422,016 | 1,100,048,384 / 1,100,048,384 | 6 / 6 |

The Qwen Q4_0 file also contains one Q8_0 tensor. Across the three files, every Q8_0 block admitted multiple packed representations, and 1,053 of the 15,436,288 Q4_0 blocks were ambiguous. The reference convention selected the original encoding for every block in these trials. The tests also exercise legal noncanonical blocks whose exact dequantized weights can be recovered while their packed fields differ. [Pinned Qwen files](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/9217f5db79a29953eb74d5343926648285ec7e67) · [Pinned TinyLlama file](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tree/52e7645ba7c309695bec7ac98f4f005b139cf465)

The standalone inference check uses the Transformers GGUF loader, which dequantizes the model for float32 inference. It requires `accelerate`. Save the recovery command's JSON to pass as `--recovery-report`.

```sh
python examples/gguf_recovery_experiment.py \
  --reference /path/to/source.gguf --candidate /path/to/recovered.gguf \
  --recovery-report /path/to/recovery.json --output /path/to/results.json \
  --device cuda --model-id Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  --revision 9217f5db79a29953eb74d5343926648285ec7e67
```

## K-quant pretrained trials

These archives mix encodings. The Qwen Q4_K_M archive includes Q5_0 and Q8_0 tensors, and the Qwen Q6_K archive includes Q8_0 tensors. TinyLlama and Lite-Mistral use Q4_K and Q6_K for their quantized tensors. CUDA float32 probes recovered every dequantized value in all four archives. Independent standalone reloads produced identical held-out logits and cached continuations. [K-quant measurements](../examples/results/gguf-k-recovery.json)

| Original archive | Exact stored fields | Exact dequantized weights | Exact cached continuations |
|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct Q6_K | 99.7899% | 630,167,424 / 630,167,424 | 6 / 6 |
| Qwen2.5-0.5B-Instruct Q4_K_M | 99.8955% | 630,167,424 / 630,167,424 | 6 / 6 |
| TinyLlama-1.1B-Chat-v1.0 Q4_K_M | 99.8242% | 1,100,048,384 / 1,100,048,384 | 6 / 6 |
| Lite-Mistral-150M-v2-Instruct Q4_K_M | 99.7246% | 156,519,168 / 156,519,168 | 6 / 6 |

The original and recovered files have different hashes. Q6_K had the largest raw differences, with alternative packed scales or codes representing the same weights. Lite-Mistral matched all 450,816 Q4_K blocks byte for byte. TinyLlama matched 3,569,653 of 3,569,664 Q4_K blocks. The Lite-Mistral GGUF declares the `llama` architecture and was evaluated through the corresponding Transformers loader. [Pinned Lite-Mistral file](https://huggingface.co/OuteAI/Lite-Mistral-150M-v2-Instruct-GGUF/tree/b76ce5212026af9de0f919463d04da12a93cd9ea)
