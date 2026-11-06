# Recover vision-language checkpoints

`recover-layers` reconstructs supported native Transformers vision-language checkpoints from chosen layer inputs. The probes cover the image encoder, the image-to-text projector or merger, and the language decoder. The output preserves the source processor, tokenizer, and chat-template files.

```sh
rephix --threads 6 recover-layers --model /path/to/source --output /path/to/recovered \
  --device cuda --stage-layers --dtype auto --batch-size 512
python examples/vlm_recovery_experiment.py --reference /path/to/source \
  --candidate /path/to/recovered --model-id "$repo" --revision "$revision" \
  --device cuda --output /path/to/behavior.json
```

The supported image-text wrappers include Idefics3, SmolVLM, Qwen2-VL, Qwen2.5-VL, Qwen3-VL, and LLaVA, plus Gemma 3, PaliGemma, and LLaVA-OneVision with headless SigLIP towers. Qwen vision patch projections use native three-dimensional convolution probes. CLIP class tokens are observed before their positional offset, and Gemma 3 projectors receive chosen features after their input normalization. LLaVA-NeXT, NeXT-Video, and OneVision newline tokens are observed through native image packing. Use the native Transformers processor dependencies for the selected family, including matching PyTorch and Torchvision builds for Qwen. The published trials used Transformers 4.57.6. Synthetic LLaVA checkpoints also exercised the older wrapper layout in Transformers 4.46.0.

When BF16 probes round away part of a biased affine response, `--dtype float32` can improve weight agreement. It also increases the storage and memory needed for the checkpoint. To compare native BF16 inference after FP32 recovery, pass `--dtype bfloat16` to the evaluator so it uses the same inference dtype for both checkpoints.

Add `--evidence-dir /path/to/evidence` during capture to journal the observations. `replay-layers` can rebuild the checkpoint and its processor assets from a sealed bundle after the source has been removed. The FP32 SmolVLM trial reproduced all 12 checkpoint and processor files byte for byte on source-free replay, using 953 cached responses and zero new calls. [Layer evidence format](LAYER_RECOVERY.md#resume-and-replay-layer-probes)

## Published checkpoint trials

The evaluator independently loaded each source and export. It compared processor outputs, full next-token logits, and up to eight greedy continuation tokens on six rendered image-text prompts. Two prompts use identical text with different images. Their logits changed in both the source and export for all models. The raw audit matches every stored reference tensor by name and shape, with absolute tolerance `1e-8` and relative tolerance `1e-5`. [Checkpoint hashes and measurements](../examples/results/vlm-recovery.json)

| Reference | Probe dtype | Stored values | Exact agreement | Within tolerance | Exact continuations |
| --- | --- | ---: | ---: | ---: | ---: |
| [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/tree/7e3e67edbbed1bf9888184d9df282b700a323964) | FP32 | 256,484,928 | 99.9946% | 100% | 6 / 6 |
| [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/tree/895c3a49bc3fa70a340399125c650a463535e71c) | FP32 | 2,208,985,600 | 99.9973% | 99.999995% | 6 / 6 |
| [Gemma 3 4B IT](https://huggingface.co/unsloth/gemma-3-4b-it/tree/bf46152c47f5dd20b896357cb51abc4c03b8ee8c) | BF16 | 4,300,079,472 | 97.1564% | 97.1660% | 6 / 6 |
| [Tiny random Qwen3-VL](https://huggingface.co/tiny-random/qwen3-vl/tree/be61f02fa193813901f6bc60707eb4ee08f27b02) | FP32 | 4,507,336 | 99.9461% | 100% | 6 / 6 |

The BF16 SmolVLM control recovered 91.2101% within tolerance. Its differences were confined to the image encoder. After FP32 recovery and common BF16 reload, only 296 coordinates in the biased vision projections differed, with a maximum weight error of `3.7253e-9`. Maximum held-out logit error was 0.5, and all six continuations matched. A deterministic CUDA comparison of the source with itself was exact.

The BF16 Qwen2-VL control reached 97.2428% within tolerance and matched five continuations. With FP32 probes, 109 stored values missed tolerance. Under a common BF16 inference dtype, the FP32 export matched all 911,616 held-out logits. The FP32 tiny Qwen3-VL export also matched all measured logits. Gemma 3 used BF16 probes, and its maximum held-out logit difference was 2.03125.

These trials use direct access to internal operators and evaluate still images with text.
