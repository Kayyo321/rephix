# Recover LiquidAI LFM2 checkpoints

`recover-layers` supports native Transformers LFM2 text models and LFM2-VL with a headless SigLIP2 vision tower. LFM2 alternates attention and gated short convolutions. The convolution kernels use the existing depthwise `Conv1d` probes, and the gates and projections use affine probes. The LFM2 RMS normalization computes the normalized activations in FP32, casts back to the input dtype, and then applies the learned scale. The vision tower and image projector use the existing affine, embedding, and LayerNorm probes.

LFM2-VL exports preserve the source processor, tokenizer, and chat template. Rephix writes the `lfm2_vl` model type in the export and evidence scaffold so the native auto-loader can reload it in Transformers 4.57.6.

```sh
rephix --threads 2 recover-layers --model /path/to/lfm2 --output /path/to/recovered \
  --device cuda --stage-layers --dtype auto --batch-size 512
rephix --threads 2 recover-layers --model /path/to/lfm2-vl --output /path/to/recovered-vl \
  --device cuda --stage-layers --dtype float32 --batch-size 512 --evidence-dir /path/to/evidence
rephix --threads 2 replay-layers --evidence-dir /path/to/evidence --output /path/to/replayed-vl
```

The published trials used PyTorch 2.10.0+cu128, Transformers 4.57.6, and an RTX 5080. The vision-language processor used Torchvision 0.25.0+cu128. The raw audit covered all reference values by name and shape. A value passes tolerance when `abs(candidate - reference) <= 1e-8 + 1e-5 * abs(reference)`.

| Checkpoint | Probe dtype | Exact values | Values within tolerance | Exact greedy continuations |
| --- | --- | ---: | ---: | ---: |
| [LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M/tree/f37d3f5c8c5484bc01dad379a595cf4c68c4e70e) | BF16 | 354,483,968 / 354,483,968 | 354,483,968 / 354,483,968 | 6 / 6 |
| [LFM2-VL-450M](https://huggingface.co/LiquidAI/LFM2-VL-450M/tree/d355d88bf86e5c7054230232120e573ca2238286) | FP32 | 450,805,284 / 450,822,656 | 450,822,655 / 450,822,656 | 6 / 6 |

The text export matched all measured logits across 24 held-out prompts. The vision-language export matched all 393,216 measured logits and all processor outputs on six image-text prompts after a common BF16 reload. Two prompts held the text fixed and changed the image. Their largest absolute logit difference was 10.9375 in both the source and export. The only weight outside tolerance was a vision MLP coordinate with absolute error `1.49e-8`.

After deleting the downloaded LFM2-VL checkpoint, replay used all 812 cached observations and made zero new calls. It reproduced the exported checkpoint and processor files byte for byte. The synthetic tests also exercise biased depthwise kernels, cached generation, and source-free replay for both wrappers.

The measurements use chosen inputs and complete outputs at internal layers. [Experiment data](../examples/results/lfm2-recovery.json) · [Text evaluator](../examples/layer_recovery_experiment.py) · [Image-text evaluator](../examples/vlm_recovery_experiment.py)
