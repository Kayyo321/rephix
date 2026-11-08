# Recover segmentation and depth checkpoints

`recover-layers` preserves the declared native task for SegFormer semantic segmentation and DPT depth estimation. The probes cover their image encoders, convolutional or affine decoders, normalization state, and dense prediction heads. The export includes the original image processor configuration. SegFormer classification and DPT semantic segmentation wrappers are also selected from their exact native architecture declarations.

```sh
rephix --threads 2 recover-layers --model /path/to/source --output /path/to/recovered \
  --device cpu --dtype float64 --restore-source-dtypes --batch-size 512 \
  --evidence-dir /path/to/evidence
rephix replay-layers --evidence-dir /path/to/evidence --output /path/to/replayed
```

Native DPT ViT embeddings receive two separate observations. Zero position features expose the class vector before addition. Zero class and patch features expose the position table through the native resize and addition at its original grid. This identifies the two stored values independently. Synthetic tests use a very small class vector with large positions to check that cancellation cannot hide the class vector. DPT decoder upsampling uses native transposed-convolution probes. Supported native backbone configurations use the corresponding existing recovery operators.

## Published checkpoint trials

The trials used CPU FP64 probes and inference with PyTorch 2.10.0 and Transformers 4.57.6. Original parameter and floating-buffer storage precision was restored before export. Every original stored tensor was compared by name and shape, and both exports matched every original value exactly. The two pinned releases provide safetensors directly. [Checkpoint hashes, raw audits, output hashes, and replay measurements](../examples/results/dense-vision-layer-recovery.json)

| Reference | Stored values | Exact agreement | Held-out images | Full output coordinates | Exact output agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| [SegFormer B0 ADE](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/tree/489d5cd81a0b59fab9b7ea758d3548ebe99677da) | 3,753,207 | 100% | 9 | 24,038,400 logits | 100% |
| [DPT Swin V2 Tiny](https://huggingface.co/Intel/dpt-swinv2-tiny-256/tree/f7f350e1a10a5ea58671b68ab5f49dc18ec00483) | 40,930,795 | 100% | 9 | 696,320 depth values | 100% |

The evaluator independently loaded both checkpoints and compared four native-size random images, four rectangular random images, and one public COCO image processed by each model's native image processor. SegFormer's 160,256 pixelwise class decisions all matched. The maximum output difference was zero for both tasks. Both processors produced identical pixels for the public image. [Public image source](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png)

The source paths were absent during replay. SegFormer consumed 226 saved observations and DPT consumed 421, with zero new calls. Each replay reproduced its weights, configuration, and processor file byte for byte. Focused tests passed on Transformers 4.46.0 and 4.57.6, including a native DPT ViT fixture and independent dense output comparisons. These paths require chosen inputs and instrumented access to native internal operators.

```sh
python examples/dense_vision_recovery_experiment.py \
  --reference /path/to/source --candidate /path/to/recovered \
  --model-id "$repo" --revision "$revision" --image /path/to/coco_sample.png \
  --device cpu --dtype float64 --threads 2 --output /path/to/results.json
```
