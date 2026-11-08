# Recover ResNet and EfficientNet classifiers

`recover-layers` supports native Transformers ResNet and EfficientNet image classifiers. The probes reconstruct convolution kernels, the classification head, affine BatchNorm values, running statistics, and integer batch counters. The export includes the native image processor configuration.

```sh
rephix --threads 2 recover-layers --model /path/to/source --output /path/to/recovered \
  --device cpu --dtype float64 --restore-source-dtypes --batch-size 512 \
  --evidence-dir /path/to/evidence
rephix replay-layers --evidence-dir /path/to/evidence --output /path/to/replayed
```

BatchNorm evaluation combines its affine values and running statistics, so ordinary evaluation outputs leave those stored coordinates ambiguous. The adapter chooses native BatchNorm primitive arguments to isolate each affine value and running mean. A separate normalized response exposes the running variance at FP64 probe precision. The counter probe intercepts the native increment, substitutes a zero increment, and observes its integer result without changing the source state. Source storage precision is restored after reconstruction for both parameters and floating persistent buffers.

EfficientNet's depthwise convolutions use the same grouped convolution probe. Native `padding="same"` is supported for stride-one Conv1d, Conv2d, and Conv3d, including asymmetric padding from even kernels and dilated taps. The probe chooses an output whose entire receptive field lies inside the chosen input and computes its complete native output shape.

These paths require instrumented access to native internal operators. Synthetic tests alter every BatchNorm running statistic and use batch counters above the exact-integer range of FP64. They verify reconstruction and source-free replay without regenerating stored state from configuration.

## Published checkpoint trials

The trials used CPU FP64 probes with PyTorch 2.10.0 and Transformers 4.57.6. `--restore-source-dtypes` restored every parameter and floating persistent buffer to its original archive precision. Independent audits matched original stored names and shapes, including all running statistics and batch counters. Both exports recovered every original stored value exactly. [Checkpoint hashes, raw audits, and behavior measurements](../examples/results/cnn-layer-recovery.json)

| Reference | Stored values | Persistent buffer values | Exact agreement | Classifier predictions | Maximum logit error |
| --- | ---: | ---: | ---: | ---: | ---: |
| [ResNet18](https://huggingface.co/microsoft/resnet-18/tree/65a5785d9156231087c481e0c7dd33a5ff6f7e3e) | 11,699,132 | 9,620 | 100% | 17 / 17 | 0 |
| [EfficientNet-B0](https://huggingface.co/google/efficientnet-b0/tree/3d8d75d0812b65064f0151b456f869c6344bf20c) | 5,330,613 | 42,065 | 100% | 17 / 17 | 0 |

The evaluator independently loaded each source and export and compared full FP64 logits on eight native-size random images, eight rectangular random images, and one public COCO image. Both checkpoint processors produced identical pixels for the public image. The EfficientNet release provides a PyTorch archive. It was loaded with `torch.load(weights_only=True)` and serialized directly to safetensors, then every tensor name, shape, dtype, and value was verified unchanged. Both original and repacked archive hashes are saved. [Public image source](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png)

With the source paths absent, replay consumed 175 saved ResNet observations and 384 saved EfficientNet observations, with zero new calls. Each replay reproduced all three weight, configuration, and processor files byte for byte. The focused CNN and convolution tests also passed with Transformers 4.46.0.

```sh
python examples/vision_recovery_experiment.py \
  --reference /path/to/source --candidate /path/to/recovered \
  --model-id "$repo" --revision "$revision" --image /path/to/coco_sample.png \
  --device cpu --threads 2 --output /path/to/results.json
```
