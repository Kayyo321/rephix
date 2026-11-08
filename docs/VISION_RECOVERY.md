# Recover image classifiers

`recover-layers` can export Hugging Face ViT, DeiT, and ConvNeXT V1 and V2 classifiers from chosen inputs to their parameterized layers. This includes the two classifier heads in distilled DeiT. It uses the model configuration and layer outputs in their native coordinates. The local adapter loads the source checkpoint to serve the layer calls. An instrumented external runtime can instead supply the callback to `recover_from_layer_calls`.

[Swin and Swin V2 recovery](SWIN_RECOVERY.md) also captures learned relative position state and Swin V1's persistent integer indices.

[ResNet and EfficientNet recovery](CNN_RECOVERY.md) includes their stored BatchNorm statistics and integer counters.

```bash
rephix --threads 4 recover-layers --model models/vit --output runs/vit-recovered \
  --device cuda --dtype float64
rephix audit-weights models/vit runs/vit-recovered
```

The vision export copies `preprocessor_config.json` verbatim, so it can be loaded through `AutoImageProcessor`. Use `--without-preprocessor` for a fixture that consumes tensors directly. The optional image evaluation script needs Pillow. The measured commands used FP64 for the probes, export, and classifier runs.

## Patch kernels and learned positions

A zero input to the patch convolution exposes its bias. A unit impulse inside a complete receptive field exposes one kernel coefficient for every output channel. The convolution probe supports numeric zero padding, stride, dilation, and channel groups. The same relative input coordinate is activated in each group, and each output channel sees its own group's impulse.

The embedding output for an all-zero image gives the patch position rows plus the patch convolution bias. The recovered projection supplies that bias independently. Subtracting it identifies the patch position rows. Mask-token embeddings are outside this classifier implementation.

For the class token, the observable value is `class_token + class_position`. Adding a vector to one and subtracting it from the other preserves the embedding. This also holds with positional interpolation because the special row is carried through separately from the resized patch grid. Distilled DeiT has a second independent sum for the distillation token. Rephix stores each observed sum in its token and sets the corresponding special position row to zero. The export report counts the affected stored coordinates and the independent dimensions of the ambiguity.

## Pretrained checkpoints

| Reference | Stored values | Raw within tolerance | Within tolerance after special-token alignment | Classifier top-1 |
| --- | ---: | ---: | ---: | ---: |
| ViT-Tiny | 5,717,416 | 99.99328% | 100% | 18/18 |
| Distilled DeiT-Tiny | 5,910,800 | 99.98701% | 100% | 18/18 |
| ViT-Base | 86,567,656 | 99.99823% | 100% | 18/18 |

The raw audit used `1e-8 + 1e-5 * abs(reference)`. The separate aligned audit moves only the recovered special-token pairs to the reference gauge and preserves each fitted sum. Reference weights are used only in that evaluation. The official distilled DeiT source file was loaded with the Transformers weights-only option and saved as safetensors before recovery. The report records both hashes. [ViT-Tiny model](https://huggingface.co/WinKawaks/vit-tiny-patch16-224) · [Distilled DeiT-Tiny model](https://huggingface.co/facebook/deit-tiny-distilled-patch16-224) · [ViT-Base model](https://huggingface.co/google/vit-base-patch16-224)

Each independent evaluation used eight native synthetic images, eight rectangular images with positional interpolation, and two real images processed through the saved processor. The two processors produced exactly equal tensors for the real images. Across the three checkpoints, the largest logit difference was `2.14e-14`. Shifting the reference special tokens and position rows in opposite directions preserved all 54 predictions, including the rectangular inputs. These checks measure agreement with the reference classifier. [Pinned revisions, image digests, and audits](../examples/results/vision-layer-recovery.json)

To reproduce an audit, pass the local source and exported checkpoint to `examples/vision_recovery_experiment.py`. Pass one `--image` per real fixture. The runs used Matplotlib's `grace_hopper.jpg` sample and the [Hugging Face parrots image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png).

## ConvNeXT channel scales

ConvNeXT uses the same grouped convolution probes for its depthwise kernels. Its LayerNorm probes respect both native channel orders. In V1, a zero residual and a unit response from the last pointwise projection expose the LayerScale vector through the native block. In V2, zero input to global response normalization exposes its bias. A separate observation captures the weighted response for equal channel magnitudes before the native bias and residual additions. This avoids cancellation when the learned scales are small.

| Reference | Stored values | Exact agreement | Within tolerance | Held-out top-1 |
| --- | ---: | ---: | ---: | ---: |
| [ConvNeXT V1 Tiny](https://huggingface.co/facebook/convnext-tiny-224/tree/6166b7613034066690a621d8bf25ffdf181a34f0) | 28,589,128 | 99.999895% | 100% | 17 / 17 |
| [ConvNeXT V2 Atto](https://huggingface.co/facebook/convnextv2-atto-1k-224/tree/c7eab08296c3bca2631738322afc3d05373acc8f) | 3,708,400 | 99.999380% | 100% | 17 / 17 |

The trials ran on CPU with FP64 probes and independently reloaded the exports. The largest held-out logit error across both runs was `3.56e-15`. The published V1 checkpoint was loaded with `torch.load(weights_only=True)` and converted to safetensors with exact agreement across all stored tensors. The evaluation reused the synthetic image panels and Matplotlib's sample image. Both exported processors produced the same input pixels as their references. [Raw audits, revisions, and hashes](../examples/results/convnext-layer-recovery.json)

For the V2 run, adding `--evidence-dir` captured 151 responses. A source-free `replay-layers` invocation reused all 151, made zero new calls, and reproduced the weights, config, and processor files byte for byte.
