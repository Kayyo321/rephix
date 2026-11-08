# Recover native Encodec checkpoints

`recover-layers` supports the native 24 kHz mono and 48 kHz stereo Encodec models. It recovers convolution and transposed-convolution kernels, weight normalization, GroupNorm, both LSTM stacks, and the residual quantizer codebooks. The exported checkpoint includes the native audio feature extractor.

```sh
rephix --threads 3 recover-layers --model /path/to/source --output /path/to/recovered \
  --device cpu --dtype float64 --restore-source-dtypes --batch-size 512 \
  --probe-inactive-state --evidence-dir /path/to/evidence
rephix replay-layers --evidence-dir /path/to/evidence --output /path/to/replayed
```

## Required observations

Transposed-convolution probes place an impulse far enough inside the chosen input to expose the full kernel. They support native ConvTranspose1d, ConvTranspose2d, and ConvTranspose3d with groups, dilation, stride, padding, and output padding. Both current and legacy weight-normalization layouts are supported for temporal transposed convolutions.

The LSTM interface observes each packed input, recurrent, and projection affine branch before nonlinear gates and before the two bias vectors are added together. Chosen basis inputs expose the matrices. Zero branch inputs expose each bias independently. A provider exposing only the ordinary fused LSTM output cannot identify those two biases independently.

The active quantizer table is observed through native codebook decoding at chosen IDs. The supported Hugging Face Encodec implementation also stores `inited`, `cluster_size`, and `embed_avg`, although inference never reads them. Their original values require a separate registered-state interface. `--probe-inactive-state` enables selected row lookups for those buffers and records every response in the journal. The report uses `chosen_layer_inputs_and_explicit_registered_state` as its observation mode and lists all inactive fields and element counts. Recovery rejects this path when that access is disabled.

The experiment changed 4,227,104 inactive values in the 24 kHz model and 2,113,552 in the 48 kHz model. Encoded codes and decoded audio remained identical. The explicit state observations supply the evidence needed to recover those stored coordinates.

## Published checkpoint trials

The trials used CPU FP64 observations, then restored original storage precision. Independent FP32 reloads compared codes, normalization scales, padding values, and decoded audio at both the minimum and maximum configured bandwidth. The checkpoints provide native safetensors, and raw audits compare every original stored tensor by name and shape. [Hashes, observations, inactive-state controls, and raw audits](../examples/results/encodec-layer-recovery.json)

| Reference | Original stored values | Exact agreement | Exact code sequences | Exact decoded waveforms | Exact decoded coordinates |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Encodec 24 kHz](https://huggingface.co/facebook/encodec_24khz/tree/c1dbe2ae3f1de713481a3b3e7c47f357092ee040) | 23,273,218 | 100% | 12 / 12 | 12 / 12 | 233,280 / 233,280 |
| [Encodec 48 kHz](https://huggingface.co/facebook/encodec_48khz/tree/c3def8e7185ac8c8efdce6eb8c4a651e487a503e) | 19,066,998 | 100% | 12 / 12 | 12 / 12 | 1,532,160 / 1,532,160 |

The six generated native-rate signals cover silence, an impulse, a tone, a chirp, a tone mixture, and seeded noise, with durations from 0.37 to 1.31 seconds. Both stereo channels are exercised. The 48 kHz processor and model also exercise normalization, multiple chunks, overlap addition, and processor padding. Every processor tensor, audio scale sequence, and padding value matched. All 12,512 mono codes and 37,800 stereo codes matched, with zero maximum decoded-audio error. Decoded coordinate counts include native processor padding.

Both sources were absent during replay. The 24 kHz replay used 384 saved responses and the 48 kHz replay used 290, with zero new calls. Each reproduced its weights, configuration, and processor file byte for byte. The 24 kHz capture also resumed from 157 cached observations after a singleton-view journal serialization failure. The fixed journal preserves the scalar value while normalizing its byte-view stride. Focused tests passed on Transformers 4.46.0 and 4.57.6.

```sh
python examples/encodec_recovery_experiment.py \
  --reference /path/to/source --candidate /path/to/recovered \
  --model-id "$repo" --revision "$revision" --device cpu --threads 3 \
  --output /path/to/results.json
```
