# Recover Wav2Vec2, HuBERT, and WavLM

`recover-layers` reconstructs native base encoders and CTC recognizers from these three families. The source configuration selects the exact native task. Base encoders export their audio feature extractor. CTC checkpoints also export their tokenizer. The probes cover waveform convolutions, GroupNorm and LayerNorm, projection and attention matrices, positional convolutions, learned mask vectors, and WavLM's relative position gates.

```sh
rephix --threads 3 recover-layers --model /path/to/source --output /path/to/recovered \
  --device cuda --stage-layers --dtype float64 --restore-source-dtypes \
  --batch-size 512 --evidence-dir /path/to/evidence
rephix replay-layers --evidence-dir /path/to/evidence --output /path/to/replayed
```

The positional convolution stores a scale `g` and a direction `v`. Its effective kernel alone leaves the magnitude of `v` ambiguous. Rephix observes the native weight-normalization primitive with chosen inputs. A unit direction reveals the signed scale. A native observation of the direction norm supplies a chosen scale that exposes the direction. This also handles a stored zero scale. FP16 and BF16 observations use a controlled FP32 primitive to avoid rounding the chosen norm. Zero direction norms are rejected because the native normalization would be nonfinite.

The export preserves the original stored names, including legacy `weight_g` and `weight_v` names that current Transformers loaders convert to parametrizations internally. Name aliases are metadata in the evidence bundle, so replay produces the same archive layout.

The published Wav2Vec2 checkpoint omits its configured `masked_spec_embed`. Its ordinary evaluation path leaves this training mask vector unused. Rephix initializes that unstored vector to zero and records it in `source_initialized_audio_masks`. It appears as an extra candidate tensor in the audit. Original stored coordinates and synthesized state have separate report fields.

## Published trials

The trials used FP64 chosen probes on an RTX 5080, followed by `--restore-source-dtypes` to export the original FP32 storage precision. The adapter reads dtype metadata from the source safetensors archive. Both capture and source-free replay apply the saved dtype map after reconstruction. Independent FP32 reloads compared full outputs on five held-out LibriSpeech clips. [Hashes, raw audits, audio traces, precision controls, and replay measurements](../examples/results/audio-encoder-recovery.json)

| Reference | Original stored values | Exact numeric agreement | Within tolerance | Measured output coordinates | Exact output agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Wav2Vec2 Base 960h](https://huggingface.co/facebook/wav2vec2-base-960h/tree/22aad52d435eb6dbaf354bdad9b0da84ce7d6156) | 94,395,552 | 100% | 100% | 76,736 logits | 100% |
| [HuBERT Base LS960](https://huggingface.co/facebook/hubert-base-ls960/tree/dba3bb02fda4248b6e082697eee756de8fe8aa8a) | 94,371,712 | 100% | 100% | 1,841,664 features | 100% |
| [WavLM Base Plus](https://huggingface.co/microsoft/wavlm-base-plus/tree/4c66d4806a428f2e922ccfa1a962776e232d487b) | 94,381,936 | 100% | 100% | 1,841,664 features | 100% |

Wav2Vec2 matched all 2,398 CTC frame decisions and all five decoded transcriptions. HuBERT and WavLM were evaluated as their published base feature encoders. All processors produced identical tensors. Replacing the last waveform with silence changed outputs in each source and export by the same amount. [Pinned audio panel](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/tree/5be91486e11a2d616f4ec5db8d3fd248585ac07a)

The FP64 export controls achieved 100% agreement within tolerance, with small residual differences below the original FP32 precision. Restoring storage dtypes removed those differences. Each model's 4,718,720 stored weight-normalization values then matched exactly.

The pinned HuBERT and WavLM releases provide PyTorch archives. They were loaded with `torch.load(weights_only=True)` and serialized to safetensors. Every original tensor name, shape, dtype, and value was checked against the repack, and both archive hashes are recorded.

The source paths were absent during replay. Wav2Vec2 consumed 345 saved observations, HuBERT 342, and WavLM 379. Each replay reproduced all checkpoint and processor files byte for byte with zero new calls. The probes require access to the native internal operators. Focused tests also passed with Transformers 4.46.0.

```sh
python examples/audio_encoder_recovery_experiment.py \
  --reference /path/to/source --candidate /path/to/recovered --device cuda \
  --model-id "$repo" --revision "$revision" \
  --dataset /path/to/validation-00000-of-00001.parquet \
  --dataset-revision 5be91486e11a2d616f4ec5db8d3fd248585ac07a \
  --output /path/to/results.json
```
