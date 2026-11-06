# Recover Whisper speech recognition models

`recover-layers` can rebuild native Whisper conditional-generation checkpoints. It recovers the encoder's two temporal convolutions and both position tables, then exports a standalone checkpoint with the decoder's token embedding and output head still tied. The export includes the tokenizer, native audio feature extractor configuration, and generation configuration. Load it locally with `AutoModelForSpeechSeq2Seq` and `AutoProcessor`.

```sh
rephix recover-layers --model /path/to/local-whisper --output /path/to/recovered \
  --device cuda --dtype float64 --batch-size 512
rephix audit-weights /path/to/local-whisper /path/to/recovered
```

Use `--without-preprocessor` to export weights and generation settings without tokenizer or feature extractor files. The regular export copies `preprocessor_config.json` and any `processor_config.json` byte-for-byte and saves the native tokenizer alongside them.

The convolution probe first observes the zero-input bias, then moves an impulse across each input channel and kernel tap. It reads one output position whose full receptive field lies inside the input, including for the stride-two convolution. The same probe supports numeric zero padding, grouped channels, and dilated kernels. The frozen encoder position table and learned decoder positions are both recovered from their lookup outputs. The unit test uses a deliberately modified frozen table to check recovery independent of sinusoidal initialization.

The recovery requires chosen inputs to known native layers. Waveforms and transcripts from a speech API expose a different observation interface and do not uniquely determine these stored weights.

## Pretrained checkpoints

The CUDA trials used an RTX 5080 with float64 for recovery and held-out inference. The weight audit compares each original safetensors coordinate by name and shape with tolerance `abs(candidate - reference) <= 1e-8 + 1e-5 * abs(reference)`. The Turbo candidate was sharded during export. The audit read both candidate shards directly. [Whisper Tiny.en checkpoint](https://huggingface.co/openai/whisper-tiny.en/tree/87c7102498dcde7456f24cfd30239ca606ed9063) · [Whisper Large V3 Turbo checkpoint](https://huggingface.co/openai/whisper-large-v3-turbo/tree/41f01f3fe87f28c78e2fbf8b568835947dd65ed9)

The evaluation used five LibriSpeech clips from three chapters by one speaker, ranging from 1.64 to 29.4 seconds. The processor produced identical log-mel features and attention masks for each original and recovered checkpoint. A 36.75-second concatenation of the two longest clips exercised timestamped generation across the 30-second window. Greedy generation used the decoder cache and a 128-token limit. The measurements compare each export with its source model on these clips. [Pinned audio dataset](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/tree/5be91486e11a2d616f4ec5db8d3fd248585ac07a)

| Checkpoint | Mel bins | Stored weights within tolerance | Teacher-forced argmax | Exact short generations | Exact long-form timestamp segments |
|---|---:|---:|---:|---:|---:|
| Whisper Tiny.en | 80 | 37,760,256 / 37,760,256 | 147 / 147 | 5 / 5 | 8 / 8 |
| Whisper Large V3 Turbo | 128 | 808,878,080 / 808,878,080 | 143 / 143 | 5 / 5 | 4 / 4 |

The long-form token sequences and transcripts also matched exactly. The largest teacher-forced logit error across both runs was `1.03e-13`.

The experiment runner uses `pyarrow` and `soundfile` to read the local Parquet fixture. The recorded results include hashes of the model shards, processor files, dataset archive, and individual audio clips. [Full measurements](../examples/results/whisper-layer-recovery.json)

```sh
python examples/whisper_recovery_experiment.py \
  --reference /path/to/local-whisper --candidate /path/to/recovered \
  --device cuda --model-id openai/whisper-tiny.en \
  --revision 87c7102498dcde7456f24cfd30239ca606ed9063 \
  --dataset /path/to/validation-00000-of-00001.parquet \
  --dataset-revision 5be91486e11a2d616f4ec5db8d3fd248585ac07a \
  --longform --output /path/to/results.json
```
