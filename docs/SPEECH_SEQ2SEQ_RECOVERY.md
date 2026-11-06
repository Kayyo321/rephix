# Recover Moonshine and Speech2Text checkpoints

`recover-layers` supports native Moonshine and Speech2Text conditional generation checkpoints. Moonshine consumes a waveform and uses affine GroupNorm in its convolutional encoder. A zero input exposes the GroupNorm bias. Paired positive and negative samples have known normalized values, so their output contrast yields the scale. Speech2Text consumes mel filter-bank features and uses convolution, embedding, linear, and LayerNorm probes. Transformers regenerates its sinusoidal positions from the model configuration.

The export preserves the audio feature extractor and tokenizer, including the native SentencePiece model for Speech2Text. Install `sentencepiece` for that tokenizer. Load the recovered checkpoint with `AutoProcessor` and `AutoModelForSpeechSeq2Seq`.

```sh
rephix --threads 4 recover-layers --model /path/to/source --output /path/to/recovered \
  --device cuda --stage-layers --dtype float64 --batch-size 512 \
  --evidence-dir /path/to/evidence
rephix replay-layers --evidence-dir /path/to/evidence --output /path/to/replayed
```

## Published checkpoint trials

The trials used FP64 probes on an RTX 5080 with PyTorch 2.10.0 and Transformers 4.57.6. The audit compares each raw stored value by name and shape using `abs(candidate - reference) <= 1e-8 + 1e-5 * abs(reference)`. Both exports covered every original tensor. [Checkpoint and audio hashes, weight audits, and behavior traces](../examples/results/speech-seq2seq-recovery.json)

| Reference | Stored values | Exact agreement | Within tolerance | Teacher-forced top-1 | Exact greedy transcriptions |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Moonshine Tiny](https://huggingface.co/moonshine-ai/moonshine-tiny/tree/390624ed33d594443aa4aa221f5b9f283b545b5a) | 27,092,736 | 99.9741% | 100% | 285 / 285 | 5 / 5 |
| [Speech2Text Small LibriSpeech](https://huggingface.co/facebook/s2t-small-librispeech-asr/tree/256e91a603c013ae09aa1e4c09cfc3f8acdf8d19) | 29,536,256 | 99.9736% | 100% | 164 / 164 | 5 / 5 |

The evaluator reloaded both checkpoints independently and compared processor outputs, full teacher-forced logits, and cached greedy generation on five held-out LibriSpeech clips. Processor tensors matched for every clip. Moonshine's FP64 reload differed by at most `6.22e-14` on the measured logits. The Speech2Text reference and export used FP32 for inference because the native FP64 decoder mask overflowed. All 1,640,000 measured Speech2Text logits matched exactly. [Pinned LibriSpeech audio panel](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/tree/5be91486e11a2d616f4ec5db8d3fd248585ac07a)

With the source paths absent, replay used 253 saved Moonshine responses and 434 saved Speech2Text responses. It reproduced all seven Moonshine checkpoint files and all eight Speech2Text checkpoint files byte for byte, including the processor assets. Both trials use chosen internal layer inputs and outputs.

```sh
python examples/speech_seq2seq_recovery_experiment.py \
  --reference /path/to/source --candidate /path/to/recovered --device cuda \
  --model-id "$repo" --revision "$revision" --dtype "$inference_dtype" \
  --dataset /path/to/validation-00000-of-00001.parquet \
  --dataset-revision 5be91486e11a2d616f4ec5db8d3fd248585ac07a \
  --output /path/to/results.json
```
