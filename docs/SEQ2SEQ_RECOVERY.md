# Recover T5-family and BART encoder–decoders

`recover-layers` accepts native Transformers T5, mT5, UMT5, and BART conditional-generation models. It builds a CPU skeleton from the source configuration, recovers weights from chosen operator inputs, restores the skeleton's embedding ties, and exports a standalone safetensors checkpoint with the generation configuration. The same oracle contract can be used with an instrumented external runtime that exposes the required operations.

```sh
rephix recover-layers --model /path/to/local-seq2seq --output /path/to/recovered \
  --device cuda --dtype float64 --batch-size 512
rephix audit-weights /path/to/local-seq2seq /path/to/recovered
```

Install `sentencepiece` when exporting a T5-family tokenizer supplied only as a SentencePiece model. The exporter preserves the native tokenizer and its byte fallback for these checkpoints. Use `--without-tokenizer` to export the recovered weights and generation configuration alone. Run `AutoModelForSeq2SeqLM.from_pretrained` on the export to load the model.

## What the probes observe

- **T5, mT5, and UMT5.** A fixed nonzero input to each native RMS normalization layer exposes its learned scale. With default `--relative-bias-mode embedding`, integer row IDs recover each relative attention bias table, including rows the attention operation never uses. Mixed-precision checkpoints retain the loaded operator dtypes, including mT5 feed-forward output projections kept in float32 by Transformers.
- **Attention-only relative bias.** `--relative-bias-mode attention` still requires direct probes for the rest of the model. For relative bias it queries the native `compute_bias` operation with cache positions that cover all reachable buckets. The standard bidirectional T5 encoder has no relative offset that maps to bucket 16. The exporter zeros unreachable rows and records them under `relative_bias_gauges`. It reports how many parameter elements were individually identified. UMT5 has a separate table per self-attention layer.
- **BART.** Scaled word embedding outputs are divided by the known scale. Contiguous positional probes include the two reserved stored rows by passing signed logical positions to the learned lookup. To recover the persistent `final_logits_bias` buffer, the local oracle runs a one-token encoder–decoder forward with zero hidden input to the biasless language-model head. Its returned logits expose the additive buffer. The report counts these buffer elements separately from learned parameters.

These operations require known architecture and instrumented access to native operators. A generation endpoint exposes neither these coordinates nor enough information to uniquely identify the stored values.

## Pretrained checkpoints

Four public checkpoints ran with a float64 oracle on an RTX 5080. The tolerance is `abs(candidate - reference) <= 1e-8 + 1e-5 * abs(reference)`. The comparison used an independent read of each reference archive after recovery. None of the audits supplied target weights to the recovery routine. [Full measurements](../examples/results/seq2seq-layer-recovery.json)

| Checkpoint and probe | Loaded parameters | Original stored values within tolerance | Held-out argmax | Exact cached generations |
|---|---:|---:|---:|---:|
| T5-small, table | 60,506,624 | 60,506,624 / 60,506,880 | 74 / 74 | 5 / 5 |
| mT5-small, table | 300,176,768 | 300,176,768 / 300,176,768 | 33 / 33 | 6 / 6 |
| BART-base | 139,420,416 | 139,420,416 / 139,420,416 | 79 / 79 | 5 / 5 |
| BART-large-CNN | 406,290,432 | 406,290,432 / 406,290,432 | 79 / 79 | 5 / 5 |
| T5-small, attention bias | 60,506,624 | 60,506,616 / 60,506,880 | 74 / 74 | 5 / 5 |

- The T5-small archive includes 256 values from an obsolete cross-attention bias table that the native Transformers loader ignores. The default export recovered all 60,506,624 loaded parameters with exact numeric equality. The attention-bias export differs at the eight values in its unreachable encoder row. Both exports produced identical held-out logits. [T5-small checkpoint](https://huggingface.co/google-t5/t5-small/tree/df1b051c49625cf57a3d0d8d3863ed4d13564fe4)
- The mT5-small source was loaded locally with `weights_only=True` and repacked into safetensors for the trial. The evidence records hashes of the upstream and repacked weights. Every repacked value matched exactly. The exported `spiece.model` was byte-for-byte identical, and the source and export produced the same IDs and masks on all six multilingual examples, including a rare-Unicode fixture. [mT5-small checkpoint](https://huggingface.co/google/mt5-small/tree/73fb5dbe4756edadc8fbe8c769b0a109493acf7a)
- The BART rows use a direct audit of every stored source value after resolving storage names. BART-base's encoder and decoder keys acquire the `model.` prefix under the generation wrapper. BART-large-CNN stores the shared token table as `model.decoder.embed_tokens.weight`, and the export uses `model.shared.weight`. Literal-name audit coverage was 0% and 87.3316% respectively. Both source archives omit `final_logits_bias`. The native loader supplies zeros, which the oracle recovered, and the canonical runtime audits included them. The largest observed logit error was `9.60e-14`. [BART-base checkpoint](https://huggingface.co/facebook/bart-base/tree/aadd2ab0ae0c8268c7c9693540e9904811f36177) · [BART-large-CNN checkpoint](https://huggingface.co/facebook/bart-large-cnn/tree/37f520fa929c961707657b28798b30c003dd100b)

The standalone tests also cover UMT5 with per-layer bias tables, a nonzero BART logit bias and nonzero reserved positional rows, and mixed-precision mT5. They reload the candidates with the original checkpoint directories moved aside.

```sh
python examples/seq2seq_recovery_experiment.py \
  --reference /path/to/local-seq2seq --candidate /path/to/recovered \
  --device cuda --model-id google-t5/t5-small \
  --revision df1b051c49625cf57a3d0d8d3863ed4d13564fe4 --output /path/to/results.json
```
