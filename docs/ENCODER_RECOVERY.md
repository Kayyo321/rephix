# Recover encoder-only language models and task heads

`recover-layers` supports native Transformers BERT, RoBERTa, DistilBERT, ALBERT, and DeBERTa v2 encoders. It selects a built-in wrapper from the checkpoint's `config.architectures`. The supported families cover base encoders, masked language modeling, classification, span question answering, multiple choice, and pretraining heads when a family provides that wrapper.

```sh
rephix recover-layers --model /path/to/local-encoder --output /path/to/recovered \
  --device cuda --dtype float64 --batch-size 512
rephix audit-weights /path/to/local-encoder /path/to/recovered
```

Use `--architecture` with the exact native class name when the source declaration is missing or describes a different head than the one you want to recover. This override still requires the source archive to load the selected wrapper's parameters. For example, the upstream BERT-base-uncased config declares `BertForMaskedLM`, but its archive also contains the pooler and next-sentence prediction head. The following command recovers both pretraining heads from that archive.

```sh
rephix recover-layers --model /path/to/bert-base-uncased --output /path/to/bert-pretraining \
  --architecture BertForPreTraining --device cuda --dtype float64 --batch-size 512
```

The encoder exports retain their selected class, tokenizer, and task configuration, including classification label maps. A masked-LM bias that shares a parameter with its decoder is recovered through the decoder's affine response. The exporter preserves the alias and any tied word embedding table. It also supports the newer DeBERTa v2 masked-LM head, where a controlled zero input to the tied vocabulary projection exposes a separately stored bias. ALBERT's repeated layer groups are recovered once per shared operator.

The report lists source keys ignored by the selected native loader. The experiment also audits original names after resolving known legacy LayerNorm names and verified ties. It reports unused learned source tensors separately from deterministic buffers that the native model rebuilds. The standard `audit-weights` command compares literal stored names and shapes.

Recovery uses chosen inputs to individual layers. An encoder API that returns only logits, class labels, or vectors does not expose those measurements.

## Pretrained checkpoints

The trials used float64 probes and inference on an RTX 5080 with Transformers 4.57.6. The table compares the original archive values to the saved candidate after resolving known names and aliases. The tolerance is `abs(candidate - reference) <= 1e-8 + 1e-5 * abs(reference)`. Every parameter in the selected native runtime matched within tolerance on all six runs, and the largest held-out logit error was `1.62e-13`.

| Checkpoint and selected task | Original archive values matched in the export | Held-out decisions matched |
|---|---:|---:|
| BERT-base uncased, declared masked LM | 109,514,298 / 110,106,428 (99.4622%) | 6 / 6 |
| BERT-base uncased, full pretraining override | 110,106,428 / 110,106,428 (100%) | 6 / 6 |
| ALBERT-base-v2, masked LM | 11,221,680 / 11,842,272 (94.7595%) | 6 / 6 |
| ALBERT-base-v2, fine-tuned extractive QA | 11,094,530 / 11,094,530 (100%) | 6 / 6 |
| DistilBERT-base uncased, SST-2 sentiment | 66,955,010 / 66,955,010 (100%) | 6 / 6 |
| RoBERTa-base, SQuAD 2.0 extractive QA | 124,056,578 / 124,057,092 (99.9996%) | 6 / 6 |

- The native masked-LM wrapper skips BERT's pooler and next-sentence head. The pretraining wrapper recovers both, including 6/6 next-sentence decisions. The archive uses legacy LayerNorm `gamma` and `beta` names, which the direct audit maps to `weight` and `bias`. [BERT checkpoint](https://huggingface.co/google-bert/bert-base-uncased/tree/86b5e0934494bd15c9632b12f734a8a67f723594)
- The original ALBERT masked-LM archive contains 590,592 unused pooler values and conflicting 30,000-element bias copies. The native loader ties the two biases and exposes the all-zero decoder copy. The recovered model matches that runtime. The fine-tuned QA archive has a single active value for each learned coordinate. [ALBERT masked-LM checkpoint](https://huggingface.co/albert/albert-base-v2/tree/8e2f239c5f8a2c0f253781ca60135db913e5c80c) · [ALBERT QA checkpoint](https://huggingface.co/Yannis98/squad_albert_finetuned2/tree/1cef3aa503eb6c44aa92e45e9fdd556491163375)
- RoBERTa's original archive stores 514 deterministic position IDs. The native model regenerates them at load time, and all 514 matched exactly. The DistilBERT export preserved the learned sentiment head and the label map. [RoBERTa QA checkpoint](https://huggingface.co/deepset/roberta-base-squad2/tree/adc3b06f79f797d1c575d5479d6f5efe54a9e3b4) · [DistilBERT SST-2 checkpoint](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/tree/714eb0fa89d2f80546fda750413ed43d93601a13)

The offline tests also cover relative-position BERT, DeBERTa v2 token classification with grouped convolution, and a nonzero standalone DeBERTa v2 MLM bias. They reload the exports after moving the source directories out of reach.

```sh
python examples/encoder_recovery_experiment.py \
  --reference /path/to/bert-base-uncased --candidate /path/to/bert-pretraining \
  --device cuda --model-id google-bert/bert-base-uncased \
  --revision 86b5e0934494bd15c9632b12f734a8a67f723594 \
  --output /path/to/results.json
```

[Full measurements](../examples/results/encoder-layer-recovery.json)
