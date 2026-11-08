# Recover native RWKV checkpoints

`recover-layers` supports the native Transformers RWKV-4 causal language model. It recovers the recurrent attention and feed-forward time mixtures, the attention `time_first` and `time_decay` tensors, and the ordinary embedding, normalization, and projection layers. The export includes the native tokenizer and generation configuration.

```sh
rephix --threads 3 recover-layers --model /path/to/source --output /path/to/recovered \
  --device cpu --dtype float64 --restore-source-dtypes --batch-size 512 \
  --evidence-dir /path/to/evidence
rephix replay-layers --evidence-dir /path/to/evidence --output /path/to/replayed
```

A single unit token has a zero shifted predecessor. The chosen time-mixing multiplication therefore exposes each mixture before its learned projection. A chosen zero key exposes `time_first` at its native addition. `time_decay` is observed through a controlled FP64 exponential and reconstructed with its inverse logarithm. Restoring the original stored dtype removes the residual rounding for the tested FP32 archive. Decay values outside the finite positive FP64 exponential range are rejected.

The capture invokes the parameterized modules individually. Native full-model inference applies RWKV's lazy weight rescaling when it first runs. The evaluator independently reloads both source and candidate and exercises their ordinary inference and generation entry points, including that rescaling.

## Published checkpoint trial

The trial used CPU FP64 probes and inference with PyTorch 2.10.0 and Transformers 4.57.6. The raw audit matched every original stored name and shape. The pinned release provides a PyTorch archive. It was loaded with `torch.load(weights_only=True)` and serialized directly to safetensors, with every tensor name, shape, dtype, and value verified unchanged. Both archive hashes are recorded. [Raw audit, prompts, checkpoint hashes, and behavior measurements](../examples/results/rwkv-layer-recovery.json)

| Reference | Original stored values | Exact agreement | Next-token top-1 | Exact greedy continuations | Maximum FP64 logit error |
| --- | ---: | ---: | ---: | ---: | ---: |
| [RWKV-4 169M Pile](https://huggingface.co/RWKV/rwkv-4-169m-pile/tree/46bdc280eb97b6141d5d51a935e0c4870ecaefcc) | 169,342,464 | 100% | 24 / 24 | 6 / 6 | 0 |

The evaluator uses one unpadded prompt at a time and compares full-vocabulary next-token logits on 24 prompts from six held-out families. Six prompts also compare cached greedy continuations of up to eight tokens. The export matched every measured FP64 logit and every continuation.

With the original source path absent, replay consumed all 453 saved observations with zero new calls. The replayed weights, configuration, generation configuration, and tokenizer files matched all six initial export files byte for byte. Synthetic tests also exercise cached recurrence and source-free replay with Transformers 4.46.0. These probes require instrumented access to native internal operators.

```sh
python examples/layer_recovery_experiment.py \
  --reference /path/to/source --candidate /path/to/recovered --device cpu --threads 3 \
  --model-id RWKV/rwkv-4-169m-pile \
  --revision 46bdc280eb97b6141d5d51a935e0c4870ecaefcc \
  --output /path/to/results.json
```
