# Choosing new reference queries

`select-queries` ranks unqueried prompts using a candidate surrogate. It does not contact the reference teacher or use a prompt's recorded answer to score that prompt. Query selection is an experimental heuristic; an uncertain candidate may be uncertain for reasons unrelated to recoverable information.

Begin with a random-query baseline. Pass every previously queried dataset and every reserved validation/test dataset with repeatable `--exclude` options:

```bash
rephix select-queries --pool data/pool.jsonl --output data/round-2-prompts.jsonl \
  --budget 20 --strategy random --seed 42 \
  --exclude data/splits/train.jsonl --exclude data/splits/validation.jsonl \
  --exclude data/splits/test.jsonl
```

The command joins pool and excluded entries by observation id, normalized prompt, and `metadata.group_id`, including transitive connections. If any part of a group touches an excluded dataset, the whole connected group is ineligible. At most one representative of each remaining group is selected. The budget is a maximum: fewer prompts are returned when fewer independent groups remain. An entirely excluded pool is an error. Similar paraphrases cannot be detected automatically; assign them a shared group id when preparing the pool.

For uncertainty selection, score the full next-token distribution at the end of the candidate's rendered prompt:

```bash
rephix select-queries --pool data/pool.jsonl --output data/entropy-prompts.jsonl \
  --budget 20 --strategy entropy --model runs/candidate --device cpu \
  --exclude data/splits/train.jsonl --exclude data/splits/validation.jsonl \
  --exclude data/splits/test.jsonl
```

Entropy is measured in nats; higher values rank first. It measures uncertainty for the next token, not uncertainty across a whole response. Prompt content alone is rendered; recorded completions, reasoning annotations, and token-level teacher evidence are ignored. Prompts exceeding `--max-seq-length` or model context capacity produce an error instead of being silently shortened.

With two independently fitted candidate surrogates, `--strategy disagreement --model runs/candidate-a --reference-candidate runs/candidate-b` ranks prompts by Jensen-Shannon divergence between their next-token distributions. The divergence is symmetric, is measured in nats, and lies between zero and `log(2)`. The two candidates must have identical tokenizers, chat templates, and vocabulary sizes. Both candidates consume the same rendered prompt. Neither model is implicitly treated as the ground-truth teacher.

The selected JSONL contains empty outputs and no completion token IDs, reasoning, or teacher probabilities. User context and group metadata are retained; stale collector metadata and `metadata.completion_text` are removed. Each row receives `metadata.query_selection` with its score, strategy, seed, and source pool hash. A companion `OUTPUT.report.json` records exclusions, eligible group counts, selected scores, and all candidate scores when model scoring is used. Existing output/report files are never overwritten.

Run `rephix collect --prompts data/round-2-prompts.jsonl ...` to query the actual reference afterward. Compare query strategies at equal independent-prompt and observed-token budgets, with the same initial candidate and a test set excluded from every selection round. Do not choose a query strategy by repeatedly inspecting the reserved test results.
