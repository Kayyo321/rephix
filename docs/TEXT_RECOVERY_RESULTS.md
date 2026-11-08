# Text-only recovery: pre-registered results

Method, contract and theorem: [TEXT_RECOVERY.md](TEXT_RECOVERY.md).

## Protocol and provenance

* **Protocol:** [version 2](../examples/text_recovery_protocol_v2.json), SHA-256
  `b430dd0c…a01a9b2`. It was committed alone as `c7815cf`, before any v2 final run,
  together with its pass criteria S1–S11, budgets and held-out seeds 201–205.
* **Run:** 2026-11-08 18:22–19:23 UTC, from a detached worktree at `c7815cf`, so later
  edits in the development checkout could not reach it. Environment: Python 3.13.5,
  NumPy, Torch 2.10.0+cu128 and Transformers 4.57.6 on CPU (float32 features).
  Prompts and corpora were read from `docs/` at commit `6aa8d55`.
* **Data:** [all runs](../examples/results/text-recovery-v2-final.json) and the
  [scored criteria](../examples/results/text-recovery-v2-summary.json), produced by
  `examples/text_recovery_analysis.py`. Pretrained backbones:
  `nickypro/tinyllama-15M@a97e01fa` and `HuggingFaceTB/SmolLM2-135M@93efa2f0`, with
  [file hashes](../examples/results/text-recovery-backbone-sha256.txt). Downloads
  were deleted after the run.
* **Isolation in every run:**
  * The private target directory had mode `000` during recovery and reload.
  * Recovery spawned only the string relay.
  * No private file was opened and no chmod was attempted.
  * Target parameters came from a per-run secret, recorded only after recovery.
  * Held-out test prompts were disjoint from every queried prompt.
* **Fixed budgets:** listed per case below. A "sampled output" is one returned
  string.

## Pre-registered criteria

| Criterion | Result | Measured |
| --- | --- | --- |
| S1 every run in its allowed status set | **Failed** | 84 of 85 runs. `ngram-random` seed 204 (in class by construction) was labelled `misspecified_approximation` at p = 0.005 (= α/2). This is the class test's designed false-alarm rate, about 1% per run: a conservative error, not a false recovery |
| S2 main in-class cases recovered in ≥ 4/5 | Passed | ngram-random 4/5; 4× budget, Qwen2, GPT-2, TinyLlama-15M, SmolLM2-135M and blind spot 5/5 |
| S3 canonical error ≤ 0.10 when recovered | Passed | 37 runs; max 0.098 (blind spot, seed 202) |
| S4 error ≤ 2.5 × reported standard error | Passed | max ratio 1.23 |
| S5 pooled 95% interval coverage in [0.90, 0.99] | Passed | 0.948 over 27,692 coordinates |
| S6 held-out TV ≤ 0.02 when recovered | Passed | max 0.0095 |
| S7 MLE beats the plug-in baseline in ≥ 80% | Passed | 100% |
| S8 error ratio 8,000 → 32,000 samples/context in [1.5, 2.7] | Passed | median 2.10 (theory: 2) |
| S9 no out-of-class, greedy, undeclared or underdetermined run recovered | Passed | 0 of 30 |
| S10 isolation and standalone reload | Passed | 0 violations; reload log-probability difference ≤ 1.4e-14 |
| S11 blind spot recovered while wrong on the trigger | Passed | 5/5 recovered; trigger TV 0.53–0.96 |

## Per case (5 held-out seeds each)

Errors are relative L2 errors of the canonical parameters. TV is the mean total
variation distance from the true output law on held-out contexts.

| Case | Target | Allowed | Recovered | Median canonical error | Median held-out TV | Sampled outputs per run |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ngram-random | log-linear bigram LM, 16 tokens | recovered | 4 | 0.044 | 0.0084 | 1,152,000 |
| ngram-random-4x-budget | same class, 4× samples | recovered | 5 | 0.024 | 0.0046 | 4,608,000 |
| ngram-random-low-budget | same class, 1/16 samples | insufficient | 0 | — | 0.035 | 72,000 |
| ngram-moderate | logit scale 1.5 | recovered or insufficient | 3 | 0.028 | 0.0043 | 4,608,000 |
| head-tiny-qwen2 | private 4-label head, random Qwen2 (RMSNorm) | recovered | 5 | 0.025 | 0.0081 | 420,000 |
| head-tiny-gpt2 | private head, random GPT-2 (LayerNorm; rank 48 of 49) | recovered | 5 | 0.031 | 0.0070 | 420,000 |
| head-tinyllama-15m | private head on pretrained TinyLlama-15M (d = 288, condition 1.2e3) | recovered | 5 | 0.017 | 0.0063 | 4,880,000 |
| head-smollm2-135m | private head on pretrained SmolLM2-135M (d = 576, condition 3.0e3) | recovered | 5 | 0.047 | 0.0055 | 12,100,000 |
| ngram-peaked | logit scale 2.5 (improbable outputs) | insufficient or recovered | 0 | — | 0.013 | 1,152,000 |
| ngram-corpus | English character model fitted to docs | insufficient or recovered | 0 | — | 0.012 | 1,856,000 |
| ngram-order3-target | depends on a third previous token | misspecified | 0 (5 misspecified) | — | 0.36 | 1,152,000 |
| ngram-mlp-target | log-linear plus a tanh layer | misspecified | 0 (5 misspecified) | — | 0.055 | 1,152,000 |
| head-perturbed-backbone | private backbone ≠ public prior | misspecified | 0 (5 misspecified) | — | 0.073 | 4,880,000 |
| ngram-greedy-target | greedy decoding | deterministic | 0 (5 deterministic) | — | 0.63 | 7,200 |
| ngram-undeclared-output | emits a string outside the classes | contract error | 0 (5 errors) | — | — | — |
| ngram-underdetermined | 12 contexts for a 31-dimensional span | insufficient | 0 | — | 0.32 | 256,000 |
| ngram-blind-spot | departs only on an unqueried 4-token prompt | recovered (limitation) | 5 | 0.050 | 0.0089 | 1,152,000 |

Median recovery time ranged from 3 s (n-gram) to 456 s (SmolLM2) on one CPU host,
including all oracle queries and 199 bootstrap refits.

## What these results establish, and what they do not

* **The four levels are measured separately.** For in-class targets:
  * behavior on held-out inputs: TV ≤ 0.0095;
  * class identification: the class test accepts in-class targets and rejects all
    15 out-of-class runs;
  * canonical parameter recovery, up to the equivalence the theorem identifies:
    errors 0.017–0.050 at the stated budgets;
  * an independently runnable export: it reloads without the target to 1.4e-14.

  None of these is presented as recovery of *original* coordinates. The theorem
  shows those are not identifiable.
* **Uncertainty is calibrated.** Reported standard errors track actual errors (ratio
  ≤ 1.23), coverage is near nominal, and error halves when the budget quadruples.
  Statements about one run's accuracy can rely on its reported SE.
* **The budget grows with the improbability of outputs.** Peaked and natural-language
  targets are refused as `insufficient_evidence`, while their *behavior* is
  approximated well (TV ≈ 0.013). The method reports this distinction instead of
  overstating parameter recovery.
* **The class test is necessary but cannot prove membership.** It caught every
  out-of-class target that departs on queried or validation inputs. It cannot catch
  departures elsewhere, and the blind-spot case shows exactly that.
* **Single false alarm.** About 1% of in-class runs are rejected by design (α = 0.01,
  two-sided). One of 50 eligible runs was. The registered S1 was stricter than the
  test's own error rate permits; it is reported as failed rather than redefined.

## Pilot (protocol version 1)

Version 1 ran the same kinds of cases on seeds 101–105. It imported the package from
the live development checkout, and source edits made in response to the independent
review changed it mid-run. Every run is labelled in the
[pilot record](../examples/results/text-recovery-v1-pilot.json).

* **Valid runs** finished before any edit. Recovered counts: ngram-random 5/5, 4×
  budget 5/5, Qwen2 head 5/5, TinyLlama-15M head 5/5, SmolLM2-135M head 1/1;
  ngram-moderate 3/5. Median errors were 0.050, 0.026, 0.024, 0.022 and 0.042.
* **Invalid runs** are the remaining boundary cases and SmolLM2 seed 105. They failed
  on a new argument check introduced by those edits.
* **Uncertain runs** are SmolLM2 seeds 102–104.

Version 1's decision rule also had a defect the review found: its class test ignored
held-out validation contexts. Version 2 was registered to fix the defects and re-run
on new seeds.
