# Local reconstruction sessions across model families

The measured baseline for this work is commit `711b5ab`. Its aligned-token Qwen and white-box results are unchanged. The experiments here use randomly initialized candidates with their own UTF-8 byte codec and no target weights, tokenizer, or hidden states. The target server alone uses the reference tokenizer and model. Behavioral measures, identity hypotheses, and subspace recovery have separate evidence and reports. [Baseline](LOCAL_RESULTS.md)

## Three-family reconstruction

The same text workflow queried Qwen2.5-0.5B-Instruct, SmolLM2-135M-Instruct and GPT-2. Qwen2 uses grouped-query attention, SmolLM2 uses a Llama-derived architecture and its own tokenizer, and GPT-2 uses learned absolute positions and a different BPE vocabulary. Their pinned snapshots ran locally in FP32 on the RTX 5080. The target adapter used each checkpoint's chat template when present and recorded greedy continuations capped at 24 native tokens. The student was a 58,371-parameter GRU with a 128-byte context. It was randomly initialized and optimized with AdamW on observed output bytes. Distribution metrics are unavailable in the student codec. [Qwen snapshot](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775) [SmolLM2 snapshot](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac) [GPT-2 snapshot](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e)

A frozen synthetic plan had 60 pool prompts, 15 validation prompts and 15 final prompts. Related templates and variants stay in one partition. The acquisition policy selected 12 pool prompts per round. Each fit had 96 updates per round at batch size 64, or 6,144 sampled byte positions per round. Each pool response exposed at least 24 bytes, so both Qwen acquisition policies had exactly 288 new eligible bytes per round and 18,432 sampled supervised positions over three rounds. Validation and test each scored 360 positions. Capped continuations supply no artificial end marker. [Prompt generator](../examples/session_prompt_plan.py) [Full measurements and commands](../examples/results/text-sessions.json)

| Target, policy and seed | Validation CE round 1 / 2 / 3 | Teacher-forced matches round 1 / 2 / 3 | Pool native tokens |
| --- | ---: | ---: | ---: |
| Qwen, active 17 | 3.303 / 3.354 / 3.407 | 72 / 74 / 82 | 816 |
| Qwen, random 17 | 3.303 / 3.240 / 3.387 | 72 / 81 / 84 | 807 |
| Qwen, active 42 | 3.281 / 3.395 / 3.352 | 59 / 71 / 72 | 814 |
| Qwen, random 42 | 3.281 / 3.258 / 3.537 | 59 / 80 / 82 | 806 |
| Qwen, active 17, recent only | 3.303 / 3.904 / 3.398 | 72 / 66 / 88 | 816 |
| SmolLM2, active 17 | 3.379 / 3.307 / 3.484 | 68 / 83 / 77 | 856 |
| GPT-2, active 17 | 1.346 / 0.956 / 0.910 | 232 / 279 / 286 | 864 |

The active heuristic starts from the same seeded random batch as its paired arm. Later it combines first-byte entropy, prompt novelty, and lexical similarity to failed validation tasks. Random reached lower best validation CE in both seeds and had higher round-three byte agreement. The budgets match in queries, capped byte evidence, and optimizer work. Actual native token spend varies because the target can terminate before its identical native cap. Two seeds on one prompt plan support a descriptive comparison only.

The replay comparison used identical acquisitions and training budgets at seed 17. Cumulative replay lowered round-two validation CE by 0.550 nats and raised matches from 66 to 74. At round three recent-only replay produced slightly lower CE and six more matches. It did not consistently avoid regression, and the best checkpoint for both policies remained round one. SmolLM2 and GPT-2 improved over their own round-one validation measurements under the same pipeline.

| Target | Locked round | Final byte CE | Final teacher-forced agreement | Exact at 24 bytes | Matched prefix bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen | 1 | 2.913 | 95 / 360 | 0 / 15 | 8 |
| SmolLM2 | 2 | 3.104 | 100 / 360 | 0 / 15 | 4 |
| GPT-2 | 3 | 2.224 | 205 / 360 | 0 / 15 | 96 |

The final rows come from the sole locked candidate in each designated arm. Even when the candidate predicted frequent bytes under teacher forcing, its continuation often diverged early. GPT-2's greedy completions sometimes echoed the prompt, creating regularities that favor next-byte agreement. None of these measurements gives a weight error or indicates successful reproduction of complete responses.

## Qwen follow-up on validation

The baseline's rising Qwen validation loss motivated a shorter schedule and a lower learning rate. This follow-up reused the original acquisition and validation prompt families and reserved five different final families before tuning. Each arm stopped after round one and restarted to finish rounds two and three. The active arms can acquire different rows after the first round because selection depends on the checkpoint. Their results compare end-to-end schedules. A separate random-acquisition arm compares two candidate architectures on identical evidence and sampled positions. [Follow-up report and lock](../examples/results/text-session-followup.json)

| Candidate, learning rate, updates per round and seed | Validation CE round 1 / 2 / 3 |
| --- | ---: |
| GRU, 0.001, 96, 17 | 3.288 / 3.091 / **2.970** |
| GRU, 0.003, 48, 17 | 3.375 / 3.135 / 3.030 |
| GRU, 0.001, 48, 17 | 3.413 / 3.191 / 3.080 |
| GRU, 0.001, 48, 42 | 3.396 / 3.193 / 3.108 |
| GRU with random acquisition, 0.001, 48, 17 | 3.413 / 3.212 / 3.091 |
| 91,203-parameter causal convolution on the same random acquisition | 3.996 / 4.238 / 4.786 |

The validation-selected GRU used 288 updates and 18,432 sampled positions. The runner locked that candidate before collecting the fresh final responses. It matched **120 of 360** teacher-forced bytes at 2.744 nats and **0 of 15** exact continuations at the common cap. The candidate generated 16 matching prefix bytes total. The fresh groups included entailment, word origins, fictional scenes, coordinate movement, and simple instrument construction. The lower validation loss supports further study of the schedule, with only one seed at the selected learning rate and update count.

A separate replay of the two controlled-agent training snapshots compared uniform byte positions with uniform output segments using the same 2,048 supervised samples, initial model, and frozen Qwen validation. The balanced arm reached 2.380 nats and 2/12 exact tool traces at round two, versus 2.370 and 2/12 for uniform bytes. Both scored 0/12 on complete tasks. This one-seed ablation did not improve the measured behavior and uniform byte sampling remains the default. [Objective ablation](../examples/results/live-objectives.json) [Live task outcomes](LIVE_SESSION_RESULTS.md)

## Restart and resources

The interrupted Qwen arm received SIGKILL after the target committed a receipt, after the first immutable dataset transition, after a committed training checkpoint, and after an optimizer update chunk but before publishing its checkpoint. A fresh process reopened the session each time. It also stopped at a requested checkpoint and resumed after a pause command. The uninterrupted comparator reproduced acquisition order, all three recovery checkpoint hashes, the best checkpoint, and per-round validation exactly. A fresh full rerun reproduced all eight arms' normalized responses and events, checkpoint hashes, validation results, and applicable final metrics. Final test request reservations followed the candidate lock. [Reproduction and interruption checks](../examples/results/session-reproduction.json) The controlled live run separately verifies SIGTERM recovery, HTTP reconciliation and standalone rollout after the target server closed. [Live restart and backend change](LIVE_SESSION_RESULTS.md)

The workstation used Debian 13.7, Ryzen 7 9800X3D, about 30 GiB RAM, an RTX 5080 with 15.55 GiB visible VRAM, Python 3.13.5, Torch 2.10.0+cu128, and Transformers 4.57.6. The pre-existing NVML mismatch did not prevent CUDA execution. Uninterrupted text collection, fitting and validation took 6.4 seconds for GPT-2, 14.5 seconds for Qwen, and 16.2 seconds for SmolLM2. Their measured peak process RSS was approximately 2.30, 3.68, and 2.55 GiB respectively. Training-stage CUDA allocator peaks can include a reference loaded earlier in the same process. The maximum observed for the interrupted Qwen arm was about 2,025 MiB. The JSON reports record individual process timings, allocator peaks, command arguments, seeds, model file hashes, snapshot identities, and measured token usage.

## Reproduce

Set up the pinned local environment from [the CUDA baseline](LOCAL_RESULTS.md) and stage the SmolLM2 and GPT-2 snapshots at the revisions above. The measured crash drivers require a Linux host. Choose fresh directories outside the checkout.

```bash
BASE="$HOME/.local/share/rephix-local-20261106"
RUN="$HOME/.local/share/rephix-sessions"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
"$BASE/venv/bin/python" examples/run_text_sessions.py \
  --output "$RUN/text-reproduction" --qwen "$BASE/qwen" \
  --smol "$RUN/models/smollm2-135m-instruct" --gpt2 "$RUN/models/gpt2"
"$BASE/venv/bin/python" examples/run_session_followup.py \
  --output "$RUN/followup-reproduction" --qwen "$BASE/qwen"
"$BASE/venv/bin/python" examples/ablate_live_objective.py \
  --session "$RUN/live-cpu-reproduction/session" --output "$RUN/objective-reproduction.json"
```

Session IDs and receipt hashes differ between independent runs. Each journal verifies its own evidence bytes and locks. The restart comparison checks semantic acquisition order and checkpoint state under the same local software and hardware.

Next experiments can vary query policy on independent plans, compare intermediate checkpoints before acquisition, and make tool-choice supervision directly comparable with free-text supervision. Keep new final cohorts separate from these measured outcomes. [Session workflow and extension points](SESSIONS.md)
