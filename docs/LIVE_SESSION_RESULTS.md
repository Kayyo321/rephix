# Controlled live agent reconstruction

A loopback HTTP endpoint served a two-tool agent backed by a local language model. An operator sent interactions independently of Rephix, and the session imported them through the endpoint's passive mirror. The session also made its own active requests. The same training process could start with visible text alone, and each sandbox trace added model-selected tool actions and subsequent environmental inputs. [Target integration](../src/rephix/session_targets.py)

## Observation and access

The wrapper gives the model three available action labels per turn. On the first turn the model selects add, multiply, or subtract. On the second it selects double, negate, or format. The wrapper supplies the original operands, passes the executed first result into the second action, and asks the model to produce final answer text. Both tools run with bounded integer arguments and have no external side effects. Tool name accuracy measures the model's choices inside this action space. Argument agreement also depends on whether the prior selected operation produced the same value. The experiment gives no estimate of unconstrained argument generation.

The HTTP receipt includes message roles and sources, output chunks, tool selections and arguments, tool results, event order, finish reason, opaque deployment epochs, and four selected generation scores for each final answer token. Tool outputs enter training only as subsequent context. The random byte-level GRU trains on exposed model decisions and text. Its tokenizer uses UTF-8 bytes, so it neither aligns nor consumes the target token scores. The integration reports reasoning and internals as unavailable. Ground-truth model identities and checkpoint paths stay in the runner's server configuration. Session records see the HTTP address, available events and opaque epoch labels. [Session model and objective](../src/rephix/session_candidate.py)

## Split, interruption and backend change

The plan freezes 60 acquisition tool prompts, 12 validation tool prompts and 12 final tool prompts before connecting. Prompt templates and operand regimes are disjoint across the three partitions. Acquisition uses two single-digit positive operands, validation moves the first operand into the teens, and final testing moves the second into the teens. A separate pool of eight fingerprint prompts serves identity investigation. The wrapper routes them through its text endpoint. [Plan generator](../examples/session_prompt_plan.py)

1. An external operator sends one pool task, one interaction with a validation prompt, and the initial fingerprint panel. The passive feed admits the pool task and diagnostic panel. It retains the validation-prompt interaction as unassigned, outside all fit and evaluation snapshots.
2. An active client receives the first validation receipt and then receives SIGKILL before saving it locally. After process restart the session finds the pending ID and reconciles the already committed receipt under the same one-query ceiling.
3. A fresh client begins collection and fitting. The experiment watches the manifest for a durable training checkpoint, sends SIGTERM, verifies the stopped checkpoint from a new process, and resumes through validation.
4. The target process switches from Qwen2.5-0.5B-Instruct to Pythia-70m behind the same URL and receipt database. The old receipt remains available. The session imports a fresh probe panel under the replacement's opaque epoch and continues a second acquisition round with cumulative replay.
5. Validation compares both rounds on the original backend's frozen evidence. The session locks the minimum validation byte cross-entropy checkpoint before querying the replacement backend with reserved final tool tasks. The final results combine a backend change and held-out task families.
6. The runner exports the locked candidate, stops the server, checks that the listening port is closed and executes the exported candidate against the local pure sandbox. Standalone execution uses no target service, pretrained candidate, retrieval or response cache.

The fingerprint profile was calibrated on three enrolled families under a specified sampling population. The live endpoint uses greedy decoding, so its hypotheses are deployment-shift observations. An opaque epoch transition independently documents the exposed backend boundary. [Identification method and calibration](IDENTIFICATION.md)

## Measured results

The seed-17 run used 48 active and 18 passive interactions. The two backends executed 100 tool decisions and emitted 1,045 final-answer or probe tokens. The endpoint recorded 100 tool results, 638 output chunks and 66 top-4 log-probability events. In two 64-update rounds, the candidate consumed 2,048 sampled supervised byte positions. End-to-end wall time was 139 seconds on two CPU threads, with 3.68 GiB peak child-process RSS. [Machine-readable report](../examples/results/live-session.json)

| Measurement | Round 1 validation | Round 2 validation | Locked final test |
| --- | ---: | ---: | ---: |
| Backend | Qwen | Qwen | Pythia |
| Accumulated training interactions | 21 | 41 | 41 |
| Teacher-forced byte cross-entropy, nats | 2.849 | 2.370 | 2.788 |
| Teacher-forced byte matches | 310 / 1,173 | 584 / 1,173 | 668 / 1,831 |
| Tool name matches | 9 / 24 | 9 / 24 | 9 / 24 |
| Constrained argument matches | 17 / 24 | 17 / 24 | 16 / 24 |
| Full tool trace and outcome matches | 3 / 12 | 2 / 12 | 1 / 12 |
| Candidate exact completion | 0 / 12 | 0 / 12 | 0 / 12 |
| Candidate / target task success | 0 / 12, 7 / 12 | 0 / 12, 7 / 12 | 0 / 12, 0 / 12 |

The frozen Qwen validation loss improved after replay and further training while full tool traces regressed. Pythia sometimes chose the intended tools but echoed the surrounding dialogue in its final message. The exported candidate often repeated a fragment of the tool JSON. For `tool-test-01-01`, the target selected subtract then negate, producing 13 and echoing the intermediate result in its reply. The candidate selected add then double, produced 50, and repeatedly emitted `{"a":`. The independently executed export reproduced that candidate trace with the target listener closed.

The Qwen probe panel supported the Qwen hypothesis. The Pythia panel rejected all three enrolled families and produced `unknown`. The overall hypothesis became `unknown`, with Qwen retained for its own epoch. The exposed epoch history records the backend change. Neither panel establishes architecture or deployed configuration.

The recovered validation observation matched its server receipt exactly, including its checksum. SIGTERM preserved the step-32 checkpoint, and the new client resumed at step 32 to complete round 1 at step 64. The journal locked the step-128 candidate at sequence 122 before the first final reservation at sequence 123. All 12 final observations were active and outside training.

The next controlled trial should separate supervision for tool choice and free-text completion while holding the evidence and optimizer budgets fixed. The current objective spends many byte positions on the serialization of constrained tool calls. It can lower teacher-forced loss without improving action choice or free-running replies.

## Reproduce

The script runs on a POSIX host with SIGTERM and SIGKILL. It uses CPU only. Install Rephix with the optional model dependencies in an isolated environment and stage the two checkpoint snapshots outside the checkout. The reference snapshots and the identity profile are pinned. [Qwen checkpoint](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775) [Pythia checkpoint](https://huggingface.co/EleutherAI/pythia-70m/tree/a39f36b100fe8a5377810d56c3f4789b9c53ac42)

```bash
BASE="$HOME/.local/share/rephix-local-20261106"
RUN="$HOME/.local/share/rephix-sessions"
ulimit -c 0
"$BASE/venv/bin/python" examples/run_live_session.py run \
  --output "$RUN/live-cpu-reproduction" \
  --qwen "$BASE/qwen" --pythia "$RUN/models/pythia-70m" \
  --identity-profile examples/results/identity-profile.json
```

The runner creates a fresh artifact directory with the frozen spec, server and client configurations, two SQLite databases, evidence, recovery history, the locked final report and the standalone export. The generated `report.json` includes the exact child commands and artifact hashes. The local loopback server is a controlled test integration. Deployments with consequential tools need receipt reconciliation backed by their actual execution history, and an unresolved request stays pending until evidence or an explicit decision resolves it.
