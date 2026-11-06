# Persistent reconstruction sessions

Use a session to connect to an unfamiliar target, accumulate evidence across process restarts, and export the best candidate selected on validation. The candidate learns from exposed text in its own UTF-8 byte space. A target family, tokenizer, or checkpoint specification is optional for this workflow. The earlier aligned-token and white-box studies use separate commands and evidence contracts. [Local Qwen baseline](LOCAL_RESULTS.md)

## Start and resume

Use Python 3.10 or later and install `.[hf]` in an isolated environment. Keep the session directory, target receipts, target weights, and candidate export outside the checkout.

```bash
RUN="$HOME/.local/share/rephix-example"
mkdir -p "$RUN"
python examples/session_prompt_plan.py --mode text --output "$RUN/spec.json"
rephix session create "$RUN/session" --spec "$RUN/spec.json"
rephix --threads 2 session resume "$RUN/session" --target "$RUN/target.json" \
  --max-rounds 1 --max-queries 27 --checkpoint-updates 24
rephix session inspect "$RUN/session" --verify --history
rephix session pause "$RUN/session"
rephix --threads 2 session resume "$RUN/session" --target "$RUN/target.json" \
  --max-rounds 3 --max-queries 51
rephix --threads 2 session evaluate "$RUN/session" --target "$RUN/target.json" --max-queries 66
rephix session export "$RUN/session" --cohort 1 --output "$RUN/standalone"
rephix --threads 2 session generate "$RUN/standalone" --prompt "Describe a solar eclipse."
```

For an open reference on the same machine, `target.json` can contain the following fields. The snapshot must already be on disk. `prompt_mode` defaults to `auto`, which uses the supplied model's chat template when available.

```json
{
  "kind": "hf",
  "model": "/absolute/path/to/model-snapshot",
  "receipts": "/absolute/path/to/target-receipts.sqlite",
  "device": "cpu",
  "dtype": "float32",
  "max_new_tokens": 24,
  "epoch": "reference-1"
}
```

The spec reserves separate groups for acquisition, validation, and final testing. Related prompt variants belong in the same group. Rephix rejects normalized duplicates and groups spanning partitions. The run freezes validation evidence before any optimization and locks a candidate before the first final request. A subsequent final test requires a fresh cohort with unused groups through `session add-prompts --fresh-test`. The original cohort and its lock remain in the session.

`--max-rounds` and `--max-queries` are cumulative ceilings. A query is charged when its ID is reserved, including a request whose execution is uncertain. `--max-seconds` is per invocation. `--validation-ce` and `--plateau` stop between rounds. `--max-storage-mib` checks the local journal and write-ahead log before each transition. Time, storage, and pause checks occur at durable boundaries, so an in-progress request or checkpoint can cross a limit. Set `--checkpoint-updates` to bound the training work between checkpoints. `--stop-after-checkpoint` provides a deliberate stop during training. Saved ceilings carry over on restart. Inspect the remaining active-query budget, supply a higher ceiling, or explicitly clear saved limits with `--reset-limits`.

Use `session add-prompts` to extend the acquisition pool and `session add-candidate` to try a new configuration in the next round. An explicit `session resume --refit` starts one cumulative replay round without a target connection, useful after exhausting a pool. A candidate JSON can choose `architecture` (`gru` or `conv`), `objective` (`byte_ce` or `segment_balanced_ce`), width, context, batch size, learning rate, seed, and device. The default samples supervised byte positions uniformly. The balanced objective first samples model-output segments uniformly, then samples a supervised position within each selected segment. Both count one supervised position per sample and use the same held-out byte cross-entropy for checkpoint selection.

## Connect a controlled agent

The HTTP target accepts this client configuration. The caller supplies credentials through an environment variable if the service requires them.

```json
{
  "kind": "http",
  "url": "http://127.0.0.1:8765",
  "timeout": 60
}
```

For the included loopback server, use the local model configuration above with `"kind": "sandbox"`. A tool interaction adds `"task": {"a": 7, "b": 3}` to the request. The optional `expect` field belongs only in the prompt spec and stays with the evaluator. Run the following commands from separate processes as appropriate.

```bash
rephix --threads 2 session-target-serve --config "$RUN/server.json" --port 8765
rephix session-target-interact --config "$RUN/target.json" --request "$RUN/interaction.json"
rephix session observe "$RUN/session" --target "$RUN/target.json" --feed agent
rephix session evidence "$RUN/session" --partition pool
rephix --threads 2 session generate "$RUN/standalone" --request "$RUN/interaction.json"
```

An interaction has `request_id`, `session_id`, a list of `messages` with roles and content, and an optional sandbox task. The standalone tool path needs the same request form and executes only the included integer sandbox. The live adapter uses `POST /v1/requests` for active submissions, `POST /v1/interactions` for the controlled external interaction ingress, `POST /v1/reconcile` for receipts, and `GET /v1/passive?after=CURSOR` for passive ingestion. External integrations can implement the Python `Target` protocol or these JSON endpoints. The loopback server is the verified integration. [Wire contract and public constructors](../src/rephix/session_targets.py)

The wire format keeps session and request IDs, ordered messages, output chunks, available log probabilities, exposed reasoning, tool calls, tool results, errors, and opaque target epochs. Model actions supply labels for training. Environment results enter later context. The adapter declares missing signals through capabilities. The bundled sandbox chooses among three preconstructed actions on each turn, with arguments supplied by the wrapper from the task and prior tool result. Closed-loop evaluation exercises the candidate's own choices and resulting tool outcomes within those constraints.

The passive feed stores unfamiliar interactions as unassigned. An exact single user prompt and matching task can enter training when assigned to the predeclared pool, including when `session add-prompts` registers an independent pool group after import. Validation and test interactions never enter this route. Multi-user-turn passive transcripts are retained for inspection and remain unassigned in this release.

A dispatched remote request can finish while Rephix is offline. Resume reconciles the original request ID first. With no confirmed receipt, the HTTP client records `uncertain_request` and keeps the request pending. After inspecting the target's actual execution history, an operator can resume reconciliation or record `session abandon --reason ...`. Local automatic retries are limited to the side-effect-free target implementations whose receipts and execution share a serialized transaction. Record observed epoch changes and consult behavior probes before drawing conclusions about a backend switch.

## Durable state and limits

The SQLite journal uses full synchronous transactions and a checksum chain. An evidence commit includes the observation and its ordered events. Training checkpoints store model, optimizer, scheduler, sampler RNG and update position. The current state keeps immutable training membership, query selection, prior knowledge, epoch and identity history, the next stage, the latest checkpoint for each candidate, and checkpoints locked for export or final testing. Old recovery payloads can be reclaimed while their history hashes remain. Verify the journal with `session inspect --verify`. Stop the writer before copying the whole session directory. Keep recovery databases from trusted local runs. Integrity hashes detect changes and have no signing authority. For portable inference, the export binds the inference manifest and safetensors weights and requires neither the session database nor target access.

| Capability | Status | Verified scope |
| --- | --- | --- |
| Family-independent text acquisition and byte learning | Demonstrated | Qwen2.5, SmolLM2 and GPT-2 through the same session workflow |
| Random GRU and causal convolution candidates | Demonstrated | Native UTF-8 codec and common validation loss, without target tokenizer or pretrained candidate |
| Local and remote-style receipt recovery | Demonstrated | Local CUDA processes and loopback HTTP with forced crashes, graceful pause, and agent restart |
| Behavior-based identification | Demonstrated within calibration scope | Ranked alternatives, abstention, and segmented mixed/routed hypothesis across selected deployments |
| Agent event capture and standalone rollout | Demonstrated with constrained tools | Two-turn integer sandbox, hidden target model, passive and active interaction, backend change |
| Aligned top-k optimization | Implemented separately | Existing HF workflow where tokenizer identity is established |
| External agent with arbitrary tools or orchestration | Extension point | Wire events can be stored; closed-loop execution and side-effect reconciliation require an adapter |
| MoE and state-space families | Untested in sessions | Target protocol and candidate boundary can accommodate new implementations |
| Architecture/version inference and whole-model weight reconstruction | Unproven | Lineage evidence and component-specific identifiability have separate reports |

[Measured sessions](SESSION_RESULTS.md), [live agent experiment](LIVE_SESSION_RESULTS.md), [identification calibration](IDENTIFICATION.md), and [structural recovery](LOGIT_SUBSPACE.md) describe access assumptions and failure cases.
