"""Durable iterative model reconstruction sessions."""

from __future__ import annotations

import math
import platform
import random
import re
import signal
import time
import unicodedata
import uuid
from itertools import islice
from pathlib import Path

from .session_candidate import ByteCandidate, candidate_config
from .session_store import Journal, digest, object_hash

DEFAULT_ROUND = {
    "acquire": 8,
    "updates": 64,
    "byte_cap": 64,
    "seed": 17,
    "policy": "active",
    "replay": "cumulative",
}
SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
PARTITIONS = ("pool", "validation", "test")


def _normalize(prompt):
    return " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())


def _prompts(rows, earlier):
    indexed = dict(earlier)
    groups = {row["group"]: row["partition"] for row in indexed.values()}
    texts = {_normalize(row["prompt"]) for row in indexed.values()}
    for row in rows:
        if not isinstance(row, dict) or set(row) - {
            "id",
            "group",
            "partition",
            "prompt",
            "task",
            "expect",
            "probe_id",
            "cohort",
        }:
            raise ValueError("prompt entries require id, group, partition and prompt")
        key, group, part, prompt = (
            row.get(field) for field in ("id", "group", "partition", "prompt")
        )
        if (
            not isinstance(key, str)
            or not SLUG.fullmatch(key)
            or not isinstance(group, str)
            or not group.strip()
            or part not in PARTITIONS
            or not isinstance(prompt, str)
            or not _normalize(prompt)
        ):
            raise ValueError("invalid prompt ID, group, partition or text")
        normalized = _normalize(prompt)
        if key in indexed or normalized in texts:
            raise ValueError(f"duplicate prompt ID or normalized text: {key}")
        if group in groups and groups[group] != part:
            raise ValueError(f"prompt group crosses reserved partitions: {group}")
        if "probe_id" in row and (part != "pool" or not isinstance(row["probe_id"], str)):
            raise ValueError("identity probe IDs belong only in the acquisition pool")
        indexed[key] = row
        groups[group] = part
        texts.add(normalized)
    return indexed


def _profile(config):
    from .identification import FingerprintProfile

    return FingerprintProfile.from_dict(config) if config is not None else None


def _check_probes(prompts, profile):
    if profile is None:
        return
    ids = set()
    for row in prompts.values():
        if "probe_id" not in row:
            continue
        key = row["probe_id"]
        if key in ids or profile.probes.get(key) != row["prompt"] or "task" in row:
            raise ValueError("identity probes must uniquely match registered IDs and text")
        ids.add(key)


def _config(spec):
    if spec.get("schema_version") != 1 or set(spec) - {
        "schema_version",
        "prompts",
        "candidates",
        "round",
        "identity_profile",
        "priors",
        "scope",
    }:
        raise ValueError("unsupported session specification")
    prompts = _prompts(spec.get("prompts", []), {})
    if {row["partition"] for row in prompts.values()} != set(PARTITIONS):
        raise ValueError(
            "reserve distinct pool, validation and test prompts when creating a session"
        )
    if any("cohort" in row for row in prompts.values()):
        raise ValueError("initial test cohort is assigned by Rephix")
    candidates = spec.get("candidates", {"byte-gru": {}})
    if not candidates or not all(SLUG.fullmatch(name) for name in candidates):
        raise ValueError("at least one valid candidate ID is required")
    options = spec.get("round", {})
    if set(options) - DEFAULT_ROUND.keys():
        raise ValueError("unsupported round option")
    options = DEFAULT_ROUND | options
    for name, maximum in (
        ("acquire", 10000),
        ("updates", 1000000),
        ("byte_cap", 4096),
        ("seed", 2**63 - 1),
    ):
        minimum = int(name != "seed")
        if type(options[name]) is not int or not minimum <= options[name] <= maximum:
            raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    if options["policy"] not in ("random", "active") or options["replay"] not in (
        "cumulative",
        "recent",
    ):
        raise ValueError("unknown acquisition policy or replay mode")
    scope = spec.get("scope", "model_observable")
    if scope not in ("model_observable", "agent_combined"):
        raise ValueError("unknown target scope")
    profile = spec.get("identity_profile")
    _check_probes(prompts, _profile(profile))
    return {
        "prompts": prompts,
        "candidates": {name: candidate_config(config) for name, config in candidates.items()},
        "round_config": options,
        "scope": scope,
        "identity_profile": profile,
        "priors": {
            "supplied": spec.get("priors", {}),
            "candidate_codec": "UTF-8 bytes; target token IDs unused",
            "candidate_initialization": "random",
            "target_internals_used_for_fitting": False,
            "selection_data": "validation only",
        },
    }


def _artifacts(state):
    hashes = {item["sha256"] for item in state["recovery"].values()}
    if state["best"]:
        hashes.add(state["best"]["sha256"])
    hashes.update(item["locked"]["sha256"] for item in state["finals"].values())
    return hashes


def _words(prompt):
    return set(re.findall(r"\w+", _normalize(prompt)))


def _similarity(words, references):
    return max((len(words & item) / max(1, len(words | item)) for item in references), default=0.0)


def _passive_prompt(observation):
    messages = [
        event["content"]
        for event in observation["events"]
        if event["kind"] == "message" and event.get("role") == "user" and event["source"] == "user"
    ]
    return messages[0] if len(messages) == 1 else None


class Session:
    def __init__(self, directory, spec=None):
        if spec is not None:
            config = _config(spec)
        self.journal = Journal(directory, create=spec is not None)
        if spec is not None:
            self.journal.commit(
                {
                    "schema_version": 1,
                    "session_id": uuid.uuid4().hex,
                    **config,
                    "phase": "validation_collection",
                    "stop": None,
                    "round": 0,
                    "current_round": None,
                    "query_count": 0,
                    "pending": None,
                    "abandoned": [],
                    "evidence": {},
                    "observed_epochs": [],
                    "passive_cursors": {},
                    "recovery": {},
                    "history": [],
                    "best": None,
                    "identity": None,
                    "finals": {},
                    "test_cohort": 1,
                },
                "created",
                {"spec_sha256": object_hash(spec), "environment": platform.platform()},
            )
        self.state = self.journal.load()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.journal.close()

    def inspect(self, verify=False, history=False):
        if verify or history:
            self.state, events = self.journal.verify(_artifacts)
        else:
            self.state = self.journal.load()
        summary = {
            key: value
            for key, value in self.state.items()
            if key not in {"prompts", "evidence", "identity_profile"}
        }
        summary["evidence_counts"] = {
            part: sum(ref["partition"] == part for ref in self.state["evidence"].values())
            for part in (*PARTITIONS, "passive_unassigned")
        }
        summary["pause_requested"] = self.journal.pause_requested()
        summary["storage_bytes"] = self.journal.size_bytes()
        limit = self.state.get("limits", {}).get("max_queries")
        summary["remaining_active_queries"] = (
            None if limit is None else max(0, limit - self.state["query_count"])
        )
        if verify:
            summary["integrity_verified"] = True
        if history:
            summary["journal"] = events
        return summary

    def _save(self, action, detail=None, *, evidence=None, artifact=None):
        self.journal.commit(
            self.state,
            action,
            detail,
            evidence=evidence,
            artifact=artifact,
            retain=_artifacts(self.state) if artifact is not None else None,
        )

    def _stop(self, reason):
        self.state["stop"] = reason
        self._save("stopped", {"reason": reason, "phase": self.state["phase"]})
        return self.inspect()

    def _reload(self):
        self.state, _ = self.journal.verify(_artifacts)

    def pause(self):
        self.journal.pause(True)
        return self.inspect()

    def add_prompts(self, rows, *, fresh_test=False):
        with self.journal.runner():
            self._reload()
            if not rows:
                raise ValueError("no new prompts provided")
            if fresh_test:
                previous = self.state["finals"].get(str(self.state["test_cohort"]))
                if not previous or "metrics" not in previous:
                    raise ValueError("finish the current test cohort before reserving another")
                groups = {row["group"] for row in self.state["prompts"].values()}
                if any(
                    row.get("partition") != "test" or row.get("group") in groups for row in rows
                ):
                    raise ValueError("a new final cohort needs previously unused test groups")
                self.state["test_cohort"] += 1
                rows = [row | {"cohort": self.state["test_cohort"]} for row in rows]
            elif any(row.get("partition") != "pool" or "cohort" in row for row in rows):
                raise ValueError("extend the pool, or explicitly reserve a fresh final test")
            prompts = _prompts(rows, self.state["prompts"])
            _check_probes(prompts, _profile(self.state["identity_profile"]))
            self.state["prompts"] = prompts
            admitted = []
            if not fresh_test:
                by_text = {row["prompt"]: row for row in rows}
                for key, reference in self.state["evidence"].items():
                    if reference["partition"] != "passive_unassigned":
                        continue
                    observation = self.journal.evidence(key, reference["sha256"])
                    row = by_text.get(_passive_prompt(observation))
                    if row and row.get("task") == observation.get("task"):
                        reference.update(prompt_id=row["id"], partition="pool")
                        admitted.append(key)
                self._update_identity()
            self._save(
                "prompts_registered",
                {
                    "manifest_sha256": object_hash(rows),
                    "count": len(rows),
                    "admitted_passive_ids": admitted,
                },
            )
            return self.inspect()

    def add_candidate(self, name, options):
        with self.journal.runner():
            self._reload()
            if not SLUG.fullmatch(name) or name in self.state["candidates"]:
                raise ValueError(
                    "candidate ID must be new and use letters, digits, dots, or hyphens"
                )
            self.state["candidates"][name] = candidate_config(options)
            self._save(
                "candidate_registered", {"id": name, "config": self.state["candidates"][name]}
            )
            return self.inspect()

    def _dispatched(self, partition, *, round_number=None, epoch=None):
        records = (*self.state["evidence"].values(), *self.state["abandoned"])
        return {
            item["prompt_id"]
            for item in records
            if item["partition"] == partition
            and (round_number is None or item.get("round") == round_number)
            and (epoch is None or item.get("target_epoch") == epoch)
        }

    def _unseen(self, partition):
        used = self._dispatched(partition)
        epochs = self.state["observed_epochs"]
        refresh = partition == "pool" and self.state["identity_profile"] and len(epochs) > 1
        current_epoch = self._dispatched(partition, epoch=epochs[-1]["epoch"]) if refresh else set()
        return [
            key
            for key, item in sorted(self.state["prompts"].items())
            if item["partition"] == partition
            and (key not in used or (refresh and "probe_id" in item and key not in current_epoch))
        ]

    def _reserve(self, prompt_id):
        prompt = self.state["prompts"][prompt_id]
        number = self.state["query_count"] + 1
        request = {
            "request_id": f"{self.state['session_id']}-{number}",
            "session_id": self.state["session_id"],
            "messages": [{"role": "user", "content": prompt["prompt"]}],
        }
        if "task" in prompt:
            request["task"] = prompt["task"]
        self.state["query_count"] = number
        self.state["pending"] = {
            "request": request,
            "prompt_id": prompt_id,
            "partition": prompt["partition"],
        }
        self._save("request_reserved", self.state["pending"])
        return request

    def _record(
        self, observation, prompt_id, partition, *, request=None, source="active", cursor=None
    ):
        from .session_targets import validate_observation

        validate_observation(observation, request=request)
        if source == "passive" and observation["mode"] != "passive":
            raise ValueError("passive feed returned an active observation")
        evidence_id = (
            observation["request_id"]
            if source == "active"
            else "passive:" + object_hash([observation["session_id"], observation["request_id"]])
        )
        checksum = object_hash(observation)
        existing = self.state["evidence"].get(evidence_id)
        if existing and existing["sha256"] != checksum:
            raise ValueError("request ID was reused for a different observation")
        if cursor is not None:
            self.state["passive_cursors"][cursor[0]] = cursor[1]
        if existing:
            if request is not None:
                raise ValueError("active request ID already committed through another feed")
            self._save("passive_duplicate", {"id": evidence_id, "cursor": cursor})
            return
        epoch = observation.get("target_epoch")
        epochs = self.state["observed_epochs"]
        if epoch is not None and (not epochs or epochs[-1]["epoch"] != epoch):
            epochs.append({"epoch": epoch, "request_id": evidence_id, "source": source})
        reference = {
            "prompt_id": prompt_id,
            "partition": partition,
            "sha256": checksum,
            "source": source,
            "round": self.state["round"],
            "usable": observation["status"] == "ok",
            "position": len(self.state["evidence"]),
            "target_epoch": epoch,
        }
        self.state["evidence"][evidence_id] = reference
        if request is not None:
            self.state["pending"] = None
        self._save(
            "observation_committed",
            {"id": evidence_id, **reference, "cursor": cursor},
            evidence=(evidence_id, observation),
        )

    def _finish_pending(self, target, hook, *, new=False):
        if target is None:
            return False
        pending = self.state["pending"]
        request = pending["request"]
        try:
            if new:
                hook("before_submit")
                result = target.submit(request)
                hook("after_submit")
            else:
                result = target.reconcile(request)
                if result is None and getattr(target, "retry_missing", False):
                    result = target.submit(request)
                    hook("after_submit")
        except OSError as exc:
            self._save(
                "target_transport_error",
                {
                    "request_id": request["request_id"],
                    "operation": "submit" if new else "reconcile",
                    "error_type": type(exc).__name__,
                    "code": getattr(exc, "code", exc.errno),
                },
            )
            return False
        if result is None:
            return False
        self._record(result, pending["prompt_id"], pending["partition"], request=request)
        hook("after_observation")
        return True

    def abandon(self, reason):
        if not reason.strip():
            raise ValueError("an abandonment reason is required")
        with self.journal.runner():
            self._reload()
            if not self.state["pending"]:
                raise ValueError("no unresolved request")
            epochs = self.state["observed_epochs"]
            record = self.state["pending"] | {
                "reason": reason,
                "round": self.state["round"],
                "target_epoch": epochs[-1]["epoch"] if epochs else None,
            }
            self.state["abandoned"].append(record)
            self.state["pending"] = None
            self._save("request_abandoned", record)
            return self.inspect()

    def observe(self, target, feed, limit):
        from .session_targets import validate_observation

        if not SLUG.fullmatch(feed) or limit < 1:
            raise ValueError("passive feed needs an ID and positive observation limit")
        with self.journal.runner():
            self._reload()
            matches = {
                row["prompt"]: prompt_id
                for prompt_id, row in self.state["prompts"].items()
                if row["partition"] == "pool"
            }
            cursor = self.state["passive_cursors"].get(feed, 0)
            for position, result in islice(target.iter_passive(cursor), limit):
                if type(position) is not int or position <= cursor:
                    raise ValueError("passive feed cursor failed to advance")
                cursor = position
                validate_observation(result)
                key = matches.get(_passive_prompt(result))
                if key and self.state["prompts"][key].get("task") != result.get("task"):
                    key = None
                self._record(
                    result,
                    key,
                    "pool" if key else "passive_unassigned",
                    source="passive",
                    cursor=(feed, position),
                )
            if self._update_identity():
                self._save("identity_updated", {"identity": self.state["identity"]})
            return self.inspect()

    def _rows(self, partition, *, cohort=None, include_errors=False):
        refs = []
        for evidence_id, item in sorted(
            self.state["evidence"].items(), key=lambda pair: pair[1]["position"]
        ):
            if item["partition"] != partition or (not include_errors and not item["usable"]):
                continue
            row = self.state["prompts"][item["prompt_id"]]
            if cohort is not None and row.get("cohort", 1) != cohort:
                continue
            refs.append(
                {"id": evidence_id, "sha256": item["sha256"], "prompt_id": item["prompt_id"]}
            )
        return refs

    def _examples(self, references):
        from .session_evaluation import examples_for_observation

        return [
            example
            for ref in references
            for example in examples_for_observation(self.journal.evidence(ref["id"], ref["sha256"]))
        ]

    def _has_supervision(self, references):
        return any(row["output"] or not row["truncated"] for row in self._examples(references))

    def _update_identity(self):
        from .identification import BehaviorObservation, identify

        profile = _profile(self.state["identity_profile"])
        observations = []
        for ref in self._rows("pool"):
            row = self.state["prompts"][ref["prompt_id"]]
            if profile is None or "probe_id" not in row:
                continue
            value = self.journal.evidence(ref["id"], ref["sha256"])
            # Opaque epoch boundaries group responses. They provide no lineage label.
            observations.append(
                BehaviorObservation(row["probe_id"], value["completion"], value["target_epoch"])
            )
        updated = identify(observations, profile)
        changed = updated != self.state["identity"]
        self.state["identity"] = updated
        return changed

    def _load_candidate(self, name, reference=None):
        artifact = self.journal.artifact(reference["sha256"]) if reference else None
        return ByteCandidate(self.state["candidates"][name], artifact)

    def _select(self, available):
        from .identification import select_identity_probes

        settings = self.state["round_config"]
        number = self.state["round"] + 1
        budget = min(settings["acquire"], len(available))
        if budget == 0:
            return []
        rng = random.Random(settings["seed"] + number)
        prompts = self.state["prompts"]
        picked = []
        profile = _profile(self.state["identity_profile"])
        if settings["policy"] == "active" and profile:
            by_probe = {prompts[key].get("probe_id"): key for key in available}
            missing = set(profile.probes) - by_probe.keys()
            report = self.state["identity"]
            segments = report["lineage"]["segments"] if report else []
            rivals = (
                [entry["family"] for entry in segments[-1]["hypotheses"][:2]] if segments else []
            )
            useful = select_identity_probes(profile, missing, budget, rivals)
            picked.extend(
                {"id": by_probe[key], "reason": "identity_disagreement"} for key in useful
            )
        chosen_ids = {item["id"] for item in picked}
        remaining = [key for key in available if key not in chosen_ids]
        count = budget - len(picked)
        best = self.state["best"]
        if count == 0:
            return picked
        if settings["policy"] == "random" or best is None:
            return picked + [
                {"id": key, "reason": "seeded_random"} for key in rng.sample(remaining, count)
            ]
        candidate = self._load_candidate(best["candidate"], best)
        texts = [prompts[key]["prompt"] for key in remaining]
        uncertainties = [
            entropy
            for start in range(0, len(texts), 256)
            for entropy in candidate.uncertainty(texts[start : start + 256])
        ]
        previous = [_words(prompts[ref["prompt_id"]]["prompt"]) for ref in self._rows("pool")]
        weak = [_words(prompts[key]["prompt"]) for key in best["metrics"]["failed_prompt_ids"]]
        scored = []
        for key, text, entropy in zip(remaining, texts, uncertainties):
            words = _words(text)
            novelty = 1 - _similarity(words, previous)
            related_failure = _similarity(words, weak)
            score = entropy / math.log(259) + novelty + related_failure
            scored.append(
                {
                    "id": key,
                    "reason": "entropy_novelty_and_validation_failure",
                    "score": score,
                    "entropy": entropy,
                    "novelty": novelty,
                    "failure_similarity": related_failure,
                }
            )
        scored.sort(key=lambda item: (-item["score"], item["id"]))
        return picked + scored[:count]

    def _previous_members(self):
        return {ref["id"] for item in self.state["history"] for ref in item["dataset"]["members"]}

    def _new_round(self, refit=False):
        available = [] if refit else self._unseen("pool")
        previous = self._previous_members()
        pool = self._rows("pool")
        if not available and (
            not pool or (not refit and all(ref["id"] in previous for ref in pool))
        ):
            return False
        selected = self._select(available)
        self.state["round"] += 1
        self.state["current_round"] = {
            "number": self.state["round"],
            "selected": selected,
            "candidates": sorted(self.state["candidates"]),
            "progress": {},
            "evaluations": {},
        }
        if refit:
            self.state["current_round"]["refit"] = True
        self.state["phase"] = "collection"
        self._save("round_planned", self.state["current_round"])
        return True

    def _seal_dataset(self):
        current = self.state["current_round"]
        refs = self._rows("pool")
        replay = "cumulative" if current.get("refit") else self.state["round_config"]["replay"]
        if replay == "recent":
            previous = self._previous_members()
            refs = [ref for ref in refs if ref["id"] not in previous]
        dataset = {
            "schema_version": 1,
            "members": refs,
            "replay": replay,
            "byte_cap": self.state["round_config"]["byte_cap"],
        }
        current["dataset"] = dataset | {"sha256": object_hash(dataset)}
        if not self._has_supervision(refs):
            current["status"] = "no_usable_training"
            self._conclude_round()
            return False
        self._update_identity()
        self.state["phase"] = "training"
        self._save(
            "dataset_sealed", {"dataset": current["dataset"], "identity": self.state["identity"]}
        )
        return True

    def _train_checkpoint(self, name, checkpoint_updates, hook):
        import torch

        current = self.state["current_round"]
        settings = self.state["round_config"]
        progress = current["progress"].get(name, 0)
        updates = min(checkpoint_updates, settings["updates"] - progress)
        examples = self._examples(current["dataset"]["members"])
        candidate = self._load_candidate(name, self.state["recovery"].get(name))
        device = candidate.config["device"]
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        metrics = candidate.train(examples, updates, settings["byte_cap"])
        metrics["seconds"] = time.monotonic() - started
        if device == "cuda":
            metrics["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        artifact = candidate.checkpoint()
        hook("before_checkpoint_commit")
        current["progress"][name] = progress + updates
        self.state["recovery"][name] = {"sha256": digest(artifact), "step": candidate.step}
        self._save(
            "training_checkpoint",
            {"candidate": name, "round": current["number"], "step": candidate.step, **metrics},
            artifact=artifact,
        )

    def _evaluate_candidate(self, reference, refs):
        from .session_evaluation import evaluate

        candidate = self._load_candidate(reference["candidate"], reference)
        values = [
            (
                self.state["prompts"][ref["prompt_id"]],
                self.journal.evidence(ref["id"], ref["sha256"]),
            )
            for ref in refs
        ]
        return evaluate(candidate, values, self.state["round_config"]["byte_cap"])

    def _validation(self, name):
        recovery = self.state["recovery"][name] | {"candidate": name}
        metrics = self._evaluate_candidate(recovery, self._rows("validation", include_errors=True))
        result = recovery | {"metrics": metrics, "round": self.state["round"]}
        current = self.state["current_round"]
        current["evaluations"][name] = result
        best = self.state["best"]
        score = metrics["teacher_forced"]["cross_entropy"]
        if score is not None and (
            best is None or score < best["metrics"]["teacher_forced"]["cross_entropy"]
        ):
            self.state["best"] = result
        self._save("candidate_validated", result)

    def _conclude_round(self):
        self.state["history"].append(self.state["current_round"])
        self.state["current_round"] = None
        self.state["phase"] = "selection"
        self._save("round_completed", {"round": self.state["round"], "best": self.state["best"]})

    def _collect_one(self, prompt_id, target, max_queries, hook):
        if target is None:
            return "target_required"
        if max_queries is not None and self.state["query_count"] >= max_queries:
            return "query_budget"
        self._reserve(prompt_id)
        return None if self._finish_pending(target, hook, new=True) else "uncertain_request"

    def _advance(
        self,
        target,
        max_queries,
        max_rounds,
        validation_ce,
        plateau,
        checkpoint_updates,
        stop_after_checkpoint,
        refit,
        hook,
    ):
        if self.state["pending"]:
            return None if self._finish_pending(target, hook) else "uncertain_request"
        phase = self.state["phase"]
        if phase == "validation_collection":
            unseen = self._unseen("validation")
            if unseen:
                return self._collect_one(unseen[0], target, max_queries, hook)
            if not self._has_supervision(self._rows("validation")):
                return "no_usable_validation"
            self.state["phase"] = "selection"
            self._save("validation_frozen", {"members": self._rows("validation")})
            return None
        if phase == "selection":
            best = self.state["best"]
            if max_rounds is not None and self.state["round"] >= max_rounds:
                return "round_budget"
            if (
                best
                and validation_ce is not None
                and (best["metrics"]["teacher_forced"]["cross_entropy"] <= validation_ce)
            ):
                return "validation_target"
            if best and plateau is not None and self.state["round"] - best["round"] >= plateau:
                return "plateau"
            if not self._new_round(refit):
                return "pool_exhausted"
            hook("after_plan")
            return None
        current = self.state["current_round"]
        if phase == "collection":
            used = self._dispatched("pool", round_number=current["number"])
            unseen = [item["id"] for item in current["selected"] if item["id"] not in used]
            if unseen:
                return self._collect_one(unseen[0], target, max_queries, hook)
            if not self._seal_dataset():
                return "no_usable_training"
            hook("after_snapshot")
            return None
        if phase == "training":
            unfinished = [
                name
                for name in current["candidates"]
                if current["progress"].get(name, 0) < self.state["round_config"]["updates"]
            ]
            if unfinished:
                self._train_checkpoint(unfinished[0], checkpoint_updates, hook)
                hook("after_checkpoint")
                return "checkpoint" if stop_after_checkpoint else None
            self.state["phase"] = "validation"
            self._save("training_completed", {"round": self.state["round"]})
            hook("after_training")
            return None
        if phase == "validation":
            missing = [name for name in current["candidates"] if name not in current["evaluations"]]
            if missing:
                self._validation(missing[0])
                hook("after_validation")
            else:
                self._conclude_round()
                hook("after_round")
            return None
        raise ValueError(f"unknown reconstruction stage {phase}")

    def run(
        self,
        target,
        *,
        max_queries=None,
        max_rounds=None,
        max_seconds=None,
        validation_ce=None,
        plateau=None,
        checkpoint_updates=16,
        stop_after_checkpoint=False,
        refit=False,
        max_storage_mib=None,
        reset_limits=False,
        hook=None,
    ):
        hook = hook or (lambda _: None)
        for value in (max_queries, max_rounds, plateau, checkpoint_updates, max_storage_mib):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError("run budgets and checkpoint interval must be positive integers")
        for value in (max_seconds, validation_ce):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("time and validation targets must be positive and finite")
        with self.journal.runner():
            self._reload()
            prior_limits = {} if reset_limits else self.state.get("limits", {})
            supplied = {
                "max_queries": max_queries,
                "max_rounds": max_rounds,
                "max_seconds": max_seconds,
                "validation_ce": validation_ce,
                "plateau": plateau,
                "max_storage_mib": max_storage_mib,
            }
            limits = prior_limits | {
                key: value for key, value in supplied.items() if value is not None
            }
            self.state["limits"] = limits
            max_queries = limits.get("max_queries")
            max_seconds = limits.get("max_seconds")
            validation_ce = limits.get("validation_ce")
            plateau = limits.get("plateau")
            max_storage_mib = limits.get("max_storage_mib")
            if refit:
                if self.state["phase"] != "selection" or self.state["pending"]:
                    raise ValueError("finish the current stage before starting a refit")
                max_rounds = min(max_rounds or math.inf, self.state["round"] + 1)
            else:
                max_rounds = limits.get("max_rounds")
            if self.state["pending"] and self.state["pending"]["partition"] == "test":
                raise ValueError("a final request is pending; resume it with session evaluate")
            self.journal.pause(False)
            self.state["stop"] = None
            self._save(
                "run_started",
                {
                    "max_queries": max_queries,
                    "max_rounds": max_rounds,
                    "max_seconds": max_seconds,
                    "validation_ce": validation_ce,
                    "plateau": plateau,
                    "checkpoint_updates": checkpoint_updates,
                    "stop_after_checkpoint": stop_after_checkpoint,
                    "refit": refit,
                    "max_storage_mib": max_storage_mib,
                    "reset_limits": reset_limits,
                },
            )
            deadline = time.monotonic() + max_seconds if max_seconds is not None else math.inf
            previous_handler = signal.getsignal(signal.SIGTERM)
            interrupted = False

            def interrupt(*_):
                nonlocal interrupted
                interrupted = True

            try:
                signal.signal(signal.SIGTERM, interrupt)
            except ValueError:
                previous_handler = None
            try:
                while True:
                    if interrupted or self.journal.pause_requested():
                        return self._stop("paused")
                    if time.monotonic() >= deadline:
                        return self._stop("time_budget")
                    if (
                        max_storage_mib is not None
                        and self.journal.size_bytes() >= max_storage_mib * 2**20
                    ):
                        return self._stop("storage_budget")
                    reason = self._advance(
                        target,
                        max_queries,
                        max_rounds,
                        validation_ce,
                        plateau,
                        checkpoint_updates,
                        stop_after_checkpoint,
                        refit,
                        hook,
                    )
                    if reason:
                        return self._stop(reason)
            except KeyboardInterrupt:
                self.state = self.journal.load()
                return self._stop("interrupted")
            finally:
                if previous_handler is not None:
                    signal.signal(signal.SIGTERM, previous_handler)

    def final(self, target, *, max_queries=None, hook=None):
        hook = hook or (lambda _: None)
        if max_queries is not None and max_queries < 1:
            raise ValueError("final query budget must be positive")
        with self.journal.runner():
            self._reload()
            if self.state["pending"] and self.state["pending"]["partition"] != "test":
                raise ValueError("resolve pending acquisition before final evaluation")
            cohort = self.state["test_cohort"]
            key = str(cohort)
            if key not in self.state["finals"]:
                if not self.state["best"] or self.state["phase"] != "selection":
                    raise ValueError("finish a validation round before locking a final candidate")
                ids = [
                    item_id
                    for item_id, row in sorted(self.state["prompts"].items())
                    if row["partition"] == "test" and row.get("cohort", 1) == cohort
                ]
                self.state["finals"][key] = {
                    "locked": self.state["best"],
                    "prompt_ids": ids,
                    "prompt_manifest_sha256": object_hash(
                        [self.state["prompts"][item] for item in ids]
                    ),
                }
                self._save("final_locked", self.state["finals"][key])
                hook("after_final_lock")
            final = self.state["finals"][key]
            if "metrics" in final:
                return final
            limits = self.state.setdefault("limits", {})
            if max_queries is None:
                max_queries = limits.get("max_queries")
            elif max_queries != limits.get("max_queries"):
                limits["max_queries"] = max_queries
                self._save("final_query_budget_updated", {"max_queries": max_queries})
            self.journal.pause(False)
            if self.state["pending"] and not self._finish_pending(target, hook):
                return self._stop("uncertain_request")
            for prompt_id in final["prompt_ids"]:
                if prompt_id in self._dispatched("test"):
                    continue
                if self.journal.pause_requested():
                    return self._stop("paused")
                reason = self._collect_one(prompt_id, target, max_queries, hook)
                if reason:
                    return self._stop(reason)
            refs = self._rows("test", cohort=cohort, include_errors=True)
            final["metrics"] = self._evaluate_candidate(final["locked"], refs)
            final["evidence"] = refs
            final["coverage"] = {
                "reserved": len(final["prompt_ids"]),
                "received": len(refs),
                "successful": sum(self.state["evidence"][item["id"]]["usable"] for item in refs),
            }
            self.state["stop"] = None
            self._save("final_evaluated", final)
            return final

    def export(self, output, *, cohort=None):
        with self.journal.runner():
            self._reload()
            reference = self.state["best"]
            if cohort is not None:
                final = self.state["finals"].get(str(cohort))
                if not final:
                    raise ValueError(f"no locked candidate for cohort {cohort}")
                reference = final["locked"]
            if reference is None:
                raise ValueError("there is no validated candidate to export")
            candidate = self._load_candidate(reference["candidate"], reference)
            manifest = candidate.export(
                Path(output),
                provenance={
                    "session_id": self.state["session_id"],
                    "candidate_id": reference["candidate"],
                    "checkpoint_sha256": reference["sha256"],
                    "selected_round": reference["round"],
                    "locked_cohort": cohort,
                    "scope": self.state["scope"],
                    "priors": self.state["priors"],
                },
            )
            return manifest | {
                "candidate_id": reference["candidate"],
                "checkpoint_sha256": reference["sha256"],
            }


def _read(path):
    import json

    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _cli(args):
    from .session_targets import target_from_config

    action = args.session_action
    if action == "generate":
        from .session_candidate import run_export

        if args.request:
            from .session_evaluation import run_export_agent

            return run_export_agent(
                args.directory, _read(args.request), args.max_bytes, args.device
            )
        return {
            "completions": run_export(
                Path(args.directory), args.prompt, args.max_bytes, args.device
            )
        }
    if action == "create":
        with Session(args.directory, _read(args.spec)) as session:
            return session.inspect(verify=True)
    with Session(args.directory) as session:
        if action == "inspect":
            return session.inspect(verify=args.verify, history=args.history)
        if action == "pause":
            return session.pause()
        if action == "abandon":
            return session.abandon(args.reason)
        if action == "add-prompts":
            return session.add_prompts(_read(args.prompts), fresh_test=args.fresh_test)
        if action == "add-candidate":
            return session.add_candidate(args.name, _read(args.config))
        if action == "export":
            return session.export(args.output, cohort=args.cohort)
        if action == "evidence":
            session._reload()
            records = []
            for key, ref in sorted(
                session.state["evidence"].items(), key=lambda pair: pair[1]["position"]
            ):
                if args.partition is None or ref["partition"] == args.partition:
                    records.append(
                        ref
                        | {
                            "evidence_id": key,
                            "observation": session.journal.evidence(key, ref["sha256"]),
                        }
                    )
            return {"evidence": records}
        target = target_from_config(_read(args.target)) if args.target else None
        if action == "observe":
            return session.observe(target, args.feed, args.limit)
        if action == "evaluate":
            return session.final(target, max_queries=args.max_queries)
        return session.run(
            target,
            max_queries=args.max_queries,
            max_rounds=args.max_rounds,
            max_seconds=args.max_seconds,
            validation_ce=args.validation_ce,
            plateau=args.plateau,
            checkpoint_updates=args.checkpoint_updates,
            stop_after_checkpoint=args.stop_after_checkpoint,
            refit=args.refit,
            max_storage_mib=args.max_storage_mib,
            reset_limits=args.reset_limits,
        )


def add_subcommands(commands):
    parser = commands.add_parser("session", help="run persistent reconstruction across stages")
    actions = parser.add_subparsers(dest="session_action", required=True)
    descriptions = {
        "create": "reserve the split and create a session",
        "inspect": "inspect progress and verify journal integrity",
        "observe": "import passive events from a target",
        "resume": "collect, train and validate until the next budget or pause",
        "pause": "ask the running session to stop at a durable boundary",
        "abandon": "record an unresolved request as abandoned",
        "add-prompts": "extend the pool or reserve a fresh final cohort",
        "add-candidate": "register an alternative for the next round",
        "evaluate": "lock the best validated candidate and collect reserved test outputs",
        "evidence": "read committed observations",
        "export": "export standalone candidate weights",
        "generate": "run an exported candidate with target access disabled",
    }
    for name, description in descriptions.items():
        command = actions.add_parser(name, help=description)
        command.add_argument("directory")
        command.set_defaults(func=_cli)
        if name in {"observe", "resume", "evaluate"}:
            command.add_argument(
                "--target",
                required=name != "resume",
                help="target adapter JSON outside the session",
            )
        if name in {"resume", "evaluate"}:
            command.add_argument("--max-queries", type=int, help="absolute active query ceiling")
        if name == "resume":
            command.add_argument("--max-rounds", type=int)
            command.add_argument("--max-seconds", type=float)
            command.add_argument("--validation-ce", type=float)
            command.add_argument(
                "--plateau", type=int, help="rounds since last validation improvement"
            )
            command.add_argument("--checkpoint-updates", type=int, default=16)
            command.add_argument("--stop-after-checkpoint", action="store_true")
            command.add_argument(
                "--reset-limits",
                action="store_true",
                help="clear saved ceilings before applying new limits",
            )
            command.add_argument(
                "--refit", action="store_true", help="one replay round without new queries"
            )
            command.add_argument(
                "--max-storage-mib",
                type=int,
                help="stop at a durable boundary above this local size",
            )
        if name == "create":
            command.add_argument("--spec", required=True)
        if name == "inspect":
            command.add_argument("--verify", action="store_true")
            command.add_argument("--history", action="store_true")
        if name == "observe":
            command.add_argument("--feed", default="default")
            command.add_argument("--limit", type=int, default=100)
        if name == "abandon":
            command.add_argument("--reason", required=True)
        if name == "add-prompts":
            command.add_argument("--prompts", required=True)
            command.add_argument("--fresh-test", action="store_true")
        if name == "add-candidate":
            command.add_argument("--name", required=True)
            command.add_argument("--config", required=True)
        if name == "evidence":
            command.add_argument("--partition", choices=(*PARTITIONS, "passive_unassigned"))
        if name == "export":
            command.add_argument("--output", required=True)
            command.add_argument("--cohort", type=int)
        if name == "generate":
            inputs = command.add_mutually_exclusive_group(required=True)
            inputs.add_argument("--prompt", action="append")
            inputs.add_argument("--request", help="standalone sandbox request JSON")
            command.add_argument("--device", default="cpu")
            command.add_argument("--max-bytes", type=int, default=64)
