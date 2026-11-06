"""Empirical behavior fingerprints for predeclared, metadata-blind diagnostic panels."""

from __future__ import annotations

import itertools
import math
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

ABSOLUTE = "prompt-conditioned-prefix-and-trigram-v1"
CONTRASTIVE = "prompt-conditioned-contrastive-fingerprint-v2"
_METHODS = {ABSOLUTE, CONTRASTIVE}
_RESERVED = {"unknown", "mixed/routed"}


@dataclass(frozen=True)
class BehaviorObservation:
    """The exposed assistant text from a known probe, in an observed time segment."""

    probe_id: str
    response: str
    segment: str = "default"

    def __post_init__(self):
        if not isinstance(self.probe_id, str) or not self.probe_id:
            raise ValueError("probe_id must be nonempty text")
        if not isinstance(self.response, str):
            raise ValueError("probe response must be text")
        if not isinstance(self.segment, str) or not self.segment:
            raise ValueError("segment must be nonempty text")


def _signature(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())[:512]


def _trigrams(text: str) -> Counter:
    return Counter(text[index : index + 3] for index in range(len(text) - 2))


def _similarity(first: str, second: str) -> float:
    first, second = _signature(first), _signature(second)
    if not first or not second:
        return 0.0
    common = 0
    for left, right in zip(first, second):
        if left != right:
            break
        common += 1
    a, b = _trigrams(first), _trigrams(second)
    total = a.total() + b.total()
    prefix = common / max(len(first), len(second))
    overlap = 2 * (a & b).total() / total if total else float(first == second)
    return (prefix + overlap) / 2


def _score(responses: Mapping[str, str], references: Mapping[str, str]) -> float:
    return math.fsum(
        _similarity(response, references[key]) for key, response in responses.items()
    ) / len(responses)


def _support(family: str, similarities: Mapping[str, float], method: str) -> float:
    score = similarities[family]
    if method == ABSOLUTE:
        return score
    rival = max((value for key, value in similarities.items() if key != family), default=0.0)
    return score * max(0.0, score - rival)


@dataclass(frozen=True)
class FingerprintProfile:
    """Named reference responses and independently collected within-lineage scores.

    Calibration units are full panels in separately assigned runs, in the
    declared reference deployment population. Changing probes, prompt text,
    response truncation, or deployment population can invalidate calibration.
    """

    probes: Mapping[str, str]
    references: Mapping[str, Mapping[str, str]]
    calibration: Mapping[str, Sequence[float]]
    calibration_population: str
    alpha: float = 0.20
    method: str = CONTRASTIVE

    def __post_init__(self):
        if (
            not isinstance(self.probes, Mapping)
            or len(self.probes) < 2
            or not all(
                isinstance(key, str) and key and isinstance(text, str) and text
                for key, text in self.probes.items()
            )
        ):
            raise ValueError("a fingerprint profile needs at least two named text probes")
        if len(set(self.probes.values())) != len(self.probes):
            raise ValueError("probe texts must be distinct")
        if type(self.alpha) not in {int, float} or not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        if not isinstance(self.method, str) or self.method not in _METHODS:
            raise ValueError("unsupported fingerprint scoring method")
        if not isinstance(self.calibration_population, str) or not self.calibration_population:
            raise ValueError("a calibration population description is required")
        if (
            not isinstance(self.references, Mapping)
            or not isinstance(self.calibration, Mapping)
            or not self.references
            or self.references.keys() != self.calibration.keys()
        ):
            raise ValueError("reference and calibration families must match")
        for family, responses in self.references.items():
            if not isinstance(family, str) or not family or family in _RESERVED:
                raise ValueError("invalid reference family")
            if (
                not isinstance(responses, Mapping)
                or responses.keys() != self.probes.keys()
                or not all(isinstance(text, str) for text in responses.values())
                or not any(_signature(text) for text in responses.values())
            ):
                raise ValueError("each reference needs a complete panel with exposed text")
            scores = self.calibration[family]
            if (
                not isinstance(scores, Sequence)
                or 1 / (len(scores) + 1) > self.alpha
                or not all(
                    type(value) in {float, int} and 0 <= value <= 1 and math.isfinite(value)
                    for value in scores
                )
            ):
                raise ValueError("insufficient or invalid full-panel calibration scores")
        object.__setattr__(self, "probes", MappingProxyType(dict(self.probes)))
        object.__setattr__(
            self,
            "references",
            MappingProxyType(
                {key: MappingProxyType(dict(panel)) for key, panel in self.references.items()}
            ),
        )
        object.__setattr__(
            self,
            "calibration",
            MappingProxyType({key: tuple(scores) for key, scores in self.calibration.items()}),
        )

    @classmethod
    def from_dict(cls, data: Mapping) -> FingerprintProfile:
        if (
            not isinstance(data, Mapping)
            or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1
        ):
            raise ValueError("unsupported fingerprint profile schema")
        try:
            return cls(
                probes=data["probes"],
                references=data["references"],
                calibration=data["calibration"],
                calibration_population=data["calibration_population"],
                alpha=data["alpha"],
                method=data["method"],
            )
        except KeyError as error:
            raise ValueError(f"missing fingerprint profile field {error}") from error

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "method": self.method,
            "probes": dict(self.probes),
            "references": {key: dict(panel) for key, panel in self.references.items()},
            "calibration": {key: list(scores) for key, scores in self.calibration.items()},
            "calibration_population": self.calibration_population,
            "alpha": self.alpha,
        }


def calibrate_fingerprints(
    probes: Mapping[str, str],
    references: Mapping[str, Mapping[str, str]],
    trials: Mapping[str, Sequence[Mapping[str, str]]],
    *,
    population: str,
    alpha: float = 0.20,
    method: str = CONTRASTIVE,
) -> FingerprintProfile:
    """Fit ranks from explicitly separated complete, same-lineage calibration runs."""
    if trials.keys() != references.keys():
        raise ValueError("calibration trials must match reference families")
    for panels in trials.values():
        for panel in panels:
            if panel.keys() != probes.keys() or not all(isinstance(x, str) for x in panel.values()):
                raise ValueError("each calibration trial must contain the complete probe panel")
    if not probes or any(
        panel.keys() != probes.keys() or not all(isinstance(x, str) for x in panel.values())
        for panel in references.values()
    ):
        raise ValueError("each reference must contain the complete probe panel")
    calibration = {
        family: [
            _support(family, {key: _score(panel, ref) for key, ref in references.items()}, method)
            for panel in panels
        ]
        for family, panels in trials.items()
    }
    return FingerprintProfile(
        probes,
        references,
        calibration,
        population,
        alpha,
        method,
    )


def _segment_report(profile: FingerprintProfile, responses: dict[str, str]) -> dict:
    responses = dict(sorted(responses.items()))
    complete = responses.keys() == profile.probes.keys()
    informative = any(_signature(response) for response in responses.values())
    scores = {
        family: {probe: _similarity(text, ref[probe]) for probe, text in responses.items()}
        for family, ref in profile.references.items()
    }
    similarities = {
        key: math.fsum(by_probe.values()) / len(by_probe) for key, by_probe in scores.items()
    }
    ranked = []
    for family, by_probe in scores.items():
        score = similarities[family]
        support = _support(family, similarities, profile.method)
        calibration = profile.calibration[family]
        compatibility = None
        if complete and informative:
            compatibility = (1 + sum(value <= support for value in calibration)) / (
                1 + len(calibration)
            )
        rival = {
            probe: max(
                (other_scores[probe] for other, other_scores in scores.items() if other != family),
                default=0.0,
            )
            for probe in by_probe
        }
        ranked.append(
            {
                "family": family,
                "similarity": score,
                "calibrated_support": support,
                "compatibility_p": compatibility,
                "calibration_panels": len(calibration),
                "probe_similarity": by_probe,
                "supports_relative_to_rivals": [p for p in by_probe if by_probe[p] > rival[p]],
                "weakens_relative_to_rivals": [p for p in by_probe if by_probe[p] < rival[p]],
            }
        )
    ranked.sort(key=lambda row: (-row["similarity"], row["family"]))
    compatible = [
        row["family"]
        for row in ranked
        if row["compatibility_p"] is not None
        and row["compatibility_p"] > profile.alpha
        and (profile.method == ABSOLUTE or row["calibrated_support"] > 0)
    ]
    if not complete:
        reason = "incomplete_panel"
    elif not informative:
        reason = "no_nonempty_assistant_text"
    elif not compatible:
        reason = "all_calibrated_lineages_rejected"
    elif len(compatible) > 1:
        reason = "multiple_compatible_lineages"
    else:
        reason = "single_compatible_lineage"
    return {
        "outcome": compatible[0] if len(compatible) == 1 else "unknown",
        "reason": reason,
        "compatible_families": compatible,
        "observed_probe_ids": list(responses),
        "missing_probe_ids": sorted(profile.probes.keys() - responses.keys()),
        "hypotheses": ranked,
    }


def identify(
    observations: Iterable[BehaviorObservation], profile: FingerprintProfile | None = None
) -> dict:
    """Rank family hypotheses without target identifiers, config, tokenizer, or internals.

    Segments must be assigned before inspecting response content, e.g. by observed
    session boundaries or fixed time blocks. The first response per probe and
    segment is retained to avoid selecting favorable repeats after the fact.
    """
    segments: dict[str, dict[str, str]] = {}
    ignored = duplicate = 0
    for observation in observations:
        if profile is None or observation.probe_id not in profile.probes:
            ignored += 1
            continue
        responses = segments.setdefault(observation.segment, {})
        if observation.probe_id in responses:
            duplicate += 1
            continue
        responses[observation.probe_id] = observation.response
    reports = [
        {"segment": name, **_segment_report(profile, responses)}
        for name, responses in segments.items()
    ]
    accepted = {row["outcome"] for row in reports} - {"unknown"}
    if len(accepted) > 1:
        outcome, reason = "mixed/routed", "different_compatible_lineages_in_observed_segments"
    elif accepted and all(report["outcome"] in accepted for report in reports):
        outcome, reason = next(iter(accepted)), "observed_segments_agree"
    else:
        outcome = "unknown"
        reason = "no_profile" if profile is None else "no_consistent_calibrated_lineage"
    return {
        "schema_version": 1,
        "method": profile.method if profile is not None else None,
        "lineage": {
            "outcome": outcome,
            "reason": reason,
            "segments": reports,
            "alternative_families": sorted(profile.references.keys() - {outcome})
            if profile is not None
            else [],
        },
        "routing": {
            "outcome": "mixed/routed" if outcome == "mixed/routed" else "unknown",
            "evidence": [row["segment"] for row in reports if row["outcome"] in accepted]
            if outcome == "mixed/routed"
            else [],
            "limit": "Configuration shifts can mimic routing. Segment evidence cannot distinguish "
            "a backend replacement from per-request routing.",
        },
        "architecture": {"outcome": "unknown", "reason": "lineage match implies no structure"},
        "deployment": {"outcome": "unknown", "reason": "deployment config was not exposed"},
        "version": {"outcome": "unknown", "reason": "no version-level calibration"},
        "calibration": {
            "population": profile.calibration_population if profile is not None else None,
            "alpha": profile.alpha if profile is not None else None,
            "interpretation": "Empirical compatibility ranks require exchangeable complete panels. "
            "They are not posterior probabilities of family membership.",
        },
        "evidence_policy": {
            "used": "first exposed assistant text per fixed probe and observed segment",
            "ignored_observations": ignored,
            "duplicate_observations": duplicate,
            "self_report_and_incidental_metadata": "not used as identity evidence",
        },
    }


def select_identity_probes(
    profile: FingerprintProfile,
    observed_probe_ids: Iterable[str],
    limit: int,
    plausible_families: Iterable[str] = (),
) -> list[str]:
    """Prioritize outstanding probes whose reference lineages disagree in text."""
    if type(limit) is not int or limit < 0:
        raise ValueError("probe limit must be a nonnegative integer")
    families = sorted(set(plausible_families) & profile.references.keys())
    if len(families) < 2:
        families = sorted(profile.references)
    pairs = list(itertools.combinations(families, 2))

    def dissimilarity(probe: str) -> float:
        return math.fsum(
            1 - _similarity(profile.references[first][probe], profile.references[second][probe])
            for first, second in pairs
        )

    unseen = profile.probes.keys() - set(observed_probe_ids)
    return sorted(unseen, key=lambda probe: (-dissimilarity(probe), probe))[:limit]
