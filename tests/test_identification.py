import pytest

from rephix.identification import (
    ABSOLUTE,
    BehaviorObservation,
    FingerprintProfile,
    calibrate_fingerprints,
    identify,
    select_identity_probes,
)

PROBES = {"p": "First probe", "q": "Second probe", "shared": "Third probe"}
A = {"p": "go north", "q": "find the moon", "shared": "same"}
B = {"p": "turn left", "q": "ask the sky", "shared": "same"}


def _profile():
    return calibrate_fingerprints(
        PROBES,
        {"alpha": A, "beta": B},
        {"alpha": [A] * 4, "beta": [B] * 4},
        population="exact toy panels",
    )


def _observed(panel, segment="first"):
    return [BehaviorObservation(probe, response, segment) for probe, response in panel.items()]


def test_profile_roundtrip_calibrated_acceptance_and_metadata_blindness():
    profile = FingerprintProfile.from_dict(_profile().to_dict())
    report = identify(
        _observed(A)
        + [BehaviorObservation("name", "I am beta"), BehaviorObservation("p", "oops", "first")],
        profile,
    )
    assert report["lineage"]["outcome"] == "alpha"
    first, second = report["lineage"]["segments"][0]["hypotheses"]
    assert first["compatibility_p"] == 1
    assert second["compatibility_p"] == 0.2
    assert first["supports_relative_to_rivals"] == ["p", "q"]
    assert second["weakens_relative_to_rivals"] == ["p", "q"]
    assert report["evidence_policy"]["ignored_observations"] == 1
    assert report["evidence_policy"]["duplicate_observations"] == 1
    assert report["architecture"]["outcome"] == "unknown"
    assert report["version"]["outcome"] == "unknown"
    assert report["deployment"]["outcome"] == "unknown"


def test_open_set_abstention_and_incomplete_panels():
    profile = _profile()
    out_of_family = dict.fromkeys(PROBES, "unrelated")
    unknown = identify(_observed(out_of_family), profile)["lineage"]
    assert unknown["outcome"] == "unknown"
    assert unknown["segments"][0]["reason"] == "all_calibrated_lineages_rejected"
    incomplete = identify(_observed({"p": A["p"]}), profile)["lineage"]["segments"][0]
    assert incomplete["reason"] == "incomplete_panel"
    assert incomplete["hypotheses"][0]["family"] == "alpha"
    assert incomplete["hypotheses"][0]["compatibility_p"] is None
    assert identify(_observed(A))["lineage"]["reason"] == "no_profile"


def test_detects_distinct_segments_and_retains_ambiguity():
    profile = _profile()
    report = identify(_observed(A) + _observed(B, "second"), profile)
    assert report["lineage"]["outcome"] == "mixed/routed"
    assert report["routing"]["evidence"] == ["first", "second"]
    assert (
        identify(_observed(A) + _observed({"p": B["p"]}, "second"), profile)["lineage"]["outcome"]
        == "unknown"
    )
    shared = calibrate_fingerprints(
        PROBES,
        {"alpha": A, "beta": A},
        {"alpha": [A] * 4, "beta": [A] * 4},
        population="same",
        method=ABSOLUTE,
    )
    result = identify(_observed(A), shared)["lineage"]["segments"][0]
    assert result["reason"] == "multiple_compatible_lineages"
    assert result["compatible_families"] == ["alpha", "beta"]


def test_contrastive_calibration_cannot_accept_an_outscored_family():
    weak_beta = [{"p": "different", "q": "answers", "shared": "entirely"}] * 4
    absolute, contrastive = [
        calibrate_fingerprints(
            PROBES,
            {"alpha": A, "beta": B},
            {"alpha": [A] * 4, "beta": weak_beta},
            population="weak beta calibration",
            **kwargs,
        )
        for kwargs in ({"method": ABSOLUTE}, {})
    ]
    assert identify(_observed(A), absolute)["lineage"]["outcome"] == "unknown"
    assert identify(_observed(A), contrastive)["lineage"]["outcome"] == "alpha"


def test_probe_selection_and_calibration_contract():
    profile = _profile()
    choices = select_identity_probes(profile, [], 3)
    assert set(choices[:2]) == {"p", "q"}
    assert choices[-1] == "shared"
    assert select_identity_probes(profile, choices[:1], 1) == choices[1:2]
    with pytest.raises(ValueError, match="insufficient"):
        calibrate_fingerprints(
            PROBES, {"alpha": A}, {"alpha": [A] * 3}, population="too few panels"
        )
    with pytest.raises(ValueError, match="complete"):
        calibrate_fingerprints(
            PROBES, {"alpha": A}, {"alpha": [{"p": "incomplete"}] * 4}, population="partial"
        )
    with pytest.raises(ValueError, match="schema"):
        FingerprintProfile.from_dict({**profile.to_dict(), "schema_version": 999})
