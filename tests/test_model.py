"""Objective semantics, including the two 0.3 soundness bugs it repairs."""

import pytest

from qshield.model import (
    DEFAULT_WEIGHTS,
    AssetModel,
    ObjectiveWeights,
    apply_migration,
    node_risk,
    objective,
    path_risk,
)
from qshield.paths import EdgeModel


def test_asset_rejects_out_of_range_parameters():
    with pytest.raises(ValueError, match="sensitivity"):
        AssetModel("a", "RSA-2048", sensitivity=1.4)
    with pytest.raises(ValueError, match="exposure"):
        AssetModel("a", "RSA-2048", exposure=-0.1)
    with pytest.raises(ValueError, match="non-negative"):
        AssetModel("a", "RSA-2048", migration_cost=-1)


def test_node_risk_is_bounded_and_increases_with_susceptibility():
    a = AssetModel("a", "RSA-2048", 1.0, 1.0, 100, 20, 0.0)
    assert 0.0 <= node_risk(a) <= 100.0
    assert node_risk(a, qf_override=1.0) > node_risk(a, qf_override=0.05)


def test_path_risk_accounts_for_the_entry_node():
    """0.3 iterated zip(path, path[1:]) reading only the target of each hop, so
    the first node on every path contributed nothing and migrating an
    internet-facing entrypoint left the path term unchanged."""
    edges = {("a", "b"): 1.0}
    safe_entry = path_risk(["a", "b"], {"a": 0.0, "b": 50.0}, edges)
    risky_entry = path_risk(["a", "b"], {"a": 99.0, "b": 50.0}, edges)
    assert risky_entry > safe_entry


def test_path_risk_increases_with_downstream_node_risk():
    edges = {("a", "b"): 1.0}
    assert path_risk(["a", "b"], {"a": 10, "b": 80}, edges) > path_risk(
        ["a", "b"], {"a": 10, "b": 20}, edges
    )


def test_path_risk_scales_with_edge_reliability():
    scores = {"a": 10.0, "b": 80.0}
    strong = path_risk(["a", "b"], scores, {("a", "b"): 1.0})
    weak = path_risk(["a", "b"], scores, {("a", "b"): 0.1})
    assert strong > weak


def test_trivial_path_has_no_risk():
    assert path_risk(["a"], {"a": 99.0}, {}) == 0.0


def test_rotation_pressure_membership_does_not_depend_on_score():
    """The 0.3 gate was `rotation_hours and score > 10`, so a migration could
    remove an asset from the average and raise the mean over the survivors."""
    low = AssetModel("low", "ML-KEM", 0.5, 0.44, 20, 1, 4.0)
    high = AssetModel("high", "RSA-2048", 0.95, 0.95, 20, 1, 168.0)
    result = objective([low, high], [])
    # Both assets carry rotation_hours > 0, so both are in the average even
    # though `low` scores far below 10.
    assert 0 < result.rotation_pressure < node_risk(high)


def test_assets_without_rotation_are_excluded_from_rotation_pressure():
    a = AssetModel("a", "RSA-2048", 0.9, 0.9, 20, 1, rotation_hours=0.0)
    assert objective([a], []).rotation_pressure == 0.0


def test_weights_are_scale_invariant():
    a = AssetModel("a", "RSA-2048", 0.9, 0.9, 20, 4, 48.0)
    b = AssetModel("b", "AES-256", 0.9, 0.9, 20, 4, 48.0)
    edges = [EdgeModel("a", "b", 0.9)]
    small = objective([a, b], edges, ["a"], ["b"], weights=ObjectiveWeights(0.2, 0.2, 0.1))
    large = objective([a, b], edges, ["a"], ["b"], weights=ObjectiveWeights(0.6, 0.6, 0.3))
    assert small.value == pytest.approx(large.value)
    assert ObjectiveWeights(0.2, 0.2, 0.1).key() == ObjectiveWeights(0.6, 0.6, 0.3).key()


def test_weights_reject_degenerate_input():
    with pytest.raises(ValueError):
        ObjectiveWeights(0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        ObjectiveWeights(-1.0, 1.0, 1.0)


def test_node_only_weighting_ignores_the_path_term():
    a = AssetModel("a", "RSA-2048", 0.9, 0.9, 20, 4, 0.0)
    b = AssetModel("b", "RSA-2048", 0.9, 0.9, 20, 4, 0.0)
    edges = [EdgeModel("a", "b", 1.0)]
    weights = ObjectiveWeights(node=1.0, path=0.0, rotation=0.0)
    result = objective([a, b], edges, ["a"], ["b"], weights=weights)
    assert result.value == pytest.approx(result.node_mean)
    assert result.path_mean > 0  # still computed and reported, just not weighted


def test_objective_components_are_bounded():
    a = AssetModel("a", "RSA-2048", 1.0, 1.0, 100, 50, 500.0)
    b = AssetModel("b", "RSA-2048", 1.0, 1.0, 100, 50, 500.0)
    r = objective([a, b], [EdgeModel("a", "b", 1.0)], ["a"], ["b"])
    for value in (r.value, r.node_mean, r.path_mean, r.path_max, r.rotation_pressure):
        assert 0.0 <= value <= 100.0


def test_apply_migration_rejects_unknown_assets_and_missing_replacements():
    a = AssetModel("a", "RSA-2048")
    with pytest.raises(KeyError, match="unknown assets"):
        apply_migration([a], ["ghost"], {"RSA-2048": "ML-KEM"})
    with pytest.raises(KeyError, match="no replacement"):
        apply_migration([a], ["a"], {"ECDSA": "ML-DSA"})


def test_apply_migration_leaves_other_assets_untouched():
    a = AssetModel("a", "RSA-2048", 0.7, 0.7, 10, 3, 5.0, 2.0, 0.8)
    b = AssetModel("b", "ECDSA", 0.4, 0.4, 4, 1, 1.0, 1.0, 0.2)
    out = apply_migration([a, b], ["a"], {"RSA-2048": "ML-KEM", "ECDSA": "ML-DSA"})
    assert out[0].algorithm == "ML-KEM"
    assert out[1] == b
    # Only the algorithm changes; every other parameter is preserved.
    assert (out[0].sensitivity, out[0].migration_cost) == (a.sensitivity, a.migration_cost)


def test_default_weights_are_documented_values():
    assert DEFAULT_WEIGHTS.as_tuple() == (0.55, 0.30, 0.15)
