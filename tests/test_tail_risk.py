"""The path aggregate.

0.4 averaged the path risks, and that was measured degrading worst-case exposure:
the cheapest way to lower an average is to improve the many moderate paths and
leave the worst one standing. These tests pin the CVaR aggregate that replaces it.
"""

import pytest

from qshield.model import (
    DEFAULT_WEIGHTS,
    TAIL_AWARE_WEIGHTS,
    AssetModel,
    ObjectiveWeights,
    aggregate_path_risk,
    objective,
)
from qshield.paths import EdgeModel

RISKS = [10.0, 20.0, 30.0, 90.0]


def test_alpha_one_is_the_mean():
    assert aggregate_path_risk(RISKS, 1.0) == pytest.approx(37.5)


def test_small_alpha_is_the_worst_path():
    assert aggregate_path_risk(RISKS, 0.01) == pytest.approx(90.0)
    assert aggregate_path_risk(RISKS, 0.25) == pytest.approx(90.0)


def test_intermediate_alpha_averages_the_worst_fraction():
    assert aggregate_path_risk(RISKS, 0.5) == pytest.approx(60.0)  # mean of 90, 30


def test_aggregate_is_monotone_non_increasing_in_alpha():
    """More of the distribution included can only pull the aggregate down."""
    previous = None
    for alpha in (0.05, 0.25, 0.5, 0.75, 1.0):
        value = aggregate_path_risk(RISKS, alpha)
        if previous is not None:
            assert value <= previous + 1e-12
        previous = value


@pytest.mark.parametrize("alpha", [1.0, 0.75, 0.5, 0.25, 0.05])
def test_aggregate_is_monotone_in_every_risk(alpha):
    """Required for the objective's monotonicity: an average of the k largest
    values is non-decreasing in every value."""
    base = aggregate_path_risk(RISKS, alpha)
    for i in range(len(RISKS)):
        raised = list(RISKS)
        raised[i] += 5.0
        assert aggregate_path_risk(raised, alpha) >= base - 1e-12


def test_aggregate_lies_between_the_mean_and_the_max():
    for alpha in (0.05, 0.3, 0.7, 1.0):
        value = aggregate_path_risk(RISKS, alpha)
        assert min(RISKS) <= value <= max(RISKS)
        assert value >= sum(RISKS) / len(RISKS) - 1e-12


def test_empty_and_single_path_cases():
    assert aggregate_path_risk([], 0.5) == 0.0
    assert aggregate_path_risk([42.0], 0.1) == pytest.approx(42.0)


def test_alpha_is_validated():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="path_alpha"):
            ObjectiveWeights(path_alpha=bad)


def test_alpha_participates_in_the_objective_identity():
    """Two settings agreeing on the weights but differing on the tail parameter
    are different objectives and must not de-duplicate together."""
    a = ObjectiveWeights(0.55, 0.30, 0.15, 1.0)
    b = ObjectiveWeights(0.55, 0.30, 0.15, 0.25)
    assert a.key() != b.key()
    assert ObjectiveWeights(0.2, 0.2, 0.1, 0.5).key() == ObjectiveWeights(0.6, 0.6, 0.3, 0.5).key()


def test_normalization_preserves_alpha():
    assert ObjectiveWeights(2.0, 2.0, 1.0, 0.3).normalized().path_alpha == pytest.approx(0.3)


def build(risky_path_count=3):
    """One very bad path alongside several mild ones."""
    assets = [
        AssetModel("entry", "RSA-2048", 0.9, 0.9, 20, 4, 24.0),
        AssetModel("hot", "RSA-2048", 0.95, 0.95, 20, 6, 24.0),
        *[AssetModel(f"mild-{i}", "AES-256", 0.4, 0.4, 5, 1, 24.0)
          for i in range(risky_path_count)],
        AssetModel("target", "AES-256", 0.9, 0.4, 15, 3, 24.0),
    ]
    edges = [EdgeModel("entry", "hot", 1.0), EdgeModel("hot", "target", 1.0)]
    for i in range(risky_path_count):
        edges += [EdgeModel("entry", f"mild-{i}", 0.9),
                  EdgeModel(f"mild-{i}", "target", 0.9)]
    return assets, edges


def test_tail_aware_objective_weights_the_worst_path_more_heavily():
    assets, edges = build()
    mean_based = objective(assets, edges, ["entry"], ["target"], weights=DEFAULT_WEIGHTS)
    tail_based = objective(assets, edges, ["entry"], ["target"], weights=TAIL_AWARE_WEIGHTS)
    assert len(mean_based.paths) > 1
    assert tail_based.path_component > mean_based.path_component
    assert tail_based.value > mean_based.value


def test_components_are_reported_independently_of_the_aggregate_used():
    """path_mean and path_max are components, not the weighted value, so they
    stay comparable across arms optimising different alphas."""
    assets, edges = build()
    a = objective(assets, edges, ["entry"], ["target"], weights=DEFAULT_WEIGHTS)
    b = objective(assets, edges, ["entry"], ["target"], weights=TAIL_AWARE_WEIGHTS)
    assert a.path_mean == pytest.approx(b.path_mean)
    assert a.path_max == pytest.approx(b.path_max)
    assert a.path_component == pytest.approx(a.path_mean)  # alpha = 1


def test_default_objective_is_unchanged_from_0_4():
    """path_alpha defaults to 1.0, so existing cases score exactly as before."""
    assert DEFAULT_WEIGHTS.path_alpha == 1.0
    assets, edges = build()
    result = objective(assets, edges, ["entry"], ["target"])
    assert result.path_component == pytest.approx(result.path_mean)
