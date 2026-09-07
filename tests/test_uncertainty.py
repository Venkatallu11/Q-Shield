"""Sampling and paired statistics.

The property that matters is common random numbers: every strategy in a
comparison must be scored against the identical sampled world, which is what
0.3's per-strategy seeds (11, 12, 13) prevented.
"""

import statistics
from dataclasses import replace

import pytest

from qshield.experiments.generator import make_instances
from qshield.model import DEFAULT_WEIGHTS, AssetModel
from qshield.optimizer import exhaustive, greedy_flat
from qshield.paths import EdgeModel
from qshield.uncertainty import (
    PerturbationModel,
    bootstrap_ci,
    paired_comparison,
    quantiles,
    sample_worlds,
    summarize,
)

ASSETS = (
    AssetModel("a", "RSA-2048", 0.8, 0.8, 15, 3, 24.0),
    AssetModel("b", "ECDSA", 0.7, 0.6, 12, 2, 48.0),
)
EDGES = (EdgeModel("a", "b", 0.9),)


def draw(trials=50, seed=1, model=PerturbationModel()):
    return sample_worlds(
        ASSETS, EDGES, trials=trials, seed=seed, weights=DEFAULT_WEIGHTS, model=model
    )


def test_sampling_is_reproducible_and_seed_sensitive():
    assert draw(seed=1) == draw(seed=1)
    assert draw(seed=1) != draw(seed=2)


def test_sampled_parameters_stay_in_range():
    for world in draw(trials=200, model=PerturbationModel(quantum_factor_spread=0.5)):
        for asset in world.assets:
            assert 0.0 <= asset.sensitivity <= 1.0
            assert 0.0 <= asset.exposure <= 1.0
            assert asset.data_lifetime_years > 0
        for edge in world.edges:
            assert 0.0 <= edge.reliability <= 1.0
        for qf in world.overrides_for(world.assets).values():
            assert 0.0 < qf <= 1.0


def test_default_model_leaves_quantum_factors_and_weights_fixed():
    """Enabling those changes what the reported interval means, so they are
    opt-in even though 0.3's README called them the least certain parameters."""
    for world in draw(trials=20):
        assert world.qf_multipliers == {}
        assert world.weights == DEFAULT_WEIGHTS


def test_quantum_factor_uncertainty_is_a_multiplier_not_a_frozen_override():
    """An absolute override sampled from the pre-migration algorithm would
    follow the asset through a migration and cancel it out."""
    world = draw(trials=1, model=PerturbationModel(quantum_factor_spread=0.3))[0]
    before = world.overrides_for(world.assets)["a"]
    migrated = (replace(world.assets[0], algorithm="ML-KEM"),)
    after = world.overrides_for(migrated)["a"]
    assert after < before


def test_common_random_numbers_make_the_difference_noise_free():
    """The same world scored twice for the same plan must give the same number;
    that is what allows a paired difference to isolate the strategy effect."""
    problem = make_instances(1, 5)[0]
    worlds = sample_worlds(
        problem.assets, problem.edges, trials=30, seed=9, weights=DEFAULT_WEIGHTS
    )
    plan = exhaustive(problem)[0].selected
    first = [problem.evaluate_world(w, plan).value for w in worlds]
    second = [problem.evaluate_world(w, plan).value for w in worlds]
    assert first == second


def test_paired_deltas_have_lower_variance_than_either_arm():
    """The reason 0.3's independently-seeded percentile bands were uninformative."""
    problem = make_instances(1, 6)[0]
    worlds = sample_worlds(
        problem.assets, problem.edges, trials=400, seed=4, weights=DEFAULT_WEIGHTS
    )
    a = [problem.evaluate_world(w, greedy_flat(problem).selected).value for w in worlds]
    b = [problem.evaluate_world(w, exhaustive(problem)[0].selected).value for w in worlds]
    deltas = [x - y for x, y in zip(a, b, strict=True)]
    assert statistics.stdev(deltas) < min(statistics.stdev(a), statistics.stdev(b))


def test_quantiles_interpolate_consistently():
    """0.3 indexed the sorted list with int(.05*n)-1 for p05 but int(.95*n) for
    p95, biasing every reported interval."""
    values = list(range(101))  # 0..100
    q = quantiles(values)
    assert q["p05"] == pytest.approx(5.0)
    assert q["p50"] == pytest.approx(50.0)
    assert q["p95"] == pytest.approx(95.0)


def test_quantiles_are_ordered_and_handle_degenerate_input():
    q = quantiles([7.0])
    assert q["p05"] == q["p50"] == q["p95"] == 7.0
    assert quantiles([]) == {"p05": 0.0, "p50": 0.0, "p95": 0.0}
    q = quantiles([3, 1, 2])
    assert q["p05"] <= q["p50"] <= q["p95"]


def test_summarize_reports_shape():
    s = summarize([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4 and s["mean"] == pytest.approx(2.5) and s["stdev"] > 0


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    values = [float(x) for x in range(100)]
    lo, hi = bootstrap_ci(values, resamples=800, seed=3)
    assert lo < statistics.fmean(values) < hi
    assert bootstrap_ci(values, resamples=800, seed=3) == (lo, hi)


def test_bootstrap_ci_narrows_as_the_sample_grows():
    tight = bootstrap_ci([1.0, 1.01] * 200, resamples=500, seed=2)
    loose = bootstrap_ci([0.0, 2.0] * 200, resamples=500, seed=2)
    assert (tight[1] - tight[0]) < (loose[1] - loose[0])


def test_paired_comparison_counts_wins_losses_and_ties():
    result = paired_comparison([1.0, -1.0, 0.0, 2.0], label_a="a", label_b="b")
    assert result["b_better"] == 2      # positive delta: b scored lower
    assert result["a_better"] == 1
    assert result["ties"] == 1
    assert result["decisive_instances"] == 3
    assert result["mean_delta"] == pytest.approx(0.5)


def test_conditional_effect_excludes_ties():
    """An unconditional mean over a mostly-tied sample understates the effect
    where it exists; a raw win rate overstates how often it applies."""
    deltas = [0.0] * 90 + [4.0] * 10
    result = paired_comparison(deltas)
    assert result["mean_delta"] == pytest.approx(0.4)
    assert result["mean_delta_when_decisive"] == pytest.approx(4.0)
    assert result["tie_rate"] == pytest.approx(0.9)
    assert result["win_rate_given_decisive"] == pytest.approx(1.0)


def test_paired_comparison_handles_an_empty_sample():
    assert paired_comparison([])["instances"] == 0


def test_all_ties_report_no_decisive_effect():
    result = paired_comparison([0.0] * 10)
    assert result["ties"] == 10
    assert result["decisive_instances"] == 0
    assert result["mean_delta_when_decisive"] == 0.0
    assert "win_rate_given_decisive" not in result
