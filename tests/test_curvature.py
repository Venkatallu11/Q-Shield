"""Curvature of the risk-reduction function.

A reviewer suggested testing for submodularity, hoping it would hold and hand
greedy planning the classic 1 - 1/e approximation guarantee. It does not hold,
and the direction of failure is the opposite of what such a guarantee needs.
"""

import random

import pytest

from qshield.experiments.curvature import (
    CurvatureSample,
    _spearman,
    sample_curvature,
)
from qshield.model import AssetModel
from qshield.optimizer import MigrationProblem
from qshield.paths import EdgeKind, EdgeModel
from qshield.threat import ThreatClass


def ca(name, validity, cost, exposure):
    return AssetModel(
        name, "ECDSA", 0.8, exposure, 10, 4, 24.0, cost, 0.9,
        threat_class=ThreatClass.AUTHENTICATION, credential_validity_years=validity,
    )


@pytest.fixture
def hierarchy():
    assets = (ca("root", 20, 1.0, 0.3), ca("leaf-a", 5, 0.5, 0.9), ca("leaf-b", 5, 0.5, 0.9))
    edges = (
        EdgeModel("root", "leaf-a", 1.0, EdgeKind.DELEGATION),
        EdgeModel("root", "leaf-b", 1.0, EdgeKind.DELEGATION),
    )
    return MigrationProblem(assets, edges, (), (), {"ECDSA": "ML-DSA"}, budget=99)


def test_delegation_gives_strictly_increasing_returns(hierarchy):
    """A leaf migrated alone reduces risk by exactly zero, because its risk is
    pinned to an unmigrated issuer. The same leaf after the root is worthwhile."""
    risk = lambda s: hierarchy.evaluate(s).value  # noqa: E731
    alone = risk(()) - risk(["leaf-a"])
    after_root = risk(["root"]) - risk(["root", "leaf-a"])
    assert alone == pytest.approx(0.0)
    assert after_root > 0
    assert after_root > alone


def test_synergy_between_an_anchor_and_its_leaf_is_positive(hierarchy):
    risk = lambda s: hierarchy.evaluate(s).value  # noqa: E731
    base = risk(())
    together = base - risk(["root", "leaf-a"])
    separate = (base - risk(["root"])) + (base - risk(["leaf-a"]))
    assert together > separate  # supermodular: the pair beats the sum of parts


def test_the_noisy_or_path_term_is_supermodular_without_any_delegation():
    """The cause is more general than delegation. Path risk composes as
    1 - prod(1 - p), so the benefit of migrating one node is proportional to
    prod(1 - p) over the others -- a factor that grows as they are migrated."""
    assets = (
        AssetModel("entry", "RSA-2048", 0.9, 0.9, 20, 4, 24.0, 1.0),
        AssetModel("mid", "RSA-2048", 0.9, 0.9, 20, 4, 24.0, 1.0),
        AssetModel("target", "AES-256", 0.9, 0.4, 20, 4, 24.0, 1.0),
    )
    edges = (EdgeModel("entry", "mid", 1.0), EdgeModel("mid", "target", 1.0))
    problem = MigrationProblem(
        assets, edges, ("entry",), ("target",), {"RSA-2048": "ML-KEM"}, budget=99
    )
    risk = lambda s: problem.evaluate(s).value  # noqa: E731
    alone = risk(()) - risk(["entry"])
    after_mid = risk(["mid"]) - risk(["mid", "entry"])
    assert after_mid > alone


def test_sampler_draws_properly_nested_pairs(hierarchy):
    drawn = sample_curvature(hierarchy, random.Random(1), samples=30)
    assert drawn
    assert all(isinstance(s, CurvatureSample) for s in drawn)


def test_sampler_needs_enough_candidates():
    assets = (ca("only", 10, 1.0, 0.5),)
    problem = MigrationProblem(assets, (), (), (), {"ECDSA": "ML-DSA"}, budget=9)
    assert sample_curvature(problem, random.Random(1)) == []


def test_no_supermodularity_violation_is_observed(hierarchy):
    """Violating supermodularity would mean diminishing returns somewhere; the
    measurement says that never happens on these instances."""
    drawn = sample_curvature(hierarchy, random.Random(3), samples=60)
    assert any(s.violates_submodularity for s in drawn)
    assert not any(s.violates_supermodularity for s in drawn)


def test_sample_classification_is_exclusive():
    assert CurvatureSample(1.0, 2.0).violates_submodularity
    assert not CurvatureSample(1.0, 2.0).violates_supermodularity
    assert CurvatureSample(2.0, 1.0).violates_supermodularity
    assert not CurvatureSample(2.0, 1.0).violates_submodularity
    equal = CurvatureSample(1.0, 1.0)
    assert not equal.violates_submodularity and not equal.violates_supermodularity


def test_spearman_matches_known_cases():
    assert _spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert _spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert _spearman([1, 2], [1, 2]) is None       # too few points
    assert _spearman([1, 1, 1], [1, 2, 3]) is None  # no variance to correlate


@pytest.mark.slow
def test_report_states_the_consequence_for_greedy():
    from qshield.experiments.curvature import run

    report = run(instances=8, samples_per_instance=10)
    verdict = report["verdict"]
    assert "1 - 1/e" in verdict["consequence"]
    for suite in report["suites"].values():
        assert suite["supermodularity_violation_rate"]["mean"] == 0.0
