"""Whether a recommendation depends on the parameters that had to be guessed.

This is what makes an uncalibrated model usable: rather than asking what the
unobservable parameters are, ask whether the answer changes across every
admissible setting of them.
"""

import pytest

from qshield.experiments.generator import make_instances
from qshield.experiments.robustness import AssetVerdict, AssumptionRanges, perturb, run
from qshield.model import AssetModel
from qshield.optimizer import MigrationProblem
from qshield.paths import EdgeKind, EdgeModel
from qshield.threat import ThreatClass


@pytest.fixture
def hierarchy():
    """A root whose migration dominates, and leaves whose does not."""
    def ca(name, validity, cost, exposure):
        return AssetModel(
            name, "ECDSA", 0.8, exposure, 10, 4, 24.0, cost, 0.9,
            threat_class=ThreatClass.AUTHENTICATION,
            credential_validity_years=validity,
        )

    assets = (
        ca("root", 20.0, 1.0, 0.2),
        *[ca(f"leaf-{i}", 0.25, 0.5, 0.9) for i in range(3)],
    )
    edges = tuple(
        EdgeModel("root", f"leaf-{i}", 1.0, EdgeKind.DELEGATION) for i in range(3)
    )
    # Ample budget: the root is affordable in every draw, so its selection rate
    # measures whether it is *wanted*, not whether it fits.
    return MigrationProblem(assets, edges, (), (), {"ECDSA": "ML-DSA"}, budget=4.0)


def test_reproducible_given_a_seed(hierarchy):
    assert run(hierarchy, draws=40, seed=3) == run(hierarchy, draws=40, seed=3)


def test_observed_fields_are_never_resampled(hierarchy):
    """Algorithm, validity, threat class, hierarchy and controllability are facts
    about the estate; the point is to find what follows from the facts alone."""
    import random

    original = {a.name: a for a in hierarchy.assets}
    for seed in range(20):
        sampled = perturb(hierarchy, random.Random(seed), AssumptionRanges())
        assert sampled.edges == hierarchy.edges
        assert sampled.budget == hierarchy.budget
        for asset in sampled.assets:
            before = original[asset.name]
            assert asset.algorithm == before.algorithm
            assert asset.credential_validity_years == before.credential_validity_years
            assert asset.effective_threat_class is before.effective_threat_class
            assert asset.dependency_count == before.dependency_count
            assert asset.controllable == before.controllable


def test_unobservable_fields_do_move(hierarchy):
    import random

    seen = {f: set() for f in ("sensitivity", "exposure", "migration_cost")}
    for seed in range(30):
        for asset in perturb(hierarchy, random.Random(seed), AssumptionRanges()).assets:
            for field in seen:
                seen[field].add(round(getattr(asset, field), 6))
    assert all(len(values) > 5 for values in seen.values())


def test_sampled_values_stay_admissible(hierarchy):
    import random

    for seed in range(25):
        for asset in perturb(hierarchy, random.Random(seed), AssumptionRanges()).assets:
            assert 0.0 <= asset.sensitivity <= 1.0
            assert 0.0 <= asset.exposure <= 1.0
            assert asset.migration_cost >= 0
            assert asset.data_lifetime_years > 0


def test_a_dominant_anchor_is_recommended_under_every_setting(hierarchy):
    """The root gates every leaf beneath it, so no admissible setting of the
    guessed parameters should move it out of the plan."""
    report = run(hierarchy, draws=200, seed=11)
    assert "root" in report["act_now"]
    verdicts = {e["asset"]: e for e in report["assets"]}
    assert verdicts["root"]["selection_rate"] > 0.9
    for i in range(3):
        assert verdicts[f"leaf-{i}"]["selection_rate"] < verdicts["root"]["selection_rate"]


def test_an_unaffordable_asset_is_reported_as_a_budget_finding(hierarchy):
    """'Would be chosen whenever affordable' is a funding problem, not a
    cryptographic one, and the two call for opposite responses."""
    # Tight enough that the root (cost 1.0, resampled 0.4x-2.5x) fits only
    # sometimes -- and is taken every time it does.
    tight = MigrationProblem(
        hierarchy.assets, hierarchy.edges, (), (), hierarchy.replacements, budget=1.0
    )
    report = run(tight, draws=200, seed=13)
    assert "root" in report["blocked_by_budget"]
    assert "root" not in report["act_now"]


def test_an_asset_no_draw_can_afford_is_not_reported_as_unhelpful(hierarchy):
    """'Never on the table' and 'assessed and rejected' are different claims."""
    starved = MigrationProblem(
        hierarchy.assets, hierarchy.edges, (), (), hierarchy.replacements, budget=0.35
    )
    report = run(starved, draws=120, seed=13)
    assert "root" in report["never_affordable"]
    verdicts = {e["asset"]: e["verdict"] for e in report["assets"]}
    assert verdicts["root"] == "never affordable"


def test_verdict_thresholds():
    def verdict(rate, affordable=1.0, when_affordable=None):
        return AssetVerdict(
            "a", rate, 1.0, affordable,
            rate if when_affordable is None else when_affordable,
        ).verdict

    assert verdict(0.95) == "act now"
    assert verdict(0.7) == "likely"
    assert verdict(0.3) == "contested"
    assert verdict(0.02) == "not indicated"
    # Rarely affordable but always chosen when it is.
    assert verdict(0.2, affordable=0.25, when_affordable=0.8) == "blocked by budget"
    assert verdict(0.0, affordable=0.0, when_affordable=0.0) == "never affordable"


def test_report_records_what_was_held_fixed(hierarchy):
    design = run(hierarchy, draws=20)["design"]
    assert "algorithm" in design["held_fixed"]
    assert "credential_validity_years" in design["held_fixed"]
    assert "sensitivity" in design["resampled_fields"]


def test_human_summary_is_ordered_by_confidence(hierarchy):
    from qshield.experiments.robustness import summarise_for_humans

    lines = summarise_for_humans(run(hierarchy, draws=60, seed=5))
    rates = [float(line.split("%")[0]) for line in lines]
    assert rates == sorted(rates, reverse=True)


def test_runs_on_a_generated_instance():
    report = run(make_instances(1, 77)[0], draws=40)
    assert report["distinct_plans"] >= 1
    assert 0.0 <= report["nominal_plan_reproduced_in_draws"] <= 1.0
