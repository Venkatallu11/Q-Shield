"""Planner selection: the exact method when it will finish, greedy when not."""

import pytest

from qshield.experiments.generator import GeneratorConfig, make_instances
from qshield.model import AssetModel
from qshield.optimizer import MigrationProblem, exhaustive
from qshield.planner import MAX_EXACT_CANDIDATES, choose, plan


def problem_with(candidate_count, budget=1e6):
    assets = tuple(
        AssetModel(f"a{i}", "RSA-2048", 0.8, 0.8, 10, migration_cost=1.0)
        for i in range(candidate_count)
    )
    return MigrationProblem(assets, (), (), (), {"RSA-2048": "ML-KEM"}, budget)


def test_small_instances_use_the_exact_planner():
    report = choose(problem_with(5))
    assert report.planner == "exhaustive" and report.exact


def test_large_instances_downgrade_and_say_why():
    report = choose(problem_with(MAX_EXACT_CANDIDATES + 4))
    assert report.planner == "greedy_marginal"
    assert not report.exact
    # Approximation must never be silent.
    assert "not tractable" in report.reason
    assert "80.3%" in report.reason


def test_the_threshold_is_the_documented_one():
    assert choose(problem_with(MAX_EXACT_CANDIDATES)).exact
    assert not choose(problem_with(MAX_EXACT_CANDIDATES + 1)).exact


def test_the_exact_planner_can_be_forced():
    report = choose(problem_with(MAX_EXACT_CANDIDATES + 2), force="exhaustive")
    assert report.exact and "explicitly" in report.reason


def test_the_greedy_planner_can_be_forced():
    report = choose(problem_with(3), force="greedy")
    assert report.planner == "greedy_marginal" and not report.exact


def test_the_chosen_plan_respects_the_budget():
    for count in (4, MAX_EXACT_CANDIDATES + 3):
        problem = problem_with(count, budget=3.0)
        assert plan(problem).cost <= 3.0 + 1e-9


@pytest.mark.parametrize("seed", range(6))
def test_below_the_threshold_the_planner_returns_the_true_optimum(seed):
    problem = make_instances(1, seed + 5000, GeneratorConfig())[0]
    assert plan(problem).objective == pytest.approx(exhaustive(problem)[0].objective)


def test_the_report_records_how_the_plan_was_made():
    detail = choose(problem_with(4)).as_dict()
    assert detail["planner"] == "exhaustive"
    assert detail["exact"] is True
    assert detail["migration_candidates"] == 4
    assert detail["planner_reason"]
