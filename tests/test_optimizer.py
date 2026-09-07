"""Planner correctness — including the structural fact that invalidates 0.3's
headline result, asserted here so it cannot be forgotten again."""

import pytest

from qshield.experiments.generator import make_instances
from qshield.model import NODE_ONLY_WEIGHTS, AssetModel, ObjectiveWeights
from qshield.optimizer import (
    MigrationProblem,
    do_nothing,
    exhaustive,
    greedy_flat,
    greedy_marginal,
    pareto_frontier,
)
from qshield.paths import EdgeModel


def two_asset_problem(budget):
    return MigrationProblem(
        assets=(
            AssetModel("a", "RSA-2048", 0.9, 0.9, 15, migration_cost=2),
            AssetModel("b", "X25519", 0.9, 0.9, 15, migration_cost=2),
        ),
        edges=(),
        entrypoints=(),
        targets=(),
        replacements={"RSA-2048": "ML-KEM", "X25519": "ML-KEM"},
        budget=budget,
    )


def test_problem_rejects_duplicate_names_and_negative_budget():
    with pytest.raises(ValueError, match="unique"):
        MigrationProblem(
            (AssetModel("a", "RSA-2048"), AssetModel("a", "ECDSA")), (), (), (), {}, 1
        )
    with pytest.raises(ValueError, match="budget"):
        MigrationProblem((AssetModel("a", "RSA-2048"),), (), (), (), {}, -1)


def test_every_enumerated_plan_respects_the_budget():
    problem = two_asset_problem(2)
    _, rows = exhaustive(problem)
    assert rows and all(plan.cost <= 2 + 1e-9 for plan in rows)


def test_enumeration_includes_the_empty_plan():
    """0.3's optimizer started at r=1 and could not return 'do nothing', while
    the duplicate copy inside weight_sensitivity started at r=0."""
    plans = two_asset_problem(4).feasible_plans()
    assert () in [names for names, _ in plans]


def test_enumeration_is_the_full_feasible_power_set():
    plans = {names for names, _ in two_asset_problem(4).feasible_plans()}
    assert plans == {(), ("a",), ("b",), ("a", "b")}


@pytest.mark.parametrize("seed", range(10))
def test_exhaustive_is_optimal_over_all_feasible_plans(seed):
    problem = make_instances(1, seed + 300)[0]
    best, _ = exhaustive(problem)
    for names, _cost in problem.feasible_plans():
        assert best.objective <= problem.evaluate(names).value + 1e-9


@pytest.mark.parametrize("seed", range(10))
def test_heuristic_plans_lie_inside_the_exhaustive_search_space(seed):
    """This is *why* 0.3 recorded zero losses in 1000 instances. The heuristic
    returns a budget-feasible subset, and the exhaustive planner minimises over
    every budget-feasible subset, so it cannot score worse on that objective. A
    win rate measured this way is a theorem, not evidence."""
    problem = make_instances(1, seed + 400)[0]
    feasible = {names for names, _cost in problem.feasible_plans()}
    for heuristic in (greedy_flat, greedy_marginal, do_nothing):
        plan = heuristic(problem)
        assert plan.selected in feasible
        assert exhaustive(problem)[0].objective <= plan.objective + 1e-9


@pytest.mark.parametrize("seed", range(10))
def test_a_path_blind_planner_can_lose_on_the_full_objective(seed):
    """The converse of the above: the node-only arm optimises a different
    function, so it has no guarantee and the ablation can genuinely fail."""
    problem = make_instances(1, seed + 500)[0]
    node_only, _ = exhaustive(problem, weights=NODE_ONLY_WEIGHTS)
    path_aware, _ = exhaustive(problem)
    # Both selections must be re-scored under the *same* weighting: a plan's
    # recorded objective is the value of the function it was chosen with.
    assert path_aware.objective <= problem.evaluate(node_only.selected).value + 1e-9


def test_ties_are_broken_deterministically_by_cost_then_name():
    problem = two_asset_problem(4)
    first, rows_a = exhaustive(problem)
    second, rows_b = exhaustive(problem)
    assert first == second
    assert [r.selected for r in rows_a] == [r.selected for r in rows_b]
    costs = [r.cost for r in rows_a]
    objectives = [r.objective for r in rows_a]
    assert objectives == sorted(objectives)
    for i in range(len(rows_a) - 1):
        if objectives[i] == pytest.approx(objectives[i + 1]):
            assert costs[i] <= costs[i + 1]


def test_greedy_flat_is_budget_feasible_and_graph_blind():
    problem = make_instances(1, 11)[0]
    plan = greedy_flat(problem)
    assert plan.cost <= problem.budget + 1e-9
    # Ranking uses only per-asset properties, so rewiring the graph must not
    # change the selection.
    rewired = MigrationProblem(
        problem.assets,
        (EdgeModel("entry", "target", 0.5),),
        problem.entrypoints,
        problem.targets,
        problem.replacements,
        problem.budget,
    )
    assert greedy_flat(rewired).selected == plan.selected


def test_greedy_marginal_respects_the_budget_and_improves_on_nothing():
    problem = make_instances(1, 13)[0]
    plan = greedy_marginal(problem)
    assert plan.cost <= problem.budget + 1e-9
    assert plan.objective <= do_nothing(problem).objective + 1e-9


def test_do_nothing_is_the_zero_cost_reference():
    plan = do_nothing(make_instances(1, 17)[0])
    assert plan.selected == () and plan.cost == 0 and plan.improvement == 0


def test_improvement_is_measured_against_the_null_plan():
    problem = two_asset_problem(4)
    best, _ = exhaustive(problem)
    assert best.improvement == pytest.approx(do_nothing(problem).objective - best.objective)


@pytest.mark.parametrize("seed", range(8))
def test_pareto_frontier_is_non_dominated_and_cheapest_first(seed):
    problem = make_instances(1, seed + 600)[0]
    _, rows = exhaustive(problem)
    frontier = pareto_frontier(rows)
    assert frontier
    costs = [p.cost for p in frontier]
    objectives = [p.objective for p in frontier]
    assert costs == sorted(costs)
    assert objectives == sorted(objectives, reverse=True)
    # No enumerated plan may dominate a frontier point on both axes.
    for point in frontier:
        for other in rows:
            dominates = (
                other.cost <= point.cost - 1e-9 and other.objective <= point.objective - 1e-9
            )
            assert not dominates


def test_evaluate_accepts_an_overridden_inventory():
    """Monte Carlo scores a fixed plan against perturbed assets."""
    problem = two_asset_problem(4)
    quieter = tuple(
        AssetModel(a.name, a.algorithm, 0.1, 0.1, 1, migration_cost=a.migration_cost)
        for a in problem.assets
    )
    assert problem.evaluate((), assets=quieter).value < problem.evaluate(()).value


def test_weights_flow_through_to_the_selection():
    """The two 0.3 optimizers disagreed on whether weights were even accepted."""
    problem = make_instances(1, 23)[0]
    a, _ = exhaustive(problem, weights=ObjectiveWeights(1.0, 0.0, 0.0))
    b, _ = exhaustive(problem, weights=ObjectiveWeights(0.0, 1.0, 0.0))
    assert a.objective != pytest.approx(b.objective)


def test_uncontrollable_assets_are_not_decision_variables():
    """An individual cannot migrate their bank's key exchange. Pricing that out
    with a huge migration_cost works only until the budget grows."""
    assets = (
        AssetModel("mine", "RSA-2048", migration_cost=1.0, controllable=True),
        AssetModel("theirs", "RSA-2048", migration_cost=1.0, controllable=False),
    )
    problem = MigrationProblem(assets, (), (), (), {"RSA-2048": "ML-KEM"}, budget=100)
    assert [a.name for a in problem.candidates] == ["mine"]
    assert [a.name for a in problem.uncontrollable] == ["theirs"]
    # Even with a budget that could afford everything, no plan names it.
    assert all("theirs" not in names for names, _ in problem.feasible_plans())


def test_uncontrollable_risk_is_still_scored():
    """Not actionable is not the same as not there; a plan that ignored it would
    look better than the situation warrants."""
    assets = (AssetModel("theirs", "RSA-2048", 0.9, 0.9, 20, controllable=False),)
    problem = MigrationProblem(assets, (), (), (), {"RSA-2048": "ML-KEM"}, budget=100)
    assert problem.evaluate(()).value > 0
