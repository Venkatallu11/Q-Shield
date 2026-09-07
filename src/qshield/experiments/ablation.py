"""Does *path-awareness* help, or was 0.3 measuring exhaustive search?

0.3's suite compared a flat-priority heuristic against an exhaustive optimizer
and reported a 48-50% win rate with zero losses across 1000 instances. Zero
losses is the tell. The heuristic returns a budget-feasible subset; the
exhaustive planner minimises the objective over *all* budget-feasible subsets;
so the heuristic's plan is one of the candidates the optimizer already
considered. ``optimizer_objective <= heuristic_objective`` is a theorem about
subset enumeration. The experiment could not have produced a loss, which means
it could not have falsified anything.

Worse, the comparison confounds two separate claims:

    A. exhaustive search beats a greedy heuristic          (search power)
    B. modelling attack paths beats ranking assets alone   (path-awareness)

Only B is the paper's thesis, and 0.3's design cannot separate them.

This experiment separates them with a third arm and held-out metrics:

    flat_priority  graph-blind heuristic       -- neither A nor B
    node_only      exhaustive, w_path = 0      -- A without B
    path_aware     exhaustive, full weights    -- A and B

``node_only`` is the control. It has identical search power to ``path_aware``
and differs *only* in whether the objective it minimises contains the path term.
The ``path_aware - node_only`` contrast is therefore the effect of B alone.

Every arm is then scored on metrics none of them optimises, so any arm can lose:

    path_max              worst-case attack path; no arm targets it
    expected_objective    mean objective under parameter uncertainty; every arm
                          plans against the nominal point estimate

The nominal ``objective`` column is still reported, flagged, and excluded from
the conclusions for the arm that minimises it.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ..model import DEFAULT_WEIGHTS, NODE_ONLY_WEIGHTS, ObjectiveWeights
from ..optimizer import MigrationProblem, Plan, exhaustive, greedy_flat
from ..uncertainty import (
    PerturbationModel,
    paired_comparison,
    sample_worlds,
    summarize,
)
from .generator import GeneratorConfig, make_instances

ARMS = ("flat_priority", "node_only", "path_aware")

# Metrics no arm minimises. Conclusions rest on these.
HELD_OUT_METRICS = ("path_max", "expected_objective")
# Minimised directly by path_aware; reported for transparency, not for inference.
IN_OBJECTIVE_METRICS = ("objective",)


@dataclass
class InstanceRecord:
    plans: dict[str, Plan]
    metrics: dict[str, dict[str, float]]  # arm -> metric -> value
    paths: int
    budget: float


def _plan_arms(
    problem: MigrationProblem, weights: ObjectiveWeights
) -> dict[str, Plan]:
    """Build one plan per arm. Each arm sees the same instance and budget."""
    path_aware, _ = exhaustive(problem, weights=weights, strategy="path_aware")
    node_only, _ = exhaustive(problem, weights=NODE_ONLY_WEIGHTS, strategy="node_only")
    return {
        "flat_priority": greedy_flat(problem, weights=weights),
        "node_only": node_only,
        "path_aware": path_aware,
    }


def _score_arms(
    problem: MigrationProblem,
    plans: Mapping[str, Plan],
    *,
    weights: ObjectiveWeights,
    trials: int,
    seed: int,
    perturbation: PerturbationModel,
) -> dict[str, dict[str, float]]:
    """Score every arm on every metric, using one shared set of sampled worlds.

    Common random numbers: the same ``worlds`` list scores all three arms, so a
    per-instance difference between arms carries no Monte Carlo noise.
    """
    worlds = sample_worlds(
        problem.assets,
        problem.edges,
        trials=trials,
        seed=seed,
        weights=weights,
        model=perturbation,
    )
    out: dict[str, dict[str, float]] = {}
    for arm, plan in plans.items():
        nominal = problem.evaluate(plan.selected, weights=weights)
        expected = statistics.fmean(
            problem.evaluate_world(w, plan.selected, weights=weights).value
            for w in worlds
        ) if trials else nominal.value
        out[arm] = {
            "objective": nominal.value,
            "path_max": nominal.path_max,
            "expected_objective": expected,
            "cost": plan.cost,
        }
    return out


def run(
    *,
    instances: int = 300,
    seed: int = 20260907,
    trials: int = 150,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    perturbation: PerturbationModel = PerturbationModel(
        quantum_factor_spread=0.25, weight_spread=0.20
    ),
    config: GeneratorConfig = GeneratorConfig(),
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Run the ablation and return a self-describing report."""
    problems = make_instances(instances, seed, config)
    records: list[InstanceRecord] = []

    for index, problem in enumerate(problems):
        if not problem.candidates:
            continue
        plans = _plan_arms(problem, weights)
        metrics = _score_arms(
            problem,
            plans,
            weights=weights,
            trials=trials,
            # A distinct world-seed per instance, shared across arms within it.
            seed=seed + 1_000_003 * (index + 1),
            perturbation=perturbation,
        )
        records.append(
            InstanceRecord(
                plans=plans,
                metrics=metrics,
                paths=len(problem.path_set),
                budget=problem.budget,
            )
        )
        if progress:
            progress(index + 1, len(problems))

    return {
        "experiment": "ablation",
        "question": (
            "Is the benefit attributable to attack-path awareness, or to "
            "exhaustive search over migration plans?"
        ),
        "design": {
            "arms": {
                "flat_priority": "graph-blind greedy heuristic (no search, no paths)",
                "node_only": "exhaustive search minimising node risk only (search, no paths)",
                "path_aware": "exhaustive search minimising the full objective (search + paths)",
            },
            "controls_for": (
                "path_aware - node_only isolates path-awareness at equal search power; "
                "node_only - flat_priority isolates search power at equal path-blindness"
            ),
            "held_out_metrics": list(HELD_OUT_METRICS),
            "in_objective_metrics": list(IN_OBJECTIVE_METRICS),
            "instances": len(records),
            "monte_carlo_trials_per_instance": trials,
            "common_random_numbers": True,
            "weights": weights.as_tuple(),
            "perturbation": dict(perturbation.__dict__),
            "generator": config.as_dict(),
            "seed": seed,
        },
        "marginal_distributions": {
            arm: {
                metric: summarize([r.metrics[arm][metric] for r in records])
                for metric in (*IN_OBJECTIVE_METRICS, *HELD_OUT_METRICS, "cost")
            }
            for arm in ARMS
        },
        "contrasts": _contrasts(records),
        "structural_check": _structural_check(records),
        "plan_agreement": _plan_agreement(records),
    }


def _contrasts(records: Sequence[InstanceRecord]) -> dict[str, object]:
    """Paired per-instance differences for each arm pair and metric."""
    pairs = (
        ("node_only", "path_aware"),      # effect of path-awareness
        ("flat_priority", "node_only"),   # effect of search power
        ("flat_priority", "path_aware"),  # the 0.3 comparison, for reference
    )
    out: dict[str, object] = {}
    for a, b in pairs:
        per_metric: dict[str, object] = {}
        for metric in (*IN_OBJECTIVE_METRICS, *HELD_OUT_METRICS):
            deltas = [r.metrics[a][metric] - r.metrics[b][metric] for r in records]
            summary = paired_comparison(deltas, label_a=a, label_b=b)
            summary["metric"] = metric
            summary["held_out"] = metric in HELD_OUT_METRICS
            per_metric[metric] = summary
        # Cost is reported separately: an arm that spends more is not "better"
        # for free, so an objective gain must be read against its cost delta.
        per_metric["cost_delta"] = summarize(
            [r.metrics[a]["cost"] - r.metrics[b]["cost"] for r in records]
        )
        out[f"{a}_vs_{b}"] = per_metric
    return out


def _structural_check(records: Sequence[InstanceRecord]) -> dict[str, object]:
    """Empirical confirmation that the 0.3-style comparison cannot produce a loss.

    If this ever reports a violation, the exhaustive planner is not exhaustive or
    the objective is not deterministic — either way a bug, not a result.
    """
    violations = sum(
        1
        for r in records
        if r.metrics["path_aware"]["objective"] > r.metrics["flat_priority"]["objective"] + 1e-9
    )
    held_out_losses = {
        metric: sum(
            1
            for r in records
            if r.metrics["path_aware"][metric] > r.metrics["flat_priority"][metric] + 1e-9
        )
        for metric in HELD_OUT_METRICS
    }
    return {
        "claim": (
            "path_aware can never score worse than flat_priority on the objective "
            "it minimises, because the heuristic's plan is inside the exhaustive "
            "search space; a 'win rate' on that metric is not evidence"
        ),
        "instances": len(records),
        "in_objective_losses": violations,
        "in_objective_losses_are_possible": False,
        "held_out_losses_observed": held_out_losses,
        "held_out_losses_are_possible": True,
    }


def _plan_agreement(records: Sequence[InstanceRecord]) -> dict[str, object]:
    """How often the arms pick the same plan.

    A high agreement rate bounds how much any downstream difference can matter,
    and explains a large tie fraction without appealing to the model."""
    def agree(a: str, b: str) -> float:
        if not records:
            return 0.0
        same = sum(
            1 for r in records if set(r.plans[a].selected) == set(r.plans[b].selected)
        )
        return round(same / len(records), 6)

    return {
        "path_aware_equals_node_only": agree("path_aware", "node_only"),
        "path_aware_equals_flat_priority": agree("path_aware", "flat_priority"),
        "node_only_equals_flat_priority": agree("node_only", "flat_priority"),
    }
