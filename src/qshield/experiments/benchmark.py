"""Single-case benchmark: full report for one named system.

Differences from 0.3's ``benchmark.run_benchmark``:

* All strategies are scored against **one** shared set of Monte Carlo worlds.
  0.3 used seeds 11, 12 and 13 for the baseline, the heuristic and the optimizer
  respectively, so the three percentile bands it printed side by side were
  computed on different random draws and their spread was not a strategy effect.
  Paired differences are reported here because that is the comparable quantity.
* The candidate list is a cost/objective Pareto frontier rather than the ten
  lowest-scoring plans, which on a binding budget are mostly the same plan with
  one asset swapped.
* Every report embeds the algorithm registry, weights and perturbation model it
  was produced with, so a number can be traced to the assumptions behind it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .. import __version__
from ..algorithms import registry_snapshot
from ..model import DEFAULT_WEIGHTS, AssetModel, ObjectiveWeights
from ..optimizer import (
    MigrationProblem,
    Plan,
    do_nothing,
    exhaustive,
    greedy_flat,
    greedy_marginal,
    pareto_frontier,
)
from ..paths import EdgeKind, EdgeModel
from ..threat import ThreatClass
from ..uncertainty import (
    PerturbationModel,
    paired_comparison,
    sample_worlds,
    summarize,
)

REQUIRED_KEYS = ("assets", "edges", "entrypoints", "targets", "replacements", "budget")


def load_case(path: str | Path) -> MigrationProblem:
    """Read a case file, validating it rather than failing deep in the model."""
    return problem_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def problem_from_dict(data: Mapping[str, Any]) -> MigrationProblem:
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"case is missing required keys: {missing}")
    assets = tuple(_asset(a) for a in data["assets"])
    edges = tuple(_edge(e) for e in data["edges"])
    names = {a.name for a in assets}
    dangling = {
        n
        for e in edges
        for n in (e.source, e.target)
        if n not in names
    }
    if dangling:
        raise ValueError(f"edges reference assets not in the inventory: {sorted(dangling)}")
    for role in ("entrypoints", "targets"):
        unknown = set(data[role]) - names
        if unknown:
            raise ValueError(f"{role} reference unknown assets: {sorted(unknown)}")
    return MigrationProblem(
        assets=assets,
        edges=edges,
        entrypoints=tuple(data["entrypoints"]),
        targets=tuple(data["targets"]),
        replacements=dict(data["replacements"]),
        budget=float(data["budget"]),
    )


def _asset(raw: Mapping[str, Any]) -> AssetModel:
    """Build an asset, resolving the threat class from its string form.

    Keys beginning with an underscore are treated as documentation, so a case
    file can carry per-asset commentary without the loader rejecting it.
    """
    fields = {k: v for k, v in raw.items() if not k.startswith("_")}
    if isinstance(fields.get("threat_class"), str):
        fields["threat_class"] = ThreatClass(fields["threat_class"])
    return AssetModel(**fields)


def _edge(raw: Mapping[str, Any]) -> EdgeModel:
    fields = {k: v for k, v in raw.items() if not k.startswith("_")}
    if isinstance(fields.get("kind"), str):
        fields["kind"] = EdgeKind(fields["kind"])
    return EdgeModel(**fields)


def run(
    problem: MigrationProblem,
    *,
    trials: int = 3000,
    seed: int = 20260907,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    perturbation: PerturbationModel = PerturbationModel(
        quantum_factor_spread=0.25, weight_spread=0.20
    ),
) -> dict[str, object]:
    best, all_plans = exhaustive(problem, weights=weights, strategy="path_aware")
    plans: dict[str, Plan] = {
        "do_nothing": do_nothing(problem, weights=weights),
        "flat_priority": greedy_flat(problem, weights=weights),
        "greedy_marginal": greedy_marginal(problem, weights=weights),
        "path_aware": best,
    }

    baseline = problem.evaluate((), weights=weights)

    # One shared world set: differences between strategies are then noise-free.
    worlds = sample_worlds(
        problem.assets,
        problem.edges,
        trials=trials,
        seed=seed,
        weights=weights,
        model=perturbation,
    )
    sampled: dict[str, list[float]] = {
        name: [problem.evaluate_world(w, plan.selected, weights=weights).value for w in worlds]
        for name, plan in plans.items()
    }

    comparisons = {}
    for arm in ("flat_priority", "greedy_marginal"):
        deltas = [x - y for x, y in zip(sampled[arm], sampled["path_aware"], strict=True)]
        comparisons[f"{arm}_vs_path_aware"] = {
            **paired_comparison(deltas, label_a=arm, label_b="path_aware"),
            "note": (
                "paired over shared Monte Carlo worlds; positive means path_aware "
                "scored lower (better) in that world"
            ),
        }

    return {
        "experiment": "benchmark",
        "qshield_version": __version__,
        "instance": {
            "assets": len(problem.assets),
            "edges": len(problem.edges),
            "entrypoints": list(problem.entrypoints),
            "targets": list(problem.targets),
            "attack_paths": len(problem.path_set),
            "migration_candidates": [a.name for a in problem.candidates],
            "budget": problem.budget,
            "feasible_plans": len(problem.feasible_plans()),
        },
        "assumptions": {
            "weights": weights.as_tuple(),
            "monte_carlo_trials": trials,
            "seed": seed,
            "common_random_numbers": True,
            "perturbation": dict(perturbation.__dict__),
            "algorithm_registry": registry_snapshot(),
        },
        "baseline": baseline.as_dict(),
        "plans": {name: plan.as_dict() for name, plan in plans.items()},
        "uncertainty": {name: summarize(values) for name, values in sampled.items()},
        "paired_comparisons": comparisons,
        "pareto_frontier": [p.as_dict() for p in pareto_frontier(all_plans)],
        "caveats": [
            "Quantum factors and objective weights are uncalibrated hypotheses; a "
            "lower objective does not establish that a system is safer in reality.",
            "path_aware minimises the reported objective over every budget-feasible "
            "plan, so it cannot score worse than any other strategy on that metric. "
            "See qshield.experiments.ablation for comparisons on held-out metrics.",
            "Path risk treats hops as independent, which real infrastructure is not.",
        ],
    }
