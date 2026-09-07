"""Migration planners: what to migrate, given a budget.

0.3 contained two exhaustive optimizers — ``optimizer.enumerate_migrations`` and
a copy inlined in ``weight_sensitivity.run`` — that had already drifted: the
first started at ``r=1`` and so could not return the empty plan, the second
started at ``r=0``; only the second accepted objective weights. This module is
the single implementation. Every planner returns a :class:`Plan`, so experiments
can treat them interchangeably.

A note the 0.3 benchmark did not make, and which its headline number depends on:
:func:`exhaustive` minimises the objective over *every* budget-feasible subset,
and each heuristic baseline returns a budget-feasible subset. The heuristic's
plan is therefore inside the exhaustive search space, so the exhaustive plan can
never score worse **on the objective it minimises**. A "win rate" measured that
way is a theorem about subset enumeration, not evidence about migration
strategy. ``qshield.experiments.ablation`` exists to measure the part that is
actually contingent; see docs/METHODOLOGY.md.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations

from .model import (
    DEFAULT_WEIGHTS,
    AssetModel,
    ObjectiveWeights,
    apply_migration,
    objective,
)
from .paths import EdgeModel, PathSet, enumerate_paths
from .uncertainty import World

__all__ = [
    "Plan",
    "MigrationProblem",
    "exhaustive",
    "greedy_flat",
    "greedy_marginal",
    "do_nothing",
    "pareto_frontier",
]


@dataclass(frozen=True)
class Plan:
    """A migration decision: which assets to move, at what cost and score."""

    selected: tuple[str, ...]
    cost: float
    objective: float
    strategy: str
    improvement: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "selected": list(self.selected),
            "cost": round(self.cost, 6),
            "objective": round(self.objective, 6),
            "improvement": round(self.improvement, 6),
            "improvement_per_cost": (
                round(self.improvement / self.cost, 6) if self.cost > 0 else None
            ),
        }


@dataclass
class MigrationProblem:
    """An instance: inventory, graph, replacement map and budget.

    Enumerates the attack paths once on construction. Migration never changes the
    graph, so every plan evaluation below reuses this single :class:`PathSet`
    instead of re-running a DFS — the dominant cost in 0.3's optimizer, which
    repeated it once per candidate subset.
    """

    assets: tuple[AssetModel, ...]
    edges: tuple[EdgeModel, ...]
    entrypoints: tuple[str, ...]
    targets: tuple[str, ...]
    replacements: Mapping[str, str]
    budget: float
    strict: bool = True
    path_set: PathSet = field(init=False)

    def __post_init__(self) -> None:
        self.assets = tuple(self.assets)
        self.edges = tuple(self.edges)
        self.entrypoints = tuple(self.entrypoints)
        self.targets = tuple(self.targets)
        if self.budget < 0:
            raise ValueError("budget must be non-negative")
        names = [a.name for a in self.assets]
        if len(set(names)) != len(names):
            raise ValueError("asset names must be unique")
        self.path_set = enumerate_paths(self.edges, self.entrypoints, self.targets)

    @property
    def candidates(self) -> tuple[AssetModel, ...]:
        """The decision variables: assets that have a replacement *and* are ours
        to change. Risk carried by assets outside our control is still scored —
        it is real — but it is not something a plan can spend budget on."""
        return tuple(
            a
            for a in self.assets
            if a.algorithm in self.replacements and a.controllable
        )

    @property
    def uncontrollable(self) -> tuple[AssetModel, ...]:
        """Assets with a replacement that we nonetheless cannot act on.

        Reported separately because the gap between "risk present" and "risk
        addressable" is the whole story for an individual, and a plan that
        silently ignores it looks better than the situation warrants."""
        return tuple(
            a
            for a in self.assets
            if a.algorithm in self.replacements and not a.controllable
        )

    def evaluate(
        self,
        selected: Sequence[str] | frozenset[str] = (),
        *,
        weights: ObjectiveWeights = DEFAULT_WEIGHTS,
        assets: Sequence[AssetModel] | None = None,
    ):
        """Score a plan. ``assets`` overrides the inventory (used by Monte Carlo,
        which perturbs parameters while holding the graph and plan fixed)."""
        base = self.assets if assets is None else assets
        migrated = apply_migration(base, selected, self.replacements)
        return objective(
            migrated,
            self.edges,
            self.entrypoints,
            self.targets,
            weights=weights,
            path_set=self.path_set,
            strict=self.strict,
        )

    def evaluate_world(
        self,
        world: World,
        selected: Sequence[str] | frozenset[str] = (),
        *,
        weights: ObjectiveWeights | None = None,
    ):
        """Score a plan inside a sampled world.

        The world supplies perturbed assets, edge reliabilities and (optionally)
        objective weights; the plan and graph topology come from the problem.
        Because every strategy is scored against the same ``world``, differences
        between strategies contain no sampling noise.
        """
        migrated = apply_migration(world.assets, selected, self.replacements)
        return objective(
            migrated,
            world.edges,
            self.entrypoints,
            self.targets,
            weights=weights if weights is not None else world.weights,
            qf_overrides=world.overrides_for(migrated),
            path_set=self.path_set.with_reliability(world.edges),
            strict=self.strict,
        )

    def cost_of(self, selected: Sequence[str] | frozenset[str]) -> float:
        chosen = frozenset(selected)
        return sum(a.migration_cost for a in self.assets if a.name in chosen)

    def feasible_plans(self) -> list[tuple[tuple[str, ...], float]]:
        """Every budget-feasible subset of the candidates, including the empty plan.

        Exponential by construction: 2^|candidates|. Callers should keep instances
        small — this is an exact reference method, not a production planner.
        """
        cands = self.candidates
        out: list[tuple[tuple[str, ...], float]] = []
        for r in range(len(cands) + 1):
            for combo in combinations(cands, r):
                cost = sum(a.migration_cost for a in combo)
                if cost <= self.budget + 1e-12:
                    out.append((tuple(sorted(a.name for a in combo)), cost))
        return out


def exhaustive(
    problem: MigrationProblem,
    *,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    strategy: str = "exhaustive",
) -> tuple[Plan, list[Plan]]:
    """Exact minimiser of ``weights``-weighted objective subject to the budget.

    Returns the best plan and the full ranked candidate list. Ties on objective
    are broken by cost, then lexicographically by asset name, so the result is
    deterministic and independent of enumeration order.
    """
    baseline = problem.evaluate((), weights=weights).value
    rows: list[Plan] = []
    for names, cost in problem.feasible_plans():
        value = problem.evaluate(names, weights=weights).value
        rows.append(
            Plan(
                selected=names,
                cost=cost,
                objective=value,
                strategy=strategy,
                improvement=baseline - value,
            )
        )
    rows.sort(key=lambda p: (p.objective, p.cost, p.selected))
    return rows[0], rows


def greedy_flat(
    problem: MigrationProblem, *, weights: ObjectiveWeights = DEFAULT_WEIGHTS
) -> Plan:
    """Flat asset-priority baseline: rank by intrinsic asset importance, take
    what fits (first-fit decreasing).

    This is the strategy the research question tests against — it is graph-blind
    by design, using only per-asset properties an inventory tool would have
    without any dependency analysis.
    """
    ranked = sorted(
        problem.candidates,
        key=lambda a: (
            a.exposure * a.sensitivity * a.data_lifetime_years,
            a.name,
        ),
        reverse=True,
    )
    chosen: list[str] = []
    spent = 0.0
    for a in ranked:
        if spent + a.migration_cost <= problem.budget + 1e-12:
            chosen.append(a.name)
            spent += a.migration_cost
    return _finish(problem, chosen, spent, weights, "flat_priority")


def greedy_marginal(
    problem: MigrationProblem, *, weights: ObjectiveWeights = DEFAULT_WEIGHTS
) -> Plan:
    """Cost-benefit greedy: repeatedly take the migration with the best objective
    reduction per unit cost.

    A tractable stand-in for :func:`exhaustive` on instances too large for 2^n
    enumeration, and a fairer baseline than :func:`greedy_flat` because it does
    see the graph. Unlike the exhaustive planner it has no optimality guarantee,
    so it can genuinely lose.
    """
    chosen: list[str] = []
    spent = 0.0
    remaining = {a.name: a for a in problem.candidates}
    current = problem.evaluate(chosen, weights=weights).value
    while remaining:
        best_name, best_gain, best_cost = None, 0.0, 0.0
        for name, asset in sorted(remaining.items()):
            if spent + asset.migration_cost > problem.budget + 1e-12:
                continue
            value = problem.evaluate(chosen + [name], weights=weights).value
            # A zero-cost migration cannot be ranked by ratio; use absolute gain.
            delta = current - value
            gain = delta / asset.migration_cost if asset.migration_cost > 0 else delta
            if gain > best_gain + 1e-12:
                best_name, best_gain, best_cost = name, gain, asset.migration_cost
        if best_name is None:
            break
        chosen.append(best_name)
        spent += best_cost
        current = problem.evaluate(chosen, weights=weights).value
        del remaining[best_name]
    return _finish(problem, chosen, spent, weights, "greedy_marginal")


def do_nothing(
    problem: MigrationProblem, *, weights: ObjectiveWeights = DEFAULT_WEIGHTS
) -> Plan:
    """The null plan — the reference every improvement is measured against."""
    return _finish(problem, [], 0.0, weights, "do_nothing")


def pareto_frontier(plans: Sequence[Plan]) -> list[Plan]:
    """Cost/objective non-dominated set, cheapest first.

    The research question asks whether path-awareness reduces exposure *more
    efficiently*, which is a statement about this frontier. 0.3 reported the ten
    lowest-objective plans instead, which is a different and mostly redundant set
    — the top rows all sit at the budget ceiling.
    """
    ordered = sorted(plans, key=lambda p: (p.cost, p.objective))
    frontier: list[Plan] = []
    best = float("inf")
    for plan in ordered:
        if plan.objective < best - 1e-12:
            frontier.append(plan)
            best = plan.objective
    return frontier


def _finish(
    problem: MigrationProblem,
    chosen: Sequence[str],
    cost: float,
    weights: ObjectiveWeights,
    strategy: str,
) -> Plan:
    names = tuple(sorted(chosen))
    baseline = problem.evaluate((), weights=weights).value
    value = problem.evaluate(names, weights=weights).value
    return Plan(
        selected=names,
        cost=cost,
        objective=value,
        strategy=strategy,
        improvement=baseline - value,
    )
