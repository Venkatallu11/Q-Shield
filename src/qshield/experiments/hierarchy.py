"""The crypto-agility trap: how much of a migration plan buys nothing?

A certificate is no more trustworthy than the authority that issued it. Migrating
leaf certificates to ML-DSA while their issuing CA still signs with ECDSA leaves
every one of those leaves forgeable through the CA. The spend is real; the risk
reduction is zero.

Every Q-SHIELD version through 0.4 had a single edge type and could not express
this. A planner using that model, given a PKI, sees cheap high-exposure leaves
and expensive low-exposure roots, and buys leaves.

The comparison is deliberately **not** framed as "which planner wins". The
hierarchy-aware planner minimises the objective it is scored on and every rival
plan is feasible for it, so winning is structurally guaranteed and says nothing —
the same defect that invalidated 0.3's headline result. What is measured instead
is the size of the modelling error, which is not guaranteed and could come out
zero:

``claimed_improvement``
    the risk reduction a hierarchy-blind planner believes its plan achieved,
    computed in its own model.
``actual_improvement``
    what that same plan achieves once trust inheritance is accounted for.
``illusory_fraction``
    ``1 - actual / claimed``. The share of the plan's benefit that does not exist.
``wasted_spend``
    budget spent on assets whose migration changed the objective by nothing at
    all, because their effective risk stayed pinned to an unmigrated issuer.

If delegation modelling does not matter — if roots are cheap enough that every
planner buys them anyway — these come out near zero and the refinement is not
worth its complexity.

Two blind arms, because a pre-0.5 analyst had two ways to encode a hierarchy with
one edge type: drop the trust relations, or draw them as ordinary dependencies.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from ..model import DEFAULT_WEIGHTS, ObjectiveWeights
from ..optimizer import MigrationProblem, exhaustive, greedy_flat
from ..uncertainty import paired_comparison, summarize
from .pki import PKIConfig, delegation_as_dependency, make_instances, without_delegation

ARMS = ("flat_priority", "blind_omits_trust", "blind_trust_as_dependency", "hierarchy_aware")


def run(
    *,
    instances: int = 300,
    seed: int = 20260907,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    config: PKIConfig = PKIConfig(),
) -> dict[str, object]:
    problems = [p for p in make_instances(instances, seed, config) if p.candidates]

    records: list[dict[str, object]] = []
    for problem in problems:
        # Each blind arm plans in its own (wrong) view of the same instance.
        omits = without_delegation(problem)
        as_dep = delegation_as_dependency(problem)
        plans = {
            "flat_priority": greedy_flat(omits, weights=weights).selected,
            "blind_omits_trust": exhaustive(omits, weights=weights)[0].selected,
            "blind_trust_as_dependency": exhaustive(as_dep, weights=weights)[0].selected,
            "hierarchy_aware": exhaustive(problem, weights=weights)[0].selected,
        }
        # ...but every plan is judged in the true model.
        truth_baseline = problem.evaluate((), weights=weights).value
        record: dict[str, object] = {"plans": plans, "detail": {}}
        for arm, plan in plans.items():
            actual = truth_baseline - problem.evaluate(plan, weights=weights).value
            blind_view = omits if arm != "blind_trust_as_dependency" else as_dep
            claimed = (
                blind_view.evaluate((), weights=weights).value
                - blind_view.evaluate(plan, weights=weights).value
            )
            record["detail"][arm] = {
                "cost": problem.cost_of(plan),
                "actual_improvement": actual,
                "claimed_improvement": claimed,
                "illusory_fraction": _illusory(claimed, actual),
                "wasted_spend": _wasted_spend(problem, plan, weights),
            }
        records.append(record)

    return {
        "experiment": "hierarchy",
        "question": (
            "In a trust hierarchy, how much of a plan chosen without modelling "
            "delegation actually reduces risk?"
        ),
        "design": {
            "instances": len(records),
            "arms": {
                "flat_priority": "graph-blind heuristic",
                "blind_omits_trust": "exhaustive search, trust edges dropped",
                "blind_trust_as_dependency": "exhaustive search, trust edges drawn as dependencies",
                "hierarchy_aware": "exhaustive search over the true model",
            },
            "scoring": "every plan is evaluated in the true (delegation-aware) model",
            "not_a_win_rate": (
                "hierarchy_aware minimises the metric it is scored on and every "
                "rival plan is feasible for it, so it cannot lose; the reported "
                "quantities are the size of the modelling error, which is not "
                "structurally determined"
            ),
            "weights": weights.as_tuple(),
            "seed": seed,
            "generator": config.as_dict(),
        },
        "per_arm": {
            arm: {
                metric: summarize([r["detail"][arm][metric] for r in records])
                for metric in (
                    "actual_improvement",
                    "claimed_improvement",
                    "illusory_fraction",
                    "wasted_spend",
                    "cost",
                )
            }
            for arm in ARMS
        },
        "wasted_spend_share": {
            arm: _share(records, arm) for arm in ARMS
        },
        "plan_agreement_with_hierarchy_aware": {
            arm: round(
                sum(
                    1
                    for r in records
                    if set(r["plans"][arm]) == set(r["plans"]["hierarchy_aware"])
                )
                / len(records),
                6,
            )
            for arm in ARMS
        },
        "contrasts": {
            f"{arm}_vs_hierarchy_aware": paired_comparison(
                [
                    r["detail"]["hierarchy_aware"]["actual_improvement"]
                    - r["detail"][arm]["actual_improvement"]
                    for r in records
                ],
                label_a=arm,
                label_b="hierarchy_aware",
                # Positive delta here means hierarchy_aware improved risk MORE.
            )
            for arm in ARMS
            if arm != "hierarchy_aware"
        },
        "note_on_contrast_sign": (
            "deltas are hierarchy_aware improvement minus the arm's improvement, so "
            "a positive mean means the blind arm achieved less real risk reduction"
        ),
    }


def _illusory(claimed: float, actual: float) -> float:
    """Share of believed benefit that does not survive the true model.

    Undefined when a plan claims nothing; reported as 0 rather than dropped, so
    the denominator of every summary is the full instance count.
    """
    if claimed <= 1e-9:
        return 0.0
    return max(0.0, min(1.0, 1.0 - actual / claimed))


def _wasted_spend(
    problem: MigrationProblem, plan: Sequence[str], weights: ObjectiveWeights
) -> float:
    """Budget spent on assets that changed the true objective by nothing.

    Measured by leave-one-out: an asset is wasted if removing it from the plan
    leaves the objective unchanged. That is the operationally meaningful
    question — 'could I have skipped this and been no worse off'.
    """
    if not plan:
        return 0.0
    full = problem.evaluate(plan, weights=weights).value
    wasted = 0.0
    for name in plan:
        without = [n for n in plan if n != name]
        if abs(problem.evaluate(without, weights=weights).value - full) <= 1e-9:
            wasted += problem.cost_of([name])
    return wasted


def _share(records: Sequence[dict[str, object]], arm: str) -> dict[str, float]:
    spent = [r["detail"][arm]["cost"] for r in records]
    wasted = [r["detail"][arm]["wasted_spend"] for r in records]
    total_spent = sum(spent)
    fully_wasted = sum(
        1 for s, w in zip(spent, wasted, strict=True) if s > 0 and w >= s - 1e-9
    )
    return {
        "wasted_share_of_total_budget": round(sum(wasted) / total_spent, 6)
        if total_spent > 0
        else 0.0,
        "mean_wasted_share_per_instance": round(
            statistics.fmean(
                [w / s if s > 0 else 0.0 for s, w in zip(spent, wasted, strict=True)]
            ),
            6,
        ),
        "instances_where_entire_spend_was_wasted": fully_wasted,
        "instances": len(records),
    }


def sweep_root_cost(
    *,
    instances: int = 150,
    seed: int = 20260907,
    root_costs: Sequence[float] = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0),
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    base_config: PKIConfig = PKIConfig(),
) -> dict[str, object]:
    """Does the trap survive making trust anchors cheap to migrate?

    The headline numbers depend on the generator pricing a root at several times
    a leaf, which is realistic but is an assumption. If roots are cheap, a
    hierarchy-blind planner may buy them anyway for their own node risk, and the
    modelling refinement stops mattering. This sweeps the price and reports where
    the effect disappears -- the honest bound on the claim.
    """
    from dataclasses import replace as _replace

    rows: list[dict[str, object]] = []
    for cost in root_costs:
        config = _replace(base_config, root_cost=cost)
        report = run(instances=instances, seed=seed, weights=weights, config=config)
        rows.append(
            {
                "root_cost": cost,
                "root_cost_in_leaf_units": round(cost / base_config.leaf_cost, 3),
                "blind_illusory_fraction": report["per_arm"]["blind_omits_trust"][
                    "illusory_fraction"
                ]["mean"],
                "blind_wasted_share": report["wasted_spend_share"]["blind_omits_trust"][
                    "wasted_share_of_total_budget"
                ],
                "blind_actual_improvement": report["per_arm"]["blind_omits_trust"][
                    "actual_improvement"
                ]["mean"],
                "aware_actual_improvement": report["per_arm"]["hierarchy_aware"][
                    "actual_improvement"
                ]["mean"],
                "plan_agreement": report["plan_agreement_with_hierarchy_aware"][
                    "blind_omits_trust"
                ],
            }
        )
    return {
        "experiment": "hierarchy_root_cost_sweep",
        "question": (
            "Does the crypto-agility trap survive when trust anchors are cheap to "
            "migrate, or is it an artifact of the generator pricing roots high?"
        ),
        "design": {
            "instances_per_point": instances,
            "root_costs": list(root_costs),
            "leaf_cost": base_config.leaf_cost,
            "budget_range": [base_config.budget_low, base_config.budget_high],
            "seed": seed,
        },
        "sweep": rows,
    }
