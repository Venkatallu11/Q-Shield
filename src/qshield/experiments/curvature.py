"""Is the risk-reduction function submodular? And does the answer explain greedy?

A reviewer suggested testing for submodularity, hoping it would hold: a monotone
submodular objective gives greedy maximisation the classic ``1 - 1/e``
approximation guarantee, which would upgrade "greedy matched the optimum on 80.3%
of our instances" into a proved bound.

It does not hold, and the direction of failure matters. Under trust delegation
the objective is **supermodular** in exactly the region the model exists to
reason about: migrating a leaf certificate alone reduces risk by zero, because
its risk is pinned to an unmigrated issuer, while migrating the same leaf *after*
its root becomes worthwhile. Marginal benefit **grows** with the set instead of
shrinking. That is increasing returns, and it is the precise opposite of what a
greedy guarantee needs.

So this experiment reports a negative theoretical result: no approximation bound
of that family is available for Q-SHIELD's objective on hierarchical estates. It
then does the thing that makes the negative result useful — checks whether the
violations *predict* where greedy actually loses. If they do, the 19.7% of
instances where greedy missed the optimum are not bad luck; they are the
supermodular ones, and they can be identified in advance.

Definitions, on the benefit function ``B(S) = R(empty) - R(S)``, which is what a
planner maximises:

* submodular: ``B(S + x) - B(S) >= B(T + x) - B(T)`` for every ``S`` inside ``T``
* supermodular: the same with the inequality reversed

Measured, the objective is supermodular on every sampled pair in both suites --
with and without delegation. The cause is more general than delegation: path risk
composes as a noisy-OR, ``risk = 1 - prod(1 - p_v)``, so the benefit of migrating
one node is proportional to ``prod(1 - p_v)`` over the *other* nodes on the path.
That factor grows as those nodes are migrated, and the effect compounds with path
length -- roughly 2.4x at length two, 35x at length five. Delegation is simply the
limiting case, where a leaf's benefit is exactly zero until its anchor moves.

Migration actions are therefore complements, not substitutes.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from ..model import DEFAULT_WEIGHTS, ObjectiveWeights
from ..optimizer import MigrationProblem, exhaustive, greedy_marginal
from ..uncertainty import bootstrap_ci, summarize
from .generator import make_instances as generic_instances
from .pki import PKIConfig
from .pki import make_instances as pki_instances

TOLERANCE = 1e-9


@dataclass
class CurvatureSample:
    """One (S, T, x) triple with S inside T and x outside T."""

    marginal_small: float   # B(S + x) - B(S)
    marginal_large: float   # B(T + x) - B(T)

    @property
    def violates_submodularity(self) -> bool:
        return self.marginal_large > self.marginal_small + TOLERANCE

    @property
    def violates_supermodularity(self) -> bool:
        return self.marginal_small > self.marginal_large + TOLERANCE


def sample_curvature(
    problem: MigrationProblem,
    rng: random.Random,
    *,
    samples: int = 40,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
) -> list[CurvatureSample]:
    """Draw nested set pairs and measure the marginal benefit of one more asset.

    Budget is ignored here on purpose: submodularity is a property of the
    objective, not of the feasible region, and testing it only inside the budget
    would confound the two.
    """
    names = [a.name for a in problem.candidates]
    if len(names) < 3:
        return []

    def risk(subset: Sequence[str]) -> float:
        return problem.evaluate(subset, weights=weights).value

    out: list[CurvatureSample] = []
    for _ in range(samples):
        pool = list(names)
        rng.shuffle(pool)
        x = pool.pop()
        cut_small = rng.randint(0, max(0, len(pool) - 1))
        cut_large = rng.randint(cut_small, len(pool))
        small = pool[:cut_small]
        large = pool[:cut_large]  # a superset of `small` by construction
        out.append(
            CurvatureSample(
                marginal_small=risk(small) - risk([*small, x]),
                marginal_large=risk(large) - risk([*large, x]),
            )
        )
    return out


def run(
    *,
    instances: int = 200,
    samples_per_instance: int = 40,
    seed: int = 20260907,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    pki_config: PKIConfig = PKIConfig(),
) -> dict[str, object]:
    suites = {
        "pki_with_delegation": pki_instances(instances, seed, pki_config),
        "generic_no_delegation": generic_instances(instances, seed),
    }

    results: dict[str, object] = {}
    for suite, problems in suites.items():
        rng = random.Random(seed)
        problems = [p for p in problems if len(p.candidates) >= 3]
        sub_violations: list[float] = []
        super_violations: list[float] = []
        instance_rows: list[dict[str, float]] = []

        for problem in problems:
            drawn = sample_curvature(
                problem, rng, samples=samples_per_instance, weights=weights
            )
            if not drawn:
                continue
            sub_rate = sum(s.violates_submodularity for s in drawn) / len(drawn)
            super_rate = sum(s.violates_supermodularity for s in drawn) / len(drawn)
            sub_violations.append(sub_rate)
            super_violations.append(super_rate)

            best, _ = exhaustive(problem, weights=weights)
            greedy = greedy_marginal(problem, weights=weights)
            instance_rows.append({
                "submodularity_violation_rate": sub_rate,
                "greedy_gap": greedy.objective - best.objective,
                "greedy_optimal": float(
                    greedy.objective <= best.objective + TOLERANCE
                ),
            })

        results[suite] = {
            "instances": len(instance_rows),
            "submodularity_violation_rate": summarize(sub_violations),
            "supermodularity_violation_rate": summarize(super_violations),
            "instances_with_any_submodularity_violation": sum(
                1 for r in sub_violations if r > 0
            ),
            "greedy_predicted_by_curvature": _predictiveness(instance_rows),
        }

    return {
        "experiment": "curvature",
        "question": (
            "Is the risk-reduction function submodular, and if not, does the "
            "failure explain where greedy planning loses?"
        ),
        "design": {
            "instances_per_suite": instances,
            "nested_pairs_per_instance": samples_per_instance,
            "definition": (
                "on the benefit function B(S) = R(empty) - R(S): submodular means "
                "B(S+x) - B(S) >= B(T+x) - B(T) for every S inside T"
            ),
            "budget_ignored": (
                "curvature is a property of the objective, not the feasible "
                "region; testing inside the budget would confound them"
            ),
            "why_it_matters": (
                "a monotone submodular objective gives greedy maximisation a "
                "1 - 1/e approximation guarantee; without submodularity no bound "
                "of that family applies"
            ),
            "weights": weights.as_tuple(),
            "seed": seed,
        },
        "suites": results,
        "verdict": _verdict(results),
    }


def _predictiveness(rows: Sequence[dict[str, float]]) -> dict[str, object]:
    """Are the strongly-complementary instances the ones greedy loses on?

    Every instance violates submodularity somewhere, so a present/absent split is
    degenerate. Split at the median violation rate instead, and also report the
    rank correlation between complementarity and the greedy gap. A null result
    here would mean the curvature finding is real but operationally inert -- worth
    knowing either way.
    """
    if len(rows) < 4:
        return {"comparable": False, "reason": "too few instances"}

    ordered = sorted(rows, key=lambda r: r["submodularity_violation_rate"])
    midpoint = len(ordered) // 2
    low, high = ordered[:midpoint], ordered[midpoint:]

    def optimal_rate(group):
        return statistics.fmean(r["greedy_optimal"] for r in group)

    lo_ci = bootstrap_ci([r["greedy_optimal"] for r in low], resamples=2000, seed=11)
    hi_ci = bootstrap_ci([r["greedy_optimal"] for r in high], resamples=2000, seed=12)
    return {
        "comparable": True,
        "split": "median submodularity-violation rate",
        "low_complementarity": {
            "instances": len(low),
            "mean_violation_rate": round(
                statistics.fmean(r["submodularity_violation_rate"] for r in low), 6
            ),
            "greedy_optimal_rate": round(optimal_rate(low), 6),
            "greedy_optimal_rate_ci95": [round(v, 6) for v in lo_ci],
            "mean_greedy_gap": round(
                statistics.fmean(r["greedy_gap"] for r in low), 6
            ),
        },
        "high_complementarity": {
            "instances": len(high),
            "mean_violation_rate": round(
                statistics.fmean(r["submodularity_violation_rate"] for r in high), 6
            ),
            "greedy_optimal_rate": round(optimal_rate(high), 6),
            "greedy_optimal_rate_ci95": [round(v, 6) for v in hi_ci],
            "mean_greedy_gap": round(
                statistics.fmean(r["greedy_gap"] for r in high), 6
            ),
        },
        "spearman_violation_vs_greedy_gap": _spearman(
            [r["submodularity_violation_rate"] for r in rows],
            [r["greedy_gap"] for r in rows],
        ),
    }


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Rank correlation, computed here to keep the package dependency-free."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        index = 0
        while index < n:
            stop = index
            while stop + 1 < n and values[order[stop + 1]] == values[order[index]]:
                stop += 1
            shared = (index + stop) / 2 + 1
            for position in range(index, stop + 1):
                out[order[position]] = shared
            index = stop + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return round(num / (dx * dy), 6) if dx and dy else None


def _verdict(results: dict[str, object]) -> dict[str, object]:
    lines = {}
    for suite, data in results.items():
        sub = data["submodularity_violation_rate"]["mean"]  # type: ignore[index]
        sup = data["supermodularity_violation_rate"]["mean"]  # type: ignore[index]
        if sub <= TOLERANCE and sup > TOLERANCE:
            verdict = "submodular on the sampled pairs"
        elif sup <= TOLERANCE and sub > TOLERANCE:
            verdict = "supermodular on the sampled pairs"
        elif sub <= TOLERANCE and sup <= TOLERANCE:
            verdict = "modular on the sampled pairs"
        else:
            verdict = "neither: mixed curvature"
        lines[suite] = verdict
    return {
        "by_suite": lines,
        "consequence": (
            "no 1 - 1/e style approximation guarantee is available for greedy "
            "planning on an objective that is not submodular; the 80.3% "
            "greedy-optimal rate measured in scripts/greedy_gap.py is an "
            "empirical observation and cannot be upgraded to a bound"
        ),
        "mechanism": (
            "trust delegation creates increasing returns: migrating a leaf "
            "certificate alone reduces risk by zero because its risk is pinned "
            "to an unmigrated issuer, while migrating it after the issuer is "
            "worthwhile, so the marginal benefit grows with the set"
        ),
    }
