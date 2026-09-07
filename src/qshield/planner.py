"""Choosing a planner that will actually finish.

:func:`qshield.optimizer.exhaustive` is exact and enumerates every
budget-feasible subset, so it costs `O(2^n)` in migration candidates. That is
fine for a hand-written case with five assets and useless on a certificate estate
with four hundred, which is what :mod:`qshield.ingest` now produces.

Measured on this machine, with the budget widened so the feasible set approaches
the full power set:

======================  =========  =========
migration candidates    exhaustive greedy
======================  =========  =========
9                          0.06 s    0.006 s
11                         0.44 s    0.016 s
13                         1.22 s    0.016 s
14                         3.08 s    0.022 s
16                        12.95 s    0.030 s
17                        21.55 s    0.027 s
======================  =========  =========

Each additional candidate roughly doubles the exact cost, so 18 would be about
45 seconds and 21 upwards of six minutes. The threshold below is set from that
table rather than chosen for tidiness.

The trade is quantified rather than assumed: over 300 instances the greedy
planner returns the exact optimum on 80.3% of them, at a mean objective gap of
0.248 (``scripts/greedy_gap.py``, ``results/greedy_gap.json``). Every plan says
which planner produced it, and :func:`plan` reports when it downgraded and why,
so a result is never silently approximate.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import DEFAULT_WEIGHTS, ObjectiveWeights
from .optimizer import MigrationProblem, Plan, exhaustive, greedy_marginal

# Above this, the exact planner takes tens of seconds and doubles per candidate.
MAX_EXACT_CANDIDATES = 16

# Enumerating attack paths is exponential in a dense graph. The path layer raises
# rather than truncating, because a truncated path set would make the path term
# depend on traversal order; this is the ceiling the planner reports against.
MAX_PATHS = 100_000


@dataclass(frozen=True)
class PlanningReport:
    """A plan plus how it was produced, so approximation is never invisible."""

    plan: Plan
    planner: str
    exact: bool
    candidates: int
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            **self.plan.as_dict(),
            "planner": self.planner,
            "exact": self.exact,
            "migration_candidates": self.candidates,
            "planner_reason": self.reason,
        }


def choose(
    problem: MigrationProblem,
    *,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    max_exact_candidates: int = MAX_EXACT_CANDIDATES,
    force: str | None = None,
) -> PlanningReport:
    """Plan, picking the exact method when it will finish in reasonable time.

    ``force`` overrides the choice with ``"exhaustive"`` or ``"greedy"``. Forcing
    the exact planner on a large estate is a legitimate thing to want; it is not
    a legitimate thing to do by accident, which is why it has to be asked for.
    """
    count = len(problem.candidates)
    if force == "exhaustive" or (force is None and count <= max_exact_candidates):
        best, _ = exhaustive(problem, weights=weights)
        reason = (
            f"{count} migration candidates is within the exact planner's budget "
            f"of {max_exact_candidates}"
            if force is None
            else "exact planner requested explicitly"
        )
        return PlanningReport(best, "exhaustive", True, count, reason)

    best = greedy_marginal(problem, weights=weights)
    reason = (
        "greedy planner requested explicitly"
        if force == "greedy"
        else (
            f"{count} migration candidates exceeds the exact planner's budget of "
            f"{max_exact_candidates}; enumerating 2^{count} subsets is not "
            "tractable. The greedy planner returns the exact optimum on 80.3% of "
            "benchmark instances at a mean objective gap of 0.248 -- treat this "
            "plan as a strong candidate, not a proven optimum."
        )
    )
    return PlanningReport(best, "greedy_marginal", False, count, reason)


def plan(
    problem: MigrationProblem,
    *,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    max_exact_candidates: int = MAX_EXACT_CANDIDATES,
    force: str | None = None,
) -> Plan:
    """Just the plan, for callers that do not need the provenance."""
    return choose(
        problem,
        weights=weights,
        max_exact_candidates=max_exact_candidates,
        force=force,
    ).plan
