"""Does the recommendation depend on the parameters we had to guess?

Ingesting a real estate produces a case in which roughly half the model's inputs
are observed and half are placeholders: a certificate records its algorithm,
validity and issuer exactly, and records nothing at all about how sensitive the
data behind it is, how exposed the host is, or what migrating it would cost.

The usual responses to that are both bad. Presenting the score as a measurement
hides the guesswork. Refusing to produce a recommendation until every parameter
is calibrated means nobody ever gets one, because that calibration data does not
exist.

There is a third option, and it is the one that makes an uncalibrated model
usable: **stop asking what the parameters are, and ask whether the answer depends
on them.** Resample every unobservable field across its full plausible range,
re-plan each time, and report how often each asset is selected. An asset chosen
in 98% of draws is a recommendation you can act on today without calibrating
anything, because no admissible setting of the guesses changes it. An asset
chosen in 40% of draws is a genuine open question, and the honest output is to
say so rather than to break the tie with a placeholder.

Observed and derived fields are held fixed throughout — the algorithm, the
credential validity, the trust hierarchy, the number of certificates an authority
issued. Those are facts about the estate, and the point of the exercise is to
find out what follows from the facts alone.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, replace

from ..model import DEFAULT_WEIGHTS, ObjectiveWeights
from ..optimizer import MigrationProblem
from ..planner import plan as choose_plan
from ..uncertainty import summarize


# Parameters that no artifact supplies, with the range a reviewer would call
# admissible. Deliberately wide: the question is whether the recommendation
# survives *any* defensible setting, not a narrow band around a chosen default.
@dataclass(frozen=True)
class AssumptionRanges:
    sensitivity: tuple[float, float] = (0.3, 1.0)
    exposure: tuple[float, float] = (0.1, 1.0)
    data_lifetime_years: tuple[float, float] = (1.0, 25.0)
    business_criticality: tuple[float, float] = (0.3, 1.0)
    rotation_hours: tuple[float, float] = (0.0, 720.0)
    # Cost is multiplicative on the ingested placeholder rather than absolute,
    # because the *relative* price of a root against a leaf is the part an
    # operator can estimate even when the absolute figures are arbitrary.
    cost_multiplier: tuple[float, float] = (0.4, 2.5)

    def as_dict(self) -> dict[str, object]:
        return {k: list(v) for k, v in self.__dict__.items()}


@dataclass
class AssetVerdict:
    asset: str
    selection_rate: float
    mean_rank: float | None
    affordable_rate: float
    selection_rate_when_affordable: float

    @property
    def verdict(self) -> str:
        """Why this asset is or is not in the plan.

        Separating "does not help" from "cannot be afforded" matters: they call
        for opposite responses. The first says look elsewhere; the second says the
        budget is the binding constraint and the finding is about money, not
        cryptography.
        """
        if self.affordable_rate == 0.0:
            # Never within budget in any draw. Reporting this as "not indicated"
            # would read as "does not help", which is a different claim and one
            # this experiment has no evidence for -- the asset was never on the
            # table to be assessed.
            return "never affordable"
        if self.affordable_rate < 0.5 and self.selection_rate_when_affordable >= 0.6:
            return "blocked by budget"
        if self.selection_rate >= 0.9:
            return "act now"
        if self.selection_rate >= 0.6:
            return "likely"
        if self.selection_rate > 0.1:
            return "contested"
        return "not indicated"

    def as_dict(self) -> dict[str, object]:
        return {
            "asset": self.asset,
            "selection_rate": round(self.selection_rate, 6),
            "mean_rank_when_selected": (
                round(self.mean_rank, 3) if self.mean_rank is not None else None
            ),
            "affordable_rate": round(self.affordable_rate, 6),
            "selection_rate_when_affordable": round(
                self.selection_rate_when_affordable, 6
            ),
            "verdict": self.verdict,
        }


def perturb(
    problem: MigrationProblem, rng: random.Random, ranges: AssumptionRanges
) -> MigrationProblem:
    """One admissible reading of the estate.

    Only the unobservable fields move. Algorithm, credential validity, threat
    class, dependency count, controllability and the whole edge set are facts and
    stay exactly as ingested.
    """
    assets = tuple(
        replace(
            asset,
            sensitivity=rng.uniform(*ranges.sensitivity),
            exposure=rng.uniform(*ranges.exposure),
            data_lifetime_years=rng.uniform(*ranges.data_lifetime_years),
            business_criticality=rng.uniform(*ranges.business_criticality),
            rotation_hours=rng.uniform(*ranges.rotation_hours),
            migration_cost=asset.migration_cost * rng.uniform(*ranges.cost_multiplier),
        )
        for asset in problem.assets
    )
    return MigrationProblem(
        assets=assets,
        edges=problem.edges,
        entrypoints=problem.entrypoints,
        targets=problem.targets,
        replacements=problem.replacements,
        budget=problem.budget,
    )


def run(
    problem: MigrationProblem,
    *,
    draws: int = 500,
    seed: int = 20260907,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    ranges: AssumptionRanges = AssumptionRanges(),
) -> dict[str, object]:
    rng = random.Random(seed)

    nominal = choose_plan(problem, weights=weights)
    selections: dict[str, int] = {a.name: 0 for a in problem.assets}
    ranks: dict[str, list[int]] = {a.name: [] for a in problem.assets}
    affordable: dict[str, int] = {a.name: 0 for a in problem.assets}
    selected_when_affordable: dict[str, int] = {a.name: 0 for a in problem.assets}
    plans: list[frozenset[str]] = []
    sizes: list[int] = []
    spends: list[float] = []

    for _ in range(draws):
        sampled = perturb(problem, rng, ranges)
        # Whether the budget could stretch to this asset at all in this draw --
        # the difference between "the model rejected it" and "it was never on
        # the table".
        for asset in sampled.candidates:
            if asset.migration_cost <= sampled.budget + 1e-12:
                affordable[asset.name] += 1
        result = choose_plan(sampled, weights=weights)
        chosen = frozenset(result.selected)
        plans.append(chosen)
        sizes.append(len(chosen))
        spends.append(result.cost)
        # Rank within the draw by how much each asset alone would have helped:
        # this distinguishes "always chosen and always first" from "always chosen
        # but marginal", which a bare selection rate cannot.
        ordered = _rank_within(sampled, chosen, weights)
        affordable_now = {
            a.name
            for a in sampled.candidates
            if a.migration_cost <= sampled.budget + 1e-12
        }
        for position, name in enumerate(ordered, start=1):
            selections[name] += 1
            ranks[name].append(position)
            if name in affordable_now:
                selected_when_affordable[name] += 1

    verdicts = [
        AssetVerdict(
            asset=name,
            selection_rate=selections[name] / draws,
            mean_rank=statistics.fmean(ranks[name]) if ranks[name] else None,
            affordable_rate=affordable[name] / draws,
            selection_rate_when_affordable=(
                selected_when_affordable[name] / affordable[name]
                if affordable[name]
                else 0.0
            ),
        )
        for name in selections
    ]
    verdicts.sort(key=lambda v: (-v.selection_rate, v.mean_rank or 1e9, v.asset))

    stable = sum(1 for p in plans if p == frozenset(nominal.selected))
    return {
        "experiment": "robustness",
        "question": (
            "Which recommendations survive every admissible setting of the "
            "parameters no artifact supplies?"
        ),
        "design": {
            "draws": draws,
            "seed": seed,
            "resampled_fields": sorted(ranges.as_dict()),
            "held_fixed": [
                "algorithm",
                "credential_validity_years",
                "verification_horizon_years",
                "threat_class",
                "dependency_count",
                "controllable",
                "edges (dependency and delegation)",
                "budget",
            ],
            "ranges": ranges.as_dict(),
            "weights": weights.as_tuple(),
            "planner": nominal.strategy,
        },
        "nominal_plan": nominal.as_dict(),
        "nominal_plan_reproduced_in_draws": round(stable / draws, 6),
        "distinct_plans": len(set(plans)),
        "plan_size": summarize([float(s) for s in sizes]),
        "spend": summarize(spends),
        "assets": [v.as_dict() for v in verdicts],
        "act_now": [v.asset for v in verdicts if v.verdict == "act now"],
        "contested": [v.asset for v in verdicts if v.verdict == "contested"],
        "blocked_by_budget": [v.asset for v in verdicts if v.verdict == "blocked by budget"],
        "never_affordable": [v.asset for v in verdicts if v.verdict == "never affordable"],
        "reading": (
            "An asset selected in nearly every draw is a recommendation that does "
            "not depend on the guessed parameters; act on it without calibrating "
            "anything. A contested asset is a real open question, and the right "
            "response is to measure its sensitivity, exposure or migration cost "
            "rather than to let a placeholder decide. An asset blocked by budget "
            "would be chosen whenever it is affordable -- that finding is about "
            "money, not cryptography. An asset marked never affordable was not "
            "assessed at all: no draw could fit it in the budget."
        ),
    }


def _rank_within(
    problem: MigrationProblem, chosen: frozenset[str], weights: ObjectiveWeights
) -> list[str]:
    """Order a plan's assets by standalone benefit, best first."""
    scored = []
    baseline = problem.evaluate((), weights=weights).value
    for name in chosen:
        gain = baseline - problem.evaluate([name], weights=weights).value
        scored.append((-gain, name))
    scored.sort()
    return [name for _, name in scored]


def summarise_for_humans(report: dict[str, object]) -> list[str]:
    """One line per asset, ordered as a work queue."""
    lines = []
    for entry in report["assets"]:  # type: ignore[index]
        if entry["selection_rate"] == 0.0 and entry["verdict"] != "never affordable":
            continue
        lines.append(
            f"{entry['selection_rate']:6.1%}  {entry['verdict']:<18} {entry['asset']}"
        )
    return lines


def stable_recommendations(report: dict[str, object]) -> Sequence[str]:
    return tuple(report["act_now"])  # type: ignore[index]
