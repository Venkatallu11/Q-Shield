"""Does correcting the threat class change what you should migrate?

Q-SHIELD through 0.4 measured every asset's longevity against
``data_lifetime_years``. That term encodes harvest-now-decrypt-later, which
applies to confidentiality and not to signatures: recording a signature gains an
adversary nothing, and forgery only becomes possible from the moment a CRQC
exists, forward. See :mod:`qshield.threat`.

The correction is defensible on its own terms, but "more principled" is not the
same as "changes the answer". If the recommendation is identical either way, the
extra fields are complexity for nothing. This measures it on PKI instances, where
the distinction is sharpest: a 90-day leaf certificate and the 20-year root above
it sit behind the same 15-year data-retention horizon, so the old term scored them
alike while the corrected one separates them by a factor of ten.

As in :mod:`qshield.experiments.hierarchy`, this is not framed as a win rate. The
corrected planner minimises the objective it is scored on and the mis-scored
plan is feasible for it, so it cannot lose — that guarantee is the thing that
invalidated 0.3's headline. What is measured is the size of the error, which is
free to be zero.
"""

from __future__ import annotations

from dataclasses import replace

from ..model import DEFAULT_WEIGHTS, ObjectiveWeights
from ..optimizer import MigrationProblem, exhaustive
from ..threat import ThreatClass
from ..uncertainty import paired_comparison, summarize
from .pki import PKIConfig, make_instances


def as_pre_0_5(problem: MigrationProblem) -> MigrationProblem:
    """The instance as every version through 0.4 would have scored it.

    Every asset forced to the confidentiality horizon, and the class-specific
    horizons discarded — which is precisely what a single shared
    ``data_lifetime_years`` term amounts to.
    """
    assets = tuple(
        replace(
            a,
            threat_class=ThreatClass.CONFIDENTIALITY,
            credential_validity_years=None,
            verification_horizon_years=None,
        )
        for a in problem.assets
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
    *,
    instances: int = 300,
    seed: int = 20260907,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    config: PKIConfig = PKIConfig(),
) -> dict[str, object]:
    problems = [p for p in make_instances(instances, seed, config) if p.candidates]

    agree = 0
    deltas: list[float] = []
    mis_scored_improvement: list[float] = []
    correct_improvement: list[float] = []
    tier_selection = {"root": [0, 0], "intermediate": [0, 0], "leaf": [0, 0]}

    for problem in problems:
        mis = as_pre_0_5(problem)
        mis_plan = exhaustive(mis, weights=weights)[0].selected
        correct_plan = exhaustive(problem, weights=weights)[0].selected
        if set(mis_plan) == set(correct_plan):
            agree += 1

        # Both judged under the corrected model.
        baseline = problem.evaluate((), weights=weights).value
        mis_actual = baseline - problem.evaluate(mis_plan, weights=weights).value
        correct_actual = baseline - problem.evaluate(correct_plan, weights=weights).value
        mis_scored_improvement.append(mis_actual)
        correct_improvement.append(correct_actual)
        deltas.append(correct_actual - mis_actual)

        for index, plan in enumerate((mis_plan, correct_plan)):
            for name in plan:
                for tier in tier_selection:
                    if name.startswith(tier):
                        tier_selection[tier][index] += 1

    return {
        "experiment": "threat_class",
        "question": (
            "Does scoring signatures against a credential-validity horizon rather "
            "than a data-retention horizon change which assets get migrated?"
        ),
        "design": {
            "instances": len(problems),
            "arms": {
                "pre_0_5_scoring": "every asset measured against data_lifetime_years",
                "threat_class_scoring": "horizon selected by the asset's threat class",
            },
            "scoring": "both plans evaluated under the corrected model",
            "not_a_win_rate": (
                "the corrected planner minimises the metric it is scored on and the "
                "mis-scored plan is feasible for it, so it cannot lose; the reported "
                "quantity is the size of the error"
            ),
            "weights": weights.as_tuple(),
            "seed": seed,
            "generator": config.as_dict(),
        },
        "plan_agreement": round(agree / len(problems), 6),
        "risk_reduction": {
            "pre_0_5_plan": summarize(mis_scored_improvement),
            "threat_class_plan": summarize(correct_improvement),
        },
        "shortfall": {
            **paired_comparison(
                deltas, label_a="pre_0_5_scoring", label_b="threat_class_scoring"
            ),
            "note": (
                "positive delta means the mis-scored plan achieved less real risk "
                "reduction than the corrected one"
            ),
        },
        "assets_selected_by_tier": {
            tier: {"pre_0_5": counts[0], "threat_class": counts[1]}
            for tier, counts in tier_selection.items()
        },
        "reading": (
            "the tier counts are the mechanism: the pre-0.5 horizon flattens a "
            "90-day leaf certificate and a 20-year root into the same score, so a "
            "planner using it buys cheap leaves"
        ),
    }
