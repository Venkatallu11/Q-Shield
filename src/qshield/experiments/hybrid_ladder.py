"""Should short-lived credentials go to hybrid rather than to pure post-quantum?

0.7 produced an uncomfortable result: under the calibrated model, migrating a
short-lived credential to a pure post-quantum primitive *raises* modelled risk.
Below a crossover horizon the chance a CRQC arrives at all is smaller than the
residual risk of a young primitive, so the swap is net-negative. For ``ECDSA ->
ML-DSA`` that crossover sits at 3.7 years, and a TLS leaf certificate lives 90
days. The model's advice was, in effect, leave most of the certificate estate
alone.

That advice is an artifact of only offering one destination. A hybrid is broken
only if **both** halves are, so it is covered against the CRQC *and* against
cryptanalysis of the young primitive. Composing the two probabilities moves the
crossover from 3.7 years to roughly 0.15 -- about eight weeks -- which is shorter
than the life of the certificate. The advice inverts: migrate them, but to a
hybrid.

This experiment checks whether that holds when a planner actually spends a
budget, rather than only in the per-primitive arithmetic. Three replacement
policies over the same instances:

``pure``
    classical -> post-quantum, the policy every version through 0.7 assumed.
``hybrid``
    classical -> hybrid only.
``ladder``
    both rungs available; the planner picks per asset.

``ladder`` has a superset of ``pure``'s options, so it cannot score worse — the
same structural guarantee that invalidated 0.3's headline. What is measured is
therefore the *size* of the improvement and which rung each asset takes, not a
win rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from ..algorithms import Family, family
from ..calibration import DEFAULT_CALIBRATION, CalibrationConfig, horizon_for
from ..hybrid import default_hybrid_for, migration_ladder
from ..model import DEFAULT_WEIGHTS, AssetModel, ObjectiveWeights
from ..optimizer import MigrationProblem
from ..planner import plan as choose_plan
from ..threat import ThreatClass
from ..uncertainty import summarize
from .pki import PKIConfig
from .pki import make_instances as pki_instances

POLICIES = ("pure", "hybrid", "ladder")


def flat_instances(count: int, seed: int) -> list[MigrationProblem]:
    """Short-lived service credentials with no trust hierarchy.

    The PKI suite turns out not to isolate the effect this experiment is about:
    on a hierarchy, leaf certificates are already excluded because their risk is
    pinned to an unmigrated issuer, so the destination on offer never gets to
    matter. A flat estate of short-lived credentials -- workload identities,
    service certificates, signing keys with no CA above them -- is where the
    crossover actually decides anything.
    """
    import random

    rng = random.Random(seed)
    out: list[MigrationProblem] = []
    for _ in range(count):
        size = rng.randint(4, 7)
        assets = tuple(
            AssetModel(
                f"svc-{i}",
                rng.choice(("ECDSA", "ED25519", "X25519")),
                sensitivity=rng.uniform(0.6, 0.95),
                exposure=rng.uniform(0.7, 1.0),
                data_lifetime_years=rng.uniform(5, 15),
                dependency_count=rng.randint(1, 4),
                rotation_hours=rng.uniform(12, 168),
                migration_cost=rng.uniform(0.5, 1.5),
                business_criticality=0.9,
                threat_class=ThreatClass.AUTHENTICATION,
                # The regime the crossover is about: credentials rotated on a
                # cadence far shorter than any plausible CRQC timeline.
                credential_validity_years=rng.uniform(0.08, 0.75),
            )
            for i in range(size)
        )
        out.append(
            MigrationProblem(assets, (), (), (), {}, budget=rng.uniform(1.5, 4.0))
        )
    return out


def replacements_for(
    assets: Sequence[AssetModel], policy: str
) -> dict[str, str]:
    """Build a replacement map under one policy.

    The model's replacement map is keyed by algorithm and offers one destination
    per source, so the ``ladder`` policy is expressed by duplicating each asset's
    source algorithm into whichever rung is better *for that asset's horizon*.
    That is done in :func:`ladder_problem`, which needs the horizons.
    """
    out: dict[str, str] = {}
    for asset in assets:
        if family(asset.algorithm) is not Family.SHOR_VULNERABLE:
            continue
        rungs = migration_ladder(asset.algorithm)
        if not rungs:
            continue
        hybrid_rung = default_hybrid_for(asset.algorithm)
        pure_rung = rungs[-1]
        if policy == "pure":
            out[asset.algorithm] = pure_rung
        elif policy == "hybrid" and hybrid_rung:
            out[asset.algorithm] = hybrid_rung
    return out


def ladder_problem(
    problem: MigrationProblem, config: CalibrationConfig
) -> MigrationProblem:
    """Per-asset choice of rung, resolved by which one the horizon favours.

    A replacement map cannot express "it depends on the asset", so the choice is
    made here, where the horizon is known, and baked in. An asset shorter-lived
    than the pure crossover takes the hybrid rung; a long-lived anchor takes the
    pure one.
    """
    # A replacement map holds one destination per source algorithm, so where
    # assets sharing an algorithm disagree the safest rung wins: that is the
    # shorter-lived asset's choice, and it is the one the pure rung would have
    # harmed.
    chosen: dict[str, str] = {}
    for asset in problem.assets:
        if family(asset.algorithm) is not Family.SHOR_VULNERABLE:
            continue
        rungs = migration_ladder(asset.algorithm)
        if not rungs:
            continue
        horizon = horizon_for(asset)
        best = min(rungs, key=lambda rung: config.probability(rung, horizon))
        current = chosen.get(asset.algorithm)
        if current is None or config.probability(best, horizon) < config.probability(
            current, horizon
        ):
            chosen[asset.algorithm] = best
    return replace(problem, replacements=chosen, calibration=config)


def run(
    *,
    instances: int = 300,
    seed: int = 20260907,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    config: CalibrationConfig = DEFAULT_CALIBRATION,
    pki_config: PKIConfig = PKIConfig(),
) -> dict[str, object]:
    suites = {
        "pki_with_delegation": [
            p for p in pki_instances(instances, seed, pki_config)
            if any(family(a.algorithm) is Family.SHOR_VULNERABLE for a in p.assets)
        ],
        "flat_short_lived": flat_instances(instances, seed),
    }
    return {
        "experiment": "hybrid_ladder",
        "question": (
            "Does offering a hybrid destination change what a budget should buy, "
            "and does it rescue the short-lived credentials the pure-post-quantum "
            "crossover told us to leave alone?"
        ),
        "design": {
            "instances_per_suite": instances,
            "suites": {
                "pki_with_delegation": "three-tier certificate authority hierarchies",
                "flat_short_lived": (
                    "service credentials rotated far faster than any plausible "
                    "CRQC timeline, with no trust hierarchy above them"
                ),
            },
            "policies": {
                "pure": "classical -> post-quantum, the policy assumed through 0.7",
                "hybrid": "classical -> hybrid only",
                "ladder": "both rungs, chosen per asset by its exposure horizon",
            },
            "not_a_win_rate": (
                "ladder has a superset of pure's options so it cannot score worse; "
                "what is measured is the size of the improvement and which rung "
                "each asset takes"
            ),
            "calibration": config.as_dict(),
            "weights": weights.as_tuple(),
            "seed": seed,
        },
        "suites": {
            name: _run_suite(problems, weights, config)
            for name, problems in suites.items()
        },
        "crossover_comparison": _crossovers(config),
    }


def _run_suite(
    problems: Sequence[MigrationProblem],
    weights: ObjectiveWeights,
    config: CalibrationConfig,
) -> dict[str, object]:
    achieved: dict[str, list[float]] = {p: [] for p in POLICIES}
    spend: dict[str, list[float]] = {p: [] for p in POLICIES}
    plan_size: dict[str, list[float]] = {p: [] for p in POLICIES}
    rung_choice = {"hybrid": 0, "pure": 0}
    short_lived_migrated = {"pure": 0, "ladder": 0}
    short_lived_total = 0

    for problem in problems:
        variants = {
            "pure": replace(
                problem,
                replacements=replacements_for(problem.assets, "pure"),
                calibration=config,
            ),
            "hybrid": replace(
                problem,
                replacements=replacements_for(problem.assets, "hybrid"),
                calibration=config,
            ),
            "ladder": ladder_problem(problem, config),
        }
        baseline = variants["pure"].evaluate((), weights=weights).value

        for policy, variant in variants.items():
            if not variant.candidates:
                achieved[policy].append(0.0)
                spend[policy].append(0.0)
                plan_size[policy].append(0.0)
                continue
            chosen = choose_plan(variant, weights=weights)
            achieved[policy].append(baseline - chosen.objective)
            spend[policy].append(chosen.cost)
            plan_size[policy].append(float(len(chosen.selected)))

        # Which rung the ladder actually took, and whether the short-lived
        # certificates the crossover is about got migrated at all.
        ladder = variants["ladder"]
        for destination in ladder.replacements.values():
            if family(destination) is Family.HYBRID:
                rung_choice["hybrid"] += 1
            else:
                rung_choice["pure"] += 1
        # Read the candidate set off a *variant*: the base problem carries no
        # replacement map on the flat suite, so problem.candidates is empty there
        # and counting from it silently reported zero short-lived credentials.
        short = [
            a for a in variants["ladder"].candidates
            if horizon_for(a) < 1.0 and family(a.algorithm) is Family.SHOR_VULNERABLE
        ]
        short_lived_total += len(short)
        for policy in ("pure", "ladder"):
            variant = variants[policy]
            if not variant.candidates:
                continue
            selected = set(choose_plan(variant, weights=weights).selected)
            short_lived_migrated[policy] += sum(1 for a in short if a.name in selected)

    return {
        "instances": len(problems),
        "risk_reduction_achieved": {
            policy: summarize(values) for policy, values in achieved.items()
        },
        "spend": {policy: summarize(values) for policy, values in spend.items()},
        "plan_size": {policy: summarize(values) for policy, values in plan_size.items()},
        "ladder_rung_chosen": rung_choice,
        "short_lived_credentials": {
            "total": short_lived_total,
            "migrated_under_pure_policy": short_lived_migrated["pure"],
            "migrated_under_ladder_policy": short_lived_migrated["ladder"],
            "note": (
                "credentials with an exposure horizon under one year -- the ones "
                "the pure-post-quantum crossover at 3.7 years told us to leave "
                "alone"
            ),
        },
    }


def _crossovers(config: CalibrationConfig) -> dict[str, object]:
    from ..calibration import crossover_horizon_years

    rows: dict[str, object] = {}
    for source in ("ECDSA", "ED25519", "X25519", "RSA-2048"):
        hybrid_rung = default_hybrid_for(source)
        pure_rung = migration_ladder(source)[-1]
        rows[source] = {
            "to_pure": _years(crossover_horizon_years(source, pure_rung, config=config)),
            "to_hybrid": _years(
                crossover_horizon_years(source, hybrid_rung, config=config)
            )
            if hybrid_rung
            else None,
        }
    rows["_reading"] = (
        "years of exposure horizon above which the migration lowers modelled "
        "risk; a TLS leaf certificate lives about 0.25 years"
    )
    return rows


def _years(value: float) -> float | None:
    return None if value == float("inf") else round(value, 2)
