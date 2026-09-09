"""What does treating attack-path hops as independent actually cost?

``METHODOLOGY.md`` has carried the independence assumption as a caveat since 0.4.
0.5's delegation work turned it into a defect: ``effective_node_risk`` correctly
raises two sibling certificates to their issuer's risk, and ``path_risk`` then
multiplies them as though they were independent events. They are the same event.

Fixing it properly changes two things at once, and they pull in **opposite
directions**, so reporting only the net would hide what happened:

``marginal``
    0.5 combined an asset's own risk with its inherited risk by taking the
    **max**, which throws away the asset's own contribution whenever the issuer
    dominates. The generative model implies a noisy-OR — compromised by its own
    weak key *or* through a cause — which is strictly larger. This **raises**
    scores.
``conditioning``
    Composing the joint distribution instead of multiplying marginals stops the
    same shared compromise being counted once per sibling on the path. This
    **lowers** path risk.

Three arms isolate them:

======================  =============================  ======================
arm                     node marginal                  path composition
======================  =============================  ======================
``legacy``              max of own and inherited       independent hops
``noisy_or_only``       noisy-OR over own and causes   independent hops
``conditioned``         noisy-OR over own and causes   conditioned on causes
======================  =============================  ======================

``noisy_or_only`` minus ``legacy`` is the marginal effect; ``conditioned`` minus
``noisy_or_only`` is the correlation effect. Neither is a win rate: there is no
ground truth here, only a coherent model and an incoherent one, and what is
measured is how far apart they are and whether the difference changes the plan.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence
from dataclasses import replace

from ..correlation import build_causes, correlated_path_risk, marginal_risk
from ..model import (
    DEFAULT_WEIGHTS,
    AssetModel,
    ObjectiveWeights,
    apply_migration,
    effective_node_risk,
    node_risk,
    path_risk,
)
from ..optimizer import MigrationProblem, exhaustive
from ..paths import EdgeKind, EdgeModel
from ..threat import ThreatClass
from ..uncertainty import paired_comparison, summarize
from .pki import PKIConfig
from .pki import make_instances as pki_instances

ARMS = ("legacy", "noisy_or_only", "conditioned")


def shared_substrate_instances(count: int, seed: int) -> list[MigrationProblem]:
    """Services sharing an HSM and an identity provider, with no CA hierarchy.

    Delegation-only instances exercise one kind of common cause. This exercises
    the other: substrate rather than trust. A hardware security module holding
    several services' keys, or one identity provider behind several accounts,
    compromises every dependent at once without any certificate being involved.
    """
    rng = random.Random(seed)
    out: list[MigrationProblem] = []
    for _ in range(count):
        services = [f"svc-{i}" for i in range(rng.randint(3, 5))]
        assets = [
            AssetModel(
                "hsm", rng.choice(("RSA-2048", "ECDSA")),
                sensitivity=0.95, exposure=rng.uniform(0.1, 0.3),
                data_lifetime_years=12, dependency_count=len(services),
                rotation_hours=0.0, migration_cost=rng.uniform(2.0, 3.5),
                business_criticality=1.0,
                threat_class=ThreatClass.AUTHENTICATION,
                credential_validity_years=rng.uniform(8, 15),
            ),
            AssetModel(
                "identity-provider", rng.choice(("ECDSA", "ED25519")),
                sensitivity=0.9, exposure=rng.uniform(0.4, 0.6),
                data_lifetime_years=10, dependency_count=len(services),
                rotation_hours=168.0, migration_cost=rng.uniform(1.5, 2.5),
                business_criticality=1.0,
                threat_class=ThreatClass.AUTHENTICATION,
                credential_validity_years=rng.uniform(3, 6),
            ),
        ]
        for name in services:
            assets.append(
                AssetModel(
                    name, rng.choice(("X25519", "ECDSA", "RSA-2048")),
                    sensitivity=rng.uniform(0.6, 0.9),
                    exposure=rng.uniform(0.6, 1.0),
                    data_lifetime_years=rng.uniform(5, 15),
                    dependency_count=rng.randint(1, 4),
                    rotation_hours=rng.uniform(24, 168),
                    migration_cost=rng.uniform(0.4, 1.0),
                    business_criticality=0.85,
                    threat_class=ThreatClass.AUTHENTICATION,
                    credential_validity_years=rng.uniform(1, 5),
                )
            )
        edges: list[EdgeModel] = []
        for name in services:
            # Every service's keys live in the same module and authenticate
            # through the same provider: two shared causes, not two hops.
            edges.append(EdgeModel("hsm", name, 1.0, EdgeKind.SHARED))
            if rng.random() < 0.8:
                edges.append(
                    EdgeModel("identity-provider", name, rng.uniform(0.8, 1.0),
                              EdgeKind.SHARED)
                )
        for u, v in zip(services, services[1:], strict=False):
            edges.append(EdgeModel(u, v, rng.uniform(0.6, 0.95)))
        out.append(
            MigrationProblem(
                tuple(assets), tuple(edges), (services[0],), (services[-1],),
                {"RSA-2048": "ML-KEM", "X25519": "ML-KEM",
                 "ECDSA": "ML-DSA", "ED25519": "ML-DSA"},
                budget=rng.uniform(1.5, 4.0),
            )
        )
    return out


def supply_chain_instances(
    count: int, seed: int, *, include_vendor: bool = True
) -> list[MigrationProblem]:
    """Vendors behind modules behind services: a cause over the causes.

    Two hardware security modules bought from one supplier, two identity
    providers in one cloud region, two authorities linking one library. The
    supplier tier does not compromise every unit when it fails -- a firmware
    defect affects some configurations, a library flaw reaches some callers --
    so strengths are partial, which is precisely the regime where a flat cause
    model goes wrong.

    ``include_vendor=False`` builds the identical estate as an analyst who never
    recorded the shared supplier would see it: the modules are there, their
    common origin is not.
    """
    rng = random.Random(seed)
    out: list[MigrationProblem] = []
    for _ in range(count):
        vendors = [f"vendor-{i}" for i in range(rng.randint(1, 2))]
        modules = [f"hsm-{i}" for i in range(rng.randint(2, 3))]
        services = [f"svc-{i}" for i in range(rng.randint(2, 4))]

        assets: list[AssetModel] = []
        for name in vendors:
            assets.append(
                AssetModel(
                    name, rng.choice(("RSA-2048", "ECDSA")),
                    sensitivity=0.95, exposure=rng.uniform(0.05, 0.2),
                    data_lifetime_years=15, dependency_count=len(modules),
                    rotation_hours=0.0, migration_cost=rng.uniform(3.0, 5.0),
                    business_criticality=1.0,
                    threat_class=ThreatClass.AUTHENTICATION,
                    credential_validity_years=rng.uniform(10, 20),
                    # A supplier is a cause, not a purchase: you cannot migrate
                    # someone else's firmware. Keeping it uncontrollable holds
                    # the candidate set identical across both views.
                    controllable=False,
                )
            )
        for name in modules:
            assets.append(
                AssetModel(
                    name, rng.choice(("RSA-2048", "ECDSA")),
                    sensitivity=0.9, exposure=rng.uniform(0.1, 0.3),
                    data_lifetime_years=12, dependency_count=len(services),
                    rotation_hours=0.0, migration_cost=rng.uniform(1.5, 3.0),
                    business_criticality=0.95,
                    threat_class=ThreatClass.AUTHENTICATION,
                    credential_validity_years=rng.uniform(6, 12),
                )
            )
        for name in services:
            assets.append(
                AssetModel(
                    name, rng.choice(("X25519", "ECDSA", "RSA-2048")),
                    sensitivity=rng.uniform(0.6, 0.9),
                    exposure=rng.uniform(0.6, 1.0),
                    data_lifetime_years=rng.uniform(5, 15),
                    dependency_count=rng.randint(1, 4),
                    rotation_hours=rng.uniform(24, 168),
                    migration_cost=rng.uniform(0.4, 1.0),
                    business_criticality=0.85,
                    threat_class=ThreatClass.AUTHENTICATION,
                    credential_validity_years=rng.uniform(1, 5),
                )
            )

        edges: list[EdgeModel] = []
        for name in modules:
            # Both draws happen either way, so the two views see the identical
            # estate: only whether the supplier edge is recorded differs. Skipping
            # the draw would desynchronise the random streams and compare two
            # different infrastructures.
            vendor = rng.choice(vendors)
            strength = rng.uniform(0.3, 0.8)
            if include_vendor:
                edges.append(EdgeModel(vendor, name, strength, EdgeKind.SHARED))
        for name in services:
            edges.append(
                EdgeModel(rng.choice(modules), name, rng.uniform(0.5, 1.0),
                          EdgeKind.SHARED)
            )
        for u, v in zip(services, services[1:], strict=False):
            edges.append(EdgeModel(u, v, rng.uniform(0.7, 1.0)))

        # The blind view keeps the supplier as an inert asset so the two estates
        # have identical inventories and identical candidate sets; only the
        # edges recording the shared origin differ.
        out.append(
            MigrationProblem(
                tuple(assets), tuple(edges), (services[0],), (services[-1],),
                {"RSA-2048": "ML-KEM", "X25519": "ML-KEM", "ECDSA": "ML-DSA"},
                budget=rng.uniform(2.0, 5.0),
            )
        )
    return out


def _unrecorded_vendor_effect(
    instances: int, seed: int, weights: ObjectiveWeights
) -> dict[str, object]:
    """What an analyst loses by not recording that two modules share a supplier.

    Both estates are the same infrastructure. One model knows the modules came
    from one supplier; the other does not. Everything else is identical, so the
    difference is the value of that single fact.
    """
    known = supply_chain_instances(instances, seed, include_vendor=True)
    unknown = supply_chain_instances(instances, seed, include_vendor=False)

    understatement: list[float] = []
    plan_matches = 0
    blind_shortfall: list[float] = []
    compared = 0
    for full, partial in zip(known, unknown, strict=True):
        if not full.candidates or not full.path_set.paths:
            continue
        if not partial.candidates or not partial.path_set.paths:
            continue
        compared += 1
        informed = replace(full, correlated=True)
        blind = replace(partial, correlated=True)

        # Path risk only: the inventories are identical, so this isolates the
        # correlation rather than picking up any change in the asset count.
        understatement.append(
            informed.evaluate((), weights=weights).path_mean
            - blind.evaluate((), weights=weights).path_mean
        )

        blind_plan = exhaustive(blind, weights=weights)[0].selected
        informed_plan = exhaustive(informed, weights=weights)[0].selected
        if frozenset(blind_plan) == frozenset(informed_plan):
            plan_matches += 1
        # What the blind plan actually achieves, judged in the informed model.
        baseline = informed.evaluate((), weights=weights).value
        blind_shortfall.append(
            (baseline - informed.evaluate(blind_plan, weights=weights).value)
            - (baseline - informed.evaluate(informed_plan, weights=weights).value)
        )

    return {
        "instances": compared,
        "path_risk_understated_by_omitting_the_supplier": summarize(understatement),
        "plan_agreement": round(plan_matches / max(1, compared), 6),
        "risk_reduction_forgone_by_the_blind_plan": summarize(blind_shortfall),
        "reading": (
            "both estates are the same infrastructure with the same inventory, "
            "budget and candidate set; the supplier is present in both but only "
            "one model records that the modules share it. The difference is what "
            "that single fact is worth"
        ),
    }


def _score(
    problem: MigrationProblem,
    selected: Sequence[str],
    arm: str,
    weights: ObjectiveWeights,
) -> tuple[float, float]:
    """Return (objective, mean path risk) for one plan under one arm."""
    migrated = apply_migration(problem.assets, selected, problem.replacements)
    own = {a.name: node_risk(a) for a in migrated}
    structure = build_causes(problem.edges)
    paths = problem.path_set

    if arm == "legacy":
        scores = effective_node_risk(own, problem.edges)
        risks = [path_risk(p, scores, paths.reliability) for p in paths.paths]
    else:
        scores = {name: marginal_risk(name, own, structure) for name in own}
        if arm == "noisy_or_only":
            risks = [path_risk(p, scores, paths.reliability) for p in paths.paths]
        else:
            risks = [
                correlated_path_risk(p, own, paths.reliability, structure)
                for p in paths.paths
            ]

    node_mean = sum(scores.values()) / len(scores) if scores else 0.0
    path_mean = sum(risks) / len(risks) if risks else 0.0
    rotation = [
        scores[a.name] * min(1.0, a.rotation_hours / 168.0)
        for a in migrated
        if a.rotation_hours > 0
    ]
    rotation_mean = sum(rotation) / len(rotation) if rotation else 0.0
    w = weights.normalized()
    value = w.node * node_mean + w.path * path_mean + w.rotation * rotation_mean
    return value, path_mean


def run(
    *,
    instances: int = 200,
    seed: int = 20260907,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    pki_config: PKIConfig = PKIConfig(),
) -> dict[str, object]:
    suites = {
        "pki_delegation": [
            p for p in pki_instances(instances, seed, pki_config)
            if p.candidates and p.path_set.paths
        ],
        "shared_substrate": [
            p for p in shared_substrate_instances(instances, seed)
            if p.candidates and p.path_set.paths
        ],
        "supply_chain": [
            p for p in supply_chain_instances(instances, seed)
            if p.candidates and p.path_set.paths
        ],
    }

    results: dict[str, object] = {}
    for suite, problems in suites.items():
        baseline_gap: dict[str, list[float]] = {a: [] for a in ARMS}
        path_by_arm: dict[str, list[float]] = {a: [] for a in ARMS}
        plans: dict[str, list[frozenset[str]]] = {a: [] for a in ARMS}
        inherited_share: list[float] = []

        for problem in problems:
            for arm in ARMS:
                value, path_mean = _score(problem, (), arm, weights)
                baseline_gap[arm].append(value)
                path_by_arm[arm].append(path_mean)
                variant = replace(
                    problem, correlated=(arm == "conditioned")
                )
                if arm == "legacy":
                    variant = replace(problem, correlated=False)
                plans[arm].append(frozenset(exhaustive(variant, weights=weights)[0].selected))

            own = {a.name: node_risk(a) for a in problem.assets}
            structure = build_causes(problem.edges)
            covered = [n for n in own if structure.influence.get(n)]
            if covered:
                inherited_share.append(
                    statistics.fmean(
                        1.0 - (own[n] / marginal_risk(n, own, structure))
                        if marginal_risk(n, own, structure) > 0
                        else 0.0
                        for n in covered
                    )
                )

        results[suite] = {
            "instances": len(problems),
            "baseline_objective": {
                arm: summarize(baseline_gap[arm]) for arm in ARMS
            },
            "baseline_path_risk": {arm: summarize(path_by_arm[arm]) for arm in ARMS},
            "marginal_effect": paired_comparison(
                [
                    a - b
                    for a, b in zip(
                        baseline_gap["legacy"], baseline_gap["noisy_or_only"],
                        strict=True,
                    )
                ],
                label_a="legacy",
                label_b="noisy_or_only",
            ),
            "correlation_effect": paired_comparison(
                [
                    a - b
                    for a, b in zip(
                        path_by_arm["noisy_or_only"], path_by_arm["conditioned"],
                        strict=True,
                    )
                ],
                label_a="independent_hops",
                label_b="conditioned",
            ),
            "net_effect_on_objective": paired_comparison(
                [
                    a - b
                    for a, b in zip(
                        baseline_gap["legacy"], baseline_gap["conditioned"],
                        strict=True,
                    )
                ],
                label_a="legacy",
                label_b="conditioned",
            ),
            "plan_agreement_legacy_vs_conditioned": round(
                sum(
                    1
                    for a, b in zip(plans["legacy"], plans["conditioned"], strict=True)
                    if a == b
                )
                / max(1, len(problems)),
                6,
            ),
            "mean_share_of_risk_that_is_inherited": summarize(inherited_share),
        }

    return {
        "experiment": "correlation_impact",
        "question": (
            "What does treating attack-path hops as independent cost, once "
            "shared causes are modelled properly?"
        ),
        "design": {
            "arms": {
                "legacy": "max-dominance marginals, independent hops (0.5 to 0.8)",
                "noisy_or_only": "noisy-OR marginals, independent hops",
                "conditioned": "noisy-OR marginals, paths conditioned on causes",
            },
            "decomposition": (
                "noisy_or_only - legacy isolates the marginal change, which raises "
                "risk; conditioned - noisy_or_only isolates the correlation "
                "correction, which lowers path risk. They pull in opposite "
                "directions, so the net is not the story"
            ),
            "not_a_win_rate": (
                "there is no ground truth: only a coherent joint model and an "
                "incoherent product of marginals. What is measured is the size of "
                "the disagreement and whether it changes the plan"
            ),
            "instances_per_suite": instances,
            "weights": weights.as_tuple(),
            "seed": seed,
        },
        "suites": results,
        "unrecorded_supplier": _unrecorded_vendor_effect(instances, seed, weights),
    }
