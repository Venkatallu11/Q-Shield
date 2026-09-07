"""Generator for public-key infrastructure instances.

Identity infrastructure has a shape the generic generator cannot produce: a small
number of very long-lived, expensive, rarely-rotated trust anchors, and a large
number of cheap, short-lived, frequently-rotated leaves that inherit all of their
trustworthiness from those anchors.

That shape is exactly where the two 0.5 model changes bite. The threat-class
correction separates a 90-day leaf certificate from the 20-year root above it,
which the shared ``data_lifetime_years`` term could not. The delegation edge
makes the inheritance explicit, so migrating leaves under an unmigrated root
correctly shows up as achieving nothing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..model import AssetModel
from ..optimizer import MigrationProblem
from ..paths import EdgeKind, EdgeModel
from ..threat import ThreatClass

# Identity infrastructure signs; it rarely does key establishment.
SIGNATURE_ALGORITHMS: tuple[str, ...] = ("ECDSA", "ED25519", "RSA-2048")
REPLACEMENTS: dict[str, str] = {
    "ECDSA": "ML-DSA",
    "ED25519": "ML-DSA",
    # RSA used as a signing key migrates to a signature algorithm, not a KEM.
    # The registry's default_replacement maps RSA to ML-KEM because key
    # transport is the usual driver; a signing hierarchy overrides that.
    "RSA-2048": "ML-DSA",
}


@dataclass(frozen=True)
class PKIConfig:
    roots: int = 1
    intermediates: int = 2
    leaves: int = 5
    services: int = 3
    root_validity_years: float = 20.0
    intermediate_validity_years: float = 8.0
    leaf_validity_years: float = 0.25       # a 90-day certificate
    root_cost: float = 3.0
    intermediate_cost: float = 1.5
    leaf_cost: float = 0.4
    budget_low: float = 1.5
    budget_high: float = 5.0

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def make_instance(rng: random.Random, config: PKIConfig = PKIConfig()) -> MigrationProblem:
    assets: list[AssetModel] = []
    edges: list[EdgeModel] = []

    roots = [f"root-ca-{i}" for i in range(config.roots)]
    intermediates = [f"intermediate-ca-{i}" for i in range(config.intermediates)]
    leaves = [f"leaf-cert-{i}" for i in range(config.leaves)]
    services = [f"service-{i}" for i in range(config.services)]

    def signing_asset(name, validity, cost, exposure, sensitivity, deps, rotation):
        return AssetModel(
            name=name,
            algorithm=rng.choice(SIGNATURE_ALGORITHMS),
            sensitivity=sensitivity,
            exposure=exposure,
            # Retention of the data behind the identity system. Deliberately long
            # and identical across tiers: under the pre-0.5 model this term drove
            # every tier's score, flattening root and leaf together.
            data_lifetime_years=15.0,
            dependency_count=deps,
            rotation_hours=rotation,
            migration_cost=cost,
            business_criticality=0.9,
            threat_class=ThreatClass.AUTHENTICATION,
            credential_validity_years=validity,
        )

    for name in roots:
        # Offline, low network exposure, but everything hangs off it.
        assets.append(signing_asset(
            name, config.root_validity_years, config.root_cost,
            exposure=rng.uniform(0.10, 0.25), sensitivity=rng.uniform(0.9, 1.0),
            deps=config.intermediates + config.leaves, rotation=0.0,
        ))
    for name in intermediates:
        assets.append(signing_asset(
            name, config.intermediate_validity_years, config.intermediate_cost,
            exposure=rng.uniform(0.3, 0.5), sensitivity=rng.uniform(0.8, 0.95),
            deps=max(1, config.leaves // max(1, config.intermediates)),
            rotation=rng.uniform(48, 168),
        ))
    for name in leaves:
        # Internet-facing, cheap, rotated constantly.
        assets.append(signing_asset(
            name, config.leaf_validity_years, config.leaf_cost,
            exposure=rng.uniform(0.8, 1.0), sensitivity=rng.uniform(0.5, 0.8),
            deps=rng.randint(1, 3), rotation=rng.uniform(4, 48),
        ))
    for name in services:
        # What the identity system protects. Data at rest, not a swap candidate.
        assets.append(AssetModel(
            name=name, algorithm="AES-256",
            sensitivity=rng.uniform(0.8, 1.0), exposure=rng.uniform(0.2, 0.4),
            data_lifetime_years=rng.uniform(10, 20), dependency_count=rng.randint(2, 5),
            rotation_hours=rng.uniform(24, 168), migration_cost=0.5,
            business_criticality=1.0,
            threat_class=ThreatClass.CONFIDENTIALITY,
        ))

    # Trust hierarchy: delegation, not traversal.
    for i, name in enumerate(intermediates):
        edges.append(EdgeModel(roots[i % len(roots)], name, 1.0, EdgeKind.DELEGATION))
    for i, name in enumerate(leaves):
        issuer = intermediates[i % len(intermediates)] if intermediates else roots[0]
        edges.append(EdgeModel(issuer, name, 1.0, EdgeKind.DELEGATION))

    # Attack surface: a leaf fronts a service, services reach each other.
    for i, service in enumerate(services):
        edges.append(EdgeModel(leaves[i % len(leaves)], service,
                               rng.uniform(0.7, 1.0), EdgeKind.DEPENDENCY))
    for u, v in zip(services, services[1:], strict=False):
        edges.append(EdgeModel(u, v, rng.uniform(0.5, 0.9), EdgeKind.DEPENDENCY))

    return MigrationProblem(
        assets=tuple(assets),
        edges=tuple(edges),
        entrypoints=(leaves[0],),
        targets=(services[-1],),
        replacements=REPLACEMENTS,
        budget=rng.uniform(config.budget_low, config.budget_high),
    )


def make_instances(
    count: int, seed: int, config: PKIConfig = PKIConfig()
) -> list[MigrationProblem]:
    rng = random.Random(seed)
    return [make_instance(rng, config) for _ in range(count)]


def without_delegation(problem: MigrationProblem) -> MigrationProblem:
    """The same instance as a planner with no concept of delegation would see it.

    Trust edges are simply dropped. This is one of the two ways a pre-0.5 tool
    would encode a PKI; :func:`delegation_as_dependency` is the other.
    """
    return _rebuild(problem, tuple(
        e for e in problem.edges if e.kind is EdgeKind.DEPENDENCY
    ))


def delegation_as_dependency(problem: MigrationProblem) -> MigrationProblem:
    """The same instance with trust edges encoded as ordinary dependencies.

    This is the more likely pre-0.5 encoding, since the model had exactly one
    edge type and an analyst would naturally draw "root signs intermediate" as an
    edge. It understates the relation twice over: the root's risk is diluted by
    the hop probability instead of dominating, and the certificates beneath it
    look independently improvable.
    """
    return _rebuild(problem, tuple(
        EdgeModel(e.source, e.target, e.reliability, EdgeKind.DEPENDENCY)
        for e in problem.edges
    ))


def _rebuild(problem: MigrationProblem, edges: tuple[EdgeModel, ...]) -> MigrationProblem:
    return MigrationProblem(
        assets=problem.assets,
        edges=edges,
        entrypoints=problem.entrypoints,
        targets=problem.targets,
        replacements=problem.replacements,
        budget=problem.budget,
    )
