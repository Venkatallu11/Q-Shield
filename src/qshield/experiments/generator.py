"""Synthetic instance generator for the evaluation suite.

Derived from 0.3's ``suite.make_case`` with three changes that affect what the
suite can conclude:

* Cross-links no longer attach arbitrarily. 0.3 drew ``rng.sample(names, 2)``,
  which could add edges *into* the entrypoint or *out of* the target. Neither
  can appear on an entry-to-target path, so a share of the "random cross-links"
  in every generated graph were inert, quietly reducing topological variety.
* Migration cost is correlated with dependency count, so the budget constraint
  actually binds against the assets a path-aware method wants. With independent
  costs the constraint is close to a random subset filter.
* The generator reports the parameters it used, so a run is reconstructible from
  its report alone.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..algorithms import default_replacement
from ..model import AssetModel
from ..optimizer import MigrationProblem
from ..paths import EdgeModel

ENTRY = "entry"
TARGET = "target"

VULNERABLE_ALGORITHMS: tuple[str, ...] = ("RSA-2048", "X25519", "ED25519", "ECDSA")
REPLACEMENTS: dict[str, str] = {
    alg: default_replacement(alg) for alg in VULNERABLE_ALGORITHMS
}


@dataclass(frozen=True)
class GeneratorConfig:
    nodes: int = 7
    cross_links: int = 7
    budget_low: float = 1.5
    budget_high: float = 4.0
    reliability_low: float = 0.2
    reliability_high: float = 1.0
    backbone_reliability_low: float = 0.7

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def make_instance(
    rng: random.Random, config: GeneratorConfig = GeneratorConfig()
) -> MigrationProblem:
    """One random instance with at least one entry-to-target path."""
    interior = [f"a{i}" for i in range(max(0, config.nodes - 2))]
    names = [ENTRY, *interior, TARGET]

    assets: list[AssetModel] = []
    for name in names:
        if name == TARGET:
            algorithm = "AES-256"  # data at rest; not a swap candidate
        elif name == ENTRY:
            algorithm = rng.choice(VULNERABLE_ALGORITHMS)
        else:
            algorithm = rng.choice((*VULNERABLE_ALGORITHMS, "AES-256"))
        dependency_count = rng.randint(1, 8)
        assets.append(
            AssetModel(
                name=name,
                algorithm=algorithm,
                sensitivity=rng.uniform(0.5, 1.0),
                exposure=rng.uniform(0.3, 1.0),
                data_lifetime_years=rng.uniform(5, 20),
                dependency_count=dependency_count,
                rotation_hours=rng.uniform(4, 96),
                # Cost grows with dependencies: widely-depended-on assets are the
                # expensive ones to migrate, which is what makes the budget bite.
                migration_cost=round(0.5 + 0.15 * dependency_count * rng.uniform(0.6, 1.4), 3),
                business_criticality=rng.uniform(0.5, 1.0),
            )
        )

    edges: list[EdgeModel] = []
    for u, v in zip(names, names[1:], strict=False):  # backbone guarantees reachability
        edges.append(
            EdgeModel(u, v, rng.uniform(config.backbone_reliability_low, 1.0))
        )

    # Cross-links may leave the entry and may enter the target, but never the
    # reverse, so every added edge can lie on an entry-to-target path.
    sources = [n for n in names if n != TARGET]
    sinks = [n for n in names if n != ENTRY]
    existing = {(e.source, e.target) for e in edges}
    for _ in range(config.cross_links):
        u = rng.choice(sources)
        v = rng.choice(sinks)
        if u == v or (u, v) in existing:
            continue
        existing.add((u, v))
        edges.append(
            EdgeModel(u, v, rng.uniform(config.reliability_low, config.reliability_high))
        )

    return MigrationProblem(
        assets=tuple(assets),
        edges=tuple(edges),
        entrypoints=(ENTRY,),
        targets=(TARGET,),
        replacements=REPLACEMENTS,
        budget=rng.uniform(config.budget_low, config.budget_high),
    )


def make_instances(
    count: int, seed: int, config: GeneratorConfig = GeneratorConfig()
) -> list[MigrationProblem]:
    rng = random.Random(seed)
    return [make_instance(rng, config) for _ in range(count)]
