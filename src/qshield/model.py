"""The Q-SHIELD risk objective.

This is a *scenario model*: a deliberately simple, monotone aggregation of
modelling assumptions. It is not a probability of compromise, a compliance score,
or a production security assessment. Its purpose is to make migration-plan
comparisons reproducible and falsifiable, not to be correct in absolute terms.

Changes from 0.3, both of which were soundness bugs rather than tuning choices:

* **The rotation-pressure term no longer gates on a risk threshold.** 0.3
  averaged the term over ``{a : rotation_hours and score(a) > 10}``. Membership
  of that set depends on the score, so migrating an asset could drop it out of
  the average and *raise* the mean over the survivors. On the reference case in
  ``tests/test_regressions.py`` that made a strict RSA-2048 -> ML-KEM migration
  increase the objective from 18.838 to 19.329 — the optimizer was penalised for
  a strictly beneficial action. Membership now depends only on
  ``rotation_hours``, which migration never changes.

* **Path risk now includes the entrypoint's own risk.** 0.3 iterated
  ``zip(path, path[1:], strict=False)`` and read the score of ``v`` only, so the first node on
  every path contributed nothing. Migrating an internet-facing entrypoint — the
  asset a path-aware method most wants to reason about — left the path term
  exactly unchanged.

Together these give the model a property 0.3 lacked and which the optimizer's
correctness rests on: :func:`objective` is monotone non-decreasing in every
asset's quantum factor, so replacing an algorithm with a strictly less
susceptible one can never increase the score. ``tests/test_monotonicity.py``
tests this by construction and over random cases.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from .algorithms import ROTATION_REFERENCE_HOURS, quantum_factor
from .paths import EdgeModel, PathSet, enumerate_paths

__all__ = [
    "AssetModel",
    "EdgeModel",
    "ObjectiveWeights",
    "ObjectiveResult",
    "node_risk",
    "path_risk",
    "objective",
    "apply_migration",
]

# Data lifetime beyond this horizon saturates the longevity term. It encodes the
# "harvest now, decrypt later" window: data that must stay confidential for
# longer than a plausible CRQC arrival is already fully exposed today.
LONGEVITY_HORIZON_YEARS = 20.0


@dataclass(frozen=True)
class AssetModel:
    """One cryptographic asset in the inventory.

    ``sensitivity`` and ``exposure`` are unit-interval scenario parameters;
    ``migration_cost`` is in arbitrary but self-consistent budget units.
    """

    name: str
    algorithm: str
    sensitivity: float = 0.5
    exposure: float = 0.5
    data_lifetime_years: float = 1.0
    dependency_count: int = 1
    rotation_hours: float = 0.0
    migration_cost: float = 1.0
    business_criticality: float = 0.5

    def __post_init__(self) -> None:
        for field in ("sensitivity", "exposure", "business_criticality"):
            value = getattr(self, field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{self.name}: {field}={value} must lie in [0, 1]"
                )
        if self.data_lifetime_years < 0 or self.rotation_hours < 0:
            raise ValueError(f"{self.name}: durations must be non-negative")
        if self.migration_cost < 0:
            raise ValueError(f"{self.name}: migration_cost must be non-negative")
        if self.dependency_count < 0:
            raise ValueError(f"{self.name}: dependency_count must be non-negative")


@dataclass(frozen=True)
class ObjectiveWeights:
    """Relative importance of the three objective components.

    Only the *ratios* matter: :func:`objective` divides by the sum, so
    ``(0.2, 0.2, 0.1)`` and ``(0.6, 0.6, 0.3)`` are the same weighting. 0.3's
    sensitivity sweep did not account for this and evaluated a redundant grid
    point. :meth:`normalized` makes the equivalence explicit and hashable.
    """

    node: float = 0.55
    path: float = 0.30
    rotation: float = 0.15

    def __post_init__(self) -> None:
        if min(self.node, self.path, self.rotation) < 0:
            raise ValueError("objective weights must be non-negative")
        if self.node + self.path + self.rotation <= 0:
            raise ValueError("at least one objective weight must be positive")

    def normalized(self) -> ObjectiveWeights:
        total = self.node + self.path + self.rotation
        return ObjectiveWeights(self.node / total, self.path / total, self.rotation / total)

    def key(self, digits: int = 9) -> tuple[float, float, float]:
        """Hashable identity on the simplex, for de-duplicating weight grids."""
        n = self.normalized()
        return (round(n.node, digits), round(n.path, digits), round(n.rotation, digits))

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.node, self.path, self.rotation)


DEFAULT_WEIGHTS = ObjectiveWeights()
# Ablation weighting: node risk only. Used as the control arm that isolates the
# contribution of path-awareness from the contribution of exhaustive search.
NODE_ONLY_WEIGHTS = ObjectiveWeights(node=1.0, path=0.0, rotation=0.0)


@dataclass(frozen=True)
class ObjectiveResult:
    """Objective value plus the components and diagnostics behind it."""

    value: float
    node_mean: float
    path_mean: float
    path_max: float
    rotation_pressure: float
    node_scores: Mapping[str, float]
    path_risks: tuple[float, ...]
    paths: tuple[tuple[str, ...], ...]

    def as_dict(self, *, include_detail: bool = True) -> dict[str, object]:
        out: dict[str, object] = {
            "objective": round(self.value, 6),
            "node_mean": round(self.node_mean, 6),
            "path_mean": round(self.path_mean, 6),
            "path_max": round(self.path_max, 6),
            "rotation_pressure": round(self.rotation_pressure, 6),
        }
        if include_detail:
            out["node_scores"] = {k: round(v, 6) for k, v in self.node_scores.items()}
            out["path_risks"] = [round(p, 6) for p in self.path_risks]
            out["paths"] = [list(p) for p in self.paths]
        return out


def node_risk(
    asset: AssetModel,
    qf_override: float | None = None,
    *,
    strict: bool = False,
) -> float:
    """Modeled standalone risk of one asset, in ``[0, 100]``.

    ``exposure x sensitivity x longevity x quantum susceptibility x dependency``.
    Every factor is in ``(0, 1]``, so the product is monotone non-decreasing in
    each — in particular in the quantum factor, which is what migration changes.
    """
    q = quantum_factor(asset.algorithm, strict=strict) if qf_override is None else qf_override
    longevity = min(1.0, max(0.1, asset.data_lifetime_years / LONGEVITY_HORIZON_YEARS))
    dependency = min(1.0, 0.5 + 0.1 * max(0, asset.dependency_count - 1))
    return min(100.0, 100.0 * asset.exposure * asset.sensitivity * longevity * q * dependency)


def path_risk(
    path: Sequence[str],
    node_scores: Mapping[str, float],
    reliability: Mapping[tuple[str, str], float],
) -> float:
    """Risk that a chain from an entrypoint to a target is traversable.

    Each node score is read as a bounded scenario probability that the node is
    compromisable, and the chain survives only if every hop fails. This is a
    model construct: node scores are not calibrated probabilities, and hops are
    treated as independent, which real infrastructure is not.

    Unlike 0.3, the entrypoint's own score is included — it is the first thing an
    adversary must defeat, and excluding it made entrypoint migrations invisible
    to the path term.
    """
    if len(path) < 2:
        return 0.0
    survive = 1.0 - _as_probability(node_scores.get(path[0], 0.0))
    for u, v in zip(path, path[1:], strict=False):
        p_node = _as_probability(node_scores.get(v, 0.0))
        rel = max(0.0, min(1.0, reliability.get((u, v), 1.0)))
        survive *= 1.0 - p_node * rel
    return 100.0 * (1.0 - survive)


def objective(
    assets: Sequence[AssetModel],
    edges: Sequence[EdgeModel],
    entrypoints: Sequence[str] = (),
    targets: Sequence[str] = (),
    *,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    qf_overrides: Mapping[str, float] | None = None,
    path_set: PathSet | None = None,
    strict: bool = False,
) -> ObjectiveResult:
    """Aggregate system objective in ``[0, 100]``; lower is better.

    Pass ``path_set`` to reuse a previously enumerated set. The graph is fixed
    across every candidate migration plan, so the optimizer enumerates once and
    threads the result through, rather than re-running a DFS per subset.
    """
    qf_overrides = qf_overrides or {}
    scores = {
        a.name: node_risk(a, qf_overrides.get(a.name), strict=strict) for a in assets
    }

    if path_set is None:
        path_set = enumerate_paths(edges, entrypoints, targets)

    risks = tuple(path_risk(p, scores, path_set.reliability) for p in path_set.paths)

    # Membership depends only on rotation_hours, never on the scores, so this
    # average cannot jump when an asset's risk falls. See the module docstring.
    rotation_terms = [
        scores[a.name] * min(1.0, a.rotation_hours / ROTATION_REFERENCE_HOURS)
        for a in assets
        if a.rotation_hours > 0
    ]

    node_mean = _mean(scores.values())
    path_mean = _mean(risks)
    path_max = max(risks) if risks else 0.0
    rotation_pressure = _mean(rotation_terms)

    w = weights.normalized()
    value = w.node * node_mean + w.path * path_mean + w.rotation * rotation_pressure

    return ObjectiveResult(
        value=value,
        node_mean=node_mean,
        path_mean=path_mean,
        path_max=path_max,
        rotation_pressure=rotation_pressure,
        node_scores=scores,
        path_risks=risks,
        paths=path_set.paths,
    )


def apply_migration(
    assets: Sequence[AssetModel],
    selected: Sequence[str] | frozenset[str],
    replacements: Mapping[str, str],
) -> tuple[AssetModel, ...]:
    """Return the inventory with ``selected`` assets moved to their replacement.

    Raises rather than skipping when a selected asset has no defined replacement:
    a plan that names an unmigratable asset is a bug in the caller, and 0.3's
    silent ``KeyError``-or-skip behaviour differed between call sites.
    """
    chosen = frozenset(selected)
    unknown = chosen - {a.name for a in assets}
    if unknown:
        raise KeyError(f"migration plan names unknown assets: {sorted(unknown)}")
    out = []
    for a in assets:
        if a.name in chosen:
            if a.algorithm not in replacements:
                raise KeyError(
                    f"no replacement defined for {a.name!r} running {a.algorithm!r}"
                )
            out.append(replace(a, algorithm=replacements[a.algorithm]))
        else:
            out.append(a)
    return tuple(out)


def _as_probability(score: float) -> float:
    # Bounded below 1 so a single node can never make a chain certain.
    return max(0.0, min(0.99, score / 100.0))


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
