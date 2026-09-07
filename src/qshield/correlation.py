"""Common causes, and the independence assumption they break.

``METHODOLOGY.md`` has admitted since 0.4 that attack-path hops are treated as
independent while real infrastructure shares platforms, libraries and operators.
That was filed as a caveat. It is not a caveat; it is a defect, and 0.5's
delegation work made it worse rather than better.

Here is the concrete failure. Two leaf certificates issued by the same authority
sit on one attack path. :func:`qshield.model.effective_node_risk` correctly
raises both to the issuer's risk — forge the root and you forge both — and then
:func:`qshield.model.path_risk` multiplies their probabilities as if they were
independent events. They are not independent; they are *the same event*. On the
worked case in ``tests/test_correlation.py`` that inflates the path risk from
16.8 to 31.8, an 89% overstatement, and the overstatement grows with the number
of siblings on the path.

The fix is to stop composing marginals and compose the generative model instead.
Each asset carries an **own** compromise probability from its own cryptography.
Each cause — an issuing authority, a shared HSM, a shared identity provider, a
cloud account, a crypto library — compromises its dependents with a propagation
strength. Path risk is then the expectation over the joint state of the causes:

    P(path at risk) = sum over cause states θ of
                      P(θ) · [1 - prod over v on path of (1 - p_v(θ)·rel_v)]

    p_v(θ) = 1 - (1 - own_v) · prod over causes c of v in θ of (1 - strength(c, v))

Inside a single state θ the nodes really are conditionally independent, which is
the whole point of conditioning: the correlation lives entirely in θ. With no
causes this collapses **exactly** to the previous formula, which is asserted by
a test rather than assumed.

Two modelling commitments, stated rather than buried:

* Causes are treated as independent of each other. Two separate HSMs failing for
  unrelated reasons is the intended reading; a shared supplier behind both is not
  modelled, and would need a cause over the causes.
* Propagation along a chain of causes multiplies the strengths, so a root that
  reaches a leaf through an intermediate at strength 1.0 each propagates at 1.0.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .paths import EdgeKind, EdgeModel

__all__ = [
    "CauseStructure",
    "CorrelationBudgetExceeded",
    "build_causes",
    "correlated_path_risk",
]

# Enumerating cause states costs 2^k. Real estates have a handful of shared
# substrates per path; a graph that exceeds this is telling you the cause model
# is wrong, so it raises rather than silently approximating.
MAX_CAUSE_STATES = 1 << 14


class CorrelationBudgetExceeded(RuntimeError):
    """Too many interacting causes on one path for exact conditioning."""


@dataclass(frozen=True)
class CauseStructure:
    """Which assets drive which, and how strongly.

    ``influence[v][c]`` is the probability that cause ``c`` being compromised
    also compromises ``v``, accumulated along chains of causes.
    """

    influence: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    causes: tuple[str, ...] = ()

    def causes_for(self, nodes: Sequence[str]) -> tuple[str, ...]:
        """The causes that touch any of ``nodes``, in deterministic order."""
        touching = {
            cause
            for node in nodes
            for cause in self.influence.get(node, {})
        }
        return tuple(c for c in self.causes if c in touching)

    def __bool__(self) -> bool:
        return bool(self.influence)


def build_causes(
    edges: Sequence[EdgeModel], *, max_depth: int = 16
) -> CauseStructure:
    """Derive the cause structure from delegation and shared-resource edges.

    Both edge kinds express "compromising the source compromises the target
    directly", which is what makes them causes rather than steps. They differ
    only in what they describe: delegation is trust (a CA signing a certificate),
    shared is substrate (an HSM holding several keys, an identity provider behind
    several accounts, one library linked into several services).
    """
    direct: dict[str, dict[str, float]] = {}
    for edge in edges:
        if edge.kind in (EdgeKind.DELEGATION, EdgeKind.SHARED):
            strength = max(0.0, min(1.0, edge.reliability))
            if strength > 0:
                direct.setdefault(edge.target, {})[edge.source] = max(
                    direct.get(edge.target, {}).get(edge.source, 0.0), strength
                )
    if not direct:
        return CauseStructure()

    # Accumulate along chains: a root reaching a leaf through an intermediate
    # influences it by the product of the two strengths. Iterated to a fixpoint
    # so cross-signed cycles terminate; strengths only rise and are bounded by 1.
    influence: dict[str, dict[str, float]] = {
        node: dict(parents) for node, parents in direct.items()
    }
    for _ in range(max_depth):
        changed = False
        for node, parents in list(influence.items()):
            for parent, strength in list(parents.items()):
                for grandparent, upper in direct.get(parent, {}).items():
                    if grandparent == node:
                        continue  # a cycle back onto the node itself is not a cause
                    combined = strength * upper
                    if combined > influence[node].get(grandparent, 0.0) + 1e-12:
                        influence[node][grandparent] = combined
                        changed = True
        if not changed:
            break

    causes = sorted({c for parents in influence.values() for c in parents})
    return CauseStructure(
        influence={node: dict(parents) for node, parents in influence.items()},
        causes=tuple(causes),
    )


def correlated_path_risk(
    path: Sequence[str],
    own_scores: Mapping[str, float],
    reliability: Mapping[tuple[str, str], float],
    structure: CauseStructure,
) -> float:
    """Path risk with shared causes conditioned on rather than multiplied through.

    ``own_scores`` are the assets' *own* risks — before any inheritance — because
    the inheritance is what this function is computing properly. Passing
    post-inheritance scores would double-count exactly what it exists to fix.
    """
    if len(path) < 2:
        return 0.0

    relevant = structure.causes_for(path)
    if not relevant:
        return _independent_path_risk(path, own_scores, reliability)

    if (1 << len(relevant)) > MAX_CAUSE_STATES:
        raise CorrelationBudgetExceeded(
            f"{len(relevant)} interacting causes on one path needs "
            f"2^{len(relevant)} states, above the cap of {MAX_CAUSE_STATES}; "
            "the cause model is too dense for exact conditioning"
        )

    total = 0.0
    for state in itertools.product((False, True), repeat=len(relevant)):
        compromised = {c for c, on in zip(relevant, state, strict=True) if on}
        weight = 1.0
        for cause, on in zip(relevant, state, strict=True):
            pi = _as_probability(own_scores.get(cause, 0.0))
            weight *= pi if on else (1.0 - pi)
        if weight <= 0.0:
            continue
        total += weight * _conditional_path_risk(
            path, own_scores, reliability, structure, compromised
        )
    return total


def _conditional_path_risk(
    path: Sequence[str],
    own_scores: Mapping[str, float],
    reliability: Mapping[tuple[str, str], float],
    structure: CauseStructure,
    compromised: set[str],
) -> float:
    """Risk given a fixed cause state, where the nodes *are* independent."""
    conditional = {
        node: _conditional_probability(node, own_scores, structure, compromised)
        for node in path
    }
    survive = 1.0 - conditional[path[0]]
    for u, v in zip(path, path[1:], strict=False):
        rel = max(0.0, min(1.0, reliability.get((u, v), 1.0)))
        survive *= 1.0 - conditional[v] * rel
    return 100.0 * (1.0 - survive)


def _conditional_probability(
    node: str,
    own_scores: Mapping[str, float],
    structure: CauseStructure,
    compromised: set[str],
) -> float:
    """P(node compromised | this cause state): its own risk, or any live cause."""
    intact = 1.0 - _as_probability(own_scores.get(node, 0.0))
    for cause, strength in structure.influence.get(node, {}).items():
        if cause in compromised:
            intact *= 1.0 - max(0.0, min(1.0, strength))
    return 1.0 - intact


def marginal_risk(
    node: str, own_scores: Mapping[str, float], structure: CauseStructure
) -> float:
    """The node's compromise probability with its causes averaged out, in [0, 100].

    A noisy-OR over the asset's own risk and each cause firing, which is what the
    generative model implies. 0.5's ``effective_node_risk`` used a max instead;
    the two agree closely when one term dominates and the noisy-OR is the one
    consistent with the path computation above.
    """
    intact = 1.0 - _as_probability(own_scores.get(node, 0.0))
    for cause, strength in structure.influence.get(node, {}).items():
        pi = _as_probability(own_scores.get(cause, 0.0))
        intact *= 1.0 - pi * max(0.0, min(1.0, strength))
    return 100.0 * (1.0 - intact)


def _independent_path_risk(
    path: Sequence[str],
    scores: Mapping[str, float],
    reliability: Mapping[tuple[str, str], float],
) -> float:
    """The pre-0.9 formula, kept so the no-cause case is provably unchanged."""
    survive = 1.0 - _as_probability(scores.get(path[0], 0.0))
    for u, v in zip(path, path[1:], strict=False):
        rel = max(0.0, min(1.0, reliability.get((u, v), 1.0)))
        survive *= 1.0 - _as_probability(scores.get(v, 0.0)) * rel
    return 100.0 * (1.0 - survive)


def _as_probability(score: float) -> float:
    return max(0.0, min(0.99, score / 100.0))
