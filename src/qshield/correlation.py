"""Common causes, causes over causes, and the joint distribution they imply.

``METHODOLOGY.md`` carried "attack-path hops are treated as independent" as a
caveat from 0.4. 0.5's delegation work turned it into a defect: two certificates
issued by one authority were raised to that authority's risk and then multiplied
through the path term as separate events, when they are one event. 0.9 fixed that
by conditioning on the state of each cause.

0.9 was itself only half right, and this module is the correction. Its cause
model was **flat**: every cause was treated as an independent source, and a cause
standing behind another cause — one vendor supplying two hardware security
modules, one cloud region hosting two identity providers, one library linked into
two authorities — was folded in by accumulating influence transitively along the
chain. Checked against a Monte Carlo simulation of the same generative model that
turns out to be exact when propagation is certain and **biased upward when it is
partial**, by up to a point of risk at nine standard errors. Two errors, in the
same place, that happen to cancel at strength 1.0:

* the transitive term gave an upstream cause a second, independent shot at a
  node through a channel that had already carried its influence via the
  intermediate cause;
* the state weight treated a cause and its own parent as independent, so
  incompatible states — vendor compromised, its module untouched — kept weight
  they should not have.

The model here is a **noisy-OR Bayesian network on the cause DAG**, which is what
the edges were always trying to describe. Every asset self-compromises with the
probability its own cryptography implies. Every cause edge fires independently
with its strength. A node is compromised if it self-compromises or if any live
parent's edge fires. Path risk is the expectation over the joint state of the
causes, computed by enumerating cause states **in topological order** and using
only **direct** parents, so each channel is counted exactly once:

    P(c compromised | θ) = 1 - (1 - own_c) · prod over direct parents d in θ
                                             of (1 - strength(d, c))

``tests/test_correlation.py`` checks the result against an independent Monte
Carlo of the generative model rather than against itself.

Cycles — cross-signed authorities that vouch for each other — are collapsed into
a single cause: if two authorities mutually certify, compromising either
compromises both, so treating them as one is the semantics, not a shortcut.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from .paths import EdgeKind, EdgeModel

__all__ = [
    "CauseStructure",
    "CorrelationBudgetExceeded",
    "build_causes",
    "correlated_path_risk",
    "marginal_risk",
]

# Enumerating cause states costs 2^k. Real estates have a handful of shared
# substrates behind any one path; a graph exceeding this is telling you the cause
# model is wrong, so it raises rather than silently approximating.
MAX_CAUSE_STATES = 1 << 14


class CorrelationBudgetExceeded(RuntimeError):
    """Too many interacting causes on one path for exact conditioning."""


@dataclass(frozen=True)
class CauseStructure:
    """The cause DAG, with cycles collapsed and an order to evaluate it in.

    ``direct`` is what the probabilities are computed from — one entry per real
    channel, so nothing is counted twice. ``influence`` is the transitive closure
    and is used only to decide *which* causes are relevant to a path.
    """

    direct: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    influence: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    order: tuple[str, ...] = ()
    #: Node -> the collapsed cause it belongs to. Identity for most nodes; a
    #: shared representative for members of a cross-signing cycle.
    group_of: Mapping[str, str] = field(default_factory=dict)
    group_members: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    causes: tuple[str, ...] = ()

    def causes_for(self, nodes: Sequence[str]) -> tuple[str, ...]:
        """Every cause that could reach any of ``nodes``, in evaluation order.

        The full ancestor closure, not just direct parents: a vendor two levels
        up has to be in the enumeration for its module's conditional probability
        to be computable.
        """
        wanted: set[str] = set()
        for node in nodes:
            group = self.group_of.get(node, node)
            wanted.update(self.influence.get(group, {}))
            if group in self.direct or any(
                group in parents for parents in self.direct.values()
            ):
                wanted.add(group)
        # A node on the path is only enumerated when it is itself a cause of
        # something; otherwise it is conditionally independent given the rest.
        sources = {p for parents in self.direct.values() for p in parents}
        wanted &= sources
        return tuple(c for c in self.order if c in wanted)

    def __bool__(self) -> bool:
        return bool(self.direct)


def build_causes(edges: Sequence[EdgeModel]) -> CauseStructure:
    """Derive the cause structure from delegation and shared-resource edges.

    Both kinds express "compromising the source compromises the target directly",
    which is what makes them causes rather than steps. They differ only in what
    they describe: delegation is trust, shared is substrate — one module holding
    several keys, one provider behind several accounts, one library linked into
    several services.
    """
    raw: dict[str, dict[str, float]] = {}
    for edge in edges:
        if edge.kind not in (EdgeKind.DELEGATION, EdgeKind.SHARED):
            continue
        strength = max(0.0, min(1.0, edge.reliability))
        if strength <= 0.0 or edge.source == edge.target:
            continue
        parents = raw.setdefault(edge.target, {})
        parents[edge.source] = max(parents.get(edge.source, 0.0), strength)
    if not raw:
        return CauseStructure()

    group_of, group_members = _collapse_cycles(raw)

    # Re-express the graph over collapsed groups, dropping self-loops created by
    # the collapse and keeping the strongest channel between any two groups.
    direct: dict[str, dict[str, float]] = {}
    for target, parents in raw.items():
        child = group_of[target]
        for source, strength in parents.items():
            parent = group_of[source]
            if parent == child:
                continue  # inside a collapsed cycle: already one cause
            row = direct.setdefault(child, {})
            row[parent] = max(row.get(parent, 0.0), strength)

    order = _topological_order(direct, group_members)
    influence = _transitive_influence(direct, order)
    causes = tuple(
        c for c in order if any(c in parents for parents in direct.values())
    )
    return CauseStructure(
        direct={k: dict(v) for k, v in direct.items()},
        influence=influence,
        order=order,
        group_of=group_of,
        group_members=group_members,
        causes=causes,
    )


def correlated_path_risk(
    path: Sequence[str],
    own_scores: Mapping[str, float],
    reliability: Mapping[tuple[str, str], float],
    structure: CauseStructure,
) -> float:
    """Path risk with shared causes conditioned on rather than multiplied through.

    ``own_scores`` are the assets' *own* risks, before any inheritance: the
    inheritance is what this function computes. Passing post-inheritance scores
    would double-count exactly what it exists to fix.
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

    enumerated = frozenset(relevant)
    total = 0.0
    for weight, live in _enumerate_states(relevant, own_scores, structure):
        if weight <= 0.0:
            continue
        total += weight * _conditional_path_risk(
            path, own_scores, reliability, structure, live, enumerated
        )
    return total


def marginal_risk(
    node: str, own_scores: Mapping[str, float], structure: CauseStructure
) -> float:
    """The node's compromise probability with its causes averaged out, in [0, 100].

    Computed by the same enumeration as the path term rather than by a closed
    form, so the marginal and the joint cannot disagree. 0.5's
    ``effective_node_risk`` took a max of own and inherited risk, which discards
    the asset's own contribution whenever its issuer dominates.
    """
    relevant = structure.causes_for([node])
    if not relevant:
        return 100.0 * _as_probability(own_scores.get(node, 0.0))
    if (1 << len(relevant)) > MAX_CAUSE_STATES:
        raise CorrelationBudgetExceeded(
            f"{len(relevant)} causes behind {node!r} exceeds the enumeration cap"
        )
    enumerated = frozenset(relevant)
    total = 0.0
    for weight, live in _enumerate_states(relevant, own_scores, structure):
        if weight <= 0.0:
            continue
        total += weight * _conditional_probability(
            node, own_scores, structure, live, enumerated
        )
    return 100.0 * total


def _enumerate_states(
    relevant: Sequence[str],
    own_scores: Mapping[str, float],
    structure: CauseStructure,
) -> Iterable[tuple[float, frozenset[str]]]:
    """Every joint state of ``relevant`` with its probability.

    ``relevant`` arrives in topological order, so each cause's conditional
    probability can be read off the parents already fixed by the state. This is
    what 0.9 got wrong: it used each cause's own risk as an unconditional
    marginal, which both mis-weights the states and lets incompatible ones —
    a vendor compromised while its own module is untouched — keep weight.
    """
    for assignment in itertools.product((False, True), repeat=len(relevant)):
        live: set[str] = set()
        weight = 1.0
        for cause, on in zip(relevant, assignment, strict=True):
            probability = _conditional_group_probability(
                cause, own_scores, structure, live
            )
            weight *= probability if on else (1.0 - probability)
            if weight <= 0.0:
                break
            if on:
                live.add(cause)
        if weight > 0.0:
            yield weight, frozenset(live)


def _conditional_group_probability(
    group: str,
    own_scores: Mapping[str, float],
    structure: CauseStructure,
    live: set[str],
) -> float:
    """P(this cause is compromised | the states already fixed).

    A collapsed cycle falls if any of its members does, so its own risk is the
    noisy-OR over the members.
    """
    members = structure.group_members.get(group, (group,))
    intact = 1.0
    for member in members:
        intact *= 1.0 - _as_probability(own_scores.get(member, 0.0))
    for parent, strength in structure.direct.get(group, {}).items():
        if parent in live:
            intact *= 1.0 - max(0.0, min(1.0, strength))
    return 1.0 - intact


def _conditional_path_risk(
    path: Sequence[str],
    own_scores: Mapping[str, float],
    reliability: Mapping[tuple[str, str], float],
    structure: CauseStructure,
    live: frozenset[str],
    enumerated: frozenset[str],
) -> float:
    """Risk given a fixed cause state, where the nodes really are independent."""
    conditional = {
        node: _conditional_probability(node, own_scores, structure, live, enumerated)
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
    live: frozenset[str],
    enumerated: frozenset[str],
) -> float:
    """P(node compromised | cause state), from its **direct** parents only.

    Using the transitive closure here is what biased 0.9: an upstream cause would
    get a second, independent shot at the node through a channel its child had
    already carried.

    ``enumerated`` matters when the node is itself one of the causes being
    conditioned on. Then its state is *decided* by the cause state: absent from
    ``live`` means it is not compromised, full stop. Recomputing its own risk
    there would count its self-compromise twice — once in the state weight and
    once again here — which is what the Monte Carlo check caught.
    """
    group = structure.group_of.get(node, node)
    if group in live:
        return 1.0
    if group in enumerated:
        return 0.0
    intact = 1.0 - _as_probability(own_scores.get(node, 0.0))
    for parent, strength in structure.direct.get(group, {}).items():
        if parent in live:
            intact *= 1.0 - max(0.0, min(1.0, strength))
    return 1.0 - intact


def _collapse_cycles(
    raw: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    """Merge each strongly connected component into one cause.

    Cross-signed authorities vouch for each other, so compromising either
    compromises both; treating the cycle as a single cause is the semantics
    rather than a way of avoiding one.
    """
    children: dict[str, list[str]] = {}
    nodes: set[str] = set()
    for target, parents in raw.items():
        nodes.add(target)
        for source in parents:
            nodes.add(source)
            children.setdefault(source, []).append(target)

    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    components: list[list[str]] = []
    counter = 0

    for root in sorted(nodes):
        if root in index:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node, child_index = work[-1]
            if child_index == 0:
                index[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on_stack.add(node)
            successors = children.get(node, ())
            if child_index < len(successors):
                work[-1] = (node, child_index + 1)
                nxt = successors[child_index]
                if nxt not in index:
                    work.append((nxt, 0))
                elif nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            else:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[node])
                if low[node] == index[node]:
                    component: list[str] = []
                    while True:
                        member = stack.pop()
                        on_stack.discard(member)
                        component.append(member)
                        if member == node:
                            break
                    components.append(sorted(component))

    group_of: dict[str, str] = {}
    group_members: dict[str, tuple[str, ...]] = {}
    for component in components:
        representative = component[0]
        group_members[representative] = tuple(component)
        for member in component:
            group_of[member] = representative
    return group_of, group_members


def _topological_order(
    direct: Mapping[str, Mapping[str, float]],
    group_members: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Parents before children. Acyclic by construction after collapsing."""
    nodes = set(group_members) | set(direct)
    for parents in direct.values():
        nodes.update(parents)
    remaining = {n: set(direct.get(n, {})) for n in nodes}
    order: list[str] = []
    while remaining:
        ready = sorted(n for n, parents in remaining.items() if not parents)
        if not ready:  # pragma: no cover - impossible once cycles are collapsed
            ready = [sorted(remaining)[0]]
        for node in ready:
            order.append(node)
            del remaining[node]
        for parents in remaining.values():
            parents.difference_update(ready)
    return tuple(order)


def _transitive_influence(
    direct: Mapping[str, Mapping[str, float]], order: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Ancestor sets, used only to decide which causes a path is exposed to."""
    influence: dict[str, dict[str, float]] = {}
    for node in order:
        accumulated: dict[str, float] = {}
        for parent, strength in direct.get(node, {}).items():
            accumulated[parent] = max(accumulated.get(parent, 0.0), strength)
            for grandparent, upper in influence.get(parent, {}).items():
                combined = strength * upper
                accumulated[grandparent] = max(
                    accumulated.get(grandparent, 0.0), combined
                )
        if accumulated:
            influence[node] = accumulated
    return influence


def _independent_path_risk(
    path: Sequence[str],
    scores: Mapping[str, float],
    reliability: Mapping[tuple[str, str], float],
) -> float:
    """The no-cause formula, kept so that case is provably unchanged."""
    survive = 1.0 - _as_probability(scores.get(path[0], 0.0))
    for u, v in zip(path, path[1:], strict=False):
        rel = max(0.0, min(1.0, reliability.get((u, v), 1.0)))
        survive *= 1.0 - _as_probability(scores.get(v, 0.0)) * rel
    return 100.0 * (1.0 - survive)


def _as_probability(score: float) -> float:
    return max(0.0, min(0.99, score / 100.0))
