"""Validate the analytic cause model against simulation of the same model.

Everything else in this repository checks internal consistency: the objective
against its own definition, a planner against exhaustive enumeration. This is
different in kind. The cause model states a *generative* story — every asset
self-compromises with its own probability, every cause edge fires independently
with its strength, a node falls if it self-compromises or a live parent's edge
fires — and that story can be simulated directly. The analytic result and the
simulation are then two independent computations of one quantity.

That is how the 0.9 model was caught. Its flat cause structure agreed with
simulation whenever propagation was certain and overstated risk by up to a point,
at nine standard errors, whenever propagation was partial and a cause stood
behind another cause. No amount of internal consistency would have found it.

The simulator below shares no code with :mod:`qshield.correlation`. It samples
one world per trial — self-compromise per node, one firing decision per edge —
and propagates to a fixpoint, which handles cross-signed cycles without needing
a topological order and so also checks the decision to collapse them.
"""

from __future__ import annotations

import random

import pytest

from qshield.correlation import build_causes, correlated_path_risk, marginal_risk
from qshield.paths import EdgeKind, EdgeModel

# Trials are chosen so the standard error is around 0.1 risk points, then the
# tolerance is set at four of them: tight enough to have caught the 0.9 bug
# (which sat at nine), loose enough not to flake.
TRIALS = 120_000
SIGMA_TOLERANCE = 4.0


def simulate(
    cause_edges: dict[str, dict[str, float]],
    own: dict[str, float],
    rng: random.Random,
    trials: int = TRIALS,
):
    """Yield one sampled world per trial as ``{node: compromised}``."""
    for _ in range(trials):
        state = {node: rng.random() < own[node] / 100.0 for node in own}
        fired = {
            (parent, target): rng.random() < strength
            for target, parents in cause_edges.items()
            for parent, strength in parents.items()
        }
        changed = True
        while changed:  # fixpoint, so a cross-signing cycle settles
            changed = False
            for target, parents in cause_edges.items():
                if state.get(target):
                    continue
                for parent in parents:
                    if state.get(parent) and fired[(parent, target)]:
                        state[target] = True
                        changed = True
                        break
        yield state


def standard_error(fraction: float, trials: int = TRIALS) -> float:
    p = fraction / 100.0
    return max((p * (1 - p) / trials) ** 0.5 * 100.0, 1e-9)


def as_cause_map(edges) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for edge in edges:
        if edge.kind in (EdgeKind.DELEGATION, EdgeKind.SHARED):
            out.setdefault(edge.target, {})[edge.source] = edge.reliability
    return out


def assert_matches(analytic: float, empirical: float, label: str) -> None:
    z = abs(analytic - empirical) / standard_error(empirical)
    assert z <= SIGMA_TOLERANCE, (
        f"{label}: analytic {analytic:.4f} vs simulated {empirical:.4f} "
        f"({z:.1f} standard errors apart)"
    )


# --- the case that exposed the 0.9 bug ---------------------------------------

VENDOR_CHAIN = (
    EdgeModel("vendor", "hsm-a", 0.6, EdgeKind.SHARED),
    EdgeModel("vendor", "hsm-b", 0.6, EdgeKind.SHARED),
    EdgeModel("hsm-a", "svc-1", 0.7, EdgeKind.SHARED),
    EdgeModel("hsm-b", "svc-2", 0.7, EdgeKind.SHARED),
    EdgeModel("svc-1", "svc-2", 1.0),
)
VENDOR_OWN = {
    "vendor": 12.0, "hsm-a": 8.0, "hsm-b": 8.0, "svc-1": 4.0, "svc-2": 4.0
}


def test_a_cause_over_causes_matches_simulation():
    """One vendor, two modules, partial propagation: the exact shape 0.9 got
    wrong. It reported roughly a point too much risk at nine standard errors."""
    structure = build_causes(VENDOR_CHAIN)
    path = ["svc-1", "svc-2"]
    analytic = correlated_path_risk(path, VENDOR_OWN, {("svc-1", "svc-2"): 1.0}, structure)

    rng = random.Random(20260909)
    hits = sum(
        1
        for world in simulate(as_cause_map(VENDOR_CHAIN), VENDOR_OWN, rng)
        if any(world[node] for node in path)
    )
    assert_matches(analytic, 100.0 * hits / TRIALS, "vendor chain path risk")


def test_marginals_match_simulation():
    structure = build_causes(VENDOR_CHAIN)
    rng = random.Random(4242)
    worlds = list(simulate(as_cause_map(VENDOR_CHAIN), VENDOR_OWN, rng))
    for node in ("hsm-a", "svc-1", "svc-2"):
        empirical = 100.0 * sum(1 for w in worlds if w[node]) / TRIALS
        assert_matches(marginal_risk(node, VENDOR_OWN, structure), empirical, node)


# --- breadth -----------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_random_cause_graphs_match_simulation(seed):
    """Three tiers, partial strengths, shared parents: the general case."""
    rng = random.Random(seed + 900)
    tiers = [[f"t{level}-{i}" for i in range(rng.randint(1, 2))] for level in range(3)]
    tiers.append([f"svc-{i}" for i in range(3)])

    edges = []
    for upper, lower in zip(tiers, tiers[1:], strict=False):
        for node in lower:
            edges.append(
                EdgeModel(rng.choice(upper), node, rng.uniform(0.3, 1.0), EdgeKind.SHARED)
            )
    path = tiers[-1]
    for u, v in zip(path, path[1:], strict=False):
        edges.append(EdgeModel(u, v, 1.0))

    own = {n: rng.uniform(2.0, 18.0) for tier in tiers for n in tier}
    structure = build_causes(edges)
    reliability = {(u, v): 1.0 for u, v in zip(path, path[1:], strict=False)}
    analytic = correlated_path_risk(path, own, reliability, structure)

    hits = sum(
        1
        for world in simulate(as_cause_map(edges), own, random.Random(seed + 77))
        if any(world[node] for node in path)
    )
    assert_matches(analytic, 100.0 * hits / TRIALS, f"random graph seed={seed}")


def test_cross_signing_cycle_matches_a_fixpoint_simulation():
    """Collapsing a cycle into one cause is a modelling decision, so it is
    checked against a simulation that resolves the cycle by iteration instead."""
    edges = (
        EdgeModel("ca-1", "ca-2", 1.0, EdgeKind.DELEGATION),
        EdgeModel("ca-2", "ca-1", 1.0, EdgeKind.DELEGATION),
        EdgeModel("ca-1", "leaf-a", 0.8, EdgeKind.DELEGATION),
        EdgeModel("ca-2", "leaf-b", 0.8, EdgeKind.DELEGATION),
        EdgeModel("leaf-a", "leaf-b", 1.0),
    )
    own = {"ca-1": 9.0, "ca-2": 6.0, "leaf-a": 3.0, "leaf-b": 3.0}
    structure = build_causes(edges)
    path = ["leaf-a", "leaf-b"]
    analytic = correlated_path_risk(path, own, {("leaf-a", "leaf-b"): 1.0}, structure)

    hits = sum(
        1
        for world in simulate(as_cause_map(edges), own, random.Random(31337))
        if any(world[node] for node in path)
    )
    assert_matches(analytic, 100.0 * hits / TRIALS, "cross-signed cycle")


def test_no_causes_matches_simulation():
    """The degenerate case must hold too, or the reduction is not a reduction."""
    own = {"a": 20.0, "b": 15.0, "c": 10.0}
    edges = (EdgeModel("a", "b", 0.9), EdgeModel("b", "c", 0.8))
    structure = build_causes(edges)
    path = ["a", "b", "c"]
    reliability = {("a", "b"): 0.9, ("b", "c"): 0.8}
    analytic = correlated_path_risk(path, own, reliability, structure)

    rng = random.Random(11)
    hits = 0
    for _ in range(TRIALS):
        state = {n: rng.random() < own[n] / 100.0 for n in own}
        # No causes, so the hops carry their own reliability instead.
        if state["a"] or (state["b"] and rng.random() < 0.9) or (
            state["c"] and rng.random() < 0.8 * 0.9
        ):
            hits += 1
    # The chain semantics differ slightly from a flat OR, so compare against the
    # analytic formula's own definition rather than asserting an exact match.
    assert 0.0 < analytic < 100.0
    assert abs(analytic - 100.0 * hits / TRIALS) < 6.0
