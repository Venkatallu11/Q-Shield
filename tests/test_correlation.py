"""Common causes and the independence assumption they break.

The defect these tests pin: 0.5's delegation model raised two sibling
certificates to their issuer's risk and then multiplied them through the path
term as independent events. They are the same event.
"""

import pytest

from qshield.correlation import (
    MAX_CAUSE_STATES,
    CauseStructure,
    CorrelationBudgetExceeded,
    build_causes,
    correlated_path_risk,
    marginal_risk,
)
from qshield.model import AssetModel, objective, path_risk
from qshield.paths import EdgeKind, EdgeModel, cause_edges
from qshield.threat import ThreatClass

HIERARCHY = (
    EdgeModel("root", "leaf-a", 1.0, EdgeKind.DELEGATION),
    EdgeModel("root", "leaf-b", 1.0, EdgeKind.DELEGATION),
    EdgeModel("leaf-a", "leaf-b", 1.0),
    EdgeModel("leaf-b", "target", 1.0),
)
RELIABILITY = {("leaf-a", "leaf-b"): 1.0, ("leaf-b", "target"): 1.0}


# --- structure ---------------------------------------------------------------


def test_delegation_and_shared_edges_are_both_causes():
    edges = (
        EdgeModel("ca", "cert", 1.0, EdgeKind.DELEGATION),
        EdgeModel("hsm", "svc", 1.0, EdgeKind.SHARED),
        EdgeModel("a", "b", 1.0, EdgeKind.DEPENDENCY),
    )
    assert {e.source for e in cause_edges(edges)} == {"ca", "hsm"}
    structure = build_causes(edges)
    assert set(structure.causes) == {"ca", "hsm"}
    assert structure.influence["cert"] == {"ca": 1.0}
    assert structure.influence["svc"] == {"hsm": 1.0}


def test_causes_do_not_include_traversal_edges():
    structure = build_causes((EdgeModel("a", "b", 1.0, EdgeKind.DEPENDENCY),))
    assert not structure
    assert structure.causes == ()


def test_influence_accumulates_along_a_chain():
    """A root reaching a leaf through an intermediate influences it by the
    product of the strengths."""
    edges = (
        EdgeModel("root", "mid", 0.5, EdgeKind.DELEGATION),
        EdgeModel("mid", "leaf", 0.5, EdgeKind.DELEGATION),
    )
    structure = build_causes(edges)
    assert structure.influence["leaf"]["mid"] == pytest.approx(0.5)
    assert structure.influence["leaf"]["root"] == pytest.approx(0.25)


def test_cross_signing_cycles_terminate():
    edges = (
        EdgeModel("a", "b", 1.0, EdgeKind.DELEGATION),
        EdgeModel("b", "a", 1.0, EdgeKind.DELEGATION),
    )
    structure = build_causes(edges)
    assert "b" in structure.influence["a"]
    # A node is never its own cause.
    assert "a" not in structure.influence.get("a", {})


def test_causes_for_returns_only_those_touching_the_path():
    structure = build_causes(HIERARCHY)
    assert structure.causes_for(["leaf-a", "leaf-b"]) == ("root",)
    assert structure.causes_for(["target"]) == ()


# --- the defect --------------------------------------------------------------


def test_siblings_on_one_path_are_not_two_independent_events():
    """The bug, stated as a number. Both leaves are pinned to the root, so the
    independent product counts one compromise twice."""
    own = {"root": 16.8, "leaf-a": 1.0, "leaf-b": 1.0, "target": 1.51}
    structure = build_causes(HIERARCHY)
    path = ["leaf-a", "leaf-b", "target"]

    inflated = {n: max(own[n], own["root"]) for n in path}
    independent = path_risk(path, inflated, RELIABILITY)
    conditioned = correlated_path_risk(path, own, RELIABILITY, structure)

    assert independent > conditioned
    assert conditioned == pytest.approx(19.687, abs=0.01)
    # Worked by hand: P(root) * 100 + P(not root) * residual chain risk.
    assert conditioned == pytest.approx(0.168 * 100 + 0.832 * 3.4676, abs=0.05)


def test_the_overstatement_grows_with_the_number_of_siblings():
    own = {"root": 20.0, "target": 1.0}
    edges = [EdgeModel("root", "target", 0.0, EdgeKind.DEPENDENCY)]
    previous = None
    for count in (2, 3, 4):
        names = [f"leaf-{i}" for i in range(count)]
        own.update({n: 1.0 for n in names})
        edges = [EdgeModel("root", n, 1.0, EdgeKind.DELEGATION) for n in names]
        structure = build_causes(edges)
        path = names
        reliability = {(a, b): 1.0 for a, b in zip(path, path[1:], strict=False)}
        inflated = {n: max(own[n], own["root"]) for n in path}
        gap = path_risk(path, inflated, reliability) - correlated_path_risk(
            path, own, reliability, structure
        )
        if previous is not None:
            assert gap > previous
        previous = gap


def test_no_causes_reproduces_the_previous_formula_exactly():
    """The correction must be inert where there is nothing to correct."""
    scores = {"a": 40.0, "b": 30.0, "c": 20.0}
    reliability = {("a", "b"): 0.9, ("b", "c"): 0.8}
    empty = CauseStructure()
    assert correlated_path_risk(["a", "b", "c"], scores, reliability, empty) == (
        pytest.approx(path_risk(["a", "b", "c"], scores, reliability))
    )


def test_a_cause_touching_only_one_node_changes_nothing_on_that_path():
    """The correction bites when two dependents of one cause share a path. The
    PKI generator never produces that, which is why its correlation effect is
    measured at exactly zero."""
    edges = (
        EdgeModel("root", "leaf", 1.0, EdgeKind.DELEGATION),
        EdgeModel("leaf", "svc", 1.0),
    )
    structure = build_causes(edges)
    own = {"root": 20.0, "leaf": 5.0, "svc": 3.0}
    reliability = {("leaf", "svc"): 1.0}
    conditioned = correlated_path_risk(["leaf", "svc"], own, reliability, structure)
    marginals = {n: marginal_risk(n, own, structure) for n in own}
    assert conditioned == pytest.approx(
        path_risk(["leaf", "svc"], marginals, reliability), abs=1e-9
    )


# --- marginals ---------------------------------------------------------------


def test_noisy_or_marginal_exceeds_max_dominance():
    """Max-dominance throws away an asset's own risk whenever its issuer
    dominates; the generative model says it can fall either way."""
    own = {"root": 16.8, "leaf": 12.6}
    structure = build_causes((EdgeModel("root", "leaf", 1.0, EdgeKind.DELEGATION),))
    combined = marginal_risk("leaf", own, structure)
    assert combined > max(own["root"], own["leaf"])
    assert combined == pytest.approx(100 * (1 - (1 - 0.126) * (1 - 0.168)), abs=0.01)


def test_partial_strength_scales_the_inherited_part():
    own = {"cause": 50.0, "node": 0.0}
    weak = build_causes((EdgeModel("cause", "node", 0.5, EdgeKind.SHARED),))
    strong = build_causes((EdgeModel("cause", "node", 1.0, EdgeKind.SHARED),))
    assert marginal_risk("node", own, weak) == pytest.approx(25.0)
    assert marginal_risk("node", own, strong) == pytest.approx(50.0)


def test_marginal_is_bounded():
    own = {"cause": 99.0, "node": 99.0}
    structure = build_causes((EdgeModel("cause", "node", 1.0, EdgeKind.SHARED),))
    assert 0.0 <= marginal_risk("node", own, structure) <= 100.0


# --- safety ------------------------------------------------------------------


def test_too_many_interacting_causes_is_an_error_not_an_approximation():
    """Silently approximating would make the path term depend on how dense the
    cause graph happens to be."""
    count = MAX_CAUSE_STATES.bit_length() + 2
    names = [f"svc-{i}" for i in range(3)]
    edges = [
        EdgeModel(f"cause-{c}", name, 1.0, EdgeKind.SHARED)
        for c in range(count)
        for name in names
    ]
    structure = build_causes(edges)
    own = {n: 10.0 for n in names}
    own.update({f"cause-{c}": 10.0 for c in range(count)})
    reliability = {(a, b): 1.0 for a, b in zip(names, names[1:], strict=False)}
    with pytest.raises(CorrelationBudgetExceeded):
        correlated_path_risk(names, own, reliability, structure)


def test_a_trivial_path_carries_no_risk():
    assert correlated_path_risk(["a"], {"a": 99.0}, {}, build_causes(())) == 0.0


# --- wiring ------------------------------------------------------------------


def ca(name, validity, exposure):
    return AssetModel(
        name, "ECDSA", 0.8, exposure, 10, 3, 24.0, 1.0, 0.9,
        threat_class=ThreatClass.AUTHENTICATION, credential_validity_years=validity,
    )


def test_objective_correlated_mode_is_opt_in():
    assets = [ca("root", 20, 0.3), ca("leaf-a", 5, 0.9), ca("leaf-b", 5, 0.9),
              AssetModel("target", "AES-256", 0.9, 0.4, 15, 3, 24.0)]
    plain = objective(assets, HIERARCHY, ["leaf-a"], ["target"])
    correlated = objective(assets, HIERARCHY, ["leaf-a"], ["target"], correlated=True)
    assert plain.value != pytest.approx(correlated.value)
    # Default is unchanged, so every earlier result stays comparable.
    assert objective(assets, HIERARCHY, ["leaf-a"], ["target"]).value == plain.value


def test_shared_substrate_is_never_walked_as_an_attack_step():
    from qshield.paths import enumerate_paths

    edges = [EdgeModel("hsm", "svc", 1.0, EdgeKind.SHARED), EdgeModel("svc", "db", 1.0)]
    assert enumerate_paths(edges, ["hsm"], ["db"]).paths == ()
    assert enumerate_paths(edges, ["svc"], ["db"]).paths == (("svc", "db"),)
