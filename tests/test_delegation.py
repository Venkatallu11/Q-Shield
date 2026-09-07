"""Trust inheritance -- the crypto-agility trap.

A certificate is no more trustworthy than the authority that issued it, so an
asset's effective risk is the worst of its own and what it inherits. Q-SHIELD had
one edge type through 0.4 and could not express this, which is why a planner
using that model spends its budget on leaves.
"""

import pytest

from qshield.model import (
    AssetModel,
    apply_migration,
    effective_node_risk,
    objective,
)
from qshield.optimizer import MigrationProblem, exhaustive
from qshield.paths import EdgeKind, EdgeModel, delegation_parents, enumerate_paths
from qshield.threat import ThreatClass


def ca(name, validity, cost, exposure=0.4):
    return AssetModel(
        name, "ECDSA", sensitivity=0.9, exposure=exposure, data_lifetime_years=15,
        dependency_count=4, rotation_hours=24.0, migration_cost=cost,
        threat_class=ThreatClass.AUTHENTICATION, credential_validity_years=validity,
    )


HIERARCHY_EDGES = (
    EdgeModel("root", "intermediate", 1.0, EdgeKind.DELEGATION),
    EdgeModel("intermediate", "leaf", 1.0, EdgeKind.DELEGATION),
)


def test_delegation_edges_are_not_traversed_as_attack_steps():
    """Trust is dominance, not a hop: it is applied by the trust closure, not by
    walking it as a step in a chain."""
    edges = [*HIERARCHY_EDGES, EdgeModel("leaf", "app", 0.9)]
    assert enumerate_paths(edges, ["root"], ["app"]).paths == ()
    assert enumerate_paths(edges, ["leaf"], ["app"]).paths == (("leaf", "app"),)


def test_delegation_parents_indexes_by_target():
    parents = delegation_parents(HIERARCHY_EDGES)
    assert parents == {"intermediate": [("root", 1.0)], "leaf": [("intermediate", 1.0)]}


def test_risk_propagates_down_the_whole_chain():
    own = {"root": 40.0, "intermediate": 10.0, "leaf": 2.0}
    effective = effective_node_risk(own, HIERARCHY_EDGES)
    assert effective == {"root": 40.0, "intermediate": 40.0, "leaf": 40.0}


def test_inheritance_never_lowers_a_nodes_own_risk():
    own = {"root": 5.0, "intermediate": 10.0, "leaf": 60.0}
    effective = effective_node_risk(own, HIERARCHY_EDGES)
    assert effective["leaf"] == 60.0
    assert effective["intermediate"] == 10.0


def test_partial_trust_scales_what_is_inherited():
    edges = (EdgeModel("root", "leaf", 0.5, EdgeKind.DELEGATION),)
    assert effective_node_risk({"root": 40.0, "leaf": 1.0}, edges)["leaf"] == pytest.approx(20.0)


def test_strongest_issuer_dominates_when_cross_signed():
    edges = (
        EdgeModel("weak-root", "leaf", 1.0, EdgeKind.DELEGATION),
        EdgeModel("strong-root", "leaf", 1.0, EdgeKind.DELEGATION),
    )
    own = {"weak-root": 5.0, "strong-root": 50.0, "leaf": 1.0}
    assert effective_node_risk(own, edges)["leaf"] == 50.0


def test_cross_signing_cycles_terminate():
    """Real hierarchies cross-sign, so the closure cannot assume a DAG."""
    edges = (
        EdgeModel("a", "b", 1.0, EdgeKind.DELEGATION),
        EdgeModel("b", "a", 1.0, EdgeKind.DELEGATION),
    )
    assert effective_node_risk({"a": 30.0, "b": 5.0}, edges) == {"a": 30.0, "b": 30.0}


def test_no_delegation_edges_leaves_scores_untouched():
    own = {"a": 10.0, "b": 20.0}
    assert effective_node_risk(own, (EdgeModel("a", "b", 0.9),)) == own


def test_migrating_leaves_under_an_unmigrated_root_achieves_nothing():
    """The crypto-agility trap, stated as an assertion.

    Three leaves migrated to ML-DSA while their issuer still signs with ECDSA:
    real spend, zero risk reduction, because their effective risk stays pinned to
    the root."""
    assets = [
        ca("root", 20.0, 3.0, exposure=0.3),
        ca("intermediate", 8.0, 1.5),
        *[ca(f"leaf-{i}", 0.25, 0.5, exposure=0.9) for i in range(3)],
    ]
    edges = [
        EdgeModel("root", "intermediate", 1.0, EdgeKind.DELEGATION),
        *[EdgeModel("intermediate", f"leaf-{i}", 1.0, EdgeKind.DELEGATION) for i in range(3)],
    ]
    replacements = {"ECDSA": "ML-DSA"}
    base = objective(assets, edges).value

    leaves_only = objective(
        apply_migration(assets, [f"leaf-{i}" for i in range(3)], replacements), edges
    ).value
    assert leaves_only == pytest.approx(base), "migrating leaves must buy nothing"

    root_first = objective(apply_migration(assets, ["root"], replacements), edges).value
    assert root_first < base


def test_a_planner_that_sees_delegation_buys_the_anchor():
    assets = (
        ca("root", 20.0, 1.4, exposure=0.3),
        *[ca(f"leaf-{i}", 0.25, 0.5, exposure=0.9) for i in range(3)],
    )
    edges = tuple(
        EdgeModel("root", f"leaf-{i}", 1.0, EdgeKind.DELEGATION) for i in range(3)
    )
    problem = MigrationProblem(assets, edges, (), (), {"ECDSA": "ML-DSA"}, budget=1.5)
    assert exhaustive(problem)[0].selected == ("root",)


def test_inherited_risk_is_reported_separately_from_own_risk():
    assets = [ca("root", 20.0, 1.0, exposure=0.5), ca("leaf", 0.25, 0.5, exposure=0.9)]
    edges = [EdgeModel("root", "leaf", 1.0, EdgeKind.DELEGATION)]
    result = objective(assets, edges)
    assert result.own_scores["leaf"] < result.node_scores["leaf"]
    detail = result.as_dict()
    assert "risk_inherited_from_issuer" in detail
    assert detail["risk_inherited_from_issuer"]["leaf"] > 0
