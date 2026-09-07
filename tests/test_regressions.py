"""Each test here pins one defect found in Q-SHIELD 0.3.

They are grouped separately from the unit tests so the failure message names the
original problem rather than a module. Numbers in the docstrings were measured
against the 0.3 source before it was replaced.
"""

import pytest

from qshield.algorithms import UNKNOWN_QUANTUM_FACTOR, quantum_factor
from qshield.experiments.generalization import weight_grid
from qshield.experiments.generator import make_instances
from qshield.model import AssetModel, ObjectiveWeights, apply_migration, objective, path_risk
from qshield.optimizer import exhaustive, greedy_flat
from qshield.paths import EdgeModel, enumerate_paths


def test_r1_migration_cannot_increase_the_objective():
    """0.3: the rotation term averaged only over assets scoring above 10, so
    migrating 'a' moved 18.838393 -> 19.329375 — the objective *rose* after a
    strict RSA-2048 -> ML-KEM replacement."""
    inventory = [
        AssetModel("a", "RSA-2048", 0.50, 0.44, 20, 1, 4.0),
        AssetModel("b", "RSA-2048", 0.95, 0.95, 20, 1, 168.0),
    ]
    before = objective(inventory, []).value
    after = objective(apply_migration(inventory, ["a"], {"RSA-2048": "ML-KEM"}), []).value
    assert after < before, "rotation-pressure gate reintroduced"


def test_r2_entrypoint_risk_reaches_the_path_term():
    """0.3: path_risk read only the target of each hop, so the first node on a
    path contributed nothing and both calls below returned 50.0."""
    edges = {("entry", "db"): 1.0}
    assert path_risk(["entry", "db"], {"entry": 99.0, "db": 50.0}, edges) > path_risk(
        ["entry", "db"], {"entry": 0.0, "db": 50.0}, edges
    )


def test_r3_one_algorithm_taxonomy_not_two():
    """0.3: model.QF and engine.quantum_factor disagreed. 'RSA' was 1.0 in one
    and the 0.5 unknown default in the other; the aliases resolved in one only."""
    for alg in ("RSA", "Kyber", "Dilithium", "SPHINCS+", "SHA-256"):
        assert quantum_factor(alg) != UNKNOWN_QUANTUM_FACTOR, alg


def test_r4_weight_grid_has_no_scale_duplicates():
    """0.3 swept 27 points of a 3x3x3 grid, but the objective normalises by the
    weight sum, so (0.2,0.2,0.1) and (0.6,0.6,0.3) are the same weighting."""
    grid = weight_grid()
    assert len({w.key() for w in grid}) == len(grid) == 26


def test_r5_path_enumerators_agree_on_the_depth_bound():
    """0.3 shipped two enumerators whose guards differed by one
    (`len(p) >= max_depth` vs `len(path) > max_depth`), so the same graph gave
    different path sets depending on the module called."""
    edges = [EdgeModel("a", "b"), EdgeModel("b", "c"), EdgeModel("c", "t")]
    assert len(enumerate_paths(edges, ["a"], ["t"], max_depth=4).paths) == 1
    assert len(enumerate_paths(edges, ["a"], ["t"], max_depth=3).paths) == 0


def test_r6_the_null_plan_is_a_candidate():
    """0.3's optimizer looped `range(1, len(candidates)+1)` and so could never
    return 'migrate nothing', while its duplicate in weight_sensitivity could."""
    problem = make_instances(1, 42)[0]
    assert () in {names for names, _ in problem.feasible_plans()}


def test_r7_the_headline_comparison_is_structurally_determined():
    """0.3 reported 483 wins, 0 losses over 1000 instances as evidence. Zero
    losses was guaranteed: the heuristic's plan is inside the exhaustive search
    space. This asserts the guarantee so the claim is not revived."""
    for seed in range(20):
        problem = make_instances(1, seed + 700)[0]
        heuristic = greedy_flat(problem)
        best, _ = exhaustive(problem)
        assert heuristic.selected in {names for names, _ in problem.feasible_plans()}
        assert best.objective <= heuristic.objective + 1e-9


def test_r8_a_case_file_with_dangling_edges_is_rejected():
    """0.3 built the graph without checking that edge endpoints were assets, so a
    typo produced a node with no score that silently contributed 0.0 risk."""
    from qshield.experiments.benchmark import problem_from_dict

    with pytest.raises(ValueError, match="not in the inventory"):
        problem_from_dict(
            {
                "assets": [{"name": "a", "algorithm": "RSA-2048"}],
                "edges": [{"source": "a", "target": "typo"}],
                "entrypoints": ["a"],
                "targets": ["a"],
                "replacements": {},
                "budget": 1,
            }
        )


def test_r9_weights_are_honoured_by_the_only_optimizer():
    """0.3's `optimizer.enumerate_migrations` ignored weights entirely, so the
    weight sweep had to re-implement the optimizer inline to pass them through —
    and the two copies then diverged."""
    problem = make_instances(1, 43)[0]
    node_heavy, _ = exhaustive(problem, weights=ObjectiveWeights(1.0, 0.0, 0.0))
    path_heavy, _ = exhaustive(problem, weights=ObjectiveWeights(0.0, 1.0, 0.0))
    assert node_heavy.objective != pytest.approx(path_heavy.objective)


def test_r10_percentiles_are_symmetric():
    """0.3 used int(.05*n)-1 for p05 and int(.95*n) for p95 on the same list."""
    from qshield.uncertainty import quantiles

    q = quantiles([float(x) for x in range(101)])
    assert q["p05"] == pytest.approx(100.0 - q["p95"])
