"""Path enumeration: correctness against a brute-force reference, and the
reuse property the optimizer's performance depends on."""

from itertools import permutations

import pytest

from qshield.paths import EdgeModel, PathBudgetExceeded, enumerate_paths


def brute_force_paths(edges, start, targets, max_depth):
    """Independent reference: all simple node sequences that are valid paths,
    stopping at the first target. Exponential, so only used on tiny graphs."""
    nodes = sorted({n for e in edges for n in (e.source, e.target)})
    allowed = {(e.source, e.target) for e in edges}
    found = set()
    for length in range(2, max_depth + 1):
        for candidate in permutations(nodes, length):
            if candidate[0] != start:
                continue
            if any(n in targets for n in candidate[:-1]):
                continue  # a path terminates at the first target it reaches
            if candidate[-1] not in targets:
                continue
            if all((u, v) in allowed for u, v in zip(candidate, candidate[1:], strict=False)):
                found.add(candidate)
    return found


def test_matches_brute_force_on_a_dense_graph():
    edges = [
        EdgeModel("a", "b"), EdgeModel("a", "c"), EdgeModel("b", "c"),
        EdgeModel("c", "d"), EdgeModel("b", "d"), EdgeModel("a", "d"),
        EdgeModel("d", "b"),
    ]
    result = enumerate_paths(edges, ["a"], ["d"], max_depth=6)
    assert set(result.paths) == brute_force_paths(edges, "a", {"d"}, 6)


def test_paths_terminate_at_the_first_target():
    edges = [EdgeModel("a", "t"), EdgeModel("t", "b"), EdgeModel("b", "t")]
    paths = enumerate_paths(edges, ["a"], ["t"]).paths
    assert paths == (("a", "t"),)


def test_cycles_do_not_diverge():
    edges = [EdgeModel("a", "b"), EdgeModel("b", "a"), EdgeModel("b", "t")]
    assert enumerate_paths(edges, ["a"], ["t"]).paths == (("a", "b", "t"),)


def test_max_depth_bounds_the_node_count():
    edges = [EdgeModel("a", "b"), EdgeModel("b", "c"), EdgeModel("c", "t")]
    assert enumerate_paths(edges, ["a"], ["t"], max_depth=4).paths == (("a", "b", "c", "t"),)
    # 0.3 had two enumerators whose guards differed by one; the bound is the
    # number of nodes in the path, and a path of exactly max_depth is kept.
    assert enumerate_paths(edges, ["a"], ["t"], max_depth=3).paths == ()


def test_unreachable_target_yields_no_paths():
    assert enumerate_paths([EdgeModel("a", "b")], ["a"], ["z"]).paths == ()


def test_multiple_entrypoints_are_all_explored():
    edges = [EdgeModel("a", "t"), EdgeModel("b", "t")]
    assert enumerate_paths(edges, ["a", "b"], ["t"]).paths == (("a", "t"), ("b", "t"))


def test_duplicate_entrypoints_do_not_duplicate_paths():
    edges = [EdgeModel("a", "t")]
    assert enumerate_paths(edges, ["a", "a"], ["t"]).paths == (("a", "t"),)


def test_output_is_deterministic_and_sorted():
    edges = [EdgeModel("a", "b"), EdgeModel("a", "c"), EdgeModel("b", "t"), EdgeModel("c", "t")]
    first = enumerate_paths(edges, ["a"], ["t"])
    second = enumerate_paths(list(reversed(edges)), ["a"], ["t"])
    assert first.paths == second.paths == tuple(sorted(first.paths))


def test_parallel_edges_collapse_to_the_strongest():
    edges = [EdgeModel("a", "t", 0.2), EdgeModel("a", "t", 0.9)]
    assert enumerate_paths(edges, ["a"], ["t"]).reliability[("a", "t")] == pytest.approx(0.9)


def test_reliability_is_clamped_to_the_unit_interval():
    edges = [EdgeModel("a", "t", 4.0), EdgeModel("t", "b", -1.0)]
    result = enumerate_paths(edges, ["a"], ["t"])
    assert result.reliability[("a", "t")] == 1.0
    assert result.reliability[("t", "b")] == 0.0


def test_path_budget_is_an_error_not_a_silent_truncation():
    """Truncating would make the path term depend on traversal order."""
    nodes = [f"n{i}" for i in range(9)]
    edges = [EdgeModel(u, v) for u in nodes for v in nodes if u != v]
    edges += [EdgeModel(n, "t") for n in nodes]
    with pytest.raises(PathBudgetExceeded):
        enumerate_paths(edges, ["n0"], ["t"], max_depth=10, max_paths=50)


def test_with_reliability_preserves_topology():
    """Monte Carlo perturbs reliabilities but never the graph, so re-weighting is
    exact and the DFS need not be repeated per trial."""
    edges = [EdgeModel("a", "b", 0.5), EdgeModel("b", "t", 0.5)]
    original = enumerate_paths(edges, ["a"], ["t"])
    perturbed = original.with_reliability(
        [EdgeModel("a", "b", 0.9), EdgeModel("b", "t", 0.1)]
    )
    assert perturbed.paths == original.paths
    assert perturbed.reliability[("a", "b")] == pytest.approx(0.9)
    assert original.reliability[("a", "b")] == pytest.approx(0.5)  # unchanged


def test_internal_and_covered_nodes():
    edges = [EdgeModel("a", "b"), EdgeModel("b", "t")]
    result = enumerate_paths(edges, ["a"], ["t"])
    assert result.internal_nodes == frozenset({"b"})
    assert result.covered_nodes == frozenset({"a", "b", "t"})
