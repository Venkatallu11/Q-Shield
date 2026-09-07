"""Attack-path enumeration over the asset dependency graph.

Two things motivate pulling this out of ``model.py``:

1. **Correctness.** 0.3 shipped two enumerators (``model.paths_from`` and
   ``graph.reachable_paths``) whose depth guards differed by one
   (``len(p) >= max_depth`` vs ``len(path) > max_depth``), so the same graph
   yielded different path sets depending on which module you called.

2. **Cost.** Migration replaces an asset's *algorithm*; it never adds or removes
   an edge. The path set is therefore invariant across every candidate plan the
   optimizer enumerates, yet 0.3 recomputed it inside ``system_objective`` for
   each of the 2^n subsets. Enumerating once and reusing it removes that factor
   entirely.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

Path = tuple[str, ...]


@dataclass(frozen=True)
class EdgeModel:
    """A directed dependency. ``reliability`` scales how well an adversary who
    holds ``source`` can leverage that position against ``target``."""

    source: str
    target: str
    reliability: float = 1.0


class PathBudgetExceeded(RuntimeError):
    """Raised when enumeration hits ``max_paths``.

    Silently truncating would make the path term depend on graph traversal order
    — an invisible bias in every downstream comparison — so this is an error, not
    a warning.
    """


@dataclass(frozen=True)
class PathSet:
    """Immutable, reusable result of one enumeration over a fixed graph."""

    paths: tuple[Path, ...]
    reliability: dict[tuple[str, str], float]

    def __len__(self) -> int:
        return len(self.paths)

    def with_reliability(self, edges: Sequence[EdgeModel]) -> PathSet:
        """Same paths, re-weighted by ``edges``.

        Monte Carlo perturbs edge reliability but never adds or removes an edge,
        so the enumerated topology is still exact and re-running the DFS per
        trial would be wasted work.
        """
        reliability: dict[tuple[str, str], float] = {}
        for e in edges:
            key = (e.source, e.target)
            reliability[key] = max(reliability.get(key, 0.0), _clamp(e.reliability))
        return PathSet(paths=self.paths, reliability=reliability)

    @property
    def internal_nodes(self) -> frozenset[str]:
        """Nodes strictly between an entrypoint and a target on some path."""
        return frozenset(n for p in self.paths for n in p[1:-1])

    @property
    def covered_nodes(self) -> frozenset[str]:
        return frozenset(n for p in self.paths for n in p)


def enumerate_paths(
    edges: Sequence[EdgeModel],
    entrypoints: Iterable[str],
    targets: Iterable[str],
    *,
    max_depth: int = 12,
    max_paths: int = 100_000,
) -> PathSet:
    """All simple paths from an entrypoint to a target.

    ``max_depth`` bounds the number of *nodes* in a path. A path terminates at
    the first target it reaches: continuing past a target would double-count the
    same compromise chain.
    """
    adjacency: dict[str, list[str]] = {}
    reliability: dict[tuple[str, str], float] = {}
    for e in edges:
        adjacency.setdefault(e.source, []).append(e.target)
        # Parallel edges collapse to the strongest available leverage.
        key = (e.source, e.target)
        reliability[key] = max(reliability.get(key, 0.0), _clamp(e.reliability))

    wanted = set(targets)
    found: list[Path] = []

    # Explicit stack rather than recursion: deep graphs blew the interpreter's
    # recursion limit in 0.3, and the stack makes the depth guard unambiguous.
    for start in dict.fromkeys(entrypoints):  # de-duplicate, preserve order
        stack: list[tuple[str, tuple[str, ...], frozenset[str]]] = [
            (start, (start,), frozenset({start}))
        ]
        while stack:
            node, path, seen = stack.pop()
            if node in wanted and len(path) > 1:
                found.append(path)
                if len(found) > max_paths:
                    raise PathBudgetExceeded(
                        f"path enumeration exceeded max_paths={max_paths}; the graph is "
                        f"too dense for exhaustive path analysis at max_depth={max_depth}"
                    )
                continue  # do not extend through a target
            if len(path) >= max_depth:
                continue
            for nxt in adjacency.get(node, ()):
                if nxt not in seen:
                    stack.append((nxt, path + (nxt,), seen | {nxt}))

    # Deterministic order regardless of stack/iteration order, so that reports
    # and hashes are reproducible across runs and Python versions.
    found.sort()
    return PathSet(paths=tuple(found), reliability=reliability)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
